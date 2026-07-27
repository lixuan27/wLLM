"""Safety-guardrail readiness and cost accounting.

A content safety chain is a *component of the request graph*, not a
footnote: it has weights that must actually be on disk, a load cost,
and a per-request latency that a serving claim has to include.  This
module holds the two pieces of that story which are pure enough to be
tested without a GPU:

1. ``check_hf_cache`` — content-level readiness of an HF *cache tree*
   (``<root>/hub/models--org--name/snapshots/<rev>/...``).  Guardrail
   loaders resolve their weights through ``snapshot_download`` and walk
   ``parents[2]/"blobs"``, so a flat directory of files is not enough;
   the cache layout itself is part of the requirement.  Missing repos,
   missing revisions, missing entries and half-written ``.incomplete``
   blobs each become an explicit blocker.  Fail closed: an empty
   requirement list cannot certify anything.

2. ``attribute`` / ``overlap_hypothesis`` — split a measured end-to-end
   delta into the guardrail stages that were instrumented, and say
   honestly how much of it stayed unattributed.  When the split does
   not add up, the attribution is marked incoherent and no downstream
   inference is allowed to rest on it.

``overlap_hypothesis`` deliberately returns a *hypothesis*, never a
claim.  Observing that a guardrail is serialized in front of and behind
generation does not measure what overlapping it would buy; that stays
an open question until someone runs the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# A guardrail overhead at or above this share of the unguarded
# end-to-end latency is worth writing down as a structural question
# rather than accepting as noise.
SIGNIFICANCE_PCT = 10.0
# Above this share of the measured overhead being unexplained by the
# instrumented stages, the split is not trustworthy enough to reason
# from (clock drift, an uninstrumented path, or a bad wrapper).
MAX_UNATTRIBUTED_SHARE = 0.25


# --------------------------------------------------------------- cache
@dataclass(frozen=True)
class CacheRequirement:
    """One repo that must be present in an HF cache tree.

    ``relpaths`` are entries required *inside* the resolved snapshot
    revision; an empty tuple means "the revision must exist" only.
    """

    repo_id: str
    relpaths: tuple[str, ...] = ()
    note: str = ""


@dataclass
class CacheReadiness:
    ready: bool
    checked: int
    blockers: list[str] = field(default_factory=list)
    resolved: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ready": self.ready, "checked": self.checked,
                "blockers": self.blockers, "resolved": self.resolved}


def repo_dirname(repo_id: str) -> str:
    """``org/name`` -> ``models--org--name`` (the HF cache folder)."""
    return "models--" + repo_id.replace("/", "--")


def _resolve_revision(repo_dir: Path) -> tuple[Path | None, str]:
    """Resolve the revision *exactly* as the hub loader does offline.

    ``huggingface_hub`` reads ``refs/<revision>`` verbatim — literally
    ``f.read()``, with no ``strip()`` — and joins the result onto
    ``snapshots/``.  A ref written with a trailing newline therefore
    names a directory that cannot exist, and the loader dies with
    ``LocalEntryNotFoundError``.  There is also no "lone snapshot"
    fallback: without a readable ref the loader has no commit hash at
    all and fails.

    Reproducing both behaviours byte-for-byte is the entire point of
    this function.  A precheck more forgiving than the loader is worse
    than no precheck: it certifies a chain that then cannot construct,
    converting a fast clear failure into an expensive opaque one.

    Returns ``(revision_dir, blocker)``; exactly one is meaningful.
    """
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None, f"no snapshots/ directory under {repo_dir}"
    ref = repo_dir / "refs" / "main"
    if not ref.is_file():
        return None, (f"no refs/main under {repo_dir}; the loader "
                      f"resolves revisions through that file only and "
                      f"has no lone-snapshot fallback")
    raw = ref.read_text(encoding="utf-8", errors="replace")
    if (snaps / raw).is_dir():
        return snaps / raw, ""
    stripped = raw.strip()
    if stripped != raw and (snaps / stripped).is_dir():
        return None, (
            f"refs/main names revision {stripped!r} but the file holds "
            f"{raw!r} — the loader reads the ref verbatim and will look "
            f"for snapshots/{raw!r}, which cannot exist. Rewrite the "
            f"ref with no trailing whitespace.")
    return None, (f"refs/main names revision {stripped!r} but "
                  f"{snaps / stripped} does not exist")


def check_hf_cache(cache_root: str | Path,
                   requirements: list[CacheRequirement]) -> CacheReadiness:
    """Content-level readiness of an HF cache tree.  Fails closed."""
    root = Path(cache_root)
    hub = root / "hub"
    blockers: list[str] = []
    resolved: dict = {}
    if not requirements:
        return CacheReadiness(
            False, 0,
            ["no cache requirements given; an empty expectation cannot "
             "certify that a guardrail can load"])
    if not hub.is_dir():
        return CacheReadiness(
            False, 0,
            [f"HF cache tree missing: {hub} (guardrail loaders resolve "
             f"weights through the cache layout, not a flat directory)"])

    checked = 0
    for req in requirements:
        repo_dir = hub / repo_dirname(req.repo_id)
        where = req.repo_id + (f" ({req.note})" if req.note else "")
        if not repo_dir.is_dir():
            blockers.append(f"missing repo in cache: {where}")
            continue
        rev, problem = _resolve_revision(repo_dir)
        if rev is None:
            blockers.append(f"{where}: {problem}")
            continue
        resolved[req.repo_id] = str(rev)
        for rel in req.relpaths:
            checked += 1
            if not (rev / rel).exists():
                blockers.append(f"{where}: missing snapshot entry {rel}")
        partial = sorted(p.name for p in repo_dir.rglob("*.incomplete"))
        if partial:
            blockers.append(
                f"{where}: {len(partial)} partially downloaded blob(s) "
                f"still present, e.g. {partial[0]}")

    return CacheReadiness(not blockers, checked, blockers, resolved)


# ---------------------------------------------------------- accounting
@dataclass(frozen=True)
class GuardrailTiming:
    """One measured A/B pair, all times in milliseconds.

    ``baseline_ms`` and ``guarded_ms`` are end-to-end request latencies
    of the same prompt/seed/steps with the safety chain off and on.
    The stage fields are the instrumented segments inside the guarded
    request; ``transfer_ms`` is the device movement the pipeline does
    around each check.
    """

    baseline_ms: float
    guarded_ms: float
    text_stage_ms: float = 0.0
    video_stage_ms: float = 0.0
    transfer_ms: float = 0.0


def attribute(t: GuardrailTiming) -> dict:
    """Split the measured overhead; report what stayed unexplained."""
    overhead_ms = t.guarded_ms - t.baseline_ms
    attributed_ms = t.text_stage_ms + t.video_stage_ms
    unattributed_ms = overhead_ms - attributed_ms
    overhead_pct = (100.0 * overhead_ms / t.baseline_ms
                    if t.baseline_ms > 0 else float("nan"))
    share = (abs(unattributed_ms) / abs(overhead_ms)
             if overhead_ms else float("inf"))
    coherent = (overhead_ms > 0) and (share <= MAX_UNATTRIBUTED_SHARE)
    return {
        "baseline_ms": t.baseline_ms,
        "guarded_ms": t.guarded_ms,
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
        "text_stage_ms": t.text_stage_ms,
        "video_stage_ms": t.video_stage_ms,
        "transfer_ms": t.transfer_ms,
        "attributed_ms": attributed_ms,
        "unattributed_ms": unattributed_ms,
        "unattributed_share": share,
        "attribution_coherent": coherent,
        "significant": coherent and overhead_pct >= SIGNIFICANCE_PCT,
    }


def overlap_hypothesis(attribution: dict) -> str | None:
    """A hypothesis about the request graph, or None.

    Returns prose only when the guardrail cost is both significant and
    coherently attributed.  The text is explicitly labelled unmeasured:
    nothing here licenses a speedup claim, and the ceiling it names is
    an upper bound on what an experiment could possibly find, not a
    result.
    """
    if not attribution.get("significant"):
        return None
    text_ms = attribution["text_stage_ms"]
    video_ms = attribution["video_stage_ms"]
    return (
        "HYPOTHESIS (unmeasured): the safety chain is a separable "
        "component of the request graph, executed strictly serially — "
        f"the prompt check ({text_ms:.0f} ms) blocks before the first "
        f"denoise step and the frame check ({video_ms:.0f} ms) blocks "
        f"after decode, together {attribution['overhead_pct']:.0f}% of "
        "unguarded end-to-end. The prompt check depends only on the "
        "prompt and the frame check only on decoded frames, so neither "
        "reads a generation intermediate; overlapping the prompt check "
        "with early denoising and the frame check with a streamed tail "
        "would remove at most that much, which is a ceiling and not a "
        "result. No overlap was implemented or measured here."
    )
