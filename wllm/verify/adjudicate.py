"""Token-disagreement adjudication (wBench level B, verifier law 2).

An autoregressive token disagreement between a reference run and a
candidate run is not automatically a divergence.  When two token ids sit
at (numerically) the same logit, which one greedy decoding emits is
**arbitration** between equally-optimal choices, and arbitration is not a
quality regression.  Divergence is when the candidate picks a token the
reference model does not consider optimal at all.

This module owns that decision so that no benchmark has to reimplement
it.  Two rules are enforced.

**Rule A — epsilon-optimal set.**  Given the logit row for the disputed
position, let ``m = max(row)``.  The disagreement is benign iff *both*
the reference token and the candidate token sit within ``epsilon`` of
``m``.  This is deliberately stronger than "the two tokens are the top-2
pair": knife edges observed in practice are degenerate beyond two tokens
(a live three-way tie was resolved three different ways by three code
paths), and a top-2 test misclassifies exactly those cases.  The size of
the epsilon-optimal set is reported as evidence.

**Rule B — dual-path consistency.**  The natural way to measure the logit
row is a teacher-forced *prefill*: replay the reference prefix in one
forward pass and read the last position.  But generation does not happen
that way — it happens on the *decode* path, incrementally, against a KV
cache.  In bf16 those two paths are not bitwise identical, so a knife
edge measured at prefill need not be the knife edge that actually
occurred during decoding.  Therefore, when both rows are available, the
verdict is only the shared verdict of the two paths; **if the paths
disagree the outcome is ``undecidable``, never a pass**.  A tie that
flips depending on how it is measured has not been proven to be a tie.

``undecidable`` is a first-class outcome, distinct from
``real_divergence``: it says "this measurement cannot certify anything",
which is the honest verdict when the evidence is self-contradictory,
degenerate, or absent.  Callers must treat it as a refusal.

The core (:func:`first_divergence`, :func:`adjudicate_row`,
:func:`adjudicate`) is backend-free: it takes plain sequences of floats
(python lists, tuples, or anything with ``tolist()`` such as numpy or
torch tensors) and runs on CPU with no torch import.  The torch-backed
adapters at the bottom of this file import torch lazily *inside* the
function bodies, so importing this module never drags in a GPU stack.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- vocabulary

#: Default epsilon.  bf16 logits around |x| ~ 10 have a representable
#: step near 6e-2 before accumulation; 1e-3 is a deliberately tight band
#: that only admits genuinely coincident logits (the field cases observed
#: so far had a gap of exactly 0.0).
EPSILON_DEFAULT = 1e-3

PATH_PREFILL = "prefill"
PATH_DECODE = "decode"

#: Adjudication verdicts.
BENIGN_TIE = "benign_tie"
REAL_DIVERGENCE = "real_divergence"
UNDECIDABLE = "undecidable"

#: Divergence-search outcomes.
IDENTICAL = "identical"
TOKEN_MISMATCH = "token_mismatch"
LENGTH_MISMATCH = "length_mismatch"

#: How many leading singleton dimensions a logit row may carry, so that
#: ``[[...]]`` (batch of one) and ``[[[...]]]`` are accepted as rows.
_MAX_SQUEEZE = 2


# ------------------------------------------------------------------- results

def _jsonable(value: float | None) -> float | None:
    """Non-finite floats are not valid JSON — publish them as null."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class PathEvidence:
    """What one measurement path says about one disputed position."""

    path: str
    verdict: str
    reason: str
    vocab_size: int
    reference_gap: float | None = None
    candidate_gap: float | None = None
    max_logit: float | None = None
    argmax_token: int | None = None
    optimal_set_size: int | None = None

    @property
    def decisive_gap(self) -> float | None:
        """The larger of the two gaps — the number the rule tests."""
        if self.reference_gap is None:
            return None
        if self.candidate_gap is None:
            return None
        return max(self.reference_gap, self.candidate_gap)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "verdict": self.verdict,
            "reason": self.reason,
            "vocab_size": self.vocab_size,
            "reference_gap": _jsonable(self.reference_gap),
            "candidate_gap": _jsonable(self.candidate_gap),
            "decisive_gap": _jsonable(self.decisive_gap),
            "max_logit": _jsonable(self.max_logit),
            "argmax_token": self.argmax_token,
            "optimal_set_size": self.optimal_set_size,
        }


@dataclass(frozen=True)
class Adjudication:
    """The verdict on one token disagreement, with its evidence.

    ``verdict`` is one of :data:`BENIGN_TIE`, :data:`REAL_DIVERGENCE`,
    :data:`UNDECIDABLE`.  Only :data:`BENIGN_TIE` may be treated as a
    pass; the other two are refusals with different diagnoses.
    """

    verdict: str
    epsilon: float
    reference_token: int
    candidate_token: int
    reason: str
    paths: tuple[str, ...] = ()
    evidence: dict[str, PathEvidence] = field(default_factory=dict)
    position: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_benign(self) -> bool:
        """True only for :data:`BENIGN_TIE` — fail closed everywhere else."""
        return self.verdict == BENIGN_TIE

    @property
    def dual_path(self) -> bool:
        """True when both measurement paths contributed evidence."""
        return len(self.paths) == 2

    def summary(self) -> str:
        where = "?" if self.position is None else str(self.position)
        return (f"[adjudicate] pos {where}: ref={self.reference_token} "
                f"cand={self.candidate_token} -> {self.verdict} "
                f"(eps={self.epsilon:g}, paths={','.join(self.paths)}) "
                f"{self.reason}")

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "epsilon": self.epsilon,
            "position": self.position,
            "reference_token": self.reference_token,
            "candidate_token": self.candidate_token,
            "paths": list(self.paths),
            "dual_path": self.dual_path,
            "evidence": {name: ev.as_dict()
                         for name, ev in self.evidence.items()},
            "reason": self.reason,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Divergence:
    """Where two token sequences first stop agreeing (if they do)."""

    kind: str
    reason: str
    position: int | None = None
    reference_token: int | None = None
    candidate_token: int | None = None

    @property
    def diverged(self) -> bool:
        return self.kind != IDENTICAL

    @property
    def adjudicable(self) -> bool:
        """Only a token-for-token disagreement has a logit row to judge."""
        return self.kind == TOKEN_MISMATCH

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "position": self.position,
            "reference_token": self.reference_token,
            "candidate_token": self.candidate_token,
            "reason": self.reason,
        }


# ------------------------------------------------------------- input hygiene

def _is_seq(obj: Any) -> bool:
    return isinstance(obj, (list, tuple))


def _row_values(logits: Any) -> list[float]:
    """Coerce ``logits`` to a plain 1-D list of floats.

    Accepts python lists/tuples, anything exposing ``tolist()`` (numpy
    and torch tensors), and rows carrying up to :data:`_MAX_SQUEEZE`
    leading singleton dimensions.  Raises ``ValueError`` on anything
    that is not a usable row — an unusable row must never be silently
    reinterpreted, because the caller would then get a verdict about a
    position that was never measured.
    """
    row = logits
    if hasattr(row, "tolist"):
        row = row.tolist()
    for _ in range(_MAX_SQUEEZE):
        if not _is_seq(row):
            break
        if len(row) != 1:
            break
        if not _is_seq(row[0]):
            break
        row = row[0]
    if not _is_seq(row):
        raise ValueError(
            f"logit row must be a 1-D sequence of numbers, got "
            f"{type(logits).__name__}")
    if len(row) == 0:
        raise ValueError("logit row is empty: there is no evidence to judge")
    out: list[float] = []
    for i, value in enumerate(row):
        if _is_seq(value):
            raise ValueError(
                f"logit row is not 1-D: element {i} is a nested sequence")
        if isinstance(value, (str, bytes)):
            raise ValueError(f"logit row element {i} is not numeric: {value!r}")
        try:
            out.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"logit row element {i} is not numeric: {value!r}") from exc
    return out


def _check_epsilon(epsilon: Any) -> float:
    try:
        eps = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"epsilon must be a number, got {epsilon!r}") from exc
    if not math.isfinite(eps):
        raise ValueError(f"epsilon must be finite, got {eps!r}")
    if eps < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {eps!r}")
    return eps


def _check_token(token: Any, vocab_size: int, role: str) -> int:
    try:
        tid = operator.index(token)
    except TypeError as exc:
        raise ValueError(
            f"{role} token id must be an integer, got {token!r}") from exc
    if tid < 0:
        raise ValueError(f"{role} token id {tid} is negative")
    if tid >= vocab_size:
        raise ValueError(
            f"{role} token id {tid} is outside the logit row "
            f"(vocab size {vocab_size})")
    return tid


# ------------------------------------------------------------ divergence scan

def first_divergence(reference: Any, candidate: Any) -> Divergence:
    """Locate the first position at which two token sequences differ.

    Returns a :class:`Divergence` whose ``kind`` is :data:`IDENTICAL`
    (the sequences are equal), :data:`TOKEN_MISMATCH` (a position holds
    different ids — the only case an adjudicator can rule on), or
    :data:`LENGTH_MISMATCH` (one sequence is a strict prefix of the
    other, i.e. an end-of-sequence timing difference rather than a token
    disagreement).
    """
    ref = list(reference)
    cand = list(candidate)
    shared = min(len(ref), len(cand))
    for i in range(shared):
        if ref[i] != cand[i]:
            return Divergence(
                kind=TOKEN_MISMATCH, position=i,
                reference_token=ref[i], candidate_token=cand[i],
                reason=(f"first token disagreement at position {i}: "
                        f"reference {ref[i]} vs candidate {cand[i]}"))
    if len(ref) == len(cand):
        return Divergence(
            kind=IDENTICAL,
            reason=f"token sequences are identical ({len(ref)} tokens)")
    ref_next = None
    if len(ref) > shared:
        ref_next = ref[shared]
    cand_next = None
    if len(cand) > shared:
        cand_next = cand[shared]
    return Divergence(
        kind=LENGTH_MISMATCH, position=shared,
        reference_token=ref_next, candidate_token=cand_next,
        reason=(f"common prefix of {shared} tokens is identical but the "
                f"lengths differ (reference {len(ref)} vs candidate "
                f"{len(cand)}): an end-of-sequence timing difference, not "
                f"a token disagreement"))


# ------------------------------------------------------------ core adjudication

def adjudicate_row(logits: Any, reference_token: Any, candidate_token: Any,
                   *, epsilon: float = EPSILON_DEFAULT,
                   path: str = PATH_PREFILL) -> PathEvidence:
    """Rule A on a single logit row: the epsilon-optimal-set criterion.

    ``logits`` is the next-token logit row measured *at the disputed
    position*.  The disagreement is :data:`BENIGN_TIE` iff both disputed
    token ids sit within ``epsilon`` of the row maximum; otherwise
    :data:`REAL_DIVERGENCE`.  A row containing a non-finite value yields
    :data:`UNDECIDABLE` — a degenerate measurement can never certify a
    tie.  Structurally unusable input raises ``ValueError``.
    """
    eps = _check_epsilon(epsilon)
    row = _row_values(logits)
    vocab = len(row)
    ref = _check_token(reference_token, vocab, "reference")
    cand = _check_token(candidate_token, vocab, "candidate")
    if ref == cand:
        raise ValueError(
            f"reference and candidate token are both {ref}: there is no "
            f"disagreement to adjudicate")

    for i, value in enumerate(row):
        if not math.isfinite(value):
            return PathEvidence(
                path=path, verdict=UNDECIDABLE, vocab_size=vocab,
                reason=(f"{path}: logit row holds a non-finite value at "
                        f"index {i} ({value!r}); a degenerate row cannot "
                        f"certify anything"))

    top = max(row)
    ref_gap = top - row[ref]
    cand_gap = top - row[cand]
    decisive = max(ref_gap, cand_gap)
    optimal_set = 0
    for value in row:
        if top - value <= eps:
            optimal_set += 1

    if decisive <= eps:
        verdict = BENIGN_TIE
        reason = (f"{path}: both disputed tokens lie in the epsilon-optimal "
                  f"set (reference gap {ref_gap:.6g}, candidate gap "
                  f"{cand_gap:.6g}, epsilon {eps:.6g}, optimal set holds "
                  f"{optimal_set} of {vocab} tokens)")
    else:
        verdict = REAL_DIVERGENCE
        reason = (f"{path}: decisive gap {decisive:.6g} exceeds epsilon "
                  f"{eps:.6g} (reference gap {ref_gap:.6g}, candidate gap "
                  f"{cand_gap:.6g}) — the candidate token is not among the "
                  f"reference model's optimal choices")
    return PathEvidence(
        path=path, verdict=verdict, reason=reason, vocab_size=vocab,
        reference_gap=ref_gap, candidate_gap=cand_gap, max_logit=top,
        argmax_token=row.index(top), optimal_set_size=optimal_set)


def adjudicate(reference_token: Any, candidate_token: Any, *,
               prefill_logits: Any = None, decode_logits: Any = None,
               epsilon: float = EPSILON_DEFAULT,
               position: int | None = None,
               notes: tuple[str, ...] = ()) -> Adjudication:
    """Rule A on every supplied path, then Rule B across them.

    Supply ``prefill_logits`` (teacher-forced whole-sequence pass) and/or
    ``decode_logits`` (incremental replay through a KV cache) for the
    *same* position.  With one path the verdict is that path's verdict,
    and the result records that the dual-path rule was not exercised.
    With both paths:

    * both say the same thing -> that verdict;
    * they say different things -> :data:`UNDECIDABLE`, carrying both
      gap sets, because a knife edge that moves when you change how you
      measure it has not been proven benign.

    Supplying no logits at all raises ``ValueError``: an adjudicator with
    no evidence must not return a verdict.
    """
    eps = _check_epsilon(epsilon)
    supplied: list[tuple[str, Any]] = []
    if prefill_logits is not None:
        supplied.append((PATH_PREFILL, prefill_logits))
    if decode_logits is not None:
        supplied.append((PATH_DECODE, decode_logits))
    if not supplied:
        raise ValueError(
            "adjudication needs at least one logit row (prefill and/or "
            "decode); refusing to decide on no evidence")

    evidence: dict[str, PathEvidence] = {}
    for name, logits in supplied:
        evidence[name] = adjudicate_row(
            logits, reference_token, candidate_token, epsilon=eps, path=name)
    paths = tuple(name for name, _ in supplied)

    if len(paths) == 1:
        only = evidence[paths[0]]
        verdict = only.verdict
        reason = (f"single-path verdict ({paths[0]} only): the prefill/decode "
                  f"consistency rule was not exercised. {only.reason}")
    else:
        pre = evidence[PATH_PREFILL]
        dec = evidence[PATH_DECODE]
        if pre.verdict == dec.verdict:
            verdict = pre.verdict
            reason = (f"prefill and decode agree ({verdict}). "
                      f"{pre.reason}. {dec.reason}")
        else:
            verdict = UNDECIDABLE
            reason = (f"prefill and decode disagree ({pre.verdict} vs "
                      f"{dec.verdict}): a knife edge that flips between the "
                      f"teacher-forced prefill and the incremental decode "
                      f"path is never promoted. {pre.reason}. {dec.reason}")

    return Adjudication(
        verdict=verdict, epsilon=eps,
        reference_token=operator.index(reference_token),
        candidate_token=operator.index(candidate_token),
        reason=reason, paths=paths, evidence=evidence,
        position=position, notes=tuple(notes))


# ------------------------------------------------------------- torch adapters
#
# Everything below needs a live model.  torch is imported *inside* each
# function so the core above stays importable (and unit-testable) on a
# CPU-only box with no torch installed.  These adapters are deliberately
# thin: they produce logit rows and hand them to the core, which owns
# every decision.

def prefill_logit_row(model: Any, inputs: dict, prefix_token_ids) -> list[float]:
    """Teacher-forced prefill row for the position after ``prefix_token_ids``.

    Runs ONE forward pass over ``inputs["input_ids"] ++ prefix_token_ids``
    (all other entries of ``inputs`` — e.g. image tensors — are passed
    through unchanged, and ``attention_mask`` is regrown to the new
    length), then returns ``logits[0, -1]`` as a python float list.  This
    is the pass the model would make if it saw the whole reference prefix
    at once; it is *not* how the token was actually produced.
    """
    import torch

    ids = inputs["input_ids"]
    prefix = list(prefix_token_ids)
    if prefix:
        ext = torch.tensor([prefix], device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, ext], dim=1)
    fwd = dict(inputs)
    fwd["input_ids"] = ids
    if "attention_mask" in fwd:
        fwd["attention_mask"] = torch.ones_like(ids)
    with torch.inference_mode():
        out = model(**fwd)
    return out.logits[0, -1].float().tolist()


def decode_logit_row(model: Any, inputs: dict, prefix_token_ids) -> list[float]:
    """Incremental-decode row for the position after ``prefix_token_ids``.

    Replays the actual generation path: one forward over the prompt with
    ``use_cache=True``, then one single-token forward per prefix token
    against the returned KV cache, growing ``attention_mask`` by one each
    step.  Returns ``logits[0, -1]`` of the final step — the row greedy
    decoding would have argmaxed to emit the disputed token.

    Raises ``RuntimeError`` if the model returns no KV cache (the replay
    would then silently degenerate into repeated prefills).  Callers are
    expected to treat any failure here as "decode path unavailable" and
    fall back to a single-path, explicitly-labelled verdict.
    """
    import torch

    ids = inputs["input_ids"]
    prefix = list(prefix_token_ids)
    step = dict(inputs)
    if "attention_mask" in step:
        step["attention_mask"] = torch.ones_like(ids)
    step["use_cache"] = True
    with torch.inference_mode():
        out = model(**step)
        past = getattr(out, "past_key_values", None)
        row = out.logits[0, -1]
        attn = step.get("attention_mask")
        for token in prefix:
            if past is None:
                raise RuntimeError(
                    "model returned no KV cache: the decode path cannot be "
                    "replayed incrementally")
            nxt = torch.tensor([[token]], device=ids.device, dtype=ids.dtype)
            kw = {"input_ids": nxt, "past_key_values": past, "use_cache": True}
            if attn is not None:
                grow = torch.ones((1, 1), device=attn.device, dtype=attn.dtype)
                attn = torch.cat([attn, grow], dim=1)
                kw["attention_mask"] = attn
            out = model(**kw)
            past = getattr(out, "past_key_values", None)
            row = out.logits[0, -1]
    return row.float().tolist()


def adjudicate_generation(model: Any, inputs: dict, reference, candidate, *,
                          epsilon: float = EPSILON_DEFAULT) -> Adjudication:
    """Adjudicate the first token disagreement between two generations.

    Finds the first mismatching position, measures the logit row there on
    BOTH paths, and applies Rules A and B.  The prefill row is required;
    if the decode replay fails (older cache API, model without a cache,
    OOM, ...) the failure is recorded in ``notes`` and the verdict falls
    back to the single-path prefill rule — which is exactly the previous,
    weaker behaviour, never a stronger claim.

    Requires an actual token mismatch: identical sequences and pure
    length differences have no disputed logit row and raise
    ``ValueError`` (call :func:`first_divergence` first).
    """
    div = first_divergence(reference, candidate)
    if not div.adjudicable:
        raise ValueError(
            f"nothing to adjudicate: {div.reason}")

    prefix = list(reference)[:div.position]
    notes: list[str] = []
    prefill_row = prefill_logit_row(model, inputs, prefix)
    decode_row = None
    try:
        decode_row = decode_logit_row(model, inputs, prefix)
    except Exception as exc:                                   # noqa: BLE001
        notes.append(
            f"decode-path replay unavailable ({type(exc).__name__}: {exc}); "
            f"the verdict rests on the prefill path alone")
    return adjudicate(
        div.reference_token, div.candidate_token,
        prefill_logits=prefill_row, decode_logits=decode_row,
        epsilon=epsilon, position=div.position, notes=tuple(notes))
