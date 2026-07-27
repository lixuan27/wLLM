"""Token-disagreement adjudication (wBench level B, verifier law 2).

An autoregressive token disagreement between a reference run and a
candidate run is not automatically a divergence.  When two token ids sit
at (numerically) the same logit, which one greedy decoding emits is
**arbitration** between equally-optimal choices, and arbitration is not a
quality regression.  Divergence is when the candidate picks a token the
reference model does not consider optimal at all.

This module owns that decision so that no benchmark has to reimplement
it.  Three rules are enforced.

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

Rule B needs a decode replay that actually works for the model family in
question, and that is a real precondition, not a formality: the replay
is driven through the model's *own* generation input preparation, so a
model that does not expose that contract cannot be dual-path verified.
When the replay is unavailable the adapters raise
:class:`DecodeReplayUnavailable` — typed and described — and the verdict
degrades to the single prefill path with that fact disclosed in
``notes``.  A single-path verdict is the older, weaker rule; it is never
reported as agreement between paths.

**Rule C — generation-level aggregation.**  Rules A and B rule on ONE
position.  A generation usually disagrees at many (a live MoE run
disagreed at 227 of 256 positions), and a verdict about the whole
generation read off a single position is not sound: the first position
can be a benign tie while a later one is a decisive divergence.  So a
bounded, deterministic sample of the disagreeing positions is examined
(see :func:`select_positions`) and aggregated **conservatively**:

* any examined position is a real divergence -> ``real_divergence``;
* else any examined position is undecidable -> ``undecidable``;
* else (every examined position benign) -> ``benign_tie``.

Positions that were not examined are never treated as benign; the count
of unexamined positions is stated in the reason so the residual risk is
visible instead of inferred away.

Rule C also caps the verdict on two whole-generation facts that no
per-position gap can see:

1. *Disagreement fraction.*  A knife-edge tie is a rare coincidence —
   two logits out of a ~150k vocabulary landing within ``epsilon``.  When
   a large fraction of positions disagree, the likelier explanation is
   that the two numerics regimes differ systematically, and in any case
   the sequences have forked: this is a trajectory phenomenon (verifier
   law 1's logic applied to AR decode), not a knife edge.  Above
   ``max_disagreement_fraction`` the best available verdict is capped at
   ``undecidable``.  The default (5%) is a stated convention, not a
   fitted constant; the observed cases sit far to either side of it
   (a historical accepted case at 0.8%, the refused MoE legs at 19–89%),
   so no verdict to date turns on its exact value.  The cap needs
   something to have compounded, so it applies only when more than one
   position disagrees: a single disagreement is a knife edge by
   construction, and a short generation must not be refused merely for
   being short.
2. *Length.*  A candidate that emits a different number of tokens
   stopped somewhere else; that is not token-exact whatever the gaps
   say, so the verdict is capped at ``undecidable``.

*Known scope limit, stated rather than papered over*: a position after
the first is adjudicated against the **reference** prefix, because that
is the only prefix both runs can be asked about.  Once the sequences
fork, the candidate's own conditioning differs, so those positions are
counterfactual checks — they can refuse a candidate but cannot by
themselves prove the candidate's actual trajectory optimal.  That
asymmetry is exactly why the fraction cap exists, and it is disclosed in
the result's ``notes``.

``undecidable`` is a first-class outcome, distinct from
``real_divergence``: it says "this measurement cannot certify anything",
which is the honest verdict when the evidence is self-contradictory,
degenerate, or absent.  Callers must treat it as a refusal.

The core (:func:`first_divergence`, :func:`divergence_positions`,
:func:`select_positions`, :func:`adjudicate_row`, :func:`adjudicate`,
:func:`aggregate_positions`) is backend-free: it takes plain sequences of
floats (python lists, tuples, or anything with ``tolist()`` such as numpy
or torch tensors) and runs on CPU with no torch import.  The torch-backed
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

#: How many disagreeing positions Rule C examines per generation.  Each
#: examined position costs one prefill forward, and the decode rows for
#: the whole sample come from ONE shared incremental replay, so the cost
#: is ``budget + max(position)`` forwards rather than the quadratic
#: ``sum(position)``.  Eight gives head/middle/tail coverage for a
#: few-hundred-token generation at a few seconds on one GPU.
POSITION_BUDGET_DEFAULT = 8

#: Above this fraction of disagreeing positions the disagreement is a
#: trajectory phenomenon rather than a knife edge, and the verdict is
#: capped at ``undecidable`` (see Rule C in the module docstring).  A
#: stated convention, not a fitted constant.
DISAGREEMENT_FRACTION_DEFAULT = 0.05

#: How many leading singleton dimensions a logit row may carry, so that
#: ``[[...]]`` (batch of one) and ``[[[...]]]`` are accepted as rows.
_MAX_SQUEEZE = 2

#: Inputs that must stay aligned one-to-one with ``input_ids``.  This is
#: not a guess: it is the framework's own list of "inputs that should have
#: the same length as input_ids" from its generation input preparation,
#: plus ``attention_mask`` (which spans the whole cache and is grown by
#: the same bookkeeping).  See :func:`plan_prefix_extension`.
PER_POSITION_INPUTS = ("attention_mask", "mm_token_type_ids",
                       "token_type_ids", "position_ids")

#: The value an appended position takes in each per-position input.  Also
#: taken from the framework's own per-step update rather than invented:
#: a generated token is real (mask 1) and is always text, never image or
#: video (``mm_token_type_ids`` 0).  ``position_ids`` is deliberately
#: absent: it is derived, and for multimodal rope it is 3-D and
#: model-specific, so this adapter refuses rather than inventing one.
_APPENDED_POSITION_VALUE = {"attention_mask": 1, "mm_token_type_ids": 0,
                            "token_type_ids": 0}

#: Disclosed on every multi-position verdict — see the module docstring's
#: "known scope limit".
_COUNTERFACTUAL_NOTE = (
    "positions after the first are adjudicated against the REFERENCE "
    "prefix; once the sequences fork the candidate's own conditioning "
    "differs, so those checks can refuse a candidate but cannot by "
    "themselves prove its actual trajectory optimal")


# ---------------------------------------------------------------- exceptions

class AdjudicationError(RuntimeError):
    """Base: the adjudicator could not measure what it needed to rule.

    Always describes the invariant that broke, so a caller's "no evidence
    is not a pass" message tells a reader *which* invariant it was.  A
    bare ``IndexError`` from inside a model is not a diagnosis.
    """


class PositionAccountingError(AdjudicationError):
    """A per-position input could not be aligned to ``input_ids``.

    Raised when an input that must hold one entry per sequence position
    (see :data:`PER_POSITION_INPUTS`) has a length this adapter cannot
    reconcile with the prompt plus the teacher-forced prefix, or when an
    unrecognised input is shaped as though it were per-position.
    """


class DecodeReplayUnavailable(AdjudicationError):
    """The incremental decode path could not be replayed for this model.

    Callers must degrade to a single-path (prefill-only) verdict and
    disclose it — never treat it as agreement between paths.
    """


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
class GenerationVerdict:
    """Rule C: one verdict about a whole generation, with its census.

    ``verdict`` uses the same vocabulary as :class:`Adjudication`.  The
    census fields are the point: a reader must be able to see how much of
    the disagreement was actually looked at, and how much was not.
    """

    verdict: str
    epsilon: float
    reason: str
    positions_compared: int
    positions_disagreeing: int
    positions_examined: tuple[int, ...] = ()
    per_position: tuple[Adjudication, ...] = ()
    max_disagreement_fraction: float = DISAGREEMENT_FRACTION_DEFAULT
    reference_length: int | None = None
    candidate_length: int | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_benign(self) -> bool:
        """True only for :data:`BENIGN_TIE` — fail closed everywhere else."""
        return self.verdict == BENIGN_TIE

    @property
    def positions_unexamined(self) -> int:
        """Disagreeing positions no measurement was made at."""
        return self.positions_disagreeing - len(self.positions_examined)

    @property
    def disagreement_fraction(self) -> float:
        return self.positions_disagreeing / self.positions_compared

    def summary(self) -> str:
        return (f"[adjudicate] generation -> {self.verdict} "
                f"(eps={self.epsilon:g}; examined "
                f"{len(self.positions_examined)} of "
                f"{self.positions_disagreeing} disagreeing positions over "
                f"{self.positions_compared} compared, "
                f"{self.positions_unexamined} unexamined) {self.reason}")

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "epsilon": self.epsilon,
            "positions_compared": self.positions_compared,
            "positions_disagreeing": self.positions_disagreeing,
            "positions_examined": list(self.positions_examined),
            "positions_unexamined": self.positions_unexamined,
            "disagreement_fraction": self.disagreement_fraction,
            "max_disagreement_fraction": self.max_disagreement_fraction,
            "reference_length": self.reference_length,
            "candidate_length": self.candidate_length,
            "per_position": [a.as_dict() for a in self.per_position],
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


def divergence_positions(reference: Any, candidate: Any) -> tuple[int, ...]:
    """Every position at which the two sequences hold different token ids.

    Only the shared prefix is compared; a length difference is reported
    separately (see :func:`first_divergence`) because it is a different
    kind of fact.  Returns positions in increasing order.
    """
    ref = list(reference)
    cand = list(candidate)
    shared = min(len(ref), len(cand))
    return tuple(i for i in range(shared) if ref[i] != cand[i])


def select_positions(positions: Any,
                     budget: int = POSITION_BUDGET_DEFAULT) -> tuple[int, ...]:
    """Bounded, deterministic sample of the disagreeing positions.

    The scheme is **first, last, and an even spread between them**:

    * the *first* disagreement is always examined — it is the only
      position whose adjudication is not confounded by prefix divergence,
      since both runs share an identical history up to it, so it carries
      the strongest evidence available;
    * the *last* is always examined — divergence compounds along a
      generation, so sampling only the head would systematically
      under-detect a trajectory that separates late;
    * the *middle* is covered by an even, arithmetic spread, so the
      sample is reproducible from the inputs alone.  Nobody — agent or
      human — gets to choose which positions are looked at, and a receipt
      can be replayed exactly.

    The same deterministic-spread idiom is used by
    ``scripts/mutation_smoke.py`` to pick mutation sites; this is that
    convention, not new machinery.  Budgets at or above the number of
    disagreements examine all of them.
    """
    seq = tuple(int(p) for p in positions)
    if budget < 1:
        raise ValueError(f"position budget must be >= 1, got {budget}")
    if not seq:
        return ()
    if budget == 1:
        return (seq[0],)                 # the one unconfounded position
    last = len(seq) - 1
    picked = {seq[round(i * last / (budget - 1))] for i in range(budget)}
    return tuple(sorted(picked))


def plan_prefix_extension(companion_lengths: Any, *, prompt_length: int,
                          prefix_length: int) -> dict[str, int]:
    """How to extend per-position inputs for a teacher-forced prefill.

    A teacher-forced prefill appends ``prefix_length`` generated tokens to
    a ``prompt_length`` prompt.  Every input that carries one entry per
    sequence position must be appended to in lockstep; leaving one at
    prompt length is the defect this function exists to prevent.

    **The assumption, stated rather than inferred**: the processor emits
    ``input_ids`` in which any multimodal placeholder is ALREADY expanded
    to one id per embedding position, so ``len(input_ids)`` equals the
    number of KV positions and a companion of that same length is
    per-position.  This holds for the model families in scope (their
    processors emit the placeholder token repeated once per patch), and
    the framework's own generation bookkeeping makes the same assumption
    when it grows these inputs by one per generated token.  If a model
    instead expands placeholders inside ``forward()``, a companion will
    match neither length and this function raises rather than guessing —
    which is the honest outcome, because an offset inferred here would
    silently move the position being adjudicated.

    ``companion_lengths`` maps input name -> current last-dim length.
    Returns ``{name: value_to_append}`` for those needing extension;
    names already at the target length are omitted.  Raises
    :class:`PositionAccountingError` naming the input and both lengths
    for anything it cannot reconcile.
    """
    if prompt_length < 1:
        raise ValueError(f"prompt_length must be >= 1, got {prompt_length}")
    if prefix_length < 0:
        raise ValueError(f"prefix_length must be >= 0, got {prefix_length}")
    target = prompt_length + prefix_length
    plan: dict[str, int] = {}
    for name in sorted(companion_lengths):
        length = int(companion_lengths[name])
        if length == target:
            continue                        # already aligned, nothing to do
        if length != prompt_length:
            raise PositionAccountingError(
                f"per-position input {name!r} has length {length}, which is "
                f"neither the prompt length {prompt_length} nor the "
                f"teacher-forced length {target} (prompt + {prefix_length} "
                f"reference tokens): this adapter cannot tell which "
                f"positions it describes, and guessing would move the "
                f"position being adjudicated")
        if name not in _APPENDED_POSITION_VALUE:
            raise PositionAccountingError(
                f"per-position input {name!r} sits at the prompt length "
                f"{prompt_length} and must grow to {target}, but this "
                f"adapter has no defined value for an appended generated "
                f"token; refusing to invent one")
        plan[name] = _APPENDED_POSITION_VALUE[name]
    return plan


def _check_fraction(value: Any) -> float:
    try:
        frac = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_disagreement_fraction must be a number, got "
            f"{value!r}") from exc
    if not math.isfinite(frac):
        raise ValueError(f"max_disagreement_fraction must be finite, got {frac!r}")
    if frac < 0.0:
        raise ValueError(f"max_disagreement_fraction must be >= 0, got {frac!r}")
    if frac > 1.0:
        raise ValueError(f"max_disagreement_fraction must be <= 1, got {frac!r}")
    return frac


def aggregate_positions(
        adjudications: Any, *, positions_compared: int,
        positions_disagreeing: int,
        epsilon: float = EPSILON_DEFAULT,
        max_disagreement_fraction: float = DISAGREEMENT_FRACTION_DEFAULT,
        reference_length: int | None = None,
        candidate_length: int | None = None,
        notes: tuple[str, ...] = ()) -> GenerationVerdict:
    """Rule C: fold per-position verdicts into one generation verdict.

    Backend-free, so the aggregation rule is unit-testable without a
    model.  Aggregation is conservative (``real_divergence`` beats
    ``undecidable`` beats ``benign_tie``) and then capped by the
    disagreement fraction and by a length difference.  Unexamined
    positions are counted and disclosed, never assumed benign.
    """
    eps = _check_epsilon(epsilon)
    cap = _check_fraction(max_disagreement_fraction)
    per_position = tuple(adjudications)
    if not per_position:
        raise ValueError(
            "no adjudicated positions: a generation verdict must rest on at "
            "least one measured position")
    if positions_compared < 1:
        raise ValueError(
            f"positions_compared must be >= 1, got {positions_compared}")
    if positions_disagreeing > positions_compared:
        raise ValueError(
            f"positions_disagreeing ({positions_disagreeing}) exceeds "
            f"positions_compared ({positions_compared})")
    if len(per_position) > positions_disagreeing:
        raise ValueError(
            f"examined {len(per_position)} positions but only "
            f"{positions_disagreeing} disagree")
    for adj in per_position:
        if adj.position is None:
            raise ValueError(
                "every adjudicated position must carry its index: a census "
                "cannot count positions it cannot name")
        if adj.position >= positions_compared:
            raise ValueError(
                f"adjudicated position {adj.position} is outside the "
                f"{positions_compared} compared positions")

    verdicts = [a.verdict for a in per_position]
    if REAL_DIVERGENCE in verdicts:
        verdict = REAL_DIVERGENCE
    elif UNDECIDABLE in verdicts:
        verdict = UNDECIDABLE
    else:
        verdict = BENIGN_TIE

    examined = tuple(a.position for a in per_position)
    unexamined = positions_disagreeing - len(per_position)
    fraction = positions_disagreeing / positions_compared
    parts = [
        f"examined {len(per_position)} of {positions_disagreeing} "
        f"disagreeing positions ({positions_disagreeing} of "
        f"{positions_compared} compared positions disagree = "
        f"{fraction:.1%}); {unexamined} disagreeing positions were NOT "
        f"examined and are NOT certified benign"]

    if verdict == BENIGN_TIE:
        # the fraction cap is about compounding, so it needs something to
        # have compounded: a SINGLE disagreement is a knife edge by
        # construction (a trajectory fork would have shown up downstream
        # too), and short generations must not be refused just for being
        # short.  Two or more is where "many positions disagree" starts
        # to mean something.
        if positions_disagreeing > 1:
            if fraction > cap:
                verdict = UNDECIDABLE
                parts.append(
                    f"every examined position is a benign tie, but "
                    f"{fraction:.1%} of positions disagree, above the "
                    f"{cap:.1%} budget: at this scale the sequences have "
                    f"forked and the disagreement is a trajectory "
                    f"phenomenon, not a knife edge — per-position tie "
                    f"adjudication cannot license an exact verdict here")
    if verdict == BENIGN_TIE:
        if reference_length != candidate_length:
            verdict = UNDECIDABLE
            parts.append(
                f"every examined position is a benign tie, but the runs "
                f"emitted different token counts (reference "
                f"{reference_length} vs candidate {candidate_length}): a "
                f"generation that stops somewhere else is not token-exact")

    deciding = [a for a in per_position if a.verdict == verdict]
    if not deciding:
        deciding = list(per_position)
    parts.append("deciding evidence: " + " | ".join(
        f"pos {a.position}: {a.reason}" for a in deciding[:3]))

    all_notes = list(notes)
    if positions_disagreeing > 1:
        if _COUNTERFACTUAL_NOTE not in all_notes:
            all_notes.append(_COUNTERFACTUAL_NOTE)

    return GenerationVerdict(
        verdict=verdict, epsilon=eps, reason="; ".join(parts),
        positions_compared=positions_compared,
        positions_disagreeing=positions_disagreeing,
        positions_examined=examined, per_position=per_position,
        max_disagreement_fraction=cap,
        reference_length=reference_length, candidate_length=candidate_length,
        notes=tuple(all_notes))


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

def _single_sequence_length(ids: Any) -> int:
    """Prompt length of a batch-of-one ``input_ids``; refuse otherwise."""
    shape = getattr(ids, "shape", None)
    if shape is None:
        raise PositionAccountingError(
            "inputs['input_ids'] has no shape: cannot account for positions")
    if len(shape) != 2:
        raise PositionAccountingError(
            f"inputs['input_ids'] must be (batch, sequence), got shape "
            f"{tuple(shape)}")
    if int(shape[0]) != 1:
        raise PositionAccountingError(
            f"adjudication is single-sequence by construction (one reference "
            f"and one candidate token list), but input_ids carries batch "
            f"{int(shape[0])}")
    return int(shape[1])


def _companion_lengths(inputs: dict, prompt_length: int) -> dict[str, int]:
    """Census of per-position inputs, with a guard for unrecognised ones."""
    lengths: dict[str, int] = {}
    for name, value in inputs.items():
        if name == "input_ids":
            continue
        shape = getattr(value, "shape", None)
        if shape is None:
            continue
        if len(shape) < 2:
            continue
        if name in PER_POSITION_INPUTS:
            lengths[name] = int(shape[-1])
            continue
        if len(shape) != 2:
            continue
        if int(shape[0]) != 1:
            continue
        if int(shape[-1]) == prompt_length:
            raise PositionAccountingError(
                f"input {name!r} is shaped (1, {prompt_length}) so it looks "
                f"per-position, but it is not one this adapter knows how to "
                f"extend for a teacher-forced prefill; refusing to forward it "
                f"unchanged, which would misalign it with input_ids")
    return lengths


def prefill_logit_row(model: Any, inputs: dict, prefix_token_ids) -> list[float]:
    """Teacher-forced prefill row for the position after ``prefix_token_ids``.

    Runs ONE forward pass over ``inputs["input_ids"] ++ prefix_token_ids``.
    Every per-position input is extended in lockstep (see
    :func:`plan_prefix_extension`); everything else — image tensors, grid
    metadata — passes through unchanged.  Returns ``logits[0, -1]`` as a
    python float list.

    This is the pass the model would make if it saw the whole reference
    prefix at once; it is *not* how the token was actually produced,
    which is why Rule B also measures the decode path.
    """
    import torch

    ids = inputs["input_ids"]
    prompt_length = _single_sequence_length(ids)
    prefix = list(prefix_token_ids)
    plan = plan_prefix_extension(
        _companion_lengths(inputs, prompt_length),
        prompt_length=prompt_length, prefix_length=len(prefix))

    fwd = dict(inputs)
    if prefix:
        ext = torch.tensor([prefix], device=ids.device, dtype=ids.dtype)
        fwd["input_ids"] = torch.cat([ids, ext], dim=1)
        for name, value in plan.items():
            companion = fwd[name]
            pad = companion.new_full((1, len(prefix)), value)
            fwd[name] = torch.cat([companion, pad], dim=-1)
    with torch.inference_mode():
        out = model(**fwd)
    return out.logits[0, -1].float().tolist()


def decode_logit_rows(model: Any, inputs: dict, reference_tokens,
                      positions) -> dict[int, list[float]]:
    """Incremental-decode rows for several positions in ONE replay.

    Replays the actual generation path once: a forward over the prompt
    with ``use_cache=True``, then one single-token forward per reference
    token against the returned KV cache, growing ``attention_mask`` by
    one each step.  The row read after consuming ``k`` reference tokens
    IS the row for position ``k`` — the row greedy decoding would have
    argmaxed to emit that token — so a single replay up to the largest
    requested position serves every smaller one.  That makes the cost
    ``1 + max(positions)`` forwards instead of ``sum(positions)``.

    **The decode path is defined as what generation itself runs**, not as
    a hand-rolled imitation of it.  Each step's inputs come from the
    model's own ``prepare_inputs_for_generation`` and each step's
    bookkeeping from its own ``_update_model_kwargs_for_generation``,
    driven exactly the way the sampling loop drives them: the full
    running ``input_ids`` with ``next_sequence_length=None`` and
    ``is_first_iteration=True`` for the prompt (so multimodal tensors are
    consumed once), then ``next_sequence_length=1`` per decode step (so
    the model slices to the new position and grows the mask, the cache
    and every per-position companion itself).  That removes every offset
    this adapter would otherwise have to infer — position accounting is
    the model's, and a row measured here is a row generation would have
    argmaxed.

    Returns ``{position: row}``.  Raises :class:`DecodeReplayUnavailable`
    — naming what could not be reconciled — if the model does not expose
    that contract or returns no KV cache.  Callers are expected to treat
    any failure here as "decode path unavailable" and fall back to a
    single-path, explicitly-labelled verdict.
    """
    import torch

    wanted = sorted({int(p) for p in positions})
    if not wanted:
        raise ValueError("no positions requested")
    if wanted[0] < 0:
        raise ValueError(f"positions must be >= 0, got {wanted[0]}")
    reference = list(reference_tokens)
    if wanted[-1] > len(reference):
        raise ValueError(
            f"position {wanted[-1]} is beyond the {len(reference)}-token "
            f"reference: there is no prefix to replay")

    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if prepare is None:
        raise DecodeReplayUnavailable(
            "model exposes no prepare_inputs_for_generation(): the decode "
            "path it actually runs cannot be reproduced, so the dual-path "
            "rule has nothing to compare the prefill row against")
    update = getattr(model, "_update_model_kwargs_for_generation", None)
    if update is None:
        raise DecodeReplayUnavailable(
            "model exposes no _update_model_kwargs_for_generation(): the "
            "per-step bookkeeping (KV cache, attention mask, per-position "
            "companions) cannot be reproduced faithfully")

    ids = inputs["input_ids"]
    _single_sequence_length(ids)
    model_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    model_kwargs["use_cache"] = True
    running = ids
    rows: dict[int, list[float]] = {}
    consumed = 0
    with torch.inference_mode():
        while True:
            first = consumed == 0
            try:
                model_inputs = prepare(
                    running, next_sequence_length=None if first else 1,
                    is_first_iteration=first, **model_kwargs)
            except TypeError as exc:
                raise DecodeReplayUnavailable(
                    f"the model's prepare_inputs_for_generation() does not "
                    f"accept the generation-loop contract this replay drives "
                    f"(next_sequence_length / is_first_iteration): {exc}"
                ) from exc
            out = model(**model_inputs)
            if consumed in wanted:
                rows[consumed] = out.logits[0, -1].float().tolist()
            if consumed >= wanted[-1]:
                break
            if getattr(out, "past_key_values", None) is None:
                raise DecodeReplayUnavailable(
                    "model returned no KV cache: the decode path cannot be "
                    "replayed incrementally, so no decode-path logit row can "
                    "be measured")
            model_kwargs = update(out, model_kwargs, is_encoder_decoder=False)
            nxt = torch.tensor([[reference[consumed]]], device=ids.device,
                               dtype=ids.dtype)
            running = torch.cat([running, nxt], dim=1)
            consumed += 1
    return rows


def decode_logit_row(model: Any, inputs: dict, prefix_token_ids) -> list[float]:
    """Incremental-decode row for the position after ``prefix_token_ids``.

    One-position convenience wrapper over :func:`decode_logit_rows`.
    """
    prefix = list(prefix_token_ids)
    return decode_logit_rows(model, inputs, prefix, [len(prefix)])[len(prefix)]


def adjudicate_generation(
        model: Any, inputs: dict, reference, candidate, *,
        epsilon: float = EPSILON_DEFAULT,
        budget: int = POSITION_BUDGET_DEFAULT,
        max_disagreement_fraction: float = DISAGREEMENT_FRACTION_DEFAULT,
) -> GenerationVerdict:
    """Adjudicate a whole generation: Rules A, B and C end to end.

    Censuses every disagreeing position, selects a bounded deterministic
    sample of them (:func:`select_positions`), measures each sampled
    position's logit row on BOTH the prefill and the decode path — the
    decode rows all come from one shared replay — and aggregates the
    per-position verdicts conservatively (:func:`aggregate_positions`).

    The prefill rows are required.  If the decode replay fails (older
    cache API, model without a cache, OOM, ...) the failure is recorded
    in ``notes`` and every position falls back to the single-path prefill
    rule — the previous, weaker behaviour, never a stronger claim.

    Requires at least one token disagreement: identical sequences and
    pure length differences have no disputed logit row and raise
    ``ValueError`` (call :func:`first_divergence` first).
    """
    ref = list(reference)
    cand = list(candidate)
    disagreeing = divergence_positions(ref, cand)
    if not disagreeing:
        raise ValueError(
            f"nothing to adjudicate: {first_divergence(ref, cand).reason}")

    examined = select_positions(disagreeing, budget=budget)
    notes: list[str] = []
    decode_rows: dict[int, list[float]] = {}
    try:
        decode_rows = decode_logit_rows(model, inputs, ref, examined)
    except Exception as exc:                                   # noqa: BLE001
        notes.append(
            f"decode-path replay unavailable ({type(exc).__name__}: {exc}); "
            f"the verdict rests on the prefill path alone")

    per_position = []
    for pos in examined:
        prefill_row = prefill_logit_row(model, inputs, ref[:pos])
        per_position.append(adjudicate(
            ref[pos], cand[pos], prefill_logits=prefill_row,
            decode_logits=decode_rows.get(pos), epsilon=epsilon,
            position=pos))
    return aggregate_positions(
        per_position, positions_compared=min(len(ref), len(cand)),
        positions_disagreeing=len(disagreeing), epsilon=epsilon,
        max_disagreement_fraction=max_disagreement_fraction,
        reference_length=len(ref), candidate_length=len(cand),
        notes=tuple(notes))
