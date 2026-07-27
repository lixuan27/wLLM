"""Token-disagreement adjudication: epsilon-optimal set + dual-path rule.

The gate this file pins down is the one a wrong answer would be most
expensive on: it is the difference between "the candidate emitted an
equally-optimal token" (a pass) and "the candidate emitted a token the
reference model does not consider optimal" (a refusal).  Every threshold
here is therefore tested from both sides, and every fail-closed path is
tested for the outcome it must NOT produce (``benign_tie``).
"""

import ast
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.verify.adjudicate import (
    BENIGN_TIE, DISAGREEMENT_FRACTION_DEFAULT, EPSILON_DEFAULT, IDENTICAL,
    LENGTH_MISMATCH, PATH_DECODE, PATH_PREFILL, POSITION_BUDGET_DEFAULT,
    REAL_DIVERGENCE, TOKEN_MISMATCH, UNDECIDABLE, Adjudication, PathEvidence,
    adjudicate, adjudicate_generation, adjudicate_row, aggregate_positions,
    decode_logit_row, decode_logit_rows, divergence_positions,
    first_divergence, prefill_logit_row, select_positions,
)

ADJUDICATE_PY = ROOT / "wllm" / "verify" / "adjudicate.py"


def _raises(fn, *frags):
    """Call ``fn`` and assert it raises ValueError mentioning ``frags``."""
    try:
        fn()
    except ValueError as exc:
        text = str(exc).lower()
        for frag in frags:
            assert frag.lower() in text, f"{frag!r} not in {text!r}"
        return exc
    raise AssertionError(f"expected ValueError mentioning {frags}")


# ------------------------------------------------------------------ rule A

def test_benign_tie_when_both_tokens_share_the_max():
    ev = adjudicate_row([4.0, 4.0, 1.0], 0, 1)
    assert ev.verdict == BENIGN_TIE
    assert ev.reference_gap == 0.0 and ev.candidate_gap == 0.0
    assert ev.decisive_gap == 0.0
    assert ev.max_logit == 4.0 and ev.vocab_size == 3
    assert ev.optimal_set_size == 2
    assert ev.path == PATH_PREFILL          # documented default

    res = adjudicate(0, 1, prefill_logits=[4.0, 4.0, 1.0])
    assert res.verdict == BENIGN_TIE and res.is_benign
    assert res.paths == (PATH_PREFILL,)
    assert res.dual_path is False
    assert "not exercised" in res.reason     # single-path is labelled


def test_real_divergence_when_gap_exceeds_epsilon():
    ev = adjudicate_row([4.0, 1.0, 0.0], 0, 1)
    assert ev.verdict == REAL_DIVERGENCE
    assert ev.reference_gap == 0.0 and ev.candidate_gap == 3.0
    assert ev.decisive_gap == 3.0
    assert ev.optimal_set_size == 1
    res = adjudicate(0, 1, prefill_logits=[4.0, 1.0, 0.0])
    assert res.verdict == REAL_DIVERGENCE
    assert res.is_benign is False


def test_epsilon_boundary_just_inside_and_just_outside():
    """<= epsilon passes, > epsilon refuses — both sides of the knife."""
    inside = adjudicate_row([0.0, -EPSILON_DEFAULT, -9.0], 0, 1)
    assert inside.candidate_gap == EPSILON_DEFAULT
    assert inside.verdict == BENIGN_TIE          # boundary is inclusive

    outside = adjudicate_row([0.0, -EPSILON_DEFAULT * 1.0001, -9.0], 0, 1)
    assert outside.candidate_gap > EPSILON_DEFAULT
    assert outside.verdict == REAL_DIVERGENCE

    # symmetric: it is the LARGER of the two gaps that decides
    both = adjudicate_row([-0.5, 0.0, -0.5], 0, 2, epsilon=0.4)
    assert both.verdict == REAL_DIVERGENCE
    assert both.reference_gap == 0.5 and both.candidate_gap == 0.5


def test_epsilon_is_an_explicit_parameter():
    row = [0.0, -0.5, -9.0]
    assert adjudicate_row(row, 0, 1).verdict == REAL_DIVERGENCE
    assert adjudicate_row(row, 0, 1, epsilon=0.5).verdict == BENIGN_TIE
    assert adjudicate(0, 1, prefill_logits=row, epsilon=0.5).epsilon == 0.5
    assert adjudicate(0, 1, prefill_logits=row).epsilon == EPSILON_DEFAULT
    # epsilon 0 admits only exactly-coincident logits
    assert adjudicate_row([4.0, 4.0], 0, 1, epsilon=0.0).verdict == BENIGN_TIE
    assert adjudicate_row([4.0, 3.999], 0, 1,
                          epsilon=0.0).verdict == REAL_DIVERGENCE


def test_three_way_degenerate_tie_is_arbitration():
    """The case that bit us: a knife edge degenerate beyond two tokens.

    A top-2-pair criterion misclassifies this — the disputed pair is not
    "the top 2", because a third token shares the same maximum.  The
    epsilon-optimal-set criterion gets it right and reports the
    degeneracy as evidence.
    """
    row = [7.0, 7.0, 7.0, 2.0]
    ev = adjudicate_row(row, 1, 2)
    assert ev.verdict == BENIGN_TIE
    assert ev.optimal_set_size == 3          # three-way, not two-way
    assert ev.argmax_token == 0              # argmax is neither disputed id
    assert ev.argmax_token not in (1, 2)
    assert ev.reference_gap == 0.0 and ev.candidate_gap == 0.0


def test_evidence_reports_the_optimal_set_size_honestly():
    near = 1e-9
    ev = adjudicate_row([1.0, 1.0 - near, 1.0 - near, -3.0], 0, 1,
                        epsilon=1e-6)
    assert ev.optimal_set_size == 3
    tight = adjudicate_row([1.0, 1.0 - near, 1.0 - near, -3.0], 0, 1,
                           epsilon=0.0)
    assert tight.optimal_set_size == 1
    assert tight.verdict == REAL_DIVERGENCE


# ------------------------------------------------------------------ rule B

def test_dual_path_agreement_on_tie():
    res = adjudicate(0, 1, prefill_logits=[4.0, 4.0, 1.0],
                     decode_logits=[3.0, 3.0, 0.5], position=36)
    assert res.verdict == BENIGN_TIE and res.is_benign
    assert res.paths == (PATH_PREFILL, PATH_DECODE)
    assert res.dual_path is True
    assert res.position == 36
    assert "agree" in res.reason
    assert set(res.evidence) == {PATH_PREFILL, PATH_DECODE}
    assert res.evidence[PATH_DECODE].max_logit == 3.0


def test_dual_path_agreement_on_divergence():
    res = adjudicate(0, 1, prefill_logits=[4.0, 1.0], decode_logits=[5.0, 1.0])
    assert res.verdict == REAL_DIVERGENCE
    assert res.dual_path is True
    assert res.is_benign is False


def test_dual_path_disagreement_is_undecidable_not_a_pass():
    """The systematic fix: a verdict that flips between paths never promotes."""
    flip = adjudicate(0, 1, prefill_logits=[4.0, 4.0],      # tie at prefill
                      decode_logits=[4.0, 1.0],             # divergence decoding
                      position=36)
    assert flip.verdict == UNDECIDABLE
    assert flip.is_benign is False
    assert "disagree" in flip.reason

    # ... and the mirror image is equally undecidable
    flop = adjudicate(0, 1, prefill_logits=[4.0, 1.0],
                      decode_logits=[4.0, 4.0])
    assert flop.verdict == UNDECIDABLE
    assert flop.is_benign is False


def test_dual_path_disagreement_carries_both_gap_sets():
    res = adjudicate(0, 1, prefill_logits=[4.0, 4.0],
                     decode_logits=[4.0, 1.0])
    pre = res.evidence[PATH_PREFILL]
    dec = res.evidence[PATH_DECODE]
    assert pre.verdict == BENIGN_TIE and pre.candidate_gap == 0.0
    assert dec.verdict == REAL_DIVERGENCE and dec.candidate_gap == 3.0
    # both per-path reasons survive into the human-readable verdict
    assert "prefill" in res.reason and "decode" in res.reason
    assert res.summary().startswith("[adjudicate] pos ?")


def test_single_path_decode_only_is_labelled_as_such():
    res = adjudicate(0, 1, decode_logits=[4.0, 4.0])
    assert res.verdict == BENIGN_TIE
    assert res.paths == (PATH_DECODE,)
    assert res.dual_path is False
    assert PATH_PREFILL not in res.evidence
    assert "not exercised" in res.reason


# ------------------------------------------------------------- fail closed

def test_no_logit_row_at_all_is_refused():
    _raises(lambda: adjudicate(0, 1), "at least one logit row")


def test_empty_row_is_refused():
    _raises(lambda: adjudicate_row([], 0, 1), "empty")
    _raises(lambda: adjudicate(0, 1, prefill_logits=[]), "empty")


def test_out_of_range_token_is_refused():
    _raises(lambda: adjudicate_row([1.0, 2.0], 0, 7), "candidate", "outside")
    _raises(lambda: adjudicate_row([1.0, 2.0], 5, 1), "reference", "outside")
    _raises(lambda: adjudicate_row([1.0, 2.0], -1, 1), "reference", "negative")
    _raises(lambda: adjudicate_row([1.0, 2.0], 0, -3), "candidate", "negative")


def test_non_integer_token_is_refused():
    _raises(lambda: adjudicate_row([1.0, 2.0], 0.5, 1), "integer")
    _raises(lambda: adjudicate_row([1.0, 2.0], 0, "1"), "integer")


def test_identical_disputed_tokens_are_refused():
    _raises(lambda: adjudicate_row([4.0, 4.0], 1, 1), "no disagreement")
    _raises(lambda: adjudicate(1, 1, prefill_logits=[4.0, 4.0]),
            "no disagreement")


def test_bad_epsilon_is_refused():
    _raises(lambda: adjudicate_row([4.0, 4.0], 0, 1, epsilon=-1e-9), ">= 0")
    _raises(lambda: adjudicate_row([4.0, 4.0], 0, 1,
                                   epsilon=float("nan")), "finite")
    _raises(lambda: adjudicate_row([4.0, 4.0], 0, 1,
                                   epsilon=float("inf")), "finite")
    _raises(lambda: adjudicate_row([4.0, 4.0], 0, 1, epsilon="wide"), "number")
    _raises(lambda: adjudicate(0, 1, prefill_logits=[4.0, 4.0],
                               epsilon=-1.0), ">= 0")


def test_non_finite_row_is_undecidable_never_a_tie():
    for bad in (float("nan"), float("inf"), float("-inf")):
        ev = adjudicate_row([4.0, bad, 1.0], 0, 1)
        assert ev.verdict == UNDECIDABLE, bad
        assert ev.reference_gap is None and ev.decisive_gap is None
        assert "non-finite" in ev.reason
    # a NaN row on ONE path cannot be rescued by a clean row on the other
    res = adjudicate(0, 1, prefill_logits=[4.0, 4.0],
                     decode_logits=[4.0, float("nan")])
    assert res.verdict == UNDECIDABLE and res.is_benign is False
    # and two degenerate rows still refuse
    both = adjudicate(0, 1, prefill_logits=[float("nan"), 4.0],
                      decode_logits=[4.0, float("nan")])
    assert both.verdict == UNDECIDABLE


def test_malformed_rows_are_refused():
    _raises(lambda: adjudicate_row(3.5, 0, 1), "1-D sequence")
    _raises(lambda: adjudicate_row("abc", 0, 1), "1-D sequence")
    _raises(lambda: adjudicate_row(("a", "b"), 0, 1), "not numeric")
    _raises(lambda: adjudicate_row([[1.0, 2.0], [3.0, 4.0]], 0, 1),
            "not 1-D")
    _raises(lambda: adjudicate_row([1.0, None], 0, 1), "not numeric")
    _raises(lambda: adjudicate_row([1.0, "2.0"], 0, 1), "not numeric")


def test_singleton_batch_dimension_is_accepted():
    flat = adjudicate_row([4.0, 4.0, 1.0], 0, 1)
    for nested in ([[4.0, 4.0, 1.0]], [[[4.0, 4.0, 1.0]]]):
        ev = adjudicate_row(nested, 0, 1)
        assert ev.verdict == flat.verdict
        assert ev.vocab_size == 3
        assert ev.optimal_set_size == flat.optimal_set_size
    # a genuine vocab-1 row is a row, not a dimension to peel
    _raises(lambda: adjudicate_row([9.0], 0, 1), "candidate", "outside")


def test_tolist_bearing_arrays_are_accepted():
    class FakeTensor:
        def __init__(self, data):
            self._data = data

        def tolist(self):
            return self._data

    ev = adjudicate_row(FakeTensor([[4.0, 4.0, 1.0]]), 0, 1)
    assert ev.verdict == BENIGN_TIE and ev.vocab_size == 3
    try:
        import numpy as np
    except ImportError:                                    # pragma: no cover
        return
    arr = np.array([4.0, 4.0, 1.0], dtype=np.float32)
    ev = adjudicate_row(arr, np.int64(0), np.int64(1))
    assert ev.verdict == BENIGN_TIE and ev.vocab_size == 3
    assert adjudicate_row(np.array([[4.0, 1.0]]), 0, 1).verdict \
        == REAL_DIVERGENCE


# ------------------------------------------------------- divergence scanning

def test_first_divergence_identical():
    d = first_divergence([1, 2, 3], [1, 2, 3])
    assert d.kind == IDENTICAL
    assert d.diverged is False and d.adjudicable is False
    assert d.position is None
    assert "identical" in d.reason
    empty = first_divergence([], [])
    assert empty.kind == IDENTICAL and empty.diverged is False


def test_first_divergence_token_mismatch_reports_first_only():
    d = first_divergence([1, 2, 3, 4], [1, 9, 3, 8])
    assert d.kind == TOKEN_MISMATCH
    assert d.position == 1
    assert d.reference_token == 2 and d.candidate_token == 9
    assert d.diverged is True and d.adjudicable is True
    assert first_divergence([5, 1], [9, 1]).position == 0


def test_first_divergence_length_mismatch_both_directions():
    longer_ref = first_divergence([1, 2, 3], [1, 2])
    assert longer_ref.kind == LENGTH_MISMATCH
    assert longer_ref.position == 2
    assert longer_ref.reference_token == 3
    assert longer_ref.candidate_token is None
    assert longer_ref.adjudicable is False and longer_ref.diverged is True

    longer_cand = first_divergence([1, 2], [1, 2, 7])
    assert longer_cand.kind == LENGTH_MISMATCH
    assert longer_cand.position == 2
    assert longer_cand.reference_token is None
    assert longer_cand.candidate_token == 7

    empty_ref = first_divergence([], [4])
    assert empty_ref.kind == LENGTH_MISMATCH and empty_ref.position == 0
    assert empty_ref.candidate_token == 4


def test_token_mismatch_wins_over_length_mismatch():
    d = first_divergence([1, 5], [1, 6, 7, 8])
    assert d.kind == TOKEN_MISMATCH and d.position == 1


# ------------------------------------------------- rule C: whole generation

def _pos(position, verdict):
    """A per-position Adjudication with a given verdict, built directly."""
    rows = {BENIGN_TIE: ([4.0, 4.0], [4.0, 4.0]),
            REAL_DIVERGENCE: ([4.0, 1.0], [4.0, 1.0]),
            UNDECIDABLE: ([4.0, 4.0], [4.0, 1.0])}[verdict]
    res = adjudicate(0, 1, prefill_logits=rows[0], decode_logits=rows[1],
                     position=position)
    assert res.verdict == verdict            # the fixture itself is checked
    return res


def _agg(adjudications, compared, disagreeing, **over):
    kw = {"positions_compared": compared, "positions_disagreeing": disagreeing,
          "reference_length": compared, "candidate_length": compared}
    kw.update(over)
    return aggregate_positions(adjudications, **kw)


def test_divergence_positions_censuses_every_disagreement():
    assert divergence_positions([1, 2, 3, 4], [1, 9, 3, 8]) == (1, 3)
    assert divergence_positions([1, 2, 3], [1, 2, 3]) == ()
    # only the shared prefix is compared; a length difference is not a
    # token disagreement
    assert divergence_positions([1, 2, 3], [1, 5]) == (1,)
    assert divergence_positions([], [1, 2]) == ()
    assert divergence_positions([4, 4], [5, 5]) == (0, 1)


def test_select_positions_takes_first_last_and_a_deterministic_spread():
    picked = select_positions(range(20), budget=8)
    assert len(picked) == 8
    assert picked[0] == 0 and picked[-1] == 19       # first and last always
    assert list(picked) == sorted(set(picked))       # ordered, deduplicated
    assert select_positions(range(20), budget=8) == picked   # deterministic
    # a different budget gives a different, still-anchored sample
    small = select_positions(range(20), budget=3)
    assert small[0] == 0 and small[-1] == 19 and len(small) == 3


def test_select_positions_budget_boundary():
    # budget at or above the number of disagreements examines all of them
    assert select_positions([3, 7, 11], budget=3) == (3, 7, 11)
    assert select_positions([3, 7, 11], budget=8) == (3, 7, 11)
    assert select_positions([4, 9], budget=8) == (4, 9)
    # budget 1 keeps the single unconfounded position
    assert select_positions([3, 7, 11], budget=1) == (3,)
    # one below the count starts dropping the middle, never the ends
    trimmed = select_positions([3, 7, 11], budget=2)
    assert trimmed == (3, 11)
    assert select_positions([], budget=8) == ()
    _raises(lambda: select_positions([1, 2], budget=0), "budget must be >= 1")
    _raises(lambda: select_positions([1, 2], budget=-3), "budget must be >= 1")


def test_aggregate_refuses_when_a_later_position_diverges():
    """A benign first position must never carry a divergent later one."""
    res = _agg([_pos(0, BENIGN_TIE), _pos(40, REAL_DIVERGENCE)], 100, 2)
    assert res.verdict == REAL_DIVERGENCE and res.is_benign is False
    assert res.positions_examined == (0, 40)
    deciding = res.reason.split("deciding evidence: ")[1]
    assert "pos 40" in deciding and "pos 0" not in deciding


def test_aggregate_passes_only_when_every_examined_position_is_benign():
    res = _agg([_pos(0, BENIGN_TIE), _pos(40, BENIGN_TIE)], 100, 2)
    assert res.verdict == BENIGN_TIE and res.is_benign is True


def test_aggregate_precedence_is_conservative():
    """real_divergence beats undecidable beats benign_tie, in both orders."""
    assert _agg([_pos(0, UNDECIDABLE), _pos(9, BENIGN_TIE)],
                100, 2).verdict == UNDECIDABLE
    assert _agg([_pos(0, BENIGN_TIE), _pos(9, UNDECIDABLE)],
                100, 2).verdict == UNDECIDABLE
    assert _agg([_pos(0, UNDECIDABLE), _pos(9, REAL_DIVERGENCE)],
                100, 2).verdict == REAL_DIVERGENCE
    assert _agg([_pos(0, REAL_DIVERGENCE), _pos(9, UNDECIDABLE)],
                100, 2).verdict == REAL_DIVERGENCE
    # a divergence is NOT softened to undecidable by a high fraction
    hot = _agg([_pos(0, REAL_DIVERGENCE), _pos(9, BENIGN_TIE)], 10, 10)
    assert hot.verdict == REAL_DIVERGENCE


def test_aggregate_reports_unexamined_positions_as_residual_risk():
    res = _agg([_pos(0, BENIGN_TIE), _pos(226, BENIGN_TIE)], 256, 227,
               max_disagreement_fraction=1.0)
    assert res.positions_unexamined == 225
    assert "225 disagreeing positions were NOT examined" in res.reason
    assert "NOT certified benign" in res.reason
    assert "examined 2 of 227 disagreeing positions" in res.reason
    assert res.as_dict()["positions_unexamined"] == 225


def test_aggregate_caps_a_high_disagreement_fraction():
    """227 of 256 positions differing is not a knife-edge phenomenon."""
    field_case = [_pos(0, BENIGN_TIE), _pos(226, BENIGN_TIE)]
    capped = _agg(field_case, 256, 227)                  # 88.7% >> 5%
    assert capped.verdict == UNDECIDABLE and capped.is_benign is False
    assert "trajectory phenomenon, not a knife edge" in capped.reason
    assert abs(capped.disagreement_fraction - 227 / 256) < 1e-12
    # the same evidence under a low fraction passes
    calm = _agg(field_case, 4000, 2)                     # 0.05% << 5%
    assert calm.verdict == BENIGN_TIE
    # the threshold is an explicit parameter, testable from both sides
    near = [_pos(0, BENIGN_TIE), _pos(9, BENIGN_TIE)]
    assert _agg(near, 100, 5).verdict == BENIGN_TIE                # 5% <= 5%
    assert _agg(near, 100, 6).verdict == UNDECIDABLE               # 6% > 5%
    assert _agg(near, 100, 6,
                max_disagreement_fraction=0.10).verdict == BENIGN_TIE
    assert _agg(near, 100, 5,
                max_disagreement_fraction=0.01).verdict == UNDECIDABLE
    assert DISAGREEMENT_FRACTION_DEFAULT == 0.05


def test_fraction_cap_needs_more_than_one_disagreement():
    """A single flip is a knife edge by construction, however short the run."""
    lone = _agg([_pos(0, BENIGN_TIE)], 1, 1)             # 100% of 1 position
    assert lone.verdict == BENIGN_TIE
    assert _agg([_pos(0, BENIGN_TIE)], 128, 1).verdict == BENIGN_TIE
    # two is where "many positions disagree" starts to mean something
    assert _agg([_pos(0, BENIGN_TIE), _pos(1, BENIGN_TIE)],
                2, 2).verdict == UNDECIDABLE


def test_aggregate_caps_a_length_difference():
    ok = _agg([_pos(0, BENIGN_TIE)], 128, 1)
    assert ok.verdict == BENIGN_TIE
    short = _agg([_pos(0, BENIGN_TIE)], 128, 1, candidate_length=140)
    assert short.verdict == UNDECIDABLE
    assert "different token counts" in short.reason
    assert short.as_dict()["candidate_length"] == 140


def test_counterfactual_caveat_is_disclosed_when_it_applies():
    one = _agg([_pos(0, BENIGN_TIE)], 128, 1)
    assert one.notes == ()
    many = _agg([_pos(0, BENIGN_TIE), _pos(9, BENIGN_TIE)], 100, 2,
                max_disagreement_fraction=1.0)
    assert len(many.notes) == 1
    assert "REFERENCE" in many.notes[0]
    carried = _agg([_pos(0, BENIGN_TIE), _pos(9, BENIGN_TIE)], 100, 2,
                   max_disagreement_fraction=1.0, notes=("decode degraded",))
    assert carried.notes[0] == "decode degraded"
    assert len(carried.notes) == 2


def test_aggregate_fails_closed_on_impossible_censuses():
    good = _pos(0, BENIGN_TIE)
    _raises(lambda: _agg([], 100, 1), "at least one measured position")
    _raises(lambda: _agg([good], 0, 0), "positions_compared must be >= 1")
    _raises(lambda: _agg([good], 10, 11), "exceeds")
    _raises(lambda: _agg([good, _pos(1, BENIGN_TIE)], 10, 1),
            "examined 2 positions but only 1 disagree")
    _raises(lambda: _agg([adjudicate(0, 1, prefill_logits=[4.0, 4.0])], 10, 1),
            "must carry its index")
    _raises(lambda: _agg([_pos(50, BENIGN_TIE)], 10, 1),
            "outside the 10 compared positions")
    for bad in (-0.01, 1.5, float("nan"), float("inf")):
        _raises(lambda bad=bad: _agg([good], 10, 1,
                                     max_disagreement_fraction=bad),
                "max_disagreement_fraction")
    _raises(lambda: _agg([good], 10, 1, max_disagreement_fraction="most"),
            "must be a number")
    _raises(lambda: _agg([good], 10, 1, epsilon=-1.0), ">= 0")


def test_generation_verdict_serialises_for_receipts():
    res = _agg([_pos(0, BENIGN_TIE), _pos(226, REAL_DIVERGENCE)], 256, 227)
    blob = json.loads(json.dumps(res.as_dict()))
    assert blob["verdict"] == REAL_DIVERGENCE
    assert blob["positions_compared"] == 256
    assert blob["positions_disagreeing"] == 227
    assert blob["positions_examined"] == [0, 226]
    assert blob["positions_unexamined"] == 225
    assert blob["max_disagreement_fraction"] == 0.05
    assert len(blob["per_position"]) == 2
    assert blob["per_position"][1]["position"] == 226
    assert res.summary().startswith("[adjudicate] generation -> ")
    assert "225 unexamined" in res.summary()


# ------------------------------------------------------------- receipt shape

def test_results_serialise_for_receipts():
    res = adjudicate(0, 1, prefill_logits=[4.0, 4.0, 1.0],
                     decode_logits=[4.0, float("nan")], position=12,
                     notes=("decode replay was degraded",))
    blob = json.loads(json.dumps(res.as_dict()))
    assert blob["verdict"] == UNDECIDABLE
    assert blob["epsilon"] == EPSILON_DEFAULT
    assert blob["position"] == 12
    assert blob["paths"] == [PATH_PREFILL, PATH_DECODE]
    assert blob["dual_path"] is True
    assert blob["notes"] == ["decode replay was degraded"]
    assert blob["evidence"][PATH_PREFILL]["reference_gap"] == 0.0
    assert blob["evidence"][PATH_PREFILL]["optimal_set_size"] == 2
    # non-finite numbers are published as null, never as bare NaN
    assert blob["evidence"][PATH_DECODE]["decisive_gap"] is None
    assert first_divergence([1], [2]).as_dict()["kind"] == TOKEN_MISMATCH


def test_non_finite_numbers_never_reach_the_json_receipt():
    """A receipt must be valid JSON — bare NaN/Infinity are not."""
    ev = PathEvidence(path=PATH_PREFILL, verdict=UNDECIDABLE, reason="x",
                      vocab_size=2, reference_gap=float("inf"),
                      candidate_gap=float("nan"), max_logit=float("-inf"))
    blob = ev.as_dict()
    for key in ("reference_gap", "candidate_gap", "max_logit",
                "decisive_gap"):
        assert blob[key] is None, key
    assert json.dumps(blob)          # would emit NaN/Infinity otherwise

    # a half-measured row decides nothing, in either direction
    only_ref = PathEvidence(path=PATH_DECODE, verdict=UNDECIDABLE, reason="x",
                            vocab_size=2, reference_gap=1.0)
    assert only_ref.decisive_gap is None
    only_cand = PathEvidence(path=PATH_DECODE, verdict=UNDECIDABLE, reason="x",
                             vocab_size=2, candidate_gap=1.0)
    assert only_cand.decisive_gap is None


def test_result_types_are_immutable():
    """A verdict is evidence: nothing downstream may edit it into a pass."""
    ev = adjudicate_row([4.0, 1.0], 0, 1)
    res = adjudicate(0, 1, prefill_logits=[4.0, 1.0])
    div = first_divergence([1], [2])
    gen = _agg([_pos(0, REAL_DIVERGENCE)], 8, 1)
    assert ev.verdict == REAL_DIVERGENCE and res.verdict == REAL_DIVERGENCE
    assert gen.verdict == REAL_DIVERGENCE
    for obj, attr in ((ev, "verdict"), (res, "verdict"), (div, "kind"),
                      (res, "epsilon"), (gen, "verdict"),
                      (gen, "positions_disagreeing")):
        try:
            setattr(obj, attr, BENIGN_TIE)
        except AttributeError:
            continue                      # dataclasses.FrozenInstanceError
        raise AssertionError(
            f"{type(obj).__name__}.{attr} is writable — a verdict must not "
            f"be editable after the fact")


def test_is_benign_is_true_only_for_benign_tie():
    for verdict, expect in ((BENIGN_TIE, True), (REAL_DIVERGENCE, False),
                            (UNDECIDABLE, False)):
        res = Adjudication(verdict=verdict, epsilon=EPSILON_DEFAULT,
                           reference_token=0, candidate_token=1, reason="x")
        assert res.is_benign is expect
        assert res.dual_path is False
    assert len({BENIGN_TIE, REAL_DIVERGENCE, UNDECIDABLE}) == 3


# ------------------------------------------------------------- layering rule

def test_core_does_not_import_torch_at_module_scope():
    """The decision core must stay unit-testable on a CPU box with no torch."""
    tree = ast.parse(ADJUDICATE_PY.read_text())
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level.append(node.module or "")
    assert "torch" not in top_level, top_level
    assert "numpy" not in top_level, top_level
    # the adapters DO import torch, just lazily inside their bodies
    source = ADJUDICATE_PY.read_text()
    assert source.count("import torch") >= 2


def test_adjudicate_generation_refuses_when_there_is_nothing_to_judge():
    """The model-backed entry point must not touch the model without a case.

    Identical sequences and pure length differences have no disputed
    logit row; asking for a verdict on them is a caller error, and it is
    caught before any forward pass (``model=None`` here proves it).
    """
    _raises(lambda: adjudicate_generation(None, {}, [1, 2, 3], [1, 2, 3]),
            "nothing to adjudicate", "identical")
    _raises(lambda: adjudicate_generation(None, {}, [1, 2, 3], [1, 2]),
            "nothing to adjudicate", "lengths differ")


def test_defaults_are_the_documented_constants():
    assert EPSILON_DEFAULT == 1e-3
    assert math.isfinite(EPSILON_DEFAULT) and EPSILON_DEFAULT > 0
    assert POSITION_BUDGET_DEFAULT == 8
    assert DISAGREEMENT_FRACTION_DEFAULT == 0.05
    assert 0.0 <= DISAGREEMENT_FRACTION_DEFAULT <= 1.0


# ------------------------------------------------- model adapters (no GPU)
#
# The adapters import torch lazily inside their bodies, so a stand-in
# module in ``sys.modules`` exercises their control flow on CPU in
# microseconds.  What is proven here is the SEQUENCING — one forward for
# prefill, prompt + one forward per prefix token for decode, image
# tensors only on the first step, the KV cache threaded through, the
# attention mask grown by one each step.  Numerical agreement with a real
# bf16 checkpoint is a GPU question and is validated separately.

class _Vec:
    """Stand-in for a 1-D logit slice."""

    def __init__(self, values):
        self.values = list(values)

    def float(self):
        return self

    def tolist(self):
        return list(self.values)


class _Mat:
    """Stand-in for a 2-D tensor: only what the adapters actually touch."""

    def __init__(self, rows, device="cpu", dtype="int64"):
        self.rows = [list(r) for r in rows]
        self.device = device
        self.dtype = dtype

    def __getitem__(self, key):
        i, j = key
        cell = self.rows[i][j]
        return _Vec(cell if isinstance(cell, list) else [cell])


def _fake_torch():
    mod = types.ModuleType("torch")

    def tensor(data, device=None, dtype=None):
        return _Mat(data, device=device, dtype=dtype)

    def cat(mats, dim=1):
        assert dim == 1, "adapters only ever concatenate along the sequence"
        rows = [sum((m.rows[i] for m in mats), [])
                for i in range(len(mats[0].rows))]
        return _Mat(rows, device=mats[0].device, dtype=mats[0].dtype)

    def ones_like(mat):
        return _Mat([[1] * len(r) for r in mat.rows],
                    device=mat.device, dtype=mat.dtype)

    def ones(shape, device=None, dtype=None):
        return _Mat([[1] * shape[1] for _ in range(shape[0])],
                    device=device, dtype=dtype)

    class _InferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    mod.tensor = tensor
    mod.cat = cat
    mod.ones_like = ones_like
    mod.ones = ones
    mod.inference_mode = _InferenceMode
    return mod


class _stub_torch:
    """Install the stand-in torch for the duration of a block."""

    def __enter__(self):
        self._saved = sys.modules.get("torch")
        sys.modules["torch"] = _fake_torch()
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._saved
        return False


class _StubModel:
    """Returns a scripted logit row per forward call and records the call."""

    def __init__(self, rows, cache=True):
        self.rows = [list(r) for r in rows]
        self.cache = cache
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        row = self.rows[min(len(self.calls) - 1, len(self.rows) - 1)]
        out = types.SimpleNamespace(logits=_Mat([[list(row)]]))
        out.past_key_values = ("cache", len(self.calls)) if self.cache else None
        return out


def _stub_inputs():
    return {"input_ids": _Mat([[5, 6, 7]]),
            "attention_mask": _Mat([[1, 1, 1]]),
            "pixel_values": "IMG"}


def test_prefill_adapter_teacher_forces_the_whole_prefix_in_one_pass():
    model = _StubModel([[1.0, 2.0]])
    inputs = _stub_inputs()
    with _stub_torch():
        row = prefill_logit_row(model, inputs, [11, 12])
    assert row == [1.0, 2.0]
    assert len(model.calls) == 1                       # exactly ONE forward
    call = model.calls[0]
    assert call["input_ids"].rows == [[5, 6, 7, 11, 12]]
    assert call["attention_mask"].rows == [[1, 1, 1, 1, 1]]
    assert call["pixel_values"] == "IMG"               # extras pass through
    assert inputs["input_ids"].rows == [[5, 6, 7]]     # caller not mutated

    empty = _StubModel([[3.0, 4.0]])
    with _stub_torch():
        assert prefill_logit_row(empty, inputs, []) == [3.0, 4.0]
    assert empty.calls[0]["input_ids"].rows == [[5, 6, 7]]


def test_decode_adapter_replays_one_token_at_a_time_through_the_cache():
    model = _StubModel([[0.0, 0.0], [0.5, 0.5], [9.0, 1.0]])
    with _stub_torch():
        row = decode_logit_row(model, _stub_inputs(), [11, 12])
    assert row == [9.0, 1.0]                  # the LAST step's row wins
    assert len(model.calls) == 3              # prompt + one per prefix token
    assert model.calls[0]["use_cache"] is True
    assert model.calls[0]["pixel_values"] == "IMG"
    assert model.calls[1]["use_cache"] is True
    assert model.calls[1]["input_ids"].rows == [[11]]
    assert "pixel_values" not in model.calls[1]        # image only once
    assert model.calls[1]["past_key_values"] == ("cache", 1)
    assert model.calls[1]["attention_mask"].rows == [[1, 1, 1, 1]]
    assert model.calls[2]["input_ids"].rows == [[12]]
    assert model.calls[2]["past_key_values"] == ("cache", 2)
    assert model.calls[2]["attention_mask"].rows == [[1, 1, 1, 1, 1]]


def test_decode_adapter_refuses_a_model_without_a_kv_cache():
    model = _StubModel([[1.0, 2.0]], cache=False)
    with _stub_torch():
        try:
            decode_logit_row(model, _stub_inputs(), [11])
        except RuntimeError as exc:
            assert "kv cache" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError without a KV cache")


def _tie_row(hot, vocab=10):
    return [5.0 if i in hot else 0.0 for i in range(vocab)]


def test_decode_rows_for_many_positions_come_from_one_replay():
    """Cost is 1 + max(position) forwards, not sum(position)."""
    scripted = [_tie_row({i}) for i in range(6)]
    model = _StubModel(scripted)
    with _stub_torch():
        rows = decode_logit_rows(model, _stub_inputs(), [1, 2, 3, 4, 5],
                                 [0, 2, 4])
    assert sorted(rows) == [0, 2, 4]
    assert len(model.calls) == 5               # prompt + 4 steps = 1 + max(4)
    assert rows[0] == scripted[0]              # prompt forward
    assert rows[2] == scripted[2]              # after consuming 2 tokens
    assert rows[4] == scripted[4]
    # duplicate / unsorted requests are normalised, not re-measured
    again = _StubModel(scripted)
    with _stub_torch():
        rows2 = decode_logit_rows(again, _stub_inputs(), [1, 2, 3, 4, 5],
                                  [4, 0, 4, 2])
    assert sorted(rows2) == [0, 2, 4] and len(again.calls) == 5


def test_decode_rows_fail_closed_on_impossible_requests():
    model = _StubModel([_tie_row({1})])
    with _stub_torch():
        _raises(lambda: decode_logit_rows(model, _stub_inputs(), [1, 2], []),
                "no positions requested")
        _raises(lambda: decode_logit_rows(model, _stub_inputs(), [1, 2], [-1]),
                "must be >= 0")
        _raises(lambda: decode_logit_rows(model, _stub_inputs(), [1, 2], [9]),
                "beyond the 2-token reference")


def test_generation_adapter_measures_both_paths():
    """prefill says tie, decode says divergence -> undecidable, not a pass."""
    model = _StubModel([_tie_row({9}),         # call 1: decode prompt
                        _tie_row({9}),         # call 2: decode @1 (diverge)
                        _tie_row({9, 4})])     # call 3: prefill @1 (tie)
    with _stub_torch():
        res = adjudicate_generation(model, _stub_inputs(), [7, 9], [7, 4])
    assert res.positions_examined == (1,)
    assert res.positions_disagreeing == 1 and res.positions_compared == 2
    only = res.per_position[0]
    assert only.paths == (PATH_PREFILL, PATH_DECODE)
    assert only.evidence[PATH_PREFILL].verdict == BENIGN_TIE
    assert only.evidence[PATH_DECODE].verdict == REAL_DIVERGENCE
    assert only.verdict == UNDECIDABLE
    assert res.verdict == UNDECIDABLE and res.is_benign is False
    assert res.notes == ()                     # single disagreement: no caveat
    assert len(model.calls) == 3               # (prompt + 1 step) + 1 prefill


def test_generation_adapter_degrades_to_single_path_with_a_note():
    """A decode replay that cannot run is disclosed, never silently dropped."""
    model = _StubModel([_tie_row({9, 4})], cache=False)
    with _stub_torch():
        res = adjudicate_generation(model, _stub_inputs(), [7, 9], [7, 4])
    assert res.per_position[0].paths == (PATH_PREFILL,)
    assert res.verdict == BENIGN_TIE           # the old, weaker behaviour
    assert len(res.notes) == 1
    assert "decode-path replay unavailable" in res.notes[0]
    assert "RuntimeError" in res.notes[0]
    assert res.as_dict()["notes"] == list(res.notes)


def test_generation_adapter_refuses_when_a_later_position_diverges():
    """THE regression test: a benign first position must not carry the day.

    Rule C end to end through the adapters.  Position 1 is a perfect tie;
    position 3 is a decisive divergence.  The old single-position gate
    returned ``benign_tie`` here and promoted the plan.
    """
    model = _StubModel([
        _tie_row({0}),          # 1: decode prompt (pos 0, unused)
        _tie_row({9, 4}),       # 2: decode @1  -> tie
        _tie_row({0}),          # 3: decode @2  (agrees, not examined)
        _tie_row({5}),          # 4: decode @3  -> divergence
        _tie_row({9, 4}),       # 5: prefill @1 -> tie
        _tie_row({5}),          # 6: prefill @3 -> divergence
    ])
    with _stub_torch():
        res = adjudicate_generation(
            model, _stub_inputs(), [7, 9, 3, 5], [7, 4, 3, 8],
            max_disagreement_fraction=1.0)     # isolate Rule C aggregation
    assert len(model.calls) == 6               # (prompt + 3 steps) + 2 prefills
    assert res.positions_examined == (1, 3)
    assert res.positions_disagreeing == 2 and res.positions_unexamined == 0
    assert res.per_position[0].verdict == BENIGN_TIE
    assert res.per_position[1].verdict == REAL_DIVERGENCE
    assert res.verdict == REAL_DIVERGENCE and res.is_benign is False
    # the counterfactual caveat is disclosed once more than one position
    # disagrees, and the deciding evidence names the divergent position
    assert any("REFERENCE" in n for n in res.notes)
    deciding = res.reason.split("deciding evidence: ")[1]
    assert "pos 3" in deciding and "pos 1" not in deciding


def test_generation_adapter_passes_when_every_examined_position_is_benign():
    model = _StubModel([_tie_row({0}), _tie_row({9, 4}), _tie_row({0}),
                        _tie_row({5, 8}), _tie_row({9, 4}), _tie_row({5, 8})])
    with _stub_torch():
        res = adjudicate_generation(
            model, _stub_inputs(), [7, 9, 3, 5], [7, 4, 3, 8],
            max_disagreement_fraction=1.0)
    assert res.verdict == BENIGN_TIE and res.is_benign is True
    assert res.positions_examined == (1, 3)


def test_generation_adapter_honours_the_position_budget():
    """Bounded work: only the sampled positions are measured, and the rest
    are counted as unexamined rather than assumed benign."""
    model = _StubModel([
        _tie_row({1, 9}),       # 1: decode prompt = pos 0 -> tie
        _tie_row({0}),          # 2: decode @1 (not sampled)
        _tie_row({0}),          # 3: decode @2 (not sampled)
        _tie_row({0}),          # 4: decode @3 (not sampled)
        _tie_row({5, 9}),       # 5: decode @4 -> tie
        _tie_row({1, 9}),       # 6: prefill @0
        _tie_row({5, 9}),       # 7: prefill @4
    ])
    with _stub_torch():
        res = adjudicate_generation(
            model, _stub_inputs(), [1, 2, 3, 4, 5], [9, 9, 9, 9, 9],
            budget=2, max_disagreement_fraction=1.0)
    assert res.positions_disagreeing == 5
    assert res.positions_examined == (0, 4)      # first and last
    assert res.positions_unexamined == 3
    assert len(model.calls) == 7                 # (1 + 4) replay + 2 prefills
    assert res.verdict == BENIGN_TIE
    assert "3 disagreeing positions were NOT examined" in res.reason


def test_generation_adapter_handles_a_divergence_at_position_zero():
    model = _StubModel([_tie_row({9, 4}),      # 1: decode prompt = pos 0
                        _tie_row({9, 4})])     # 2: prefill @0
    with _stub_torch():
        res = adjudicate_generation(model, _stub_inputs(), [9], [4])
    assert res.positions_examined == (0,)
    assert len(model.calls) == 2                # prompt only, no decode steps
    assert res.verdict == BENIGN_TIE


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:200]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
