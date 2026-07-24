"""Composition search over verified techniques, graded fail-closed.

Techniques that each pass verification in isolation can interfere when
stacked: residual caches feeding each other compound drift, and every
wrapper changes the trajectory its neighbours were verified on.
Composition legality is therefore measured per combination, never
assumed from the members' single verdicts.

The composer owns nothing the orchestrator already owns. Every single
and every combination is graded by the same ``TechniqueOrchestrator``
(reference run, deviation budget, authenticity signals, fail-closed
rejection reasons); the composer only builds combined runners and
specs, then applies a selection policy over the verdicts:

* combined runners apply member wraps innermost-first in the given
  order and build all wrapper state freshly inside each timed run;
* combined specs prefix every member signal as ``"<member>.<signal>"``
  so signals never collide, and a member that never engages inside the
  combo trips the orchestrator's own missing-signal rejection;
* a combo is flagged as interference — and excluded from ``chosen`` —
  when its measured deviation exceeds the sum of its members' single
  deviations by more than ``margin``, or when it beats no single on
  wall time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence

from .base import QUALITY_CLASSES, QualityBudget, TechniqueSpec
from .orchestrator import CandidateVerdict, TechniqueOrchestrator
from .step_cache import run_loop

StepFn = Callable[[list, int], list]
AuthReader = Callable[[], dict]
MakeFn = Callable[[StepFn], tuple]


@dataclass
class Composable:
    """A technique packaged as a fresh-state wrap factory.

    ``make(inner)`` returns ``(wrapped_step, authenticity_reader)``:
    ``wrapped_step(x, k)`` instruments or replaces calls to ``inner``,
    and the reader reports the counters observed by THAT instance.

    The orchestrator times every runner ``repeats`` times, so ``make``
    is invoked once per timed run, inside the runner closure. Sharing
    one wrapper instance across runs would leak warm cache state and
    stale counters into timing and authenticity evidence — the factory
    shape makes that impossible by construction.
    """

    spec: TechniqueSpec
    make: MakeFn


@dataclass
class ComposerReport:
    """Outcome of a composition search.

    ``chosen`` is the fastest accepted verdict among singles and
    interference-free combos, or ``None`` when nothing was accepted.
    ``interference`` carries one reason string per combo that was
    rejected for combined drift or excluded from ``chosen``.
    """

    singles: list[CandidateVerdict]
    combos: list[CandidateVerdict]
    chosen: CandidateVerdict | None
    interference: list[str] = field(default_factory=list)


@dataclass
class TechniqueComposer:
    """Searches technique compositions; delegates all grading.

    Built on a ``TechniqueOrchestrator`` whose reference is
    ``run_loop(step_fn, x0, iterations)``. The composer never
    reimplements deviation, authenticity, or timing checks — a combo
    can only be selected after surviving the exact fail-closed
    protocol that graded its members, on the same inputs.
    """

    step_fn: StepFn
    x0: Sequence[float]
    iterations: int
    budget: QualityBudget = field(default_factory=QualityBudget.exact)
    repeats: int = 3

    def __post_init__(self):
        self.x0 = list(self.x0)
        self._orchestrator = TechniqueOrchestrator(
            reference=lambda: run_loop(self.step_fn, self.x0,
                                       self.iterations),
            budget=self.budget, repeats=self.repeats)

    # ---------------------------------------------------------- singles
    def evaluate_singles(self, composables: Sequence[Composable]
                         ) -> list[CandidateVerdict]:
        """Grade each composable alone through the orchestrator."""
        comps = self._check_composables(composables)
        return self._orchestrator.evaluate(
            [(c.spec, self._single_runner(c)) for c in comps])

    # ----------------------------------------------------------- combos
    def combined_spec(self, members: Sequence[Composable]
                      ) -> TechniqueSpec:
        """Spec for a combination: joined name, worst quality class,
        and per-member-prefixed authenticity signals.

        The prefixes are what makes non-engagement inside a combo
        detectable: the combined runner reports every member's
        counters under ``"<member>.<signal>"``, so a member that never
        engaged leaves its prefixed signal at zero and the orchestrator
        rejects the combo on its own missing-signal check.
        """
        members = list(members)
        if len(members) < 2:
            raise ValueError("a combo needs >= 2 members; got "
                             f"{len(members)}")
        names = [m.spec.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate member names {names}; prefixed "
                             f"authenticity keys would collide")
        for m in members:
            if m.spec.quality_class not in QUALITY_CLASSES:
                raise ValueError(
                    f"member {m.spec.name!r} has unknown quality class "
                    f"{m.spec.quality_class!r}")
        families = {m.spec.family for m in members}
        family = families.pop() if len(families) == 1 else "scheduling"
        quality = max((m.spec.quality_class for m in members),
                      key=QUALITY_CLASSES.index)
        signals = [f"{m.spec.name}.{s}" for m in members
                   for s in m.spec.authenticity_signals]
        return TechniqueSpec(
            name="+".join(names), family=family, quality_class=quality,
            params={"members": names}, authenticity_signals=signals)

    def evaluate_combo(self, members: Sequence[Composable]
                       ) -> CandidateVerdict:
        """Grade one explicit combination, bypassing singles gating.

        Probing a suspect pairing directly is legitimate — the
        orchestrator still grades it fail-closed; only the search-time
        shortcut of skipping known-bad members is bypassed.
        """
        members = list(members)
        spec = self.combined_spec(members)
        return self._orchestrator.evaluate(
            [(spec, self._combo_runner(members))])[0]

    def evaluate_combos(self, composables: Sequence[Composable],
                        max_combo_size: int = 2,
                        singles: list[CandidateVerdict] | None = None
                        ) -> list[CandidateVerdict]:
        """Grade every combination (sizes 2..max_combo_size) of the
        composables whose SINGLE verdicts were accepted.

        ``singles`` may carry precomputed single verdicts to avoid
        regrading; when omitted they are evaluated here. Members that
        failed alone are excluded from the search — stacking cannot
        repair a technique the orchestrator already rejected.
        """
        comps = self._check_composables(composables)
        if max_combo_size < 2:
            raise ValueError("max_combo_size must be >= 2 (a combo has "
                             "at least two members)")
        if singles is None:
            singles = self.evaluate_singles(comps)
        accepted = {v.spec.name for v in singles if v.accepted}
        pool = [c for c in comps if c.spec.name in accepted]
        candidates = []
        for size in range(2, max_combo_size + 1):
            for members in combinations(pool, size):
                members = list(members)
                candidates.append((self.combined_spec(members),
                                   self._combo_runner(members)))
        if not candidates:
            return []
        return self._orchestrator.evaluate(candidates)

    # -------------------------------------------------------- selection
    def select(self, composables: Sequence[Composable],
               max_combo_size: int = 2,
               margin: float = 0.0) -> ComposerReport:
        """Full search: singles, combos, interference, and a choice.

        A combo is flagged as interference (and excluded from
        ``chosen``) when its measured deviation exceeds the sum of its
        members' single deviations by more than ``margin`` — the
        default 0.0 means any superadditive drift counts — or when it
        is slower than the best single, i.e. composing bought nothing.
        Combos the orchestrator rejected on the deviation budget are
        reported as interference too; every exclusion carries a
        reason. ``chosen`` is the fastest surviving verdict, or
        ``None`` when nothing was accepted at all.
        """
        comps = self._check_composables(composables)
        if max_combo_size < 2:
            raise ValueError("max_combo_size must be >= 2 (a combo has "
                             "at least two members)")
        if margin < 0:
            raise ValueError("margin must be >= 0")
        singles = self.evaluate_singles(comps)
        combos = self.evaluate_combos(comps, max_combo_size,
                                      singles=singles)
        accepted_singles = [v for v in singles if v.accepted]
        single_dev = {v.spec.name: v.max_rel_deviation
                      for v in accepted_singles}
        best_single_wall = min(
            (v.wall_ms for v in accepted_singles
             if v.wall_ms is not None), default=None)
        interference: list[str] = []
        eligible: list[CandidateVerdict] = list(accepted_singles)
        for v in combos:
            if not v.accepted:
                # deviation is only measured once signals were present,
                # so a set deviation on a rejection == budget violation
                if v.max_rel_deviation is not None:
                    interference.append(
                        f"{v.spec.name}: interference — combined drift "
                        f"rejected by the orchestrator ({v.reason})")
                continue
            names = list(v.spec.params.get("members", []))
            member_sum = sum(single_dev.get(n) or 0.0 for n in names)
            dev = v.max_rel_deviation or 0.0
            reasons = []
            if dev > member_sum + margin:
                reasons.append(
                    f"interference: combo deviation {dev:.3e} exceeds "
                    f"member single-deviation sum {member_sum:.3e} + "
                    f"margin {margin:.3e}")
            if (best_single_wall is not None and v.wall_ms is not None
                    and v.wall_ms > best_single_wall):
                reasons.append(
                    f"no composition benefit: wall {v.wall_ms:.3f} ms "
                    f"slower than best single {best_single_wall:.3f} ms")
            if reasons:
                interference.append(f"{v.spec.name}: "
                                    + "; ".join(reasons))
            else:
                eligible.append(v)
        chosen = min((v for v in eligible if v.wall_ms is not None),
                     key=lambda v: v.wall_ms, default=None)
        return ComposerReport(singles=singles, combos=combos,
                              chosen=chosen, interference=interference)

    # ---------------------------------------------------------- helpers
    def _single_runner(self, comp: Composable):
        """Runner closure: fresh wrapper per invocation (purity)."""
        def runner():
            wrapped, read = comp.make(self.step_fn)
            out = run_loop(wrapped, self.x0, self.iterations)
            return out, dict(read())
        return runner

    def _combo_runner(self, members: list[Composable]):
        """Combined runner: wraps innermost-first in member order,
        merges authenticity under per-member name prefixes.

        All wrapper instances are constructed inside the closure, so
        each of the orchestrator's timed repeats runs on cold state.
        """
        def runner():
            fn = self.step_fn
            readers = []
            for comp in members:
                fn, read = comp.make(fn)
                readers.append((comp.spec.name, read))
            out = run_loop(fn, self.x0, self.iterations)
            auth: dict = {}
            for name, read in readers:
                for sig, val in dict(read()).items():
                    key = f"{name}.{sig}"
                    if key in auth:
                        raise ValueError(
                            f"authenticity key collision on {key!r}")
                    auth[key] = val
            return out, auth
        return runner

    @staticmethod
    def _check_composables(composables: Sequence[Composable]
                           ) -> list[Composable]:
        comps = list(composables)
        if not comps:
            raise ValueError("no composables given; refusing to report "
                             "an empty search as a success")
        names = [c.spec.name for c in comps]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate composable names {names}; "
                             f"verdicts and prefixes would be ambiguous")
        return comps
