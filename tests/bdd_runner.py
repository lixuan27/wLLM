"""Minimal gherkin runner: .feature files drive real control-plane code.

Zero external dependencies (offline cluster): parses Feature/Scenario/
Given-When-Then-And-But, matches steps against regex-registered Python
functions, and executes scenarios against a per-scenario context dict.
Unmatched steps fail the scenario — a feature file can never silently
document behavior that nothing verifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_KEYWORDS = ("Given ", "When ", "Then ", "And ", "But ")


class StepRegistry:
    def __init__(self):
        self._steps: list[tuple[re.Pattern, object]] = []

    def step(self, pattern: str):
        def deco(fn):
            self._steps.append((re.compile(pattern), fn))
            return fn
        return deco

    # gherkin keywords are aliases: the text decides the match
    given = when = then = step

    def resolve(self, text: str):
        matches = [(pat, fn) for pat, fn in self._steps
                   if pat.fullmatch(text)]
        if not matches:
            raise LookupError(f"no step definition matches: {text!r}")
        if len(matches) > 1:
            raise LookupError(f"ambiguous step: {text!r}")
        pat, fn = matches[0]
        return pat, fn


@dataclass
class Scenario:
    name: str
    steps: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    scenario: str
    passed: bool
    failed_step: str = ""
    error: str = ""


def parse_feature(path: str | Path) -> tuple[str, list[Scenario]]:
    feature = ""
    scenarios: list[Scenario] = []
    current: Scenario | None = None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Feature:"):
            feature = line[len("Feature:"):].strip()
        elif line.startswith("Scenario:"):
            current = Scenario(name=line[len("Scenario:"):].strip())
            scenarios.append(current)
        elif any(line.startswith(k) for k in _KEYWORDS):
            if current is None:
                raise ValueError(f"{path}: step before first Scenario")
            text = line.split(" ", 1)[1].strip()
            current.steps.append(text)
        # freeform description lines are ignored
    if not feature:
        raise ValueError(f"{path}: no Feature declared")
    return feature, scenarios


def run_feature(path: str | Path, registry: StepRegistry
                ) -> list[ScenarioResult]:
    _, scenarios = parse_feature(path)
    if not scenarios:
        raise ValueError(f"{path}: feature has no scenarios")
    results: list[ScenarioResult] = []
    for sc in scenarios:
        ctx: dict = {}
        outcome = ScenarioResult(scenario=sc.name, passed=True)
        for text in sc.steps:
            try:
                pat, fn = registry.resolve(text)
                fn(ctx, *pat.fullmatch(text).groups())
            except Exception as exc:  # noqa: BLE001
                outcome.passed = False
                outcome.failed_step = text
                outcome.error = f"{type(exc).__name__}: {exc}"
                break
        results.append(outcome)
        teardown = ctx.get("_teardown")
        if callable(teardown):
            teardown()
    return results
