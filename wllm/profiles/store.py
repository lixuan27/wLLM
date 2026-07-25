"""Profile store: load, validate, match, and expire model profiles."""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import ModelProfile

DATA_DIR = Path(__file__).parent / "data"


def load_profiles(directory: str | Path = DATA_DIR
                  ) -> dict[str, ModelProfile]:
    """Load every profile YAML; any validation problem rejects the load.

    Aggregates problems per file so a broken pack reports everything at
    once instead of failing one field at a time.
    """
    profiles: dict[str, ModelProfile] = {}
    problems: list[str] = []
    for f in sorted(Path(directory).glob("*.yaml")):
        doc = yaml.safe_load(f.read_text()) or {}
        if not isinstance(doc, dict):
            problems.append(f"{f.name}: profile root is not a mapping")
            continue
        try:
            prof = ModelProfile.from_dict(doc, origin=f.name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        errs = prof.validate()
        if errs:
            problems.extend(f"{f.name}: {e}" for e in errs)
            continue
        if prof.model_family in profiles:
            problems.append(f"{f.name}: duplicate model_family "
                            f"{prof.model_family!r}")
            continue
        profiles[prof.model_family] = prof
    if problems:
        raise ValueError("profile pack rejected:\n  " +
                         "\n  ".join(problems))
    return profiles


def match(profiles: dict[str, ModelProfile],
          model_id: str) -> ModelProfile | None:
    """First profile whose detection ids match; None means unknown model
    (the caller goes to diagnose-only, never to a guessed profile)."""
    for prof in profiles.values():
        if prof.matches(model_id):
            return prof
    return None


def stale_report(profiles: dict[str, ModelProfile], today: str,
                 max_age_days: int = 90) -> list[str]:
    """Families whose binding has expired, with their last_validated."""
    out = []
    for family, prof in sorted(profiles.items()):
        if prof.is_stale(today, max_age_days):
            out.append(f"{family}: last validated "
                       f"{prof.binding.last_validated or '(never)'}")
    return out
