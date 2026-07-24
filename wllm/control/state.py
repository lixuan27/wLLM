"""Apply / rollback: the user always has a correct path back.

Deployment state is a small on-disk machine under `.wllm/state.json`:

    active plan  ->  last-known-good  ->  reference

`apply` promotes a receipt-backed plan (fail-closed on promote
problems), demoting the previous active plan to last-known-good.
`rollback` walks the chain one step; from `reference` it is a no-op
that still succeeds — the reference path can never be rolled away.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .receipt import Receipt

REFERENCE = "reference"


@dataclass
class DeployState:
    active: str = REFERENCE
    last_known_good: str = REFERENCE
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DeployManager:
    def __init__(self, workdir: str | Path):
        self.workdir = Path(workdir)
        self.state_path = self.workdir / "state.json"
        self.receipts_dir = self.workdir / "receipts"

    # ---------------------------------------------------------------- state
    def state(self) -> DeployState:
        if not self.state_path.exists():
            return DeployState()
        d = json.loads(self.state_path.read_text())
        return DeployState(**d)

    def _write(self, st: DeployState, event: str, detail: str = "") -> None:
        st.history.append({"t": time.time(), "event": event,
                           "detail": detail, "active": st.active})
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(st.to_dict(), indent=1))

    # ---------------------------------------------------------------- apply
    def apply(self, receipt: Receipt, quality_policy: str = "exact") -> DeployState:
        problems = receipt.promote_problems(quality_policy)
        if problems:
            raise PermissionError(
                "refusing to apply plan "
                f"{receipt.plan_id!r}: " + "; ".join(problems))
        st = self.state()
        if st.active != receipt.plan_id:
            st.last_known_good = st.active
        st.active = receipt.plan_id
        receipt.save(self.receipts_dir)
        self._write(st, "apply", receipt.plan_id)
        return st

    # -------------------------------------------------------------- rollback
    def rollback(self) -> DeployState:
        st = self.state()
        if st.active == REFERENCE:
            self._write(st, "rollback", "already at reference (no-op)")
            return st
        if st.last_known_good != st.active and st.last_known_good != REFERENCE:
            st.active, st.last_known_good = st.last_known_good, REFERENCE
        else:
            st.active = REFERENCE
            st.last_known_good = REFERENCE
        self._write(st, "rollback", f"now {st.active}")
        return st

    # ---------------------------------------------------------------- query
    def active_receipt(self) -> Receipt | None:
        st = self.state()
        if st.active == REFERENCE:
            return None
        path = self.receipts_dir / f"{st.active}.json"
        if not path.exists():
            return None
        return Receipt.load(path)
