"""wGraph state contracts.

A state is any value that survives beyond a single node invocation and
therefore constrains how the program may be re-scheduled, re-placed, or
parallelized.  The planner consumes only states whose contract has been
*verified* by counterfactual probes (see wllm.capture.probes); an agent
hypothesis alone never unlocks a transformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StateKind(str, Enum):
    IMMUTABLE_SESSION = "immutable_session"    # write-once at session start
    RECOMPUTABLE_FEATURE = "recomputable_feature"  # cache; can be rebuilt from inputs
    KV = "kv"                                  # attention KV cache, append-ordered
    RECURRENT = "recurrent"                    # fixed-size hidden state (e.g. GDN)
    ROLLING_CONTEXT = "rolling_context"        # sliding window / sink tokens
    STOCHASTIC = "stochastic"                  # RNG state; replay-sensitive
    FEEDBACK_CRITICAL = "feedback_critical"    # real-world observation; staleness-bound
    DEADLINE_BOUND = "deadline_bound"          # value only valid before a deadline
    MULTI_AGENT = "multi_agent"                # partitioned per agent/view identity


class StateScope(str, Enum):
    REQUEST = "request"
    SESSION = "session"
    CHUNK = "chunk"
    AGENT = "agent"


class DeadlinePolicy(str, Enum):
    NONE = "none"
    REJECT_STALE = "reject_stale"
    RECOMPUTE = "recompute"
    BEST_EFFORT = "best_effort"


@dataclass
class StateSpec:
    """Declarative contract for one piece of persistent state."""

    id: str
    kind: StateKind
    scope: StateScope = StateScope.SESSION
    ordered: bool = True          # updates must apply in sequence order
    recomputable: bool = False    # can be rebuilt from upstream inputs
    migratable: bool = False      # may move across devices mid-session
    forkable: bool = False        # may be copied for branched rollouts
    owner: str | None = None      # node/region id with exclusive write right
    max_staleness_ms: float | None = None
    deadline_policy: DeadlinePolicy = DeadlinePolicy.NONE
    memory_bytes: int | None = None
    partition_key: str | None = None   # MULTI_AGENT: identity field, e.g. "player_id"
    # Contract provenance: planner trusts only verified=True.
    verified: bool = False
    evidence: str | None = None   # path to probe evidence (log/json)
    description: str = ""

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.id:
            errs.append("state id must be non-empty")
        if self.kind == StateKind.FEEDBACK_CRITICAL and self.max_staleness_ms is None:
            errs.append(f"state '{self.id}': feedback_critical requires max_staleness_ms")
        if self.kind == StateKind.MULTI_AGENT and not self.partition_key:
            errs.append(f"state '{self.id}': multi_agent requires partition_key")
        if self.max_staleness_ms is not None and self.max_staleness_ms <= 0:
            errs.append(f"state '{self.id}': max_staleness_ms must be > 0")
        if self.memory_bytes is not None and self.memory_bytes < 0:
            errs.append(f"state '{self.id}': memory_bytes must be >= 0")
        if self.verified and not self.evidence:
            errs.append(f"state '{self.id}': verified=True requires evidence pointer")
        return errs
