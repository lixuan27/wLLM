"""Model profile pack: verified compatibility contracts, not docs.

A profile records what a model family is, where it runs, which
optimizations are lossless vs bounded, which combinations are known
incompatible, which authenticity signals prove an optimization engaged,
and what validation applies — every claim with an evidence pointer,
every profile with a binding that expires. Claims without evidence are
invalid; stale profiles are flagged, never silently trusted.
"""

from .schema import (
    EvidenceRef, ModelProfile, OptimizationEntry, ProfileBinding,
    RuntimeSupport,
)
from .store import load_profiles, match, stale_report

__all__ = [
    "EvidenceRef", "ModelProfile", "OptimizationEntry", "ProfileBinding",
    "RuntimeSupport", "load_profiles", "match", "stale_report",
]
