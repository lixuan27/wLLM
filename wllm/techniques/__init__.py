"""Optimization technique executors with authenticity accounting.

Each technique wraps a workload (an iterative denoise loop, a set of
linear layers) and reports *authenticity signals* — counters proving the
optimization actually engaged (cache reuses > 0, layers quantized > 0).
The orchestrator runs the exact reference and every candidate on the
same inputs, measures deviation against an explicit quality budget, and
fails closed: no signals, budget exceeded, or crash means the candidate
is rejected with a reason, never quietly kept.

    base.py          TechniqueSpec / TechniqueResult / quality budgets
    step_cache.py    residual step cache for iterative loops
    quant_sim.py     int8 weight-quantization simulation for linears
    orchestrator.py  reference-vs-candidate evaluation, fail-closed
"""

from .base import QualityBudget, TechniqueResult, TechniqueSpec
from .orchestrator import CandidateVerdict, TechniqueOrchestrator
from .quant_sim import QuantizedLinears
from .step_cache import StepResidualCache

__all__ = [
    "TechniqueSpec", "TechniqueResult", "QualityBudget",
    "StepResidualCache", "QuantizedLinears",
    "TechniqueOrchestrator", "CandidateVerdict",
]
