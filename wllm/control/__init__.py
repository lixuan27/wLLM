"""wLLM control plane: the agent-facing, agent-independent command surface.

The agent's only job is to turn intent into a typed OptimizeSpec and call
these entrypoints; everything that decides whether an optimization is
legal, faster, and quality-preserving lives here and runs identically
in CI with no agent present.

    spec.py      typed OptimizeSpec (the only thing the optimizer reads)
    inspect.py   project discovery -> evidence-listing manifest
    registry.py  declarative backend/pass capabilities + fail-closed scans
    receipt.py   measured-evidence receipts + deployment fingerprints
    state.py     apply / rollback chain (optimized -> known-good -> reference)
    cli.py       wllm inspect|optimize|verify|apply|rollback|report
"""
