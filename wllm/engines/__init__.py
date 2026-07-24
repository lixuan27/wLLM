"""Late bindings to external serving engines.

wLLM orchestrates model stages; some stages delegate heavy serving work
to external engines installed in the worker environment. Engines are
never vendored into this tree — each binding module resolves the
installed package at runtime from an environment variable, so the same
app code runs against whichever compatible engine build a site provides.
"""
