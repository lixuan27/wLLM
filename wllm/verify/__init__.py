"""wBench verification core.

* :mod:`wllm.verify.numerical` — level-B tolerance comparison of nested
  reference/candidate outputs under a quality contract.
* :mod:`wllm.verify.adjudicate` — token-disagreement adjudication
  (epsilon-optimal set + prefill/decode dual-path consistency).  The
  decision core is backend-free; torch is imported lazily by its model
  adapters only.
"""
