Feature: Agent-operable optimization control plane
  A coding agent triggers optimization with one sentence; the wLLM
  control plane — not the agent — decides what is legal, measures what
  is real, and always leaves a correct path back.

  Scenario: One-sentence optimize on a known model
    Given a project directory with a runnable entrypoint and a model config
    When the agent runs wllm inspect
    Then a project manifest exists with at least 1 entrypoint
    When the agent plans for model "Wan-AI/Wan2.2-TI2V-5B" with 2 GPUs and CFG enabled
    Then the plan keeps pass "cfg_branch_parallel"
    And the plan rejects pass "torch_compile_max_autotune" citing the quality policy

  Scenario: Measured receipt promotes and rolls back
    Given a measured receipt with passing checks and a real speedup
    When the agent applies the receipt
    Then the active plan is the receipt's plan
    When the agent rolls back
    Then the active plan is "reference"

  Scenario: Silent fallback invalidates a candidate
    Given a measured receipt whose log matched a forbidden fallback pattern
    When the agent tries to apply the receipt
    Then the apply is refused citing "silent fallback"
    And the active plan is "reference"

  Scenario: Unmeasured claims are void
    Given a receipt claiming success but carrying no performance numbers
    When the agent tries to apply the receipt
    Then the apply is refused citing "without measurement"

  Scenario: Unknown model gets diagnose-only, never a fake win
    Given a project directory with a runnable entrypoint and a model config
    When the agent plans for unknown model "nobody/mystery"
    Then planning ends in diagnose-only mode with the reference path intact
