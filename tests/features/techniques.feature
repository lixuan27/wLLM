Feature: Optimization techniques cannot grade themselves
  Every technique candidate runs against the frozen exact reference on
  identical inputs; authenticity counters must prove it engaged, and the
  measured deviation must fit the declared quality budget.

  Scenario: An engaged cache within budget is accepted with evidence
    Given a smooth iterative workload and a step cache candidate
    When the technique orchestrator evaluates the candidates
    Then the cache candidate is accepted with nonzero reuse evidence
    And its receipt reports a bounded quality verdict

  Scenario: A candidate that never engaged is rejected
    Given a jumpy iterative workload and a step cache candidate
    When the technique orchestrator evaluates the candidates
    Then the cache candidate is rejected because it never engaged

  Scenario: A candidate exceeding the quality budget is rejected
    Given a smooth iterative workload and an over-aggressive cache candidate under a strict budget
    When the technique orchestrator evaluates the candidates
    Then the cache candidate is rejected for exceeding the budget
