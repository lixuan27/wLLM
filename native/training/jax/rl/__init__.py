"""JAX-side RL primitives mirroring ``training.rl`` for parity.

Most of the algorithm — ACP tags (`wllm_native.core.rl.acp_tags`),
N-step advantage, per-task threshold (`wllm_native.core.rl.advantage`),
and the numpy reward primitives (`compute_episode_value_targets`,
`compute_dense_rewards_from_targets` in `wllm_native.core.rl.reward`)
is **framework-agnostic** and imported directly from the shared
package; this module only re-implements the bits that touched
torch tensors (distributional value-function head + soft loss).
"""
