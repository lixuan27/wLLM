"""IR operators for the WorldPlay reactive-decoding pipeline.

These operators are a faithful, op-by-op re-expression of the reference
backend's per-chunk computation:

    worker.predict  (camera-action decode)            -> IngestActions
      pipeline.step (DiT denoise: KV-fill + 4 steps)   -> KVFill, DenoiseStep x4, WriteBack
      pipeline.step (VAE decode, latent-by-latent)     -> VAEDecodeLatent x chunk_size
      worker writes the chunk to the video buffer      -> (harness collects frames)

The whole point of the decomposition is to make the **state** structure
explicit so the analysis tools can see what is and isn't separable:

  * the DiT KV cache  (state ``kv``)        -- written by KVFill + every DenoiseStep
  * the latent history (state ``latents``)  -- written by WriteBack, read for context
  * the VAE feat cache (state ``vae``)      -- written by every VAEDecodeLatent
  * the camera accumulators (state ``cam``) -- written by IngestActions

The chunk's freshly-generated latents are handed from WriteBack to the VAE
ops as an **ephemeral data edge** (``chunk_latents``), NOT through the
persistent ``latents`` state. That is the dependency that lets
``find_pipeline_stages`` recognise that VAE-decode(chunk N) and DiT-denoise
(chunk N+1) touch disjoint persistent state and can therefore run as two
pipelined stages on different devices.

Every op carries its own ``execute`` and declares exactly the state it
touches, per wllm/ir/AGENTS.md. Implementations call the real shared-runtime
runners (DiTRunner / VAERunner) so the executor reproduces the reference's
numerics, not an approximation of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import wllm.kernels_t
from wllm.serving.ir import IROperator, OpType, TensorPort, StreamMode
from wllm.serving.utils.fov import select_mem_frames_wan


# ----------------------------------------------------------------------------
# context: everything the ops need that isn't persistent IR state
# ----------------------------------------------------------------------------

@dataclass
class WPContext:
    """Per-session helpers + per-chunk scratch handed to every op's execute().

    Persistent tensors (KV cache, VAE feat cache, latent history, camera
    accumulators) live in IR ``state``; this object holds the model runners,
    static config, and small per-chunk scratch that the executor doesn't need
    to reason about.
    """
    cfg: Any
    device: torch.device
    dtype: torch.dtype
    dit_runner: Any
    vae_runner: Any
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    points_local: torch.Tensor

    # set by the harness before each run_chunk
    chunk_i: int = 0

    # per-chunk scratch (filled by IngestActions, consumed downstream)
    scratch: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# camera accumulator state object (mirrors worker.{T,C_inv,_viewmats,...})
# ----------------------------------------------------------------------------

class CamAccum:
    def __init__(self, cfg, device):
        self.viewmats = torch.zeros(1, cfg.max_num_actions, 4, 4, device=device, dtype=torch.float32)
        self.Ks = torch.zeros(1, cfg.max_num_actions, 3, 3, device=device, dtype=torch.float32)
        self.action = torch.zeros(1, cfg.max_num_actions, device=device, dtype=torch.float32)
        self.T = torch.eye(4, dtype=torch.float32)
        self.C_inv = torch.zeros((4, 4), dtype=torch.float32)

    def reset(self):
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)


# ----------------------------------------------------------------------------
# preamble ops (session init)
# ----------------------------------------------------------------------------

class EncodeSession(IROperator):
    """Mirror BasePipeline.init_session: encode prompt + first image, fill the
    DiT cross-attention encoder KV. Writes write-once session state."""

    def __init__(self):
        super().__init__(
            name="encode_session",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("prompt"), TensorPort("image")],
            outputs=[TensorPort("first_image_condition")],
            state_reads=[],
            state_writes=["enc_kv", "first_img"],
            stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, ctx: WPContext, state):
        pipe = inputs["pipe"]
        # init_session already ran on the pipe; pull its results into IR state
        state.set("enc_kv", True)  # encoder KV lives on dit_runner.kv_memory (in place)
        fic = pipe._session_ctx["first_image_condition"]
        state.set("first_img", fic)
        return {"first_image_condition": fic}


# ----------------------------------------------------------------------------
# chunk ops
# ----------------------------------------------------------------------------

class IngestActions(IROperator):
    """worker.predict's camera-action decode + accumulator update.

    Decodes the chunk's discrete action codes into camera (view / intrinsics /
    action-label) tensors via the shared CUDA kernels and writes them into the
    running camera accumulators. Also computes the memory-frame selection for
    this chunk and (chunk 0 only) seeds the first latent with the image
    condition.
    """

    def __init__(self):
        super().__init__(
            name="ingest_actions",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("action_codes")],
            outputs=[TensorPort("ready")],
            state_reads=["first_img"],
            state_writes=["cam", "latents"],
            stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, ctx: WPContext, state):
        cfg = ctx.cfg
        chunk_i = ctx.chunk_i
        cam: CamAccum = state.get("cam")
        latents = state.get("latents")

        curr_pose = inputs["action_codes"]
        translation_codes, rotation_codes = (
            wllm.kernels_t.camera_action.decode_combined_actions(curr_pose)
        )
        curr_viewmats, curr_Ks, curr_action = (
            wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
                translation_codes, rotation_codes, cam.T, cam.C_inv,
                first_chunk=(chunk_i == 0),
            )
        )
        curr_action = wllm.kernels_t.camera_action.compute_worldplay_combined_label(
            curr_action, rotation_codes,
        )
        viewmats = curr_viewmats.unsqueeze(0)
        Ks = curr_Ks.unsqueeze(0)
        action = curr_action.unsqueeze(0)

        start_idx = chunk_i * cfg.chunk_size
        end_idx = start_idx + cfg.chunk_size
        cam.viewmats[:, start_idx:end_idx, ...] = viewmats
        cam.Ks[:, start_idx:end_idx, ...] = Ks
        cam.action[:, start_idx:end_idx, ...] = action

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = cfg.first_chunk_size
            generate_latent_num = cfg.first_chunk_size
            first_image_condition = state.get("first_img")
            latents[:, :, :1] = first_image_condition
            latents_curr = latents[:, :, :already_generate_num].to(device=ctx.device, dtype=ctx.dtype)
        else:
            already_generate_num = chunk_i * cfg.chunk_size + cfg.first_chunk_size
            latents_curr = latents[:, :, :already_generate_num].to(device=ctx.device, dtype=ctx.dtype)
            generate_latent_num = cfg.chunk_size
            current_frame_idx = chunk_i * cfg.chunk_size
            if cfg.context_window_size <= current_frame_idx < cfg.max_num_actions:
                selected_frame_indices = select_mem_frames_wan(
                    cam.viewmats[0], current_frame_idx,
                    memory_frames=cfg.context_window_size,
                    temporal_context_size=(cfg.context_window_size - cfg.chunk_size),
                    pred_latent_size=cfg.chunk_size,
                    points_local=ctx.points_local, device=ctx.device,
                )
            else:
                selected_frame_indices = list(range(0, current_frame_idx))

        ctx.scratch.update(
            start_idx=start_idx, end_idx=end_idx,
            already_generate_num=already_generate_num,
            generate_latent_num=generate_latent_num,
            selected_frame_indices=selected_frame_indices,
            latents_curr=latents_curr,
            all_window_latent_num=latents_curr.shape[2],
        )
        return {"ready": True}


def _build_timestep(ctx: WPContext, t: torch.Tensor) -> torch.Tensor:
    cfg = ctx.cfg
    chunk_i = ctx.chunk_i
    if chunk_i > 0:
        t_now = torch.full((1, cfg.chunk_size), t, device=ctx.device, dtype=ctx.timesteps.dtype)
        t_ctx = torch.full((1, cfg.first_chunk_size + (chunk_i - 1) * cfg.chunk_size),
                           cfg.stabilization_level - 1, device=ctx.device, dtype=ctx.timesteps.dtype)
        return torch.cat([t_ctx, t_now], dim=1)
    t_now = torch.full((1, cfg.chunk_size - 1), t, device=ctx.device, dtype=ctx.timesteps.dtype)
    t_ctx = torch.full((1, 1), cfg.stabilization_level - 1, device=ctx.device, dtype=ctx.timesteps.dtype)
    return torch.cat([t_ctx, t_now], dim=1)


class KVFill(IROperator):
    """First-denoise-step KV-cache fill for the selected memory frames
    (chunk_i > 0 only). Writes the DiT KV cache; produces no latent output."""

    def __init__(self):
        super().__init__(
            name="kv_fill",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("ready")],
            outputs=[TensorPort("filled")],
            state_reads=["cam"],
            state_writes=["kv"],
            stream_mode=StreamMode.BATCH,
        )

    def should_run(self, ctx: WPContext) -> bool:
        return ctx.chunk_i > 0

    def execute(self, inputs, ctx: WPContext, state):
        cfg = ctx.cfg
        cam: CamAccum = state.get("cam")
        _ = state.get("kv")  # declare in-place mutation of the KV cache
        s = ctx.scratch
        sel = s["selected_frame_indices"]
        latents_curr = s["latents_curr"]
        t = ctx.timesteps[0]
        timestep = _build_timestep(ctx, t)

        latents_cache = latents_curr[:, :, sel].clone()
        timestep_cache = timestep[:, sel].flatten()
        action_cache = cam.action[:, sel]
        viewmats_cache = cam.viewmats[:, sel]
        Ks_cache = cam.Ks[:, sel]
        kv_start_rope = 0
        kv_end_rope = len(sel) * cfg.kv_spatial
        ctx.dit_runner.run(
            latents=latents_cache, timestep=timestep_cache, is_cache=True,
            cache_start=kv_start_rope, cache_end=kv_end_rope,
            rope_start=kv_start_rope, rope_end=kv_end_rope,
            viewmats=viewmats_cache, Ks=Ks_cache, action=action_cache,
            i2v_condition=None,
        )
        return {"filled": True}


class DenoiseStep(IROperator):
    """One Euler denoising step over the chunk's generate latents (reads/writes
    the DiT KV cache and the in-flight latent scratch).

    Below-IR parallelism note (Ulysses SP): the model parallelises this op by
    sharding the *frame* dimension (`sequence_model_parallel_shard(..., dim=2)`)
    across the SP group and doing an equal-split all-to-all. The generate step
    has exactly cfg.chunk_size (=4) frames, so frame-dim SP is only feasible for
    **SP that evenly divides 4 → SP ∈ {1,2,4}**. SP=3/6/8 split 4 frames unevenly
    (or leave ranks with zero frames) and break the all-to-all. For >4 GPUs,
    scale via the DiT∥VAE stage-split (find_pipeline_stages) instead, not
    deeper frame-SP."""

    def __init__(self, step_idx: int):
        super().__init__(
            name=f"denoise_step_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("prev")],
            outputs=[TensorPort("stepped")],
            state_reads=["cam", "enc_kv"],
            state_writes=["kv", "latents"],
            stream_mode=StreamMode.BATCH,
        )
        self.step_idx = step_idx

    def execute(self, inputs, ctx: WPContext, state):
        cfg = ctx.cfg
        chunk_i = ctx.chunk_i
        i = self.step_idx
        cam: CamAccum = state.get("cam")
        _ = state.get("kv")
        _ = state.get("latents")
        s = ctx.scratch
        latents_curr = s["latents_curr"]
        generate_latent_num = s["generate_latent_num"]
        sel = s["selected_frame_indices"]
        all_window_latent_num = latents_curr.shape[2]

        t = ctx.timesteps[i]
        timestep = _build_timestep(ctx, t)

        now_window_latent_num = (len(sel) + cfg.chunk_size) if sel is not None else cfg.chunk_size
        latent_model_input = latents_curr[:, :, -generate_latent_num:].clone()
        timestep_slice = timestep[:, -generate_latent_num:].flatten()
        gen_frame_start = all_window_latent_num - generate_latent_num
        gen_frame_end = all_window_latent_num
        generate_rope_start = (now_window_latent_num - generate_latent_num) * cfg.kv_spatial
        generate_rope_end = now_window_latent_num * cfg.kv_spatial

        noise_pred = ctx.dit_runner.run(
            latents=latent_model_input, timestep=timestep_slice, is_cache=False,
            cache_start=generate_rope_start, cache_end=generate_rope_end,
            rope_start=generate_rope_start, rope_end=generate_rope_end,
            viewmats=cam.viewmats[:, gen_frame_start:gen_frame_end],
            Ks=cam.Ks[:, gen_frame_start:gen_frame_end],
            action=cam.action[:, gen_frame_start:gen_frame_end],
            i2v_condition=None,
        )

        sigma = ctx.sigmas[i]
        sigma_next = ctx.sigmas[i + 1]
        dt = sigma_next - sigma
        if chunk_i == 0:
            prev_sample = latent_model_input + dt * noise_pred
            # first_image_condition is always set in this app
            latents_curr[:, :, -cfg.first_chunk_size + 1:] = prev_sample[:, :, 1:]
        else:
            noise_pred_chunk = noise_pred[:, :, -cfg.chunk_size:]
            latents_curr_chunk = latent_model_input[:, :, -cfg.chunk_size:]
            latents_curr[:, :, -cfg.chunk_size:] = latents_curr_chunk + dt * noise_pred_chunk
        s["latents_curr"] = latents_curr
        return {"stepped": True}


class WriteBackLatents(IROperator):
    """Commit the denoised chunk into the latent history and emit the chunk's
    fresh latents as an ephemeral data edge to the VAE stage."""

    def __init__(self):
        super().__init__(
            name="writeback_latents",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("stepped")],
            outputs=[TensorPort("chunk_latents")],
            state_reads=[],
            state_writes=["latents"],
            stream_mode=StreamMode.STREAMING,  # hands latents frame-by-frame to VAE
        )

    def execute(self, inputs, ctx: WPContext, state):
        latents = state.get("latents")
        s = ctx.scratch
        latents_curr = s["latents_curr"]
        already = s["already_generate_num"]
        latents[:, :, :already, :, :] = latents_curr
        start_idx, end_idx = s["start_idx"], s["end_idx"]
        # the chunk's frames are latents[:, :, start:end]; clone so downstream VAE
        # decode is decoupled from further mutation of the history buffer.
        chunk_latents = latents[:, :, start_idx:end_idx, :, :].clone()
        return {"chunk_latents": chunk_latents}


class VAEDecodeLatent(IROperator):
    """Decode a single latent frame of the chunk through the causal VAE decoder
    (writes the VAE feat cache, which chains the per-latent decodes)."""

    def __init__(self, j: int, chunk_size: int):
        super().__init__(
            name=f"vae_decode_{j}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("chunk_latents")],
            outputs=[TensorPort("frames")],
            state_reads=[],
            state_writes=["vae"],
            stream_mode=StreamMode.STREAMING,
        )
        self.j = j
        self.chunk_size = chunk_size

    def execute(self, inputs, ctx: WPContext, state):
        _ = state.get("vae")  # declare in-place mutation of the VAE feat cache
        chunk_latents = inputs["chunk_latents"]
        s = ctx.scratch
        start_idx = s["start_idx"]
        global_l = start_idx + self.j
        latent_i = chunk_latents[:, :, self.j:self.j + 1, :, :].clone()
        video_i = ctx.vae_runner.run(latent_i, (global_l == 0))  # latent2rgb -> uint8 [B,T,H,W,3]
        frames = video_i[0].cpu().numpy()
        return {"frames": frames}


class CollectFrames(IROperator):
    """Concatenate the per-latent VAE frame bursts into the chunk's video, in
    temporal order (mirrors ``np.concatenate(chunk_video, axis=0)``)."""

    def __init__(self, chunk_size: int):
        super().__init__(
            name="collect_frames",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort(f"f{j}") for j in range(chunk_size)],
            outputs=[TensorPort("chunk_video")],
            state_reads=[],
            state_writes=[],
            stream_mode=StreamMode.STREAMING,
        )
        self.chunk_size = chunk_size

    def execute(self, inputs, ctx: WPContext, state):
        parts = [inputs[f"f{j}"] for j in range(self.chunk_size)]
        return {"chunk_video": np.concatenate(parts, axis=0)}


# ----------------------------------------------------------------------------
# worker-level COMPOSITE stages (coarse view; reuse the fine-grained ops)
# ----------------------------------------------------------------------------

class DiTStage(IROperator):
    """High-level DiT denoise stage: KV-fill + ``num_inference_steps`` Euler
    steps + write-back. A COMPOSITE wrapper over the dit_model sub-graph; its
    execute() sequences the fine-grained ops sharing the same state store."""

    def __init__(self, num_steps: int):
        super().__init__(
            name="dit_denoise",
            op_type=OpType.COMPOSITE,
            inputs=[TensorPort("ready")],
            outputs=[TensorPort("chunk_latents")],
            state_reads=["cam", "enc_kv"],
            state_writes=["kv", "latents"],
            stream_mode=StreamMode.BATCH,  # all chunk latents materialize together
            sub_graph="dit_model",
        )
        self._kv_fill = KVFill()
        self._steps = [DenoiseStep(i) for i in range(num_steps)]
        self._writeback = WriteBackLatents()

    def execute(self, inputs, ctx: WPContext, state):
        if self._kv_fill.should_run(ctx):
            self._kv_fill.execute({"ready": inputs["ready"]}, ctx, state)
        prev = {"prev": inputs["ready"]}
        for st in self._steps:
            prev = st.execute(prev, ctx, state)
        out = self._writeback.execute({"stepped": prev["stepped"]}, ctx, state)
        return {"chunk_latents": out["chunk_latents"]}


class VAEStage(IROperator):
    """High-level VAE decode stage: causal latent-by-latent decode + concat.
    COMPOSITE wrapper over the vae_model sub-graph."""

    def __init__(self, chunk_size: int):
        super().__init__(
            name="vae_decode",
            op_type=OpType.COMPOSITE,
            inputs=[TensorPort("chunk_latents")],
            outputs=[TensorPort("chunk_video")],
            state_reads=[],
            state_writes=["vae"],
            stream_mode=StreamMode.STREAMING,  # frames stream out latent-by-latent
            sub_graph="vae_model",
        )
        self._decoders = [VAEDecodeLatent(j, chunk_size) for j in range(chunk_size)]
        self._collect = CollectFrames(chunk_size)

    def execute(self, inputs, ctx: WPContext, state):
        parts = {}
        for j, dec in enumerate(self._decoders):
            out = dec.execute({"chunk_latents": inputs["chunk_latents"]}, ctx, state)
            parts[f"f{j}"] = out["frames"]
        return self._collect.execute(parts, ctx, None)


class VideoWrite(IROperator):
    """Terminal stage: the chunk's frames are published to the video buffer.
    In the IR executor this is a pass-through that surfaces the external
    output; the real backend writes them to the shared-memory video buffer."""

    def __init__(self):
        super().__init__(
            name="video_write",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("chunk_video")],
            outputs=[TensorPort("published")],
            state_reads=[],
            state_writes=[],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: WPContext, state):
        return {"published": inputs["chunk_video"]}
