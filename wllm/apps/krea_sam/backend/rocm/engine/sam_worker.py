"""Decoupled SAM3 worker process.

Loads a SAM3 stream predictor on its own GPU and continuously segments the raw
frame stream the orchestrator pushes through a SamLink, writing one [H,W] uint8
body mask per frame back. Runs a single continuous session (SAM2-style tracker
memory is sequential), so this is intrinsically one GPU per session — its value
is running *concurrently* with Krea (IR: sam_segment || all Krea ops).

Per-frame masking logic is a verbatim port of KreaSAMWorker._run_sam so the
masks match the reference bit-for-bit given identical frames.

Optional: KREA_SAM_COMPILE=1 torch.compile()s the SAM image backbone.
"""

from __future__ import annotations

import os
from wllm.serving.paths import app_dir, repo_root
import sys
import time

import numpy as np

from wllm.serving.logger import init_logger
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig
from wllm.apps.krea_sam.backend.rocm.engine.sam_link import SamLink

logger = init_logger("sam_worker")

_REPO_ROOT = repo_root()
# A committed still of a clearly detectable person, used to warm SAM's
# per-object modules on a realistic object count (see _load_warmup_frames).
_WARMUP_IMAGE = os.path.join(_REPO_ROOT, "assets", "fashion_blogger.jpg")


def _mask_for_frame(sam, session_id, frame_index, prompt_set, frame_np, cfg, cv2_box):
    """One SAM step + mask post-process (port of worker._run_sam body, per frame).
    Returns (mask[H,W] uint8, prompt_set)."""
    H, W, _ = frame_np.shape
    sam.handle_request({"type": "add_frame", "session_id": session_id, "frame": frame_np})
    if not prompt_set:
        resp = sam.handle_request({"type": "add_prompt", "session_id": session_id,
                                   "frame_index": frame_index, "text": cfg.sam_text_prompt})
        prompt_set = True
    else:
        resp = sam.handle_request({"type": "run_inference", "session_id": session_id,
                                   "frame_index": frame_index})
    out = np.zeros((H, W), dtype=np.uint8)
    outputs = (resp or {}).get("outputs") or {}
    raw_masks = outputs.get("out_binary_masks")
    raw_probs = outputs.get("out_probs")
    masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
    probs = list(raw_probs) if raw_probs is not None and len(raw_probs) > 0 else []
    if not masks or not probs:
        return out, prompt_set
    score_thresh = float(cfg.sam_min_score)
    mask_thresh = float(cfg.sam_mask_threshold)
    dilate_px = int(cfg.sam_dilate_pixels)
    cv2 = cv2_box[0]
    mask_union = np.zeros((H, W), dtype=bool)
    for m, s in zip(masks, probs):
        try:
            score = float(s)
        except (TypeError, ValueError):
            score = 0.0
        if score < score_thresh:
            continue
        m_np = np.asarray(m)
        if m_np.ndim == 3 and m_np.shape[0] == 1:
            m_np = m_np[0]
        if m_np.shape != (H, W):
            if cv2 is None:
                import cv2 as _cv2
                cv2 = _cv2; cv2_box[0] = _cv2
            m_np = cv2.resize(m_np.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_union |= (m_np > mask_thresh)
    if dilate_px > 0:
        if cv2 is None:
            import cv2 as _cv2
            cv2 = _cv2; cv2_box[0] = _cv2
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
        mask_union = cv2.dilate(mask_union.astype(np.uint8), k) > 0
    out[:] = mask_union.astype(np.uint8) * 255
    return out, prompt_set


def _load_warmup_frames(n):
    """``n`` copies of a real person image to warm SAM with.

    SAM resizes every input to a fixed image_size before its backbone, so the
    image's own resolution is irrelevant. What matters is that it contains a
    detectable person: SAM's ``dynamic=False`` per-object modules (mask decoder,
    tracker memory, detector decoder) then specialize on the live object count
    up front instead of recompiling — which takes seconds — on the first real
    frame. Noise and zeros detect no objects, so they warm only the per-frame
    image backbone. Returns [] if the asset or cv2 is unavailable, and the
    caller falls back.
    """
    try:
        import cv2
        img = cv2.imread(_WARMUP_IMAGE)   # BGR uint8, native resolution
        if img is None:
            logger.warning("SAM warmup image missing at %s; falling back to noise",
                           _WARMUP_IMAGE)
            return []
        return [np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     dtype=np.uint8) for _ in range(n)]
    except Exception as e:
        logger.warning("SAM warmup image load failed (%s); falling back to noise", e)
        return []


def _warmup_sam(sam, cfg, H, W):
    """Drive one chunk's worth of frames through a throwaway SAM session so the
    lazy-compile and first-frame costs are paid before the backend reports
    ready. The session is independent of the real one (keyed by session_id), so
    tracking correctness is unaffected."""
    n_frames = int(cfg.chunk_size) * int(cfg.vae_config.scale_factor_temporal)
    frames = _load_warmup_frames(n_frames)
    if not frames:
        # Some structure, so the detector has content to work on: a zero frame
        # can error inside run_inference when nothing is detected.
        noise = np.random.default_rng(0).integers(0, 255, (H, W, 3)).astype(np.uint8)
        frames = [noise for _ in range(n_frames)]
    resp = sam.handle_request({"type": "start_session"})
    sid = resp["session_id"]
    prompt_set = False
    for i, frame in enumerate(frames):
        try:
            sam.handle_request({"type": "add_frame", "session_id": sid, "frame": frame})
            if not prompt_set:
                sam.handle_request({"type": "add_prompt", "session_id": sid,
                                    "frame_index": i, "text": cfg.sam_text_prompt})
                prompt_set = True
            else:
                sam.handle_request({"type": "run_inference", "session_id": sid, "frame_index": i})
        except Exception as e:
            logger.warning("SAM warmup frame %d: %s", i, e)
    try:
        sam.handle_request({"type": "close_session", "session_id": sid})
    except Exception:
        pass


def main():
    cfg_path = sys.argv[1]
    link_name = sys.argv[2]
    device_str = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"

    ref_cfg = KreaSAMReferenceConfig.from_yaml(cfg_path, is_path=True)
    cfg = ref_cfg.to_runtime_config()
    H, W = int(cfg.height), int(cfg.width)
    max_len = int(cfg.video_input_max_frames)

    import torch
    torch.cuda.set_device(torch.device(device_str))
    from sam3.model_builder import build_sam3_stream_predictor
    compile_sam = os.environ.get("KREA_SAM_COMPILE", "0") == "1"
    logger.info("SAM worker loading predictor on %s (compile=%s)", device_str, compile_sam)
    # NB: SAM's own compile=True uses mode="max-autotune" (CUDA graphs), which
    # crashes on the neck's tensor cache ("accessing tensor output of CUDAGraphs
    # that has been overwritten", necks.py). We instead compile the same hot
    # submodules ourselves with "max-autotune-no-cudagraphs".
    sam = build_sam3_stream_predictor(device=device_str, compile=False)
    if compile_sam:
        _compile_sam_no_cudagraphs(sam)

    # Warm SAM before reporting ready, on both paths: the compiled model
    # autotunes lazily on its first real inference (minutes), and the eager
    # model tunes cudnn.benchmark on first use. Either way the first live frame
    # would otherwise pay it.
    try:
        _warmup_sam(sam, cfg, H, W)
    except Exception:
        logger.warning("SAM warmup failed (non-fatal)", exc_info=True)

    link = SamLink(link_name, H, W, max_len, create=False)
    link.set_ready(0)  # loaded, waiting for a session
    logger.info("SAM worker up (loaded)")

    my_epoch = 0
    session_id = None
    prompt_set = False
    frame_index = 0
    read_cursor = 0
    cv2_box = [None]

    def close_session():
        nonlocal session_id
        if session_id is not None:
            try:
                sam.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
            session_id = None

    try:
        while True:
            if link.terminate_flag:
                break
            if link.epoch > my_epoch:
                # new session (start or reset)
                close_session()
                my_epoch = link.epoch
                resp = sam.handle_request({"type": "start_session"})
                session_id = resp["session_id"]
                prompt_set = False
                frame_index = 0
                read_cursor = 0
                link.set_done(0)
                link.set_ready(my_epoch)
                logger.info("SAM worker started session epoch=%d", my_epoch)

            if session_id is None:
                time.sleep(0.002)
                continue

            # process any newly-arrived frames
            avail = link.frames.num - read_cursor
            if avail <= 0:
                time.sleep(0.001)
                continue
            nxt, frames = link.frames.read(read_cursor, avail)
            if frames is None:
                time.sleep(0.001)
                continue
            for i in range(frames.shape[0]):
                mask, prompt_set = _mask_for_frame(
                    sam, session_id, frame_index, prompt_set,
                    np.ascontiguousarray(frames[i]), cfg, cv2_box)
                link.masks.write(mask)
                frame_index += 1
                link.set_done(frame_index)
            read_cursor = nxt
    finally:
        close_session()
        link.close()
        logger.info("SAM worker exiting")


def _compile_sam_no_cudagraphs(sam):
    """torch.compile the hot SAM detector submodules (the per-frame ViT backbone
    dominates) with max-autotune BUT no CUDA graphs, avoiding the cudagraph
    output-aliasing crash SAM's own compile=True path hits on the neck cache."""
    import torch
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.accumulated_cache_size_limit = 2048
    torch._dynamo.config.suppress_errors = True
    model = getattr(sam, "model", sam)
    det = getattr(model, "detector", None)
    if det is None:
        logger.warning("SAM: no detector to compile; running eager")
        return
    mode = "max-autotune-no-cudagraphs"
    targets = [
        ("backbone.vision_backbone", getattr(getattr(det, "backbone", None), "vision_backbone", None)),
        ("transformer.encoder", getattr(getattr(det, "transformer", None), "encoder", None)),
        ("transformer.decoder", getattr(getattr(det, "transformer", None), "decoder", None)),
        ("segmentation_head", getattr(det, "segmentation_head", None)),
    ]
    n = 0
    for name, mod in targets:
        if mod is not None and isinstance(mod, torch.nn.Module):
            try:
                mod.forward = torch.compile(mod.forward, fullgraph=True, mode=mode)
                n += 1
            except Exception as e:
                logger.warning("SAM compile of %s failed: %s", name, e)
    logger.info("SAM: compiled %d submodules with %s", n, mode)


def _maybe_compile_backbone(sam):
    """Best-effort torch.compile of the SAM image backbone (the per-frame ViT
    that dominates SAM latency). Kept defensive: if the attribute path or
    compile fails, SAM still runs eager."""
    import torch
    model = getattr(sam, "model", sam)
    candidates = ["image_encoder", "backbone", "vision_encoder", "image_backbone", "trunk"]
    for attr in candidates:
        mod = getattr(model, attr, None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            try:
                setattr(model, attr, torch.compile(mod, mode="max-autotune-no-cudagraphs", dynamic=True))
                logger.info("SAM: torch.compiled backbone '%s'", attr)
                return
            except Exception as e:
                logger.warning("SAM compile of %s failed: %s", attr, e)
    logger.warning("SAM: no compileable backbone attribute found (tried %s)", candidates)


if __name__ == "__main__":
    main()
