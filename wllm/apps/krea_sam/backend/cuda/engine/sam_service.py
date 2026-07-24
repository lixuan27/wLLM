"""SAM 3 segmentation service (one process, one GPU).

Runs the SAM stream predictor concurrently with the Krea service. The
per-frame SAM inference + mask post-processing is vendored verbatim from
the reference worker so output is byte-faithful; only the *scheduling*
(running on its own device, overlapped with Krea) differs.

Launched as an independent subprocess by the coordinator with its own
CUDA_VISIBLE_DEVICES and a scrubbed torch.distributed env. Talks to the
coordinator over an AF_UNIX connection (``ipc.py``).
"""

from __future__ import annotations

import argparse
import os
from wllm.serving.paths import app_dir, repo_root
import sys
import time

import numpy as np


from wllm.apps.krea_sam.backend.cuda.engine.ipc import connect_to_coordinator
from wllm.apps.krea_sam.reference.config import KreaSAMReferenceConfig

_REPO_ROOT = repo_root()
# A committed still image containing a clearly-detectable person, used to warm
# SAM's compiled per-object modules on the live object count (see below).
_WARMUP_IMAGE = os.path.join(_REPO_ROOT, "assets", "fashion_blogger.jpg")


def _load_warmup_frames(n):
    """`n` copies of a real person image for compile warmup. SAM resizes every
    input to a fixed image_size x image_size before its compiled backbone
    (_preprocess_raw_image in sam3_stream_inference.py), so the compile is
    dimension-agnostic — the warmup image can be any resolution. What matters is
    only that it contains a detectable person, so SAM's dynamic=False per-object
    modules (mask decoder, tracker memory, detector decoder) compile for the live
    object count (1 person) rather than 0 — which is what zeros/noise detect, and
    what forced the multi-second recompile on the first real frame. Returns [] if
    the asset / cv2 is unavailable (caller falls back to zeros, which still warms
    the per-frame image backbone, just not the per-object path)."""
    try:
        import cv2
        img = cv2.imread(_WARMUP_IMAGE)  # BGR uint8, native resolution
        if img is None:
            print(f"[sam] warmup image not found at {_WARMUP_IMAGE}; falling back to zeros",
                  flush=True)
            return []
        img = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.uint8)
        return [img for _ in range(n)]
    except Exception as e:
        print(f"[sam] warmup-frame load failed ({e}); falling back to zeros", flush=True)
        return []


class SAMRunner:
    def __init__(self, cfg, device: str, compile_model: bool):
        from sam3.model_builder import build_sam3_stream_predictor
        self.cfg = cfg
        self.predictor = build_sam3_stream_predictor(device=device, compile=compile_model)
        self.session_id = None
        self._prompt_set = False
        self._frame_index = 0
        if compile_model:
            try:
                self.predictor.handle_request({"type": "warm_up_compilation"})
            except Exception as e:
                print(f"[sam] warm_up_compilation failed (continuing): {e}", flush=True)
        # Drive a throwaway session of real-shaped frames so SAM's kernels are
        # tuned BEFORE we report ready — for BOTH the compiled model (whose
        # max-autotune-no-cudagraphs autotunes lazily on the first real inference,
        # sam3_stream_inference.py:548) AND the eager model (whose cudnn.benchmark
        # tunes on first use). Otherwise the first live frame pays it. One-time
        # cost; the inductor/cudnn cache stays warm.
        print("[sam] warming SAM on a real person image ...", flush=True)
        H, W = int(cfg.height), int(cfg.width)
        n_frames = int(cfg.chunk_size) * int(cfg.vae_config.scale_factor_temporal)
        # A real person image so the compiled per-object modules (dynamic=False)
        # specialize on the live object count NOW, not on the first live frame.
        # Zeros only warm the per-frame image backbone/encoder; they detect 0
        # objects, so the mask decoder / tracker recompile (many seconds) the
        # first time a real person is detected. SAM resizes any input to a fixed
        # image_size internally, so the image's own resolution doesn't matter.
        # Falls back to zeros if the image is missing.
        warm_frames = _load_warmup_frames(n_frames)
        if not warm_frames:
            warm_frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(n_frames)]
        self.start_session()
        for fr in warm_frames:
            self.run(fr[np.newaxis])   # [1, H, W, 3]
        self.close_session()
        print("[sam] warm", flush=True)

    def start_session(self):
        self.close_session()
        resp = self.predictor.handle_request({"type": "start_session"})
        self.session_id = resp["session_id"]
        self._prompt_set = False
        self._frame_index = 0

    def close_session(self):
        if self.session_id is None:
            return
        try:
            self.predictor.handle_request({"type": "close_session", "session_id": self.session_id})
        except Exception:
            pass
        self.session_id = None
        self._prompt_set = False
        self._frame_index = 0

    def run(self, frames_np: np.ndarray):
        if self.session_id is None:
            return None
        T, H, W, _ = frames_np.shape
        out = np.zeros((T, H, W), dtype=np.uint8)
        score_thresh = float(self.cfg.sam_min_score)
        mask_thresh = float(self.cfg.sam_mask_threshold)
        dilate_px = int(self.cfg.sam_dilate_pixels)
        cv2 = None
        for i in range(T):
            self.predictor.handle_request(
                {"type": "add_frame", "session_id": self.session_id, "frame": frames_np[i]})
            if not self._prompt_set:
                resp = self.predictor.handle_request(
                    {"type": "add_prompt", "session_id": self.session_id,
                     "frame_index": self._frame_index, "text": self.cfg.sam_text_prompt})
                self._prompt_set = True
            else:
                resp = self.predictor.handle_request(
                    {"type": "run_inference", "session_id": self.session_id,
                     "frame_index": self._frame_index})
            self._frame_index += 1
            outputs = (resp or {}).get("outputs") or {}
            raw_masks = outputs.get("out_binary_masks")
            raw_probs = outputs.get("out_probs")
            masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
            probs = list(raw_probs) if raw_probs is not None and len(raw_probs) > 0 else []
            if not masks or not probs:
                continue
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
                        cv2 = _cv2
                    m_np = cv2.resize(m_np.astype(np.float32), (W, H),
                                      interpolation=cv2.INTER_NEAREST)
                mask_union |= (m_np > mask_thresh)
            if dilate_px > 0:
                if cv2 is None:
                    import cv2 as _cv2
                    cv2 = _cv2
                k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
                mask_union = cv2.dilate(mask_union.astype(np.uint8), k) > 0
            out[i] = mask_union.astype(np.uint8) * 255
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--address", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    ref_cfg = KreaSAMReferenceConfig.from_yaml(args.cfg, is_path=True)
    cfg = ref_cfg.to_runtime_config()

    import torch
    from wllm.serving.utils.torch_utils import set_torch_options
    set_torch_options()
    torch.cuda.set_device(0)

    if args.compile:
        # SAM's compiled neck caches a tensor that a subsequent cudagraph run
        # overwrites (necks.py:116 -> "accessing tensor output of CUDAGraphs
        # that has been overwritten"). Keep inductor kernel fusion but disable
        # cudagraph capture so the cache stays valid across frames. Set both
        # the env var (authoritative, read at compile time) and the configs.
        os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
        for attr, val in (("triton.cudagraphs", False), ("triton.cudagraph_trees", False)):
            try:
                obj = torch._inductor.config
                *path, leaf = attr.split(".")
                for p in path:
                    obj = getattr(obj, p)
                setattr(obj, leaf, val)
            except Exception:
                pass

    print("[sam] building SAM stream predictor ...", flush=True)
    runner = SAMRunner(cfg, device="cuda:0", compile_model=args.compile)
    print("[sam] ready", flush=True)

    conn = connect_to_coordinator(args.address)
    conn.send({"ack": "ready"})

    try:
        while True:
            msg = conn.recv()
            cmd = msg.get("cmd")
            if cmd == "stop":
                break
            elif cmd == "start":
                runner.start_session()
                conn.send({"ack": "start"})
            elif cmd == "reset":
                runner.start_session()
                conn.send({"ack": "reset"})
            elif cmd == "chunk":
                masks = runner.run(msg["frames"])
                conn.send({"id": msg["id"], "out": masks})
    finally:
        runner.close_session()
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
