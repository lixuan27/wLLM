"""VLA action server — runs in the model's own environment (multi-env
worker pattern): loads the LIBERO-finetuned VLA once, serves action
predictions over a Unix domain socket with a length-prefixed pickle
protocol.  Dtype comes from WLLM_VLA_DTYPE (float32 | bfloat16) so the
same server script anchors both arms of the precision comparison."""

from __future__ import annotations

import os
import pickle
import socket
import struct
import sys
import time

MODEL_DIR = os.environ.get(
    "WLLM_VLA_CKPT",
    "/public/home/lixuan/lixuan/pretrained-model/openvla-7b-oft-libero-all")
SOCK = os.environ.get("WLLM_VLA_SOCK", "/tmp/wllm_vla.sock")
DTYPE = os.environ.get("WLLM_VLA_DTYPE", "bfloat16")


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return buf


def _send_obj(conn, obj):
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">Q", len(payload)) + payload)


def _recv_obj(conn):
    (n,) = struct.unpack(">Q", _recv_exact(conn, 8))
    return pickle.loads(_recv_exact(conn, n))


def main() -> int:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    dtype = getattr(torch, DTYPE)
    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=dtype,
        low_cpu_mem_usage=True).to("cuda").eval()
    keys = sorted(getattr(model, "norm_stats", {}).keys())
    unnorm = next((k for k in keys if "libero_spatial" in k),
                  keys[0] if keys else None)
    print(f"[server] loaded dtype={DTYPE} in {time.monotonic()-t0:.0f}s "
          f"unnorm={unnorm} keys={keys[:4]}", flush=True)

    def to_host(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy().reshape(-1)
        if isinstance(x, dict):
            return np.concatenate([to_host(x[k]) for k in sorted(x)])
        if isinstance(x, (list, tuple)):
            return np.concatenate([to_host(i) for i in x])
        return np.asarray(x, dtype=np.float64).reshape(-1)

    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(1)
    print("[server] READY", flush=True)

    while True:
        conn, _ = srv.accept()
        try:
            while True:
                req = _recv_obj(conn)
                if req.get("op") == "shutdown":
                    _send_obj(conn, {"ok": True})
                    print("[server] shutdown requested", flush=True)
                    return 0
                img = Image.fromarray(req["image"])
                prompt = (f"In: What action should the robot take to "
                          f"{req['instruction'].lower()}?\nOut:")
                inputs = processor(prompt, img).to("cuda", dtype=dtype)
                t1 = time.monotonic()
                with torch.inference_mode():
                    out = model.predict_action(
                        **inputs, unnorm_key=req.get("unnorm_key") or unnorm,
                        do_sample=False)
                torch.cuda.synchronize()
                _send_obj(conn, {"action": to_host(out),
                                 "ms": (time.monotonic() - t1) * 1000.0})
        except (ConnectionError, EOFError):
            continue
        finally:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
