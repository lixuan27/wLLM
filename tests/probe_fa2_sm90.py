"""Hopper probe for the vendored FA2 forward (.so built with GPU_ARCH=90).

Correctness: fwd_bf16 / fwd_fp16 / fwd_bf16_causal against torch SDPA on
representative VLA shapes (head_dim 96/128/256). Performance: median
kernel time vs SDPA on the same shapes. Prints FA2_SM90_PROBE_OK only if
every shape passes the tolerance gate.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

from wllm_native import wllm_native_fa2 as fa2  # noqa: E402

SHAPES = [  # (batch, seqlen_q, seqlen_k, heads_q, heads_kv, head_dim)
    (4, 1024, 1024, 16, 16, 128),
    (1, 800, 800, 8, 8, 96),        # pi0-class prefix
    (2, 512, 2048, 16, 4, 128),     # GQA decode-ish
    (1, 512, 512, 8, 8, 256),
]
TOL = 2e-2  # bf16 accumulation vs fp32 SDPA reference


def run_one(b, sq, sk, hq, hkv, d, dtype, causal):
    fn = {
        (torch.bfloat16, False): fa2.fwd_bf16,
        (torch.bfloat16, True): fa2.fwd_bf16_causal,
        (torch.float16, False): fa2.fwd_fp16,
    }.get((dtype, causal))
    if fn is None:
        return None
    q = torch.randn(b, sq, hq, d, device="cuda", dtype=dtype)
    k = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype)
    v = torch.randn(b, sk, hkv, d, device="cuda", dtype=dtype)
    o = torch.empty_like(q)
    lse = torch.empty(b, hq, sq, device="cuda", dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)
    stream = torch.cuda.current_stream().cuda_stream

    def call():
        fn(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
           lse.data_ptr(), 0, 0,
           b, sq, sk, hq, hkv, d,
           tuple(q.stride()[:3]), tuple(k.stride()[:3]),
           tuple(v.stride()[:3]), tuple(o.stride()[:3]),
           scale, 0, stream)

    call()
    torch.cuda.synchronize()
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(),
        v.transpose(1, 2).float(), is_causal=causal,
        enable_gqa=(hq != hkv)).transpose(1, 2)
    err = (o.float() - ref).abs().max().item()

    def bench(f, iters=50):
        for _ in range(5):
            f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            f()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[iters // 2] * 1e3

    t_fa2 = bench(call)
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))

    def sdpa():
        torch.nn.functional.scaled_dot_product_attention(
            qt, kt, vt, is_causal=causal, enable_gqa=(hq != hkv))

    t_sdpa = bench(sdpa)
    tag = f"{dtype}".replace("torch.", "") + ("/causal" if causal else "")
    ok = err < TOL
    print(f"[probe] b{b} q{sq} k{sk} h{hq}/{hkv} d{d} {tag}: "
          f"max_err={err:.4f} fa2={t_fa2:.3f}ms sdpa={t_sdpa:.3f}ms "
          f"ratio={t_sdpa / t_fa2:.2f}x {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


def main() -> int:
    print(f"[probe] device={torch.cuda.get_device_name(0)} "
          f"cc={torch.cuda.get_device_capability(0)}", flush=True)
    results = []
    for shape in SHAPES:
        for dtype, causal in ((torch.bfloat16, False), (torch.bfloat16, True),
                              (torch.float16, False)):
            r = run_one(*shape, dtype, causal)
            if r is not None:
                results.append(r)
    n_pass = sum(results)
    print(f"[probe] {n_pass}/{len(results)} passed")
    print("FA2_SM90_PROBE_OK" if n_pass == len(results) else "FA2_SM90_PROBE_FAIL")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
