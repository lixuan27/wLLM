import torch
import triton
import triton.language as tl

@triton.jit
def fuse_prope_qkv_kernel(
    q_ptr, k_ptr, v_ptr,                  # [1, S, H, 128]
    out_q_ptr, out_k_ptr, out_v_ptr,      # out_q: [1,S,H,128], out_k/out_v: [1,S_cache,H,128]
    Pq_ptr, Pkv_ptr,                      # [1,C,4,4]
    S,                                    # runtime
    S_cache,                              # runtime
    C,                                    # runtime (4 or 16)
    offset,                               # runtime (token offset into cache)
    OUT_DTYPE: tl.constexpr,              # tl.bfloat16 / tl.float16 / tl.float32
    H: tl.constexpr = 24,
    D: tl.constexpr = 4,
    HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)                # 0..S-1
    s = pid

    # camera id (tokens are contiguous blocks per camera)
    P = S // C
    cam = s // P

    # ---- load P matrices once: [4,4] ----
    i = tl.arange(0, D)[:, None]
    j = tl.arange(0, D)[None, :]
    # P layout [1,C,4,4] contiguous; batch=0 => drop batch term
    Pq  = tl.load(Pq_ptr  + (cam * D + i) * D + j).to(tl.float32)
    Pkv = tl.load(Pkv_ptr + (cam * D + i) * D + j).to(tl.float32)

    # ---- base offsets (batch is always 0) ----
    # input base for token s in [1,S,H,HD]
    base_in  = (s * H) * HD
    # output q base is same shape as input
    base_qo  = base_in
    # cache base for token (s + offset) in [1,S_cache,H,HD]
    base_kvc = ((s + offset) * H) * HD

    d128 = tl.arange(0, HD)[None, :]      # (1,128)

    # =========================
    # heads 0..15
    # =========================
    h0 = tl.arange(0, 16)[:, None]        # (16,1)
    offs0_in  = base_in  + h0 * HD + d128
    offs0_qo  = base_qo  + h0 * HD + d128
    offs0_kvc = base_kvc + h0 * HD + d128

    q0 = tl.load(q_ptr + offs0_in).to(tl.float32)
    k0 = tl.load(k_ptr + offs0_in).to(tl.float32)
    v0 = tl.load(v_ptr + offs0_in).to(tl.float32)

    q0 = tl.reshape(q0, (16, HD // D, D))
    k0 = tl.reshape(k0, (16, HD // D, D))
    v0 = tl.reshape(v0, (16, HD // D, D))

    q0_out = tl.sum(Pq[None, None, :, :]  * q0[:, :, None, :], axis=3)
    k0_out = tl.sum(Pkv[None, None, :, :] * k0[:, :, None, :], axis=3)
    v0_out = tl.sum(Pkv[None, None, :, :] * v0[:, :, None, :], axis=3)

    tl.store(out_q_ptr + offs0_qo,  tl.reshape(q0_out, (16, HD)).to(OUT_DTYPE))

    # ---- store to cache only if within S_cache ----
    # prevent OOB when s+offset >= S_cache
    valid_cache0 = (s + offset) < S_cache
    tl.store(out_k_ptr + offs0_kvc, tl.reshape(k0_out, (16, HD)).to(OUT_DTYPE), mask=valid_cache0)
    tl.store(out_v_ptr + offs0_kvc, tl.reshape(v0_out, (16, HD)).to(OUT_DTYPE), mask=valid_cache0)

    # =========================
    # heads 16..23
    # =========================
    h1 = (16 + tl.arange(0, 8))[:, None]  # (8,1)
    offs1_in  = base_in  + h1 * HD + d128
    offs1_qo  = base_qo  + h1 * HD + d128
    offs1_kvc = base_kvc + h1 * HD + d128

    q1 = tl.load(q_ptr + offs1_in).to(tl.float32)
    k1 = tl.load(k_ptr + offs1_in).to(tl.float32)
    v1 = tl.load(v_ptr + offs1_in).to(tl.float32)

    q1 = tl.reshape(q1, (8, HD // D, D))
    k1 = tl.reshape(k1, (8, HD // D, D))
    v1 = tl.reshape(v1, (8, HD // D, D))

    q1_out = tl.sum(Pq[None, None, :, :]  * q1[:, :, None, :], axis=3)
    k1_out = tl.sum(Pkv[None, None, :, :] * k1[:, :, None, :], axis=3)
    v1_out = tl.sum(Pkv[None, None, :, :] * v1[:, :, None, :], axis=3)

    tl.store(out_q_ptr + offs1_qo,  tl.reshape(q1_out, (8, HD)).to(OUT_DTYPE))
    tl.store(out_k_ptr + offs1_kvc, tl.reshape(k1_out, (8, HD)).to(OUT_DTYPE), mask=valid_cache0)
    tl.store(out_v_ptr + offs1_kvc, tl.reshape(v1_out, (8, HD)).to(OUT_DTYPE), mask=valid_cache0)


@torch.compile(disable=True)
def fuse_prope_qkv(q, k, v, P_q, P_kv, out_k, out_v, offset):
    # q,k,v: [1,S,24,128]
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert P_q.is_contiguous() and P_kv.is_contiguous()
    assert out_k.is_contiguous() and out_v.is_contiguous()

    B, S, H, HD = q.shape
    assert B == 1 and (H, HD) == (24, 128)

    # P: [1,C,4,4]
    assert P_q.shape[0] == 1 and P_kv.shape[0] == 1
    C = P_q.shape[1]
    assert S % C == 0
    assert P_q.shape == (1, C, 4, 4)
    assert P_kv.shape == (1, C, 4, 4)

    # cache: [S_cache,24,128]
    assert out_k.shape[1:] == (24, 128)
    assert out_k.shape == out_v.shape
    S_cache = out_k.shape[0]

    if q.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif q.dtype == torch.float16:
        out_dtype = tl.float16
    elif q.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(q.dtype)

    out_q = torch.empty_like(q)

    grid = (S,)
    fuse_prope_qkv_kernel[grid](
        q, k, v,
        out_q, out_k, out_v,
        P_q, P_kv,
        S=S,
        S_cache=S_cache,
        C=C,
        offset=offset,
        OUT_DTYPE=out_dtype,
        num_warps=4,
        num_stages=2,
    )
    return out_q


@triton.jit
def fuse_prope_o_kernel(
    o_ptr,                               # [B,S,H,128]
    P_ptr,                               # [B,C,4,4]
    B: tl.constexpr,
    S: tl.constexpr,                                   # runtime
    H: tl.constexpr,                     # = 24
    C: tl.constexpr,                                   # runtime (4 or 16)
    OUT_DTYPE: tl.constexpr,
    D: tl.constexpr = 4,
    HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)               # 0 .. B*S-1
    b = pid // S
    s = pid - b * S

    # camera id (tokens are contiguous per camera)
    Pcam = S // C
    cam = s // Pcam

    # ---- load P_O once: [4,4] ----
    i = tl.arange(0, D)[:, None]         # (4,1)
    j = tl.arange(0, D)[None, :]         # (1,4)
    # flatten index: (((b*C + cam)*D + i)*D + j)
    P = tl.load(P_ptr + ((b * C + cam) * D + i) * D + j)

    # base offset for this (b,s) tile in [B,S,H,HD]
    base = ((b * S + s) * H) * HD
    d128 = tl.arange(0, HD)[None, :]     # (1,128)

    # =========================================================
    # heads 0..15
    # =========================================================
    h0 = tl.arange(0, 16)[:, None]       # (16,1)
    offs0 = base + h0 * HD + d128        # (16,128)

    o0 = tl.load(o_ptr + offs0)
    o0 = tl.reshape(o0, (16, HD // D, D))   # (16,32,4)

    o0_out = tl.sum(P[None, None, :, :] * o0[:, :, None, :], axis=3)
    o0_out = tl.reshape(o0_out, (16, HD)).to(OUT_DTYPE)

    tl.store(o_ptr + offs0, o0_out)

    # =========================================================
    # heads 16..23
    # =========================================================
    h1 = (16 + tl.arange(0, 8))[:, None] # (8,1)
    offs1 = base + h1 * HD + d128        # (8,128)

    o1 = tl.load(o_ptr + offs1)
    o1 = tl.reshape(o1, (8, HD // D, D))    # (8,32,4)

    o1_out = tl.sum(P[None, None, :, :] * o1[:, :, None, :], axis=3)
    o1_out = tl.reshape(o1_out, (8, HD)).to(OUT_DTYPE)

    tl.store(o_ptr + offs1, o1_out)

@torch.compile(disable=True)
def fuse_prope_o(O: torch.Tensor, P_O: torch.Tensor) -> torch.Tensor:
    """
    O   : [S, H, 128]
    P_O : [1, C, 4, 4]
    """
    assert O.is_contiguous()
    assert P_O.is_contiguous()

    S, H, HD = O.shape
    assert (H, HD) == (24, 128)

    C = P_O.shape[1]
    assert S % C == 0
    assert P_O.shape == (1, C, 4, 4)

    if O.dtype == torch.float16:
        out_dtype = tl.float16
    elif O.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    elif O.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"unsupported dtype: {O.dtype}")

    grid = (S,)
    fuse_prope_o_kernel[grid](
        O,
        P_O,
        B=1,
        S=S,              # runtime
        H=H,              # constexpr
        C=C,              # runtime (4 / 16)
        OUT_DTYPE=out_dtype,
        num_warps=4,
        num_stages=2,
    )
