import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl


@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BH": 2}, num_warps=4, num_stages=2),
        triton.Config({"BH": 2}, num_warps=4, num_stages=3),
    ]),
    key=["S", "H"],
    cache_results=True
)
@triton.jit
def apply_rope_prope_qkv_kernel(
    q_ptr, k_ptr, v_ptr,            # [1, S, H, 128]
    cos_sin_cache_ptr,         # [S, 128]
    P_q_ptr, P_kv_ptr, # [1, C, M, M]
    out_q_all_ptr, out_k_ptr, out_v_ptr, 
    outp_k_ptr, outp_v_ptr,
    offset,
    C: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr
):  
    

    D: tl.constexpr = 128
    M: tl.constexpr = 4

    pid = tl.program_id(0)

    Pcam = S // C
    cid = pid // Pcam

    
    rh = tl.arange(0, BH)
    rd = tl.arange(0, D)
    rd_half = tl.arange(0, D // 2)
    rm = tl.arange(0, M)
    lane = tl.arange(0, 2)[None, None, :]   # [1,1,2]
    out_q_ptr = out_q_all_ptr
    outp_q_ptr = out_q_all_ptr + S * H * D



    c_cache_ptr = cos_sin_cache_ptr + pid * D + rd_half
    s_cache_ptr = cos_sin_cache_ptr + pid * D + (D // 2) + rd_half
    c_cache = tl.load(c_cache_ptr)[None, :].to(tl.float32) # [1, 64]
    s_cache = tl.load(s_cache_ptr)[None, :].to(tl.float32) # [1, 64]

    p_q_cache_ptr = P_q_ptr + cid * M * M + rm[:,None] * M + rm[None,:]
    p_kv_cache_ptr = P_kv_ptr + cid * M * M + rm[:,None] * M + rm[None,:]
    p_q_cache = tl.load(p_q_cache_ptr).to(tl.float32) # [4, 4]
    p_kv_cache = tl.load(p_kv_cache_ptr).to(tl.float32) # [4, 4]



    for i in tl.static_range(H // BH):

        q_ptr_i = q_ptr + (pid * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        k_ptr_i = k_ptr + (pid * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        v_ptr_i = v_ptr + (pid * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        
        q_i = tl.load(q_ptr_i).to(tl.float32)
        k_i = tl.load(k_ptr_i).to(tl.float32)
        v_i = tl.load(v_ptr_i).to(tl.float32)

        q_i = tl.reshape(q_i, (BH, 64, 2))
        k_i = tl.reshape(k_i, (BH, 64, 2))

        q_i_0, q_i_1 = tl.split(q_i) # [BH, 64]
        k_i_0, k_i_1 = tl.split(k_i) # [BH, 64]

        q_i_out_0 = q_i_0 * c_cache - q_i_1 * s_cache
        q_i_out_1 = q_i_0 * s_cache + q_i_1 * c_cache
        q_i_out = tl.where(lane == 0, q_i_out_0[:,:,None], q_i_out_1[:,:,None])
        q_i_out = tl.reshape(q_i_out, (BH, 128))

        k_i_out_0 = k_i_0 * c_cache - k_i_1 * s_cache
        k_i_out_1 = k_i_0 * s_cache + k_i_1 * c_cache
        k_i_out = tl.where(lane == 0, k_i_out_0[:,:,None], k_i_out_1[:,:,None])
        k_i_out = tl.reshape(k_i_out, (BH, 128))

        out_q_ptr_i = out_q_ptr + (pid * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        out_k_ptr_i = out_k_ptr + ((pid + offset) * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        out_v_ptr_i = out_v_ptr + ((pid + offset) * H + i * BH) * D + rh[:,None] * D + rd[None,:]

        tl.store(out_q_ptr_i, q_i_out.to(tl.bfloat16))
        tl.store(out_k_ptr_i, k_i_out.to(tl.bfloat16))
        tl.store(out_v_ptr_i, v_i.to(tl.bfloat16))


        pq_i = tl.reshape(q_i, (BH, 32, 4))
        pk_i = tl.reshape(k_i, (BH, 32, 4))
        pv_i = tl.reshape(v_i, (BH, 32, 4))

        pq_i_out = tl.sum(p_q_cache[None, None, :, :]  * pq_i[:, :, None, :], axis=3)
        pk_i_out = tl.sum(p_kv_cache[None, None, :, :] * pk_i[:, :, None, :], axis=3)
        pv_i_out = tl.sum(p_kv_cache[None, None, :, :] * pv_i[:, :, None, :], axis=3)

        pq_i_out = tl.reshape(pq_i_out, (BH, 128))
        pk_i_out = tl.reshape(pk_i_out, (BH, 128))
        pv_i_out = tl.reshape(pv_i_out, (BH, 128))

        outp_q_ptr_i = outp_q_ptr + (pid * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        outp_k_ptr_i = outp_k_ptr + ((pid + offset) * H + i * BH) * D + rh[:,None] * D + rd[None,:]
        outp_v_ptr_i = outp_v_ptr + ((pid + offset) * H + i * BH) * D + rh[:,None] * D + rd[None,:]

        tl.store(outp_q_ptr_i, pq_i_out.to(tl.bfloat16))
        tl.store(outp_k_ptr_i, pk_i_out.to(tl.bfloat16))
        tl.store(outp_v_ptr_i, pv_i_out.to(tl.bfloat16))



@torch.compile(disable=True)
def apply_rope_prope_qkv(
q: torch.Tensor, # [1, S, H, D]
k: torch.Tensor, # [1, S, H, D]
v: torch.Tensor, # [1, S, H, D]
cos_sin_cache: torch.Tensor, # [S, D],
P_q: torch.Tensor, # [1, C, M, M],
P_kv: torch.Tensor, # [1, C, M, M],
k_cache: torch.Tensor, #[S_cache, H, D]
v_cache: torch.Tensor, #[S_cache, H, D]
kp_cache: torch.Tensor, #[S_cache, H, D]
vp_cache: torch.Tensor, #[S_cache, H, D]
offset: int
):
    
    B, S, H, D = q.shape
    _, C, M, _ = P_q.shape
    assert B == 1
    assert D == 128
    assert M == 4
    assert S % C == 0
    assert k.shape == (B, S, H, D)
    assert P_q.shape == (B, C, M, M)
    assert P_kv.shape == (B, C, M, M)
    assert cos_sin_cache.shape == (S, D)

    grid = (S,)
    out_q_all = torch.empty(size=(2, S, H, D), device=q.device, dtype=q.dtype)
    
    apply_rope_prope_qkv_kernel[grid](
        q, k, v, cos_sin_cache, P_q, P_kv, out_q_all, k_cache, v_cache, kp_cache, vp_cache, offset, C, S, H
    )

    return out_q_all


@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BH": 2}, num_warps=4, num_stages=2),
        triton.Config({"BH": 2}, num_warps=4, num_stages=3),
    ]),
    key=["S", "H"],
    cache_results=True,
)
@triton.jit
def apply_rope_qkv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cos_sin_cache_ptr,
    out_q_ptr,
    out_k_ptr,
    out_v_ptr,
    offset,
    S: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
):
    D: tl.constexpr = 128

    pid = tl.program_id(0)

    rh = tl.arange(0, BH)
    rd = tl.arange(0, D)
    rd_half = tl.arange(0, D // 2)
    lane = tl.arange(0, 2)[None, None, :]

    c_cache_ptr = cos_sin_cache_ptr + pid * D + rd_half
    s_cache_ptr = cos_sin_cache_ptr + pid * D + (D // 2) + rd_half
    c_cache = tl.load(c_cache_ptr)[None, :].to(tl.float32)
    s_cache = tl.load(s_cache_ptr)[None, :].to(tl.float32)

    for i in tl.static_range(H // BH):
        q_ptr_i = q_ptr + (pid * H + i * BH) * D + rh[:, None] * D + rd[None, :]
        k_ptr_i = k_ptr + (pid * H + i * BH) * D + rh[:, None] * D + rd[None, :]
        v_ptr_i = v_ptr + (pid * H + i * BH) * D + rh[:, None] * D + rd[None, :]

        q_i = tl.load(q_ptr_i).to(tl.float32)
        k_i = tl.load(k_ptr_i).to(tl.float32)
        v_i = tl.load(v_ptr_i).to(tl.float32)

        q_i = tl.reshape(q_i, (BH, 64, 2))
        k_i = tl.reshape(k_i, (BH, 64, 2))

        q_i_0, q_i_1 = tl.split(q_i)
        k_i_0, k_i_1 = tl.split(k_i)

        q_i_out_0 = q_i_0 * c_cache - q_i_1 * s_cache
        q_i_out_1 = q_i_0 * s_cache + q_i_1 * c_cache
        q_i_out = tl.where(lane == 0, q_i_out_0[:, :, None], q_i_out_1[:, :, None])
        q_i_out = tl.reshape(q_i_out, (BH, D))

        k_i_out_0 = k_i_0 * c_cache - k_i_1 * s_cache
        k_i_out_1 = k_i_0 * s_cache + k_i_1 * c_cache
        k_i_out = tl.where(lane == 0, k_i_out_0[:, :, None], k_i_out_1[:, :, None])
        k_i_out = tl.reshape(k_i_out, (BH, D))

        out_q_ptr_i = out_q_ptr + (pid * H + i * BH) * D + rh[:, None] * D + rd[None, :]
        out_k_ptr_i = out_k_ptr + ((pid + offset) * H + i * BH) * D + rh[:, None] * D + rd[None, :]
        out_v_ptr_i = out_v_ptr + ((pid + offset) * H + i * BH) * D + rh[:, None] * D + rd[None, :]

        tl.store(out_q_ptr_i, q_i_out.to(tl.bfloat16))
        tl.store(out_k_ptr_i, k_i_out.to(tl.bfloat16))
        tl.store(out_v_ptr_i, v_i.to(tl.bfloat16))


@torch.compile(disable=True)
def apply_rope_qkv(
    q: torch.Tensor,  # [1, S, H, D]
    k: torch.Tensor,  # [1, S, H, D]
    v: torch.Tensor,  # [1, S, H, D]
    cos_sin_cache: torch.Tensor,  # [S, D]
    k_cache: torch.Tensor,  # [S_cache, H, D]
    v_cache: torch.Tensor,  # [S_cache, H, D]
    offset: int,
):

    B, S, H, D = q.shape
    assert B == 1
    assert D == 128
    assert k.shape == (B, S, H, D)
    assert v.shape == (B, S, H, D)
    assert cos_sin_cache.shape == (S, D)

    out_q = torch.empty(size=(B, S, H, D), device=q.device, dtype=q.dtype)
    grid = (S,)
    apply_rope_qkv_kernel[grid](
        q,
        k,
        v,
        cos_sin_cache,
        out_q,
        k_cache,
        v_cache,
        offset,
        S,
        H,
    )
    return out_q



@triton.jit
def apply_prope_o_kernel(
    o_ptr,                               # [B,S,H,128]
    P_ptr,                               # [B,C,4,4]
    B: tl.constexpr,
    S: tl.constexpr,                                   # runtime
    H: tl.constexpr,                     # = 24
    C: tl.constexpr,                                   # runtime (4 or 16)
    OUT_DTYPE: tl.constexpr,
    D: tl.constexpr = 4,
    HD: tl.constexpr = 128,
    BLOCK_H: tl.constexpr = 16,
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
    d_range = tl.arange(0, HD)[None, :]     # (1,128)
    h_range = tl.arange(0, BLOCK_H)[:, None]

    for h_start in tl.static_range(0, H, BLOCK_H):
        h = tl.arange(0, BLOCK_H)[:, None]
        offs = base + h * HD + d_range
        o = tl.load(o_ptr + offs)  # (BLOCK_H, HD)
        o = tl.reshape(o, (BLOCK_H, HD // D, D))
        o_out = tl.sum(P[None, None, :, :] * o[:, :, None, :], axis=3)
        o_out = tl.reshape(o_out, (BLOCK_H, HD)).to(OUT_DTYPE)
        tl.store(o_ptr + offs, o_out, mask=(h_range + h_start) < H)

@torch.compile(disable=True)
def apply_prope_o(O: torch.Tensor, P_O: torch.Tensor) -> torch.Tensor:
    """
    O   : [S, H, 128]
    P_O : [1, C, 4, 4]
    """
    assert O.is_contiguous()
    assert P_O.is_contiguous()

    S, H, HD = O.shape
    assert triton.next_power_of_2(HD) == HD
    assert HD % 4 == 0

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
    apply_prope_o_kernel[grid](
        O,
        P_O,
        B=1,
        S=S,              # runtime
        H=H,              # constexpr
        C=C,              # runtime (4 / 16)
        OUT_DTYPE=out_dtype,
        HD=HD,
        num_warps=4,
        num_stages=2,
    )
