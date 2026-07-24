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
def fuse_rope_qkv_kernel(
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
def fuse_rope_qkv(
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
    #assert H == 24
    assert D == 128
    assert M == 4
    assert S % C == 0
    assert k.shape == (B, S, H, D)
    assert P_q.shape == (B, C, M, M)
    assert P_kv.shape == (B, C, M, M)
    assert cos_sin_cache.shape == (S, D)

    grid = (S,)
    out_q_all = torch.empty(size=(2, S, H, D), device=q.device, dtype=q.dtype)

    fuse_rope_qkv_kernel[grid](
        q, k, v, cos_sin_cache, P_q, P_kv, out_q_all, k_cache, v_cache, kp_cache, vp_cache, offset, C, S, H
    )

    return out_q_all








    

