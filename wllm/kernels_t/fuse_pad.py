import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl


@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"ROWS_PER_PROG": 8},  num_warps=4,  num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=4,  num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=8,  num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8,  num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
    ]),
    key=["n_rows", "W_OUT"],
)
@triton.jit
def padcat_then_pad6_kernel(
    x_ptr, y_ptr, out_ptr,
    B, C, H,
    Tx, Ty,
    n_rows,
    # 原始 W 作为 constexpr（更贴近你写法）
    W: tl.constexpr,

    # pad list for F.pad(h, [pwL,pwR, phL,phR, ptL,ptR])
    PW_L: tl.constexpr, PW_R: tl.constexpr,
    PH_L: tl.constexpr, PH_R: tl.constexpr,
    PT_L: tl.constexpr, PT_R: tl.constexpr,

    # final sizes (constexpr 也行；这里传进来做 constexpr 更像你原来的写法)
    T_OUT: tl.constexpr,      # Tx+Ty
    H_OUT: tl.constexpr,      # H + PH_L + PH_R
    W_OUT: tl.constexpr,      # (W+2) + PW_L + PW_R

    BLOCK_W_OUT: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)

    # rows over (b,c,t_final,h_final)
    row_ids = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)  # [R]
    row_mask = row_ids < n_rows

    tmp = row_ids
    h_f = tmp % H_OUT
    tmp //= H_OUT
    t_f = tmp % (T_OUT + PT_L + PT_R)
    tmp //= (T_OUT + PT_L + PT_R)
    c = tmp % C
    b = tmp // C

    # map final -> h (after padW+cat) coordinates
    # h is [B,C,T_OUT,H,W2] with W2=W+2 (from the first padW(1,1))
    t_in = t_f - PT_L
    h_in = h_f - PH_L

    valid_t = (t_in >= 0) & (t_in < T_OUT)
    valid_h = (h_in >= 0) & (h_in < H)

    # choose x or y along dim=2 cat
    is_x = t_in < Tx
    valid_xrow = valid_t & valid_h & is_x
    valid_yrow = valid_t & valid_h & (~is_x)

    # clamp indices for safe address (masked load will ignore anyway)
    t_in_x = tl.where(valid_xrow, t_in, 0)
    t_in_y = tl.where(valid_yrow, t_in - Tx, 0)
    h_in_c = tl.where(valid_t & valid_h, h_in, 0)

    base_x = (((b * C + c) * Tx + t_in_x) * H + h_in_c) * W
    base_y = (((b * C + c) * Ty + t_in_y) * H + h_in_c) * W

    # width in final output
    offs_w = tl.arange(0, BLOCK_W_OUT)
    col_mask = offs_w < W_OUT

    # final_w -> original_w:
    # final_w --PW_L--> h_w (in h) ; then h_w --1--> original_w  (because padW left=1)
    # so original_w = final_w - PW_L - 1
    w_src = offs_w - PW_L - 1
    valid_w = (w_src >= 0) & (w_src < W)

    # offsets
    x_offsets = base_x[:, None] + w_src[None, :]
    y_offsets = base_y[:, None] + w_src[None, :]
    out_offsets = (row_ids * W_OUT)[:, None] + offs_w[None, :]

    x_mask = row_mask[:, None] & valid_xrow[:, None] & col_mask[None, :] & valid_w[None, :]
    y_mask = row_mask[:, None] & valid_yrow[:, None] & col_mask[None, :] & valid_w[None, :]
    o_mask = row_mask[:, None] & col_mask[None, :]

    xv = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    yv = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)

    out = xv + yv
    tl.store(out_ptr + out_offsets, out, mask=o_mask)


def padcat_then_pad6(x: torch.Tensor, y: torch.Tensor, padding_list):
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16
    assert x.ndim == 5 and y.ndim == 5
    assert len(padding_list) == 6

    pwL, pwR, phL, phR, ptL, ptR = padding_list
    assert all(v >= 0 for v in (pwL, pwR, phL, phR, ptL, ptR))

    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] and x.shape[3] == y.shape[3] and x.shape[4] == y.shape[4]

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    B, C, Tx, H, W = x.shape
    Ty = y.shape[2]
    T_OUT = Tx + Ty

    W2 = W + 2  # padW(1,1)
    T_FINAL = T_OUT + ptL + ptR
    H_OUT = H + phL + phR
    W_OUT = W2 + pwL + pwR

    out = torch.empty((B, C, T_FINAL, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    n_rows = B * C * T_FINAL * H_OUT
    BLOCK_W_OUT = triton.next_power_of_2(W_OUT)

    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_PROG"]),)

    padcat_then_pad6_kernel[grid](
        x, y, out,
        B=B, C=C, H=H,
        Tx=Tx, Ty=Ty,
        n_rows=n_rows,
        W=W,
        PW_L=pwL, PW_R=pwR,
        PH_L=phL, PH_R=phR,
        PT_L=ptL, PT_R=ptR,
        T_OUT=T_OUT,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        BLOCK_W_OUT=BLOCK_W_OUT,
    )
    return out



@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"ROWS_PER_PROG": 8}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 8}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
    ]),
    key=["n_rows", "W2"],
    cache_results=True
)
@triton.jit
def pad_lastdim_lr1_then_pad6_multirow_kernel(
    x_ptr, y_ptr,
    # n_rows 是最终输出的 rows: B*C*T_out*H_out
    n_rows,                      # runtime scalar
    W: tl.constexpr,             # original W
    W2: tl.constexpr,            # FINAL output width = (W+2) + pwL + pwR
    STRIDE_ROW_X: tl.constexpr,  # x row stride (contiguous => W)
    STRIDE_ROW_Y: tl.constexpr,  # y row stride (contiguous => W2)
    BLOCK_W2: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,

    # shapes
    B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    T_OUT: tl.constexpr, H_OUT: tl.constexpr,

    # padding_list for F.pad(h, [pwL,pwR, phL,phR, ptL,ptR])
    PW_L: tl.constexpr, PW_R: tl.constexpr,
    PH_L: tl.constexpr, PH_R: tl.constexpr,
    PT_L: tl.constexpr, PT_R: tl.constexpr,
):
    pid = tl.program_id(0)

    # rows handled by this program
    row_ids = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)   # [R]
    row_mask = row_ids < n_rows

    # unravel row -> (b,c,t_out,h_out)
    tmp = row_ids
    h_out = tmp % H_OUT
    tmp //= H_OUT
    t_out = tmp % T_OUT
    tmp //= T_OUT
    c = tmp % C
    b = tmp // C

    # map final (t_out,h_out) -> original (t,h) through pad on T/H
    t = t_out - PT_L
    h = h_out - PH_L
    valid_th = (t >= 0) & (t < T) & (h >= 0) & (h < H)

    # base offset into x for each row (elements)
    # x layout: [B,C,T,H,W] contiguous
    # base = ((((b*C + c)*T + t)*H + h) * W)
    t_safe = tl.where(valid_th, t, 0)
    h_safe = tl.where(valid_th, h, 0)
    base_x = (((b * C + c) * T + t_safe) * H + h_safe) * W

    # columns this program covers (final width)
    offs_w2 = tl.arange(0, BLOCK_W2)   # [Cw]
    col_mask = offs_w2 < W2

    # final_w -> original_w:
    # final_w --PW_L--> h_w (in h after pad6 on W)
    # h_w corresponds to (pad_lastdim_lr1) output w index, which is original_w+1
    # so original_w = final_w - PW_L - 1
    w = offs_w2 - PW_L - 1
    valid_w = (w >= 0) & (w < W)

    # 2D offsets
    x_offsets = base_x[:, None] + w[None, :]
    y_offsets = row_ids[:, None] * STRIDE_ROW_Y + offs_w2[None, :]

    x_mask = row_mask[:, None] & valid_th[:, None] & col_mask[None, :] & valid_w[None, :]
    y_mask = row_mask[:, None] & col_mask[None, :]

    x_val = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    tl.store(y_ptr + y_offsets, x_val, mask=y_mask)


def pad_lastdim_lr1_then_pad6(x: torch.Tensor, padding_list) -> torch.Tensor:
    """
    x: [B,C,T,H,W] bf16 CUDA contiguous
    padding_list: [pwL,pwR, phL,phR, ptL,ptR] each >= 0
    returns:
      y = F.pad( pad_lastdim_lr1(x) (i.e. pad W by (1,1)), padding_list, value=0 )
      shape = [B, C, T+ptL+ptR, H+phL+phR, (W+2)+pwL+pwR]
    """
    assert x.is_cuda
    assert x.dtype == torch.bfloat16
    assert x.ndim == 5
    assert x.is_contiguous()
    assert len(padding_list) == 6

    pwL, pwR, phL, phR, ptL, ptR = [int(v) for v in padding_list]
    assert all(v >= 0 for v in (pwL, pwR, phL, phR, ptL, ptR))

    B, C, T, H, W = x.shape

    # first pad_lastdim_lr1 gives W+2, then F.pad adds pwL+pwR
    W2 = (W + 2) + pwL + pwR
    T_OUT = T + ptL + ptR
    H_OUT = H + phL + phR

    y = torch.empty((B, C, T_OUT, H_OUT, W2), device=x.device, dtype=x.dtype)

    n_rows = B * C * T_OUT * H_OUT
    stride_row_x = W
    stride_row_y = W2
    BLOCK_W2 = triton.next_power_of_2(W2)

    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_PROG"]),)

    pad_lastdim_lr1_then_pad6_multirow_kernel[grid](
        x, y,
        n_rows=n_rows,
        W=W,
        W2=W2,
        STRIDE_ROW_X=stride_row_x,
        STRIDE_ROW_Y=stride_row_y,
        BLOCK_W2=BLOCK_W2,

        B=B, C=C, T=T, H=H,
        T_OUT=T_OUT, H_OUT=H_OUT,

        PW_L=pwL, PW_R=pwR,
        PH_L=phL, PH_R=phR,
        PT_L=ptL, PT_R=ptR,
    )
    return y



if __name__ == "__main__":
    import torch.nn.functional as F
    torch.manual_seed(0)

    B, C, Tx, Ty, H, W = 2, 3, 4, 5, 6, 7
    padding_list = [0, 3, 0, 2, 0, 5]   # [pwL,pwR, phL,phR, ptL,ptR] all > 0

    x = torch.randn(B, C, Tx, H, W, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, C, Ty, H, W, device="cuda", dtype=torch.bfloat16)

    out = padcat_then_pad6(x, y, padding_list)

    # reference: h = cat(padW(x,1,1), padW(y,1,1), dim=2); then F.pad(h, padding_list)
    h = torch.cat([F.pad(x, (1, 1), value=0), F.pad(y, (1, 1), value=0)], dim=2)
    ref = F.pad(h, padding_list, value=0)

    max_diff = (out.float() - ref.float()).abs().max().item()
    print("fused padW+cat then F.pad(6) max diff:", max_diff)

    assert max_diff == 0.0, "Mismatch detected!"
    print("✓ correctness check passed")
