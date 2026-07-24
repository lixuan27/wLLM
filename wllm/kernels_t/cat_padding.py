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
    key=["n_rows", "W2"],
)
@triton.jit
def padcat_lastdim_lr1_tdim2_kernel(
    x_ptr, y_ptr, out_ptr,
    B, C, H,
    Tx, Ty,
    n_rows,               # = B*C*(Tx+Ty)*H
    W: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_W2: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr
):
    pid = tl.program_id(0)

    # rows handled by this program
    row_ids = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)   # [R]
    row_mask = row_ids < n_rows

    # unravel row -> (b, c, t_out, h)
    Tout = Tx + Ty
    tmp = row_ids
    h = tmp % H
    tmp //= H
    t_out = tmp % Tout
    tmp //= Tout
    c = tmp % C
    b = tmp // C

    is_x = t_out < Tx
    t_x = t_out
    t_y = t_out - Tx

    # base offsets (elements)
    base_x = (((b * C + c) * Tx + t_x) * H + h) * W
    base_y = (((b * C + c) * Ty + t_y) * H + h) * W
    base_o = row_ids * W2

    # width
    offs_w2 = tl.arange(0, BLOCK_W2)
    col_mask = offs_w2 < W2

    xw = offs_w2 - 1
    in_mask = (xw >= 0) & (xw < W)

    x_offsets = base_x[:, None] + xw[None, :]
    y_offsets = base_y[:, None] + xw[None, :]
    o_offsets = base_o[:, None] + offs_w2[None, :]

    x_mask = row_mask[:, None] & is_x[:, None] & col_mask[None, :] & in_mask[None, :]
    y_mask = row_mask[:, None] & (~is_x)[:, None] & col_mask[None, :] & in_mask[None, :]
    o_mask = row_mask[:, None] & col_mask[None, :]

    xv = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    yv = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)

    out = xv + yv
    tl.store(out_ptr + o_offsets, out, mask=o_mask)

def padcat_lastdim_lr1_tdim2(x: torch.Tensor, y: torch.Tensor):
    """
    x, y: [B,C,T,H,W] bf16 CUDA
    out:  [B,C,Tx+Ty,H,W+2]
    """
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.bfloat16 and y.dtype == torch.bfloat16
    assert x.ndim == 5 and y.ndim == 5
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    assert x.shape[3] == y.shape[3]
    assert x.shape[4] == y.shape[4]

    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    B, C, Tx, H, W = x.shape
    Ty = y.shape[2]
    W2 = W + 2
    Tout = Tx + Ty

    out = torch.empty((B, C, Tout, H, W2),
                      device=x.device,
                      dtype=x.dtype)

    n_rows = B * C * Tout * H
    BLOCK_W2 = triton.next_power_of_2(W2)

    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_PROG"]),)

    padcat_lastdim_lr1_tdim2_kernel[grid](
        x, y, out,
        B=B, C=C, H=H,
        Tx=Tx, Ty=Ty,
        n_rows=n_rows,
        W=W,
        W2=W2,
        BLOCK_W2=BLOCK_W2,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---------- small case ----------
    B, C, Tx, Ty, H, W = 2, 3, 4, 5, 6, 7
    x = torch.randn(B, C, Tx, H, W, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, C, Ty, H, W, device="cuda", dtype=torch.bfloat16)

    out = padcat_lastdim_lr1_tdim2(x, y)

    ref = torch.cat(
        [
            torch.nn.functional.pad(x, (1, 1)),
            torch.nn.functional.pad(y, (1, 1)),
        ],
        dim=2,
    )

    max_diff = (out.float() - ref.float()).abs().max().item()
    print("pad+cat (small) max diff:", max_diff)

    # ---------- larger / non power-of-2 ----------
    B, C, Tx, Ty, H, W = 2, 4, 7, 11, 9, 513
    x = torch.randn(B, C, Tx, H, W, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(B, C, Ty, H, W, device="cuda", dtype=torch.bfloat16)

    out = padcat_lastdim_lr1_tdim2(x, y)
    ref = torch.cat(
        [
            torch.nn.functional.pad(x, (1, 1)),
            torch.nn.functional.pad(y, (1, 1)),
        ],
        dim=2,
    )

    max_diff = (out.float() - ref.float()).abs().max().item()
    print("pad+cat (large) max diff:", max_diff)

    # ---------- assert correctness ----------
    assert max_diff == 0.0, "Mismatch detected!"
    print("✓ correctness check passed")
