import torch
import torch.nn.functional as F

def conv3d_explicit_im2col(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """
    Explicit im2col conv3d WITHOUT unfold, for the exact non-overlap case:
      x:      (N, C, D, H, W)
      weight: (O, C, 1, 2, 2)
      bias:   (O,) or None

    Assumes:
      kernel=(1,2,2), stride=(1,2,2), padding=0, dilation=1
      H and W are divisible by 2
    Returns:
      y: (N, O, D, H/2, W/2)
    """
    assert x.dim() == 5, "x must be (N,C,D,H,W)"
    assert weight.dim() == 5, "weight must be (O,C,kD,kH,kW)"

    N, C, D, H, W = x.shape
    O, Cw, kD, kH, kW = weight.shape

    assert C == Cw, f"channel mismatch: x.C={C}, weight.C={Cw}"
    assert (kD, kH, kW) == (1, 2, 2), "this function is specialized for kernel=(1,2,2)"
    assert H % 2 == 0 and W % 2 == 0, "H and W must be divisible by 2 for non-overlap blocking"

    Hout, Wout = H // 2, W // 2

    # ---- 1) 显式 im2col（非重叠分块）----
    # 把 H 分成 (Hout, 2)，W 分成 (Wout, 2)
    # x_blocks: (N, C, D, Hout, 2, Wout, 2)
    x_blocks = x.contiguous().view(N, C, D, Hout, 2, Wout, 2)

    # 让每个空间位置对应一列：先把 (D,Hout,Wout) 展平为 M
    # patches: (N, M, K) 其中 K=C*2*2=192, M=D*Hout*Wout
    patches = (
        x_blocks.permute(0, 2, 3, 5, 1, 4, 6)   # (N, D, Hout, Wout, C, 2, 2)
                .reshape(N, D * Hout * Wout, C * 2 * 2)
    )

    # X_col: (N, K, M)
    X_col = patches.transpose(1, 2).contiguous()  # (N, 192, 3520) for your sizes

    # ---- 2) weight 展平 ----
    W_mat = weight.reshape(O, C * 2 * 2)  # (O, 192)

    # ---- 3) GEMM： (N,O,M) = (O,K) @ (N,K,M) ----
    Y = torch.bmm(W_mat.unsqueeze(0).expand(N, -1, -1), X_col)  # (N, O, M)

    if bias is not None:
        Y = Y + bias.view(1, O, 1)

    # ---- 4) reshape 回输出 ----
    y = Y.view(N, O, D, Hout, Wout)
    return y


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # input
    x = torch.randn(1, 48, 4, 44, 80, device="cpu")

    # conv params
    conv = torch.nn.Conv3d(48, 3072, kernel_size=(1,2,2), stride=(1,2,2), bias=True)
    weight = conv.weight
    bias = conv.bias

    y_ref = conv(x)
    y_im2col = conv3d_explicit_im2col(x, weight, bias, stride=(1,2,2), padding=(0,0,0))

    print("ref :", y_ref.shape)
    print("im2 :", y_im2col.shape)
    print("max abs diff:", (y_ref - y_im2col).abs().max().item())
