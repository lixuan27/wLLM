import torch
from wllm.serving.layers.custom_op import CustomOp

def w2c_to_framewise_relative_c2w(viewmats: torch.Tensor) -> torch.Tensor:
    """Convert absolute w2c matrices to framewise-relative c2w matrices.

    Given world-to-camera matrices for a sequence of frames, computes the
    framewise-relative camera-to-world transformation for each frame:
      - Frame 0: identity (camera defines the reference frame)
      - Frame i (i>0): relative c2w from frame i-1 to frame i,
        i.e. ``inv(w2c_{i-1}) @ w2c_i`` inverted, or equivalently
        ``inv(w2c_i) @ w2c_{i-1}`` ... but more precisely:
        the c2w of frame i expressed in frame (i-1)'s camera coordinate system.

    Translations are normalized so that the maximum norm across all frames
    is 1.0, following the convention used in the original LingBot-World training.

    Args:
        viewmats: Absolute world-to-camera 4x4 matrices [B, F, 4, 4].

    Returns:
        Framewise-relative c2w matrices [B, F, 4, 4].
    """
    B, F = viewmats.shape[:2]

    viewmats = viewmats.float()
    eye = torch.eye(4, device=viewmats.device, dtype=torch.float32)

    # inv(w2c) = c2w;  relative c2w for frame i in frame (i-1)'s coords:
    #   rel_c2w[i] = w2c_{i-1} @ inv(w2c_i) = w2c_{i-1} @ c2w_i
    # We compute inv(w2c) via SE3 inverse (R^T, -R^T @ t)
    R = viewmats[:, :, :3, :3]           # [B, F, 3, 3]
    t = viewmats[:, :, :3, 3:4]          # [B, F, 3, 1]
    R_inv = R.transpose(-1, -2)           # [B, F, 3, 3]  -- c2w rotation
    t_inv = -torch.matmul(R_inv, t)       # [B, F, 3, 1]  -- c2w translation

    # Build c2w (inv of w2c)
    c2w = eye.unsqueeze(0).unsqueeze(0).expand(B, F, -1, -1).clone()
    c2w[:, :, :3, :3] = R_inv
    c2w[:, :, :3, 3:4] = t_inv

    # rel_c2w[i] = w2c_{i-1} @ c2w_i
    # For frame 0, use identity (no previous frame)
    prev_w2c = viewmats[:, :-1]  # [B, F-1, 4, 4]
    curr_c2w = c2w[:, 1:]        # [B, F-1, 4, 4]
    rel = torch.matmul(prev_w2c, curr_c2w)  # [B, F-1, 4, 4]

    out = eye.unsqueeze(0).unsqueeze(0).expand(B, F, -1, -1).clone()
    out[:, 1:] = rel

    # Normalize translations (following original LingBot convention)
    trans = out[:, :, :3, 3]  # [B, F, 3]
    max_norm = trans.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
    max_norm = max_norm.clamp(min=1e-8)
    out[:, :, :3, 3] = out[:, :, :3, 3] / max_norm

    return out

def fold_spatial_to_channels(
    tensor: torch.Tensor,
    stride_h: int,
    stride_w: int,
) -> torch.Tensor:
    """Fold spatial sub-pixels into the channel dimension.

    Rearranges [B, C, F, pixel_h, pixel_w]
            -> [B, C * stride_h * stride_w, F, lat_h, lat_w]
    """
    B, C, F, pH, pW = tensor.shape
    lat_h = pH // stride_h
    lat_w = pW // stride_w
    tensor = tensor.reshape(B, C, F, lat_h, stride_h, lat_w, stride_w)
    # Move sub-pixel dims next to channels: [B, C, sh, sw, F, lh, lw]
    tensor = tensor.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    return tensor.reshape(B, C * stride_h * stride_w, F, lat_h, lat_w)

@CustomOp.register("plucker_rays")
class PluckerRayDirections(CustomOp):
    def __init__(self, vae_stride_h: int, vae_stride_w: int):
        super().__init__()
        self.vae_stride_h = vae_stride_h
        self.vae_stride_w = vae_stride_w
        self._grid_xy: torch.Tensor | None = None

    def _get_grid(
        self, pixel_h: int, pixel_w: int, device: torch.device
    ) -> torch.Tensor:
        if self._grid_xy is not None:
            return self._grid_xy
        y = torch.arange(pixel_h, device=device, dtype=torch.float32) + 0.5
        x = torch.arange(pixel_w, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        self._grid_xy = torch.stack([grid_x, grid_y], dim=-1)
        return self._grid_xy

    def forward_native(
        self,
        rel_c2ws: torch.Tensor,
        Ks: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Compute ray directions folded to latent resolution.

        Args:
            rel_c2ws: Framewise-relative camera-to-world matrices [B, F, 4, 4].
            Ks: Intrinsic matrices [B, F, 3, 3].
            latent_h: Latent spatial height.
            latent_w: Latent spatial width.

        Returns:
            [B, 3 * vae_stride_h * vae_stride_w, F, latent_h, latent_w]
        """
        pixel_h = latent_h * self.vae_stride_h
        pixel_w = latent_w * self.vae_stride_w
        B, F = rel_c2ws.shape[:2]

        R_c2w = rel_c2ws[:, :, :3, :3].float()  # [B, F, 3, 3]
        Ks = Ks.float()

        fx = Ks[:, :, 0, 0, None, None] * pixel_w  # [B, F, 1, 1]
        fy = Ks[:, :, 1, 1, None, None] * pixel_h
        cx = Ks[:, :, 0, 2, None, None] * pixel_w
        cy = Ks[:, :, 1, 2, None, None] * pixel_h

        grid = self._get_grid(pixel_h, pixel_w, rel_c2ws.device)
        gx = grid[..., 0]  # [H, W]
        gy = grid[..., 1]

        dirs_x = (gx[None, None] - cx) / fx  # [B, F, H, W]
        dirs_y = (gy[None, None] - cy) / fy
        dirs_z = torch.ones_like(dirs_x)

        dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # [B, F, H, W, 3]
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)

        dirs_flat = dirs.reshape(B, F, -1, 3)  # [B, F, H*W, 3]
        rays_d = torch.matmul(dirs_flat, R_c2w.transpose(-1, -2))  # [B, F, H*W, 3]
        rays_d = rays_d.reshape(B, F, pixel_h, pixel_w, 3)

        rays_d = rays_d.permute(0, 4, 1, 2, 3)

        return fold_spatial_to_channels(rays_d, self.vae_stride_h, self.vae_stride_w)
    
    def forward_cuda(self,
        rel_c2ws: torch.Tensor,
        Ks: torch.Tensor,
        latent_h: int,
        latent_w: int,
    ):
        return self.forward_native(rel_c2ws, Ks, latent_h, latent_w)


_ACTION_LUT = torch.tensor([
    [0, 0, 0, 0],  # 0: noop
    [1, 0, 0, 0],  # 1: W (forward)
    [0, 0, 1, 0],  # 2: S (backward)
    [0, 0, 0, 1],  # 3: D (right)
    [0, 1, 0, 0],  # 4: A (left)
    [1, 0, 0, 1],  # 5: W + D (forward + right)
    [1, 1, 0, 0],  # 6: W + A (forward + left)
    [0, 0, 1, 1],  # 7: S + D (backward + right)
    [0, 1, 1, 0],  # 8: S + A (backward + left)
], dtype=torch.float32)

@CustomOp.register("discrete_action_spatial_emb")
class DiscreteActionSpatialEmbedding(CustomOp):
    def __init__(self, action_dim: int, vae_stride_h: int, vae_stride_w: int):
        super().__init__()
        self.action_dim = action_dim
        self.vae_stride_h = vae_stride_h
        self.vae_stride_w = vae_stride_w
        self.register_buffer("lut", _ACTION_LUT.clone(), persistent=False)

    def forward_native(
        self,
        action: torch.Tensor,
        pixel_h: int,
        pixel_w: int,
    ) -> torch.Tensor:
        B, F = action.shape
        action_int = action.long().clamp(min=0, max=8)

        if self.lut.device != action.device:
            self.lut = self.lut.to(action.device)

        multihot = self.lut[action_int.reshape(-1)].reshape(B, F, self.action_dim)
        multihot = multihot.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        tiled = multihot.expand(B, self.action_dim, F, pixel_h, pixel_w).contiguous()

        return fold_spatial_to_channels(tiled, self.vae_stride_h, self.vae_stride_w)
    
    def forward_cuda(self, action: torch.Tensor, pixel_h: int, pixel_w: int):
        return self.forward_native(action, pixel_h, pixel_w)
