import torch
import math
from typing import List

def generate_points_in_sphere(n_points: int, radius: float) -> torch.Tensor:
    """
    Uniformly sample points within a sphere of a specified radius.

    :param n_points: The number of points to generate.
    :param radius: The radius of the sphere.
    :return: A tensor of shape (n_points, 3), representing the (x, y, z) coordinates of the points.
    """
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    # transfer the coordinates from spherical to cartesian
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points


def invert_se3_batch(T: torch.Tensor) -> torch.Tensor:
    """
    Fast inverse for SE(3) batch:
    T: (B,4,4) = [R t; 0 1]
    inv(T) = [R^T, -R^T t; 0 1]
    """
    assert T.ndim == 3 and T.shape[-2:] == (4, 4)
    R = T[:, :3, :3]              # (B,3,3)
    t = T[:, :3, 3]               # (B,3)
    Rt = R.transpose(1, 2)        # (B,3,3)

    out = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
    out[:, :3, :3] = Rt
    out[:, :3, 3]  = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return out


def fov_mask_world_angles(
    points_world: torch.Tensor,     # (N,3)
    center: torch.Tensor,           # (...,3)
    pitch_deg: torch.Tensor,        # (...)
    yaw_deg: torch.Tensor,          # (...)
    fov_half_h: torch.Tensor,       # scalar tensor
    fov_half_v: torch.Tensor,       # scalar tensor
) -> torch.Tensor:
    """
    Vectorized version of is_inside_fov_3d_hv but supports arbitrary leading dims on center/pitch/yaw.
    Returns mask of shape (..., N).
    """
    # vectors: (..., N, 3)
    v = points_world.unsqueeze(0)  # (1,N,3) for broadcasting
    # expand center to (...,1,3)
    c = center.unsqueeze(-2)       # (...,1,3)
    vec = v - c                    # (...,N,3)

    x = vec[..., 0]
    y = vec[..., 1]
    z = vec[..., 2]

    azimuth = torch.atan2(x, z) * (180.0 / math.pi)
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180.0 / math.pi)

    diff_az = azimuth - yaw_deg.unsqueeze(-1)
    diff_az = torch.remainder(diff_az + 180.0, 360.0) - 180.0

    diff_el = elevation - pitch_deg.unsqueeze(-1)
    diff_el = torch.remainder(diff_el + 180.0, 360.0) - 180.0

    in_h = diff_az.abs() < fov_half_h
    in_v = diff_el.abs() < fov_half_v
    return in_h & in_v


def calculate_fov_overlap_similarity_batch(
    w2c_curr: torch.Tensor,          # (B1,4,4)
    w2c_hist: torch.Tensor,          # (B2,4,4)
    points_local: torch.Tensor,      # (N,3) 以 curr 为中心的局部点（你原逻辑里 points_world=points_local）
    fov_h_deg: float = 105.0,
    fov_v_deg: float = 75.0,
    dist_thresh: float = 8.0,
    chunk_b1: int = 8,               # 分块，防止显存爆
) -> torch.Tensor:
    """
    Output: (B1,B2) overlap ratio = |curr_FOV ∩ hist_FOV| / |curr_FOV|
    完全利用 curr 归一化为单位相机的恒等性质。
    """
    device = w2c_curr.device
    dtype = w2c_curr.dtype

    B1 = w2c_curr.shape[0]
    B2 = w2c_hist.shape[0]
    N = points_local.shape[0]

    fov_half_h = torch.tensor(fov_h_deg / 2.0, device=device, dtype=dtype)
    fov_half_v = torch.tensor(fov_v_deg / 2.0, device=device, dtype=dtype)

    # ---- curr 恒等：center=(0,0,0), yaw=pitch=0 ----
    zero3 = torch.zeros(3, device=device, dtype=dtype)
    zero1 = torch.zeros((), device=device, dtype=dtype)

    # curr mask 对所有 pair 都一样，只算一次：shape (N,)
    curr_mask = fov_mask_world_angles(
        points_world=points_local,
        center=zero3,          # (3,)
        pitch_deg=zero1,       # ()
        yaw_deg=zero1,         # ()
        fov_half_h=fov_half_h,
        fov_half_v=fov_half_v,
    )  # (N,)

    denom = curr_mask.sum().to(torch.float32)  # 标量
    if denom.item() == 0:
        return torch.zeros((B1, B2), device=device, dtype=torch.float32)

    # 预先算 inv(w2c_curr): (B1,4,4)
    c2w_curr = invert_se3_batch(w2c_curr)

    # 输出
    out = torch.empty((B1, B2), device=device, dtype=torch.float32)

    # 分块遍历 B1（避免一次性构造 B1*B2*N）
    for i0 in range(0, B1, chunk_b1):
        i1 = min(i0 + chunk_b1, B1)
        c2w_c = c2w_curr[i0:i1]                       # (b,4,4)
        b = c2w_c.shape[0]

        # hist_rel: (b,B2,4,4) = w2c_hist (B2,4,4) @ c2w_curr (b,4,4)
        # 用广播实现： (1,B2,4,4) @ (b,1,4,4)
        hist_rel = w2c_hist.unsqueeze(0) @ c2w_c.unsqueeze(1)  # (b,B2,4,4)

        R = hist_rel[:, :, :3, :3]    # (b,B2,3,3)  这是 R_w2c(rel)
        t = hist_rel[:, :, :3, 3]     # (b,B2,3)

        # 相机中心（在 curr-rel world 中）：C = -R^T t
        Rt = R.transpose(-2, -1)      # (b,B2,3,3)
        C = -(Rt @ t.unsqueeze(-1)).squeeze(-1)  # (b,B2,3)

        # forward 向量（世界系）= R_c2w[:,2] = (R_w2c^T)[:,2]
        fwd = Rt[..., :, 2]           # (b,B2,3)
        x, y, z = fwd[..., 0], fwd[..., 1], fwd[..., 2]

        yaw = torch.atan2(x, z) * (180.0 / math.pi)  # (b,B2)
        pitch = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180.0 / math.pi)  # (b,B2)

        # hist FOV mask: (b,B2,N)
        hist_mask = fov_mask_world_angles(
            points_world=points_local,   # (N,3)
            center=C,                    # (b,B2,3)
            pitch_deg=pitch,             # (b,B2)
            yaw_deg=yaw,                 # (b,B2)
            fov_half_h=fov_half_h,
            fov_half_v=fov_half_v,
        )

        # distance gate: dist < thresh
        # vectors to points: (b,B2,N,3)
        vec = points_local.unsqueeze(0).unsqueeze(0) - C.unsqueeze(-2)  # (b,B2,N,3)
        dist_ok = (vec.norm(dim=-1) < dist_thresh)                      # (b,B2,N)
        hist_mask = hist_mask & dist_ok

        # overlap: curr_mask (N,) 与 hist_mask (b,B2,N)
        overlap = (hist_mask & curr_mask.view(1, 1, N)).sum(dim=-1).to(torch.float32)  # (b,B2)
        out[i0:i1] = overlap / denom

    return out

def select_mem_frames_wan(
    w2c_list: torch.Tensor,
    current_frame_idx: int,
    memory_frames: int,
    temporal_context_size: int,
    pred_latent_size: int,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    device=None,
    points_local=None,
) -> List[int]:
    """
    为给定帧选择记忆帧和上下文帧，基于复杂的四帧片段距离计算。

    参数:
        w2c_list (List[np.ndarray]): 包含所有N个4x4外参矩阵的列表。
        current_frame_idx (int): 当前要处理的帧的索引。
        memory_frames (int): 需要选择的记忆帧总数。
        context_size (int): 需要选择的上下文帧总数。
        pos_weight (float): 空间距离的权重。
        ang_weight (float): 角度距离的权重。

    返回:
        List[int]: 包含选定记忆帧和上下文帧索引的列表。
    """
    num_total_frames = len(w2c_list)
    # 检查当前帧是否能构成一个完整的4帧片段
    if current_frame_idx >= num_total_frames or current_frame_idx < 3:
        raise ValueError("当前帧索引必须在 w2c_list 的有效范围内，且至少为3。")

    # 1. 选择上下文帧 (Context Frames)
    start_context_idx = max(0, current_frame_idx - temporal_context_size)
    context_frames_indices = list(range(start_context_idx, current_frame_idx))

    # 2. 计算记忆帧 (Memory Frames) 的候选池
    query_clip_indices = list(
        range(
            current_frame_idx,
            (
                current_frame_idx + pred_latent_size
                if current_frame_idx + pred_latent_size <= num_total_frames
                else num_total_frames
            ),
        )
    )

    historical_clip_indices = list(
        range(0, current_frame_idx - temporal_context_size, pred_latent_size)
    )

    memory_frames = memory_frames - temporal_context_size
    
    historical_clip_indices_tensor_raw = torch.tensor(historical_clip_indices, dtype=torch.int32, device=device)
    historical_clip_indices_tensor = torch.stack(
        [
            historical_clip_indices_tensor_raw,
            historical_clip_indices_tensor_raw + 2
        ],
        dim=1
    ).flatten()

    query_clip_indices_tensor = torch.tensor(query_clip_indices, dtype=torch.int32, device=device)
    
    historical_w2c = w2c_list[historical_clip_indices_tensor]
    query_w2c = w2c_list[query_clip_indices_tensor]
    
    fov_simularities = calculate_fov_overlap_similarity_batch(
        query_w2c,
        historical_w2c,
        points_local,
        fov_h_deg=60.0,
        fov_v_deg=35.0,
    )

    distance = (1 - fov_simularities.reshape(
        query_w2c.shape[0],
        -1,
        2
    ) ).mean(dim=[0, 2])
    
    _indices = distance.topk(k = (memory_frames // 4), largest=False).indices
    _offsets = torch.arange(4, dtype=torch.int32, device=device)  # [0,1,2,3]
    _selected_indices = (_indices.unsqueeze(1) * 4 + _offsets).reshape(-1)
    selected_frames_set = set(context_frames_indices)
    selected_frames_set.update(_selected_indices.cpu().tolist())

    final_selected_frames = sorted(list(selected_frames_set))
    assert len(final_selected_frames) == memory_frames + temporal_context_size
    return final_selected_frames