import torch
import math
from typing import Iterable, List, Tuple
from .motions.camera_trajectory import generate_camera_trajectory_local
from .motions.one_hot_to_label import trans_one_hot_to_label9_cpp
from .motions.motion_fuse import motions_to_matrix_cpp
import numpy as np


def decode_combined_actions(combined: np.ndarray):
    combined = np.asarray(combined, dtype=np.int64)
    rotation_codes = combined // 9
    translation_codes = combined % 9
    return translation_codes, rotation_codes

# Maps 9-class rotation codes to the WorldPlay 9-class rotation label.
# The WorldPlay one-hot-to-label mapping for rotation uses the same LUT as
# translation.  One-hot bits are [yaw-right, yaw-left, pitch-up, pitch-down]:
#   code 0 (noop)       -> [0,0,0,0] -> idx 0  -> label 0
#   code 1 (up)         -> [0,0,1,0] -> idx 4  -> label 3
#   code 2 (down)       -> [0,0,0,1] -> idx 8  -> label 4
#   code 3 (right)      -> [1,0,0,0] -> idx 1  -> label 1
#   code 4 (left)       -> [0,1,0,0] -> idx 2  -> label 2
#   code 5 (up+right)   -> [1,0,1,0] -> idx 5  -> label 5
#   code 6 (up+left)    -> [0,1,1,0] -> idx 6  -> label 7
#   code 7 (down+right) -> [1,0,0,1] -> idx 9  -> label 6
#   code 8 (down+left)  -> [0,1,0,1] -> idx 10 -> label 8
_ROTATION_CODE_TO_LABEL9 = np.array([0, 3, 4, 1, 2, 5, 7, 6, 8], dtype=np.int64)

# Index bits: [bit0=forward, bit1=backward, bit2=right, bit3=left]
_LUT16_TO_9 = [
    0, 1, 2, -1,
    3, 5, 7, -1,
    4, 6, 8, -1,
    -1, -1, -1, -1,
]


def compute_worldplay_combined_label(
    trans_label9: torch.Tensor,
    rotation_codes: np.ndarray,
) -> torch.Tensor:
    rot_label9 = _ROTATION_CODE_TO_LABEL9[np.asarray(rotation_codes, dtype=np.int64)]
    rot_label9_t = torch.from_numpy(rot_label9).to(dtype=trans_label9.dtype)
    return trans_label9 * 9.0 + rot_label9_t


def motions_to_matrix(
    motions: np.ndarray,
    T: torch.Tensor,
    C_inv: torch.Tensor,
    first_chunk: bool = False,
    step: float = 0.08,
    tps: bool = False,
):
    """
    外层接口保持：
    def motions_to_matrix(motions: list, T: torch.Tensor, C_inv: torch.Tensor, first_chunk=False)

    行为：
    - 会 in-place 修改传入的 T（与原 generate_camera_trajectory_local 一致）
    - 会 in-place 更新 C_inv 为本chunk最后一帧的 w2c（与 pose_to_input 逻辑一致）
    """
    motions_tensor = torch.from_numpy(motions)
    viewmats, Ks, action = motions_to_matrix_cpp(motions_tensor, T, C_inv, first_chunk, step)
    return viewmats, Ks, action


def _apply_rotation(R: torch.Tensor, rc: int, yaw_step: float, pitch_step: float) -> torch.Tensor:
    """Apply a single rotation to a 3x3 c2w rotation matrix.

    Both yaw and pitch use post-multiply (camera-local frame), matching the
    original WorldPlay convention: ``R_new = R_old @ rot(angle)``.
    When both yaw and pitch are active, yaw is applied first.

    Args:
        R: ``(3,3)`` float32 rotation matrix (camera-to-world).
        rc: Rotation code (0=noop, 1=up, 2=down, 3=right, 4=left,
            5=up+right, 6=up+left, 7=down+right, 8=down+left).
        yaw_step: Yaw angle per frame in radians.
        pitch_step: Pitch angle per frame in radians.

    Returns:
        Updated ``(3,3)`` rotation matrix (new tensor, does not modify input).
    """
    if rc == 0:
        return R

    # Decompose compound codes into yaw/pitch components
    yaw_sign = 0    # +1 = right, -1 = left
    pitch_sign = 0  # +1 = up, -1 = down

    if rc == 1:   pitch_sign = 1          # up
    elif rc == 2: pitch_sign = -1         # down
    elif rc == 3: yaw_sign = 1            # right
    elif rc == 4: yaw_sign = -1           # left
    elif rc == 5: pitch_sign = 1;  yaw_sign = 1   # up + right
    elif rc == 6: pitch_sign = 1;  yaw_sign = -1  # up + left
    elif rc == 7: pitch_sign = -1; yaw_sign = 1   # down + right
    elif rc == 8: pitch_sign = -1; yaw_sign = -1  # down + left
    else: return R

    # Apply yaw first, then pitch (matching WorldPlay ordering)
    if yaw_sign != 0:
        angle = yaw_sign * yaw_step
        c, s = math.cos(angle), math.sin(angle)
        Ry = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        R = R @ Ry

    if pitch_sign != 0:
        angle = pitch_sign * pitch_step
        c, s = math.cos(angle), math.sin(angle)
        Rx = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        R = R @ Rx

    return R


def _classify_relative_translation(rel_x: float, rel_z: float) -> int:
    """Classify a relative translation (x, z) into a 9-class label.
    If the squared component exceeds 25% of the total squared norm, it counts.
    """
    norm2 = rel_x * rel_x + rel_z * rel_z
    idx = 0
    if norm2 > 1e-8:
        thr2 = 0.25 * norm2
        z2 = rel_z * rel_z
        if z2 > thr2:
            idx |= 1 if rel_z > 0 else 2
        x2 = rel_x * rel_x
        if x2 > thr2:
            idx |= 4 if rel_x > 0 else 8
    return _LUT16_TO_9[idx]


# WorldPlay intrinsic base (normalized)
_WORLDPLAY_Kbase = torch.tensor([
    [0.5051, 0.0,    0.5],
    [0.0,    0.8979, 0.5],
    [0.0,    0.0,    1.0],
], dtype=torch.float32)


def motions_to_matrix_with_rotation(
    translation_codes: np.ndarray,
    rotation_codes: np.ndarray,
    T: torch.Tensor,
    C_inv: torch.Tensor,
    first_chunk: bool = False,
    step: float = 0.08,
    yaw_step: float = 0.04,
    pitch_step: float = 0.04,
):
    """Convert WASD translation and arrow-key rotation codes to camera matrices.

    Handles per-frame rotation followed by translation, matching the original
    WorldPlay convention (rotation applied before translation each frame, both
    yaw and pitch post-multiplied in the camera's local frame).

    Args:
        translation_codes: ``(N,)`` int64 array.
            0=noop, 1=W, 2=S, 3=D, 4=A, 5=W+D, 6=W+A, 7=S+D, 8=S+A.
        rotation_codes: ``(N,)`` int64 array.
            0=noop, 1=up, 2=down, 3=right, 4=left,
            5=up+right, 6=up+left, 7=down+right, 8=down+left.
        T: ``(4,4)`` float32 CPU tensor, camera-to-world. Updated in-place.
        C_inv: ``(4,4)`` float32 CPU tensor, previous w2c. Updated in-place.
        first_chunk: True for the very first chunk of a session.
        step: Translation step size per frame.
        yaw_step: Yaw rotation per frame (radians).
        pitch_step: Pitch rotation per frame (radians).

    Returns:
        viewmats: ``(N, 4, 4)`` float32 – world-to-camera matrices.
        Ks: ``(N, 3, 3)`` float32 – intrinsic matrices (broadcast view).
        action: ``(N,)`` float32 – 9-class WASD action labels.
    """
    N = len(translation_codes)
    assert len(rotation_codes) == N

    viewmats = torch.empty((N, 4, 4), dtype=torch.float32)
    action = torch.empty((N,), dtype=torch.float32)
    Ks = _WORLDPLAY_Kbase.unsqueeze(0).expand(N, -1, -1)

    # State: rotation and translation from T (camera-to-world)
    R = T[:3, :3].clone()
    tx, ty, tz = T[0, 3].item(), T[1, 3].item(), T[2, 3].item()

    # Previous w2c translation components for relative motion classification
    # (rows 0 and 2 of w2c translation column)
    prev_w2c = C_inv.clone()

    for i in range(N):
        # 1. Apply rotation (before translation, matching original WorldPlay)
        rc = int(rotation_codes[i])
        R = _apply_rotation(R, rc, yaw_step, pitch_step)

        # 2. Apply translation in the (possibly rotated) camera local frame
        tc = int(translation_codes[i])
        lx, lz = 0.0, 0.0
        if tc == 1:    lz = step              # W → forward (+z)
        elif tc == 2:  lz = -step             # S → backward (-z)
        elif tc == 3:  lx = step              # D → right (+x)
        elif tc == 4:  lx = -step             # A → left (-x)
        elif tc == 5:  lz = step;  lx = step  # W+D → forward + right
        elif tc == 6:  lz = step;  lx = -step # W+A → forward + left
        elif tc == 7:  lz = -step; lx = step  # S+D → backward + right
        elif tc == 8:  lz = -step; lx = -step # S+A → backward + left

        # World delta = R @ local_delta (R columns 0 and 2)
        tx += R[0, 0].item() * lx + R[0, 2].item() * lz
        ty += R[1, 0].item() * lx + R[1, 2].item() * lz
        tz += R[2, 0].item() * lx + R[2, 2].item() * lz

        # 3. Compute w2c = inv(c2w).  c2w = [R | t; 0 0 0 1]
        #    w2c = [R^T | -R^T @ t; 0 0 0 1]
        Rt = R.T
        w2c_t = -(Rt @ torch.tensor([tx, ty, tz], dtype=torch.float32))
        viewmats[i, :3, :3] = Rt
        viewmats[i, :3, 3] = w2c_t
        viewmats[i, 3, :] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

        # 4. Classify relative translation for 9-class action label
        if i == 0 and first_chunk:
            action[i] = 0.0
        else:
            # Relative translation in previous camera's frame:
            # rel = prev_w2c @ [tx, ty, tz, 1]
            pw = prev_w2c
            rel_x = pw[0, 0].item() * tx + pw[0, 1].item() * ty + pw[0, 2].item() * tz + pw[0, 3].item()
            rel_z = pw[2, 0].item() * tx + pw[2, 1].item() * ty + pw[2, 2].item() * tz + pw[2, 3].item()
            action[i] = float(_classify_relative_translation(rel_x, rel_z))

        # Update prev_w2c for next frame
        prev_w2c[:] = viewmats[i]

    # In-place update T and C_inv for next chunk
    T[:3, :3] = R
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    C_inv[:] = viewmats[N - 1] if N > 0 else C_inv

    return viewmats, Ks, action
