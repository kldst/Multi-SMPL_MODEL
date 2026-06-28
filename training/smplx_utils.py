"""Differentiable SMPL-X layer + parameter losses (Phase 2).

Uses the `smplx` package directly (verified to reproduce the Mamma_dataset
camera-space vertices3d to ~0.6mm when fed the full GT params).  The Mamma GT is
FULL SMPL-X: pose (165) = global(3)+body(63)+jaw(3)+leye(3)+reye(3)+lhand(45)+rhand(45),
betas(16), trans(3).  Hands/face are NON-zero in this data, so we keep the full
SMPL-X forward (clche's pose72 zero-pad port cannot represent them).

Coordinate convention:
  decode_smplx(...) returns WORLD-space joints/vertices (trans applied).
  camera-space = R_w2c @ X_world + t_w2c   (cam_ext is world->camera)

The differentiable layer flows gradients to global_orient / body_pose / betas /
transl / (optional) hand & face poses; the model buffers are frozen.
"""
import os
import os.path as osp
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx

# SMPL-X model root: resolve relative to the repo (training/ -> repo root) so it is
# portable across platforms; override with env var SMPLX_MODEL_ROOT if needed.
DEFAULT_MODEL_ROOT = os.environ.get(
    "SMPLX_MODEL_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "body_models", "smplx_locked_head"),
)

# pose(165) slice layout
POSE_SLICES = {
    "global_orient": (0, 3),
    "body_pose": (3, 66),
    "jaw_pose": (66, 69),
    "leye_pose": (69, 72),
    "reye_pose": (72, 75),
    "left_hand_pose": (75, 120),
    "right_hand_pose": (120, 165),
}

_SMPLX_CACHE: Dict[tuple, nn.Module] = {}


def get_smplx(device, gender: str = "neutral", num_betas: int = 16,
              model_root: str = DEFAULT_MODEL_ROOT) -> nn.Module:
    """Lazily build & cache a frozen SMPL-X module (per device/gender/num_betas)."""
    gender = str(gender).strip().lower()
    if gender.startswith("m"):
        gender = "male"
    elif gender.startswith("f"):
        gender = "female"
    else:
        gender = "neutral"
    key = (str(device), gender, int(num_betas))
    if key not in _SMPLX_CACHE:
        m = smplx.create(
            model_path=model_root, model_type="smplx", gender=gender,
            use_pca=False, flat_hand_mean=True,
            num_betas=int(num_betas), num_expression_coeffs=0,
            batch_size=1,
        ).to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _SMPLX_CACHE[key] = m
    return _SMPLX_CACHE[key]


def split_pose165(pose165: torch.Tensor) -> Dict[str, torch.Tensor]:
    """(B,165) axis-angle -> dict of the 7 SMPL-X pose parts."""
    return {name: pose165[:, a:b] for name, (a, b) in POSE_SLICES.items()}


def decode_smplx(
    global_orient: torch.Tensor,      # (B,3) axis-angle
    body_pose: torch.Tensor,          # (B,63)
    betas: torch.Tensor,              # (B,num_betas)
    transl: Optional[torch.Tensor] = None,   # (B,3) WORLD translation
    *,
    jaw_pose: Optional[torch.Tensor] = None,
    leye_pose: Optional[torch.Tensor] = None,
    reye_pose: Optional[torch.Tensor] = None,
    left_hand_pose: Optional[torch.Tensor] = None,
    right_hand_pose: Optional[torch.Tensor] = None,
    gender: str = "neutral",
    model_root: str = DEFAULT_MODEL_ROOT,
):
    """Differentiable SMPL-X forward. Missing parts default to zero.
    Returns (vertices_world (B,10475,3), joints_world (B,127,3))."""
    device = global_orient.device
    B = global_orient.shape[0]
    nb = betas.shape[-1]
    m = get_smplx(device, gender, nb, model_root)

    def z(n):
        return torch.zeros(B, n, device=device, dtype=global_orient.dtype)

    out = m(
        global_orient=global_orient,
        body_pose=body_pose,
        jaw_pose=jaw_pose if jaw_pose is not None else z(3),
        leye_pose=leye_pose if leye_pose is not None else z(3),
        reye_pose=reye_pose if reye_pose is not None else z(3),
        left_hand_pose=left_hand_pose if left_hand_pose is not None else z(45),
        right_hand_pose=right_hand_pose if right_hand_pose is not None else z(45),
        betas=betas,
        transl=transl if transl is not None else z(3),
        expression=z(0) if m.num_expression_coeffs == 0 else z(m.num_expression_coeffs),
        return_verts=True,
    )
    return out.vertices, out.joints


def decode_smplx_from_pose165(pose165, betas, transl, gender="neutral",
                              model_root=DEFAULT_MODEL_ROOT, body_only=False):
    """Convenience: decode from the full (B,165) GT pose vector.
    body_only=True zeroes hands/face (to measure the body-only approximation gap)."""
    p = split_pose165(pose165)
    kw = dict(global_orient=p["global_orient"], body_pose=p["body_pose"],
              betas=betas, transl=transl, gender=gender, model_root=model_root)
    if not body_only:
        kw.update(jaw_pose=p["jaw_pose"], leye_pose=p["leye_pose"], reye_pose=p["reye_pose"],
                  left_hand_pose=p["left_hand_pose"], right_hand_pose=p["right_hand_pose"])
    return decode_smplx(**kw)


def decode_smplx_grouped(pose165, betas, transl, genders: List[str],
                         model_root=DEFAULT_MODEL_ROOT):
    """Batched decode with per-sample gender (splits the batch by gender)."""
    B = pose165.shape[0]
    verts = torch.zeros(B, 10475, 3, device=pose165.device, dtype=pose165.dtype)
    joints = torch.zeros(B, 127, 3, device=pose165.device, dtype=pose165.dtype)
    genders = [str(g) for g in genders]
    for g in set(genders):
        idx = torch.tensor([i for i, gg in enumerate(genders) if gg == g],
                           device=pose165.device, dtype=torch.long)
        v, j = decode_smplx_from_pose165(pose165[idx], betas[idx], transl[idx], gender=g,
                                         model_root=model_root)
        verts[idx] = v
        joints[idx] = j
    return verts, joints


def project_world_to_image(points_world: torch.Tensor, extr: torch.Tensor,
                           K: torch.Tensor) -> torch.Tensor:
    """points_world (B,N,3); extr (B,4,4 or 3,4) world->cam; K (B,3,3) -> (B,N,2) px."""
    R = extr[:, :3, :3]
    t = extr[:, :3, 3]
    cam = torch.einsum("bij,bnj->bni", R, points_world) + t[:, None, :]
    img = torch.einsum("bij,bnj->bni", K, cam)
    return img[..., :2] / img[..., 2:3].clamp(min=1e-8)


# ----------------------------------------------------------------------------
# Rotation helpers (axis-angle <-> rotmat <-> 6D)  -- used by the param losses
# ----------------------------------------------------------------------------
def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """(...,J*3) or (...,3) -> (...,J,3,3) or (...,3,3) via Rodrigues."""
    squeeze = aa.shape[-1] == 3
    J = aa.shape[-1] // 3
    a = aa.reshape(*aa.shape[:-1], J, 3)
    theta = a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = a / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    Kx = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(*k.shape[:-1], 3, 3)
    th = theta[..., 0][..., None, None]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(Kx)
    R = eye + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)
    return R[..., 0, :, :] if squeeze else R


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(...,6) -> (...,3,3) (Zhou et al. 2019)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,6)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(...,3,3) -> (...,3) axis-angle via quaternion (Shepperd, stable near pi)."""
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)  # w,x,y,z
    c0 = trace > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = (~c0) & (~c1) & (~c2)
    if c0.any():
        s = torch.sqrt((trace[c0] + 1.0).clamp(min=eps)) * 2
        q[c0, 0] = 0.25 * s
        q[c0, 1] = (R[c0, 2, 1] - R[c0, 1, 2]) / s
        q[c0, 2] = (R[c0, 0, 2] - R[c0, 2, 0]) / s
        q[c0, 3] = (R[c0, 1, 0] - R[c0, 0, 1]) / s
    for ci, (a, b, cc) in [(c1, (0, 1, 2)), (c2, (1, 2, 0)), (c3, (2, 0, 1))]:
        if ci.any():
            s = torch.sqrt((1.0 + R[ci, a, a] - R[ci, b, b] - R[ci, cc, cc]).clamp(min=eps)) * 2
            q[ci, 0] = (R[ci, cc, b] - R[ci, b, cc]) / s
            q[ci, 1 + a] = 0.25 * s
            q[ci, 1 + b] = (R[ci, b, a] + R[ci, a, b]) / s
            q[ci, 1 + cc] = (R[ci, cc, a] + R[ci, a, cc]) / s
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=eps)
    w = q[:, 0].clamp(-1, 1)
    angle = 2 * torch.acos(w)
    sin_half = torch.sqrt((1 - w * w).clamp(min=0))
    small = sin_half < 1e-6
    axis = q[:, 1:] / torch.where(small, torch.ones_like(sin_half), sin_half).unsqueeze(-1)
    aa = axis * angle.unsqueeze(-1)
    aa = torch.where(small.unsqueeze(-1), torch.zeros_like(aa), aa)
    return aa.reshape(*shape, 3)


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """(...,J*6) -> (...,J*3) axis-angle (per-joint 6D rotation regression output)."""
    J = d6.shape[-1] // 6
    R = rotation_6d_to_matrix(d6.reshape(*d6.shape[:-1], J, 6))   # (...,J,3,3)
    return matrix_to_axis_angle(R).reshape(*d6.shape[:-1], J * 3)


# ----------------------------------------------------------------------------
# SMPL-X parameter loss (world space; direct world translation, no gauge norm)
# ----------------------------------------------------------------------------
def smplx_param_loss(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    *,
    genders: Optional[List[str]] = None,
    w_pose: float = 1.0,
    w_beta: float = 0.1,
    w_trans: float = 1.0,
    w_joints3d: float = 1.0,
    w_vertices: float = 1.0,
    loss_type: str = "l1",
    model_root: str = DEFAULT_MODEL_ROOT,
) -> Dict[str, torch.Tensor]:
    """pred/gt dicts need: global_orient(B,3), body_pose(B,63), betas(B,Nb), transl(B,3)
    (all axis-angle).  Optional hand/face parts decoded if present.
    pose loss = rotmat Frobenius on global+body; beta/trans = L1; joints3d/vertices
    compared in WORLD space after SMPL-X decode."""
    def cat_gb(d):
        return torch.cat([d["global_orient"], d["body_pose"]], dim=-1)  # (B,66)

    R_pred = axis_angle_to_matrix(cat_gb(pred))   # (B,22,3,3)
    R_gt = axis_angle_to_matrix(cat_gb(gt))
    loss_pose = ((R_pred - R_gt) ** 2).sum(dim=(-1, -2)).mean()

    if loss_type == "l1":
        loss_beta = (pred["betas"] - gt["betas"]).abs().mean()
        loss_trans = (pred["transl"] - gt["transl"]).abs().mean()
    else:
        loss_beta = ((pred["betas"] - gt["betas"]) ** 2).mean()
        loss_trans = ((pred["transl"] - gt["transl"]) ** 2).mean()

    out = {"loss_pose": loss_pose, "loss_beta": loss_beta, "loss_trans": loss_trans}

    if w_joints3d > 0 or w_vertices > 0:
        B = pred["global_orient"].shape[0]
        if genders is None:
            genders = ["neutral"] * B

        def decode(d):
            verts = torch.zeros(B, 10475, 3, device=d["betas"].device, dtype=d["betas"].dtype)
            joints = torch.zeros(B, 127, 3, device=d["betas"].device, dtype=d["betas"].dtype)
            for g in set(genders):
                idx = torch.tensor([i for i, gg in enumerate(genders) if gg == g],
                                   device=d["betas"].device, dtype=torch.long)
                v, j = decode_smplx(
                    d["global_orient"][idx], d["body_pose"][idx], d["betas"][idx], d["transl"][idx],
                    jaw_pose=d.get("jaw_pose", None) if d.get("jaw_pose") is None else d["jaw_pose"][idx],
                    left_hand_pose=None if d.get("left_hand_pose") is None else d["left_hand_pose"][idx],
                    right_hand_pose=None if d.get("right_hand_pose") is None else d["right_hand_pose"][idx],
                    gender=g, model_root=model_root,
                )
                verts[idx] = v
                joints[idx] = j
            return verts, joints

        v_p, j_p = decode(pred)
        v_g, j_g = decode(gt)
        out["loss_joints3d"] = (j_p - j_g).abs().mean() if loss_type == "l1" else ((j_p - j_g) ** 2).mean()
        out["loss_vertices"] = (v_p - v_g).abs().mean() if loss_type == "l1" else ((v_p - v_g) ** 2).mean()
    else:
        out["loss_joints3d"] = loss_pose.new_zeros(())
        out["loss_vertices"] = loss_pose.new_zeros(())

    out["loss_smplx"] = (
        w_pose * out["loss_pose"] + w_beta * out["loss_beta"] + w_trans * out["loss_trans"]
        + w_joints3d * out["loss_joints3d"] + w_vertices * out["loss_vertices"]
    )
    return out
