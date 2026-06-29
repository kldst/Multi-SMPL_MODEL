"""Model components for multi-view feed-forward SMPL-X (built incrementally).

Phase 4: MaskFusion — per-view local mask injection on the VGGT patch tokens.
  F'_v = F_v + conv(M_v)   (each view independent; NO cross-view attention)

Mask-encoder architecture mirrors VGGT-S's `mask_downscaling`
(Conv2d k2s2 -> LayerNorm2d -> GELU -> Conv2d 1x1), fed a mask resized to 2x the
patch grid so the stride-2 conv lands exactly on the 37x37 grid.
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channels-first LayerNorm over C for (N, C, H, W) — same as VGGT-S/SAM2."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MaskFusion(nn.Module):
    """Per-view target-person mask injection onto VGGT patch tokens.

    tokens : [B, S, P, D]   (P = patch_start_idx + grid*grid ; D = token_dim)
    mask   : [B, S, H, W] or [B, S, 1, H, W]  (binary, target person)
    returns: [B, S, P, D]   patch tokens get + conv(mask); special tokens untouched.
    """
    def __init__(self, token_dim: int = 2048, mask_in_chans: int = 16,
                 grid: int = 37, patch_start_idx: int = 5, zero_init: bool = False):
        super().__init__()
        self.grid = grid
        self.patch_start_idx = patch_start_idx
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            nn.GELU(),
            nn.Conv2d(mask_in_chans, token_dim, kernel_size=1),
        )
        if zero_init:
            # residual-safe: at init F' = F, so a pretrained backbone is untouched
            nn.init.zeros_(self.mask_downscaling[-1].weight)
            nn.init.zeros_(self.mask_downscaling[-1].bias)

    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """[N,1,H,W] -> [N, D, grid, grid] via resize-to-2xgrid + downscaling conv."""
        m = F.interpolate(mask, size=(self.grid * 2, self.grid * 2),
                          mode="area")
        return self.mask_downscaling(m)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, S, P, D = tokens.shape
        if mask.dim() == 4:                 # [B,S,H,W] -> [B,S,1,H,W]
            mask = mask.unsqueeze(2)
        m = mask.reshape(B * S, 1, mask.shape[-2], mask.shape[-1]).to(tokens.dtype)
        enc = self.encode_mask(m)            # [B*S, D, g, g]
        enc = enc.flatten(2).transpose(1, 2).reshape(B, S, self.grid * self.grid, D)

        out = tokens.clone()
        ps = self.patch_start_idx
        out[:, :, ps:ps + self.grid * self.grid, :] = tokens[:, :, ps:ps + self.grid * self.grid, :] + enc
        return out


class MLP(nn.Module):
    """Simple multi-layer perceptron (MAMMA-style)."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DenseLandmarkHead(nn.Module):
    """MAMMA-style dense landmark head (Phase 6).

    512 learnable per-landmark queries cross-attend the (mask-fused) patch
    feature map of ONE view and decode 2D landmarks + uncertainty + visibility.
    Operates per view (input flattened to N = B*S).

    patch_tokens : [N, grid*grid, token_dim]
    returns dict:
      joints2d   : [N, n_landmarks, 3]  (x_norm, y_norm in [0,1], log_sigma)
      visibility : [N, n_landmarks, 1]  (logit)
    Coordinates are NORMALISED to [0,1] (sigmoid) so the Gaussian-NLL stays
    well-posed; multiply by image size when projecting/visualising.
    """
    def __init__(self, token_dim=2048, d_model=256, n_heads=8, n_layers=6,
                 n_landmarks=512, grid=37, img_size=518, dropout=0.0):
        super().__init__()
        self.grid = grid
        self.img_size = img_size
        self.n_landmarks = n_landmarks
        self.in_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.query = nn.Embedding(n_landmarks, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, nn.LayerNorm(d_model))
        self.coord = MLP(d_model, d_model, 3, num_layers=3)   # x, y, log_sigma
        self.vis = nn.Linear(d_model, 1)

    def forward(self, patch_tokens: torch.Tensor) -> dict:
        N = patch_tokens.shape[0]
        mem = self.in_proj(patch_tokens) + self.pos_embed
        q = self.query.weight.unsqueeze(0).expand(N, -1, -1)
        hs = self.decoder(q, mem)                              # [N, n_landmarks, d_model]
        c = self.coord(hs)
        xy = torch.sigmoid(c[..., :2])                         # normalised [0,1]
        log_sigma = c[..., 2:3]
        joints2d = torch.cat([xy, log_sigma], dim=-1)          # [N, n_landmarks, 3]
        return {"joints2d": joints2d, "visibility": self.vis(hs)}


def gnll_landmark_loss(pred_joints2d, target_xy, weights, kpts_loss_thresh=25.0):
    """MAMMA Gaussian-NLL landmark loss.
    pred_joints2d : [N,K,3] (x,y,log_sigma) ; target_xy : [N,K,2] ; weights : [N,K]"""
    sq = ((target_xy - pred_joints2d[..., :2]) ** 2).sum(-1)            # [N,K]
    sqw = sq * weights
    log_sigma = pred_joints2d[..., 2].clamp(min=math.log(1e-6))
    two_sigma_sq = 2.0 * torch.exp(log_sigma) ** 2
    kpts_loss = torch.clip(sqw / two_sigma_sq, max=kpts_loss_thresh).mean()
    sigma_loss = (2.0 * log_sigma * weights).mean()
    denom = weights.sum().clamp(min=1.0)
    raw_px = (sq.sqrt() * weights).sum() / denom                        # mean px error on valid lms
    return kpts_loss + sigma_loss, {"loss_kpts": kpts_loss, "loss_sigma": sigma_loss, "px_err": raw_px}


def visibility_bce_loss(vis_logit, vis_target):
    """vis_logit/vis_target : [N,K,1]"""
    return F.binary_cross_entropy_with_logits(vis_logit, vis_target.float())


# ============================================================================
# Phase 7 — per-view SMPL token + camera pose embedding
# ============================================================================
class SMPLViewEncoder(nn.Module):
    """One learnable SMPL token per view cross-attends that view's (mask-fused)
    patch feature map -> a per-view pose EMBEDDING (params decoded later, after
    fusion, so world-frame consistency is guaranteed by the fusion stage).

    patch_tokens : [N, grid*grid, token_dim]  (N = B*S)  ->  [N, d_model]
    """
    def __init__(self, token_dim=2048, d_model=512, n_heads=8, n_layers=2, grid=37, dropout=0.0):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, nn.LayerNorm(d_model))

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        N = patch_tokens.shape[0]
        mem = self.in_proj(patch_tokens) + self.pos_embed
        q = self.query.expand(N, -1, -1)
        return self.decoder(q, mem)[:, 0]                       # [N, d_model]


class CameraPoseEmbedding(nn.Module):
    """Encode a view's camera (extrinsic 6D rot + translation + focal) -> d_model,
    added to the per-view SMPL token BEFORE fusion so the reference token knows each
    view's geometry to translate per-camera evidence into one world frame."""
    def __init__(self, d_model=512, img_size=518):
        super().__init__()
        self.img_size = img_size
        self.mlp = MLP(6 + 3 + 2, d_model, d_model, num_layers=2)

    def forward(self, extr: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
        # extr [N,4,4] world->cam ; intr [N,3,3]
        R6 = extr[:, :3, :3][:, :2, :].reshape(-1, 6)
        t = extr[:, :3, 3]
        f = torch.stack([intr[:, 0, 0], intr[:, 1, 1]], dim=-1) / self.img_size
        return self.mlp(torch.cat([R6, t, f], dim=-1))         # [N, d_model]


# ============================================================================
# Phase 8 — reference-query fusion + SMPL-X regression
# ============================================================================
class ReferenceFusion(nn.Module):
    """A single learnable reference token (the target person) cross-attends the V
    per-view tokens (key/value) -> one world-frame embedding. View-count agnostic:
    attention over a variable number of view tokens needs no retraining.

    view_tokens : [B, S, d_model]  ->  [B, d_model]
    """
    def __init__(self, d_model=512, n_heads=8, n_layers=2, dropout=0.0):
        super().__init__()
        self.ref = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.ref, std=0.02)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, nn.LayerNorm(d_model))

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        B = view_tokens.shape[0]
        q = self.ref.expand(B, -1, -1)
        return self.decoder(q, view_tokens)[:, 0]              # [B, d_model]


class SMPLXRegressionHead(nn.Module):
    """Decode the fused person token into SMPL-X params with Iterative Error Feedback
    (IEF, HMR/SPIN-style). Starts from an INIT pose and predicts RESIDUALS conditioned
    on the current estimate, repeated num_iter times. Rotations are 6D (Invariant #8);
    hands/face not regressed.

    The init pose is either:
      * MEAN pose (HMR2/TRAM-style) loaded from `mean_params_path` (smpl_mean_params.npz),
        which avoids the rotation singularity at identity and matches the data
        distribution -> faster convergence; OR
      * the REST pose (identity 6D rotations, betas=0, transl=0) when no path is given.

    x : [B, d_model] -> dict(global_orient_6d, body_pose_6d[21*6], betas, transl)

    num_iter=1 reduces to a single-shot regressor (equivalent to the old behaviour).
    """
    def __init__(self, d_model=512, num_body_joints=21, num_betas=16,
                 num_iter=3, rest_pose_init=True, mean_params_path=None):
        super().__init__()
        self.num_body_joints = num_body_joints
        self.num_betas = num_betas
        self.num_iter = max(1, int(num_iter))
        self.g_dim = 6
        self.b_dim = num_body_joints * 6
        param_dim = self.g_dim + self.b_dim + num_betas + 3
        self.param_dim = param_dim

        # residual MLPs (predict a delta each iteration)
        self.global_orient = MLP(d_model, d_model, self.g_dim, num_layers=2)
        self.body_pose = MLP(d_model, d_model, self.b_dim, num_layers=2)
        self.betas = MLP(d_model, d_model, num_betas, num_layers=2)
        self.transl = MLP(d_model, d_model, 3, num_layers=2)
        # embed the current params back into the token (IEF feedback)
        self.param_embed = nn.Linear(param_dim, d_model)

        # starting point for IEF: mean pose (if file given) else rest pose.
        init = self._build_init_params(mean_params_path)
        self.register_buffer("init_params", init.view(1, -1))

        if rest_pose_init:
            # zero-init residual heads -> first residual is 0 -> output starts AT the
            # init pose, so loss_pose starts low instead of from random rotations.
            for head in (self.global_orient, self.body_pose, self.betas, self.transl):
                nn.init.zeros_(head.layers[-1].weight)
                nn.init.zeros_(head.layers[-1].bias)

    def _build_init_params(self, mean_params_path):
        """Return the flat init param vector [g6 | body6*nbj | betas | transl(3)] in
        THIS head's 6D convention (Zhou et al. / rotation_6d_to_matrix = first 2 rows)."""
        nbj, nb = self.num_body_joints, self.num_betas
        if mean_params_path and os.path.isfile(os.path.expanduser(mean_params_path)):
            d = np.load(os.path.expanduser(mean_params_path))
            # smpl_mean_params.npz: pose=(144,)=24*6 in HMR2/TRAM 6D convention
            # (reshape (3,2) -> two COLUMNS); shape=(10,) mean betas.
            pose = torch.tensor(np.asarray(d["pose"], dtype=np.float32)).reshape(-1, 6)  # (24,6)
            shape = torch.tensor(np.asarray(d["shape"], dtype=np.float32)).reshape(-1)   # (10,)
            R = self._hmr2_6d_to_matrix(pose)                # (24,3,3)
            my6d = R[:, :2, :].reshape(-1, 6)                # (24,6) in our row convention
            # SMPL(24) -> SMPL-X(22): joint0=global, joints1..nbj = body (drop SMPL hands 22,23)
            global6 = my6d[0]
            body6 = my6d[1:1 + nbj].reshape(-1)
            betas = torch.zeros(nb)
            m = min(nb, shape.numel())
            betas[:m] = shape[:m]
            print(f"[SMPLXRegressionHead] mean-pose init from {mean_params_path} "
                  f"(global+{nbj} body joints, {m} mean betas)")
            return torch.cat([global6, body6, betas, torch.zeros(3)])
        if mean_params_path:
            print(f"[SMPLXRegressionHead] mean_params_path '{mean_params_path}' not found; "
                  f"falling back to REST-pose init")
        # REST pose: identity 6D per joint, betas=0, transl=0
        id6 = torch.tensor([1., 0., 0., 0., 1., 0.])
        return torch.cat([id6, id6.repeat(nbj), torch.zeros(nb), torch.zeros(3)])

    @staticmethod
    def _hmr2_6d_to_matrix(d6):
        """HMR2/TRAM 6D -> rotation matrix. Their convention reshapes the 6 numbers as
        (3,2) so the two halves are the first two COLUMNS of R (differs from our Zhou
        row-split rotation_6d_to_matrix). d6: (...,6) -> (...,3,3)."""
        x = d6.reshape(*d6.shape[:-1], 3, 2)
        a1, a2 = x[..., 0], x[..., 1]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)            # columns

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]
        params = self.init_params.expand(B, -1).clone()        # [B, param_dim] rest pose
        for _ in range(self.num_iter):
            h = x + self.param_embed(params)                   # feed current estimate back
            delta = torch.cat([self.global_orient(h), self.body_pose(h),
                               self.betas(h), self.transl(h)], dim=-1)
            params = params + delta                            # accumulate residual
        g, b, nb = self.g_dim, self.b_dim, self.num_betas
        return {
            "global_orient_6d": params[:, :g],
            "body_pose_6d": params[:, g:g + b],
            "betas": params[:, g + b:g + b + nb],
            "transl": params[:, g + b + nb:],
        }
