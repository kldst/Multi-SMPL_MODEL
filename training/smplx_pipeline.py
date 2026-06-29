"""Phase 9: unified multi-view feed-forward SMPL-X model + multitask loss.

VGGTSMPLX assembles the whole architecture (Claude.md §3):
  images -> VGGT aggregator (trainable/freezable)
        -> camera_head (reads CLEAN camera token; person-agnostic)   [Invariant #2]
        -> MaskFusion (per-view local add on patch tokens)           [Invariant #3]
        -> { DenseLandmarkHead (512 aux landmarks)  ||  SMPLViewEncoder } [Invariant #4]
        -> per-view token + CameraPoseEmbedding (GT cam early)        [Invariant #5]
        -> ReferenceFusion (1 ref token, variable-V)                 [Invariant #7]
        -> SMPLXRegressionHead (6D) -> SMPL-X params

SMPLXMultitaskLoss combines: SMPL-X param loss (Phase 2) + aux landmark GNLL +
visibility BCE (Phase 6) + camera loss (pose_enc L1 vs GT).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from smplx_model import (MaskFusion, DenseLandmarkHead, SMPLViewEncoder,
                         CameraPoseEmbedding, ReferenceFusion, SMPLXRegressionHead,
                         gnll_landmark_loss, visibility_bce_loss)
from smplx_utils import decode_smplx, smplx_param_loss, rotation_6d_to_axis_angle, split_pose165


class VGGTSMPLX(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, d_model=512,
                 n_landmarks=512, num_betas=16, num_body_joints=21, reg_num_iter=3,
                 mask_zero_init=False, load_pretrained=True, freeze_encoder=True,
                 pretrained_path=None,
                 pretrained_repo="facebook/VGGT-1B", pretrained_file="model.pt", **kwargs):
        super().__init__()
        # **kwargs swallows stray keys merged from default.yaml (enable_camera, ...).
        self.vggt = VGGT(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                         enable_camera=True, enable_depth=False, enable_point=False, enable_track=False)
        self.patch_start_idx = self.vggt.aggregator.patch_start_idx          # 5
        self.grid = img_size // patch_size                                   # 37
        self.token_dim = 2 * embed_dim                                       # 2048
        self.img_size = img_size
        self.d_model = d_model

        self.mask_fusion = MaskFusion(self.token_dim, grid=self.grid,
                                      patch_start_idx=self.patch_start_idx, zero_init=mask_zero_init)
        self.dense_head = DenseLandmarkHead(self.token_dim, d_model=256, n_landmarks=n_landmarks,
                                            grid=self.grid, img_size=img_size)
        self.view_enc = SMPLViewEncoder(self.token_dim, d_model=d_model, grid=self.grid)
        self.cam_emb = CameraPoseEmbedding(d_model=d_model, img_size=img_size)
        self.ref_fusion = ReferenceFusion(d_model=d_model)
        self.reg_head = SMPLXRegressionHead(d_model=d_model, num_body_joints=num_body_joints,
                                            num_betas=num_betas, num_iter=reg_num_iter)

        if load_pretrained:
            # Priority: explicit local pretrained_path -> HF cache -> download from HF.
            if pretrained_path and os.path.isfile(os.path.expanduser(pretrained_path)):
                path = os.path.expanduser(pretrained_path)
            else:
                if pretrained_path:
                    print(f"[VGGTSMPLX] pretrained_path '{pretrained_path}' not found; "
                          f"falling back to HF ({pretrained_repo}/{pretrained_file})")
                from huggingface_hub import try_to_load_from_cache, hf_hub_download
                cached = try_to_load_from_cache(pretrained_repo, pretrained_file)
                path = cached if isinstance(cached, str) else hf_hub_download(pretrained_repo, pretrained_file)
            miss, unexp = self.load_pretrained_vggt(path)
            print(f"[VGGTSMPLX] loaded {path} (missing={len(miss)} unexpected={len(unexp)})")
        if freeze_encoder:
            self.set_encoder_trainable(False)   # Stage A: train heads + camera_head only

    def set_encoder_trainable(self, flag: bool):
        for p in self.vggt.aggregator.parameters():
            p.requires_grad_(flag)

    def load_pretrained_vggt(self, model_pt_path: str):
        sd = torch.load(model_pt_path, map_location="cpu")
        return self.vggt.load_state_dict(sd, strict=False)

    def forward(self, images, person_masks, gt_extrinsics, gt_intrinsics):
        """images [B,S,3,H,W]; person_masks [B,S,H,W]; gt_extr [B,S,4,4]; gt_intr [B,S,3,3]."""
        B, S = images.shape[:2]
        G2 = self.grid * self.grid
        ps = self.patch_start_idx
        tokens_list, _ = self.vggt.aggregator(images)
        clean = tokens_list[-1]                                              # [B,S,P,2048]
        pose_enc = self.vggt.camera_head(tokens_list)[-1]                    # [B,S,9] (clean)

        fused = self.mask_fusion(clean, person_masks)
        patch = fused[:, :, ps:ps + G2, :].reshape(B * S, G2, self.token_dim)

        dense = self.dense_head(patch)                                       # joints2d/visibility
        vtok = self.view_enc(patch) + self.cam_emb(
            gt_extrinsics.reshape(B * S, 4, 4), gt_intrinsics.reshape(B * S, 3, 3))
        person = self.ref_fusion(vtok.reshape(B, S, self.d_model))           # [B,d]
        params = self.reg_head(person)

        return {
            "pose_enc": pose_enc,
            "dense_joints2d": dense["joints2d"].reshape(B, S, -1, 3),        # normalised [0,1]
            "dense_visibility": dense["visibility"].reshape(B, S, -1, 1),
            "global_orient_6d": params["global_orient_6d"],
            "body_pose_6d": params["body_pose_6d"],
            "betas": params["betas"],
            "transl": params["transl"],
        }


class SMPLXMultitaskLoss(nn.Module):
    def __init__(self, w_smplx=1.0, w_landmark=1.0, w_vis=0.1, w_camera=1.0,
                 w_pose=1.0, w_beta=0.1, w_trans=1.0, w_joints3d=1.0, w_vertices=1.0,
                 smplx_model_root=None, **kwargs):
        super().__init__()
        # **kwargs swallows stray keys merged from default.yaml (camera, depth, ...).
        self.w = dict(smplx=w_smplx, landmark=w_landmark, vis=w_vis, camera=w_camera)
        self.smplx_model_root = smplx_model_root   # None -> use smplx_utils default / SMPLX_MODEL_ROOT env
        self.smplx_w = dict(w_pose=w_pose, w_beta=w_beta, w_trans=w_trans,
                            w_joints3d=w_joints3d, w_vertices=w_vertices)

    def forward(self, pred, batch):
        # SMPL-X geometry (decode/projection) must run in fp32: under bf16 autocast the
        # smplx model (fp32 buffers) returns fp32 while our gather buffers would be bf16
        # -> "Index put dtype mismatch". Disable autocast and upcast predictions.
        with torch.cuda.amp.autocast(enabled=False):
            pred = {k: (v.float() if torch.is_tensor(v) else v) for k, v in pred.items()}
            return self._compute(pred, batch)

    def _compute(self, pred, batch):
        dev = pred["betas"].device
        genders = batch.get("smplx_gender", ["neutral"] * pred["betas"].shape[0])
        if isinstance(genders, str):
            genders = [genders]

        # --- SMPL-X param loss ---
        pred_dict = dict(
            global_orient=rotation_6d_to_axis_angle(pred["global_orient_6d"]),
            body_pose=rotation_6d_to_axis_angle(pred["body_pose_6d"]),
            betas=pred["betas"], transl=pred["transl"])
        gp = split_pose165(batch["smplx_pose"].to(dev))
        gt_dict = dict(global_orient=gp["global_orient"], body_pose=gp["body_pose"],
                       betas=batch["smplx_betas"].to(dev), transl=batch["smplx_trans"].to(dev))
        smplx_kw = dict(self.smplx_w)
        if self.smplx_model_root:
            smplx_kw["model_root"] = self.smplx_model_root
        Lsmplx = smplx_param_loss(pred_dict, gt_dict, genders=genders, **smplx_kw)

        # --- aux dense landmark (GNLL, normalised) + visibility BCE ---
        B, S = pred["dense_joints2d"].shape[:2]
        N = pred["dense_joints2d"].shape[2]
        pj = pred["dense_joints2d"].reshape(B * S, N, 3)
        pv = pred["dense_visibility"].reshape(B * S, N, 1)
        tgt = (batch["landmarks_512"].to(dev).reshape(B * S, N, 2)) / self.img_size_of(batch)
        w = batch["landmarks_512_weights"].to(dev).reshape(B * S, N)        # weight (incl bodypart)
        l_lm, lm_stats = gnll_landmark_loss(pj, tgt, w)
        l_vis = visibility_bce_loss(pv, (batch["landmarks_512_weights"].to(dev).reshape(B * S, N, 1) > 0).float())

        # --- camera loss (pose_enc L1 vs GT) ---
        # use full_extrinsics (RAW world->cam, NOT gauge-normalised) so everything
        # stays in one consistent world frame with the SMPL-X path.
        gt_extr = batch.get("full_extrinsics", batch["extrinsics"])
        gt_pose_enc = extri_intri_to_pose_encoding(
            gt_extr[:, :, :3, :4].to(dev), batch["intrinsics"].to(dev),
            (self.img_size_of(batch), self.img_size_of(batch)))
        l_cam = (pred["pose_enc"] - gt_pose_enc).abs().mean()

        total = (self.w["smplx"] * Lsmplx["loss_smplx"] + self.w["landmark"] * l_lm
                 + self.w["vis"] * l_vis + self.w["camera"] * l_cam)
        out = {"objective": total, "loss_objective": total, "loss_total": total,
               "loss_smplx": Lsmplx["loss_smplx"],
               "loss_pose": Lsmplx["loss_pose"], "loss_beta": Lsmplx["loss_beta"],
               "loss_trans": Lsmplx["loss_trans"], "loss_joints3d": Lsmplx["loss_joints3d"],
               "loss_vertices": Lsmplx["loss_vertices"], "loss_landmark": l_lm,
               "loss_vis": l_vis, "loss_camera": l_cam, "lm_px": lm_stats["px_err"]}
        return out

    @staticmethod
    def img_size_of(batch):
        return batch["images"].shape[-1]
