# 多視角 Feed-Forward SMPL-X（VGGT base）實作 TODO

> 架構設計見 [Claude.md](Claude.md)。本檔是可逐步 debug 的實作清單，每個 Phase 通過 debug 關卡才往下走。

## 已定案的設計決定
- **參數範圍**：SMPL **body-72 axis-angle（24 關節）+ beta10**；手/臉/expression **不回歸**（zero-pad）。重用 `clche_vggt_smpl/training/loss.py` 的 `_TorchSMPLX` + `compute_smpl_loss`。
- **Translation**：**直接回歸 world translation（`trans_world`）**；**不用** clche 的 `mesh_translate` / `normalize_joints_world_to_batch_gauge` cam0-gauge 正規化。decode 時 `verts = body(pose,beta) + trans_world`。
- **3D/2D loss 語意**：param → 可微分 SMPL-X decode → **SMPL space mesh** 比 pred vs GT（驗證參數正確性），**非** world-frame 逐視角重投影 <1px。
- **Dataset 輸入**：**full scene image → pad518**（非 per-person crop）；K 只跟 full-image resize/pad 調整；mask 取目標人層（`mask==pid+1`）走同變換，fusion 時降採到 patch grid（37×37）。
- **Mask 注入**：per-view local add `F' = F + conv(M)`，**不做 cross-view attention**（第一版）。
- **2D reproj（若用）**：早期用 GT camera。

## 來源對照
- VGGT encoder / camera head / pose_enc / rotation：`vggt/models/aggregator.py`、`vggt/heads/camera_head.py`、`vggt/utils/`
- Mask fusion 抄：`VGGT-S/src/model/prompt_encoder.py` 的 `mask_downscaling` 或 `mamma/landmarks/lib/models/models_2d/mask_proc.py` 的 `MaskEmbedding`
- Dense aux head + landmark loss：`mamma/landmarks/lib/models/models_2d/mvhead.py`（`MammaNetDecoder`）、`loss.py`（`JointGNLLLoss`）
- SMPL-X decode + param loss：`clche_vggt_smpl/training/loss.py`（`_TorchSMPLX`、`compute_smpl_loss`、`smpl_losses_plus_from_axis_angle`）
- SMPL-X 模型檔：`data/body_models/smplx_locked_head/smplx/SMPLX_*.npz`（`smplx` 套件未裝）

---

## Phase 0 — 環境 + 資料 sanity ✅ 完成 (2026-06-28)
- [x] `smplx` 已裝於 `mamma` env（`/mnt/train-data-4-hdd/yian/anaconda/envs/mamma`，py3.11, torch2.5.1+cu124）。載入 `model_path=.../smplx_locked_head`（父目錄，smplx 會自接 `smplx/`）→ 10475 verts / 127 joints OK
- [x] `MAMMA_DIR` 修正：建 symlink `Mamma_dataset/tmp/bedlam_lab_20251031_191436 -> ../bedlam_lab_20251031_191436`
- [x] dataloader 跑通，502 samples（251 frame × 2 person），schema 印出
- [x] V=4 masks 都對應目標 `person_id=0`（Invariant #11 成立），frac≈0.16（因目前仍是 crop）
- **Debug 產物**：`scratchpad/phase0_schema.py`

### Phase 0 記錄給後續 Phase 的重點
- **GT 是 full SMPL-X**：`smplx_pose (165,)`、`smplx_betas (16,)`、`smplx_trans (3,)`、`smplx_gender`。但定案回歸目標是 SMPL-72 + beta10 → **Phase 2 需處理 165→72 / 16→10 的對應**（或改用 full smplx 套件直接 decode，Phase 2 再定）。
- **Phase 1 所需 bookkeeping 已存在於 sample**：`full_intrinsics`/`full_extrinsics`（全景相機）、`crop_transforms`/`crop_bboxes`/`pad_offsets`/`original_sizes`。
- **目前 images 是 per-person crop pad 到 518**（`images` list ×4 × (518,518,3)）→ Phase 1 換 full-scene。
- 其他 GT：`depths`/`cam_points`/`world_points`/`point_masks`（VGGT 幾何）、`landmarks_512`+`landmarks_512_weights`（per-view 512 點，crop 座標）。

## Phase 1 — Dataset 改 full-scene pad518（取代 per-person crop）✅ 完成 (2026-06-28)
- [x] 讀 full image（2056×1504）→ letterbox resize/pad 到 518（保持長寬比）
- [x] K 跟 letterbox 變換調整（`crop_K = affine(ltrans) @ K`）；extrinsics 不變
- [x] mask 取目標人層 `mask==pid+1`（讀真實 `{frame}.mask.jpg`，值 {0,1,2}）→ 同 letterbox 變換
- [x] landmark 跟著同變換 + visibility 權重
- [x] 加 `full_scene` 參數（預設 True，保留 legacy crop 路徑可切換）；config 兩處設 `full_scene: True`
- **Debug 關卡 ✅**：
  - 相機一致性：調整後 `crop_K` 投影 camera-space `vertices3d` vs letterbox 後 `vertices2d`，4 view 全 **0.0000px**
  - landmarks 落在 mask 內 88–93%；mask frac 降到 0.007–0.024（crop 時 0.16）
  - 疊圖確認多人場景 + 紅 mask 只標目標人 + landmark 貼合
- **實作**：`mamma_dataset.py` 新增 `_letterbox_transform`/`_warp_image`/`_warp_binary_mask`/`_load_scene_person_mask`，`_get_bedlam_data` 加 full_scene 分支
- **尺寸決定**：維持 **letterbox 518×518**（上下黑邊）。考慮過 native-aspect 518×714（無黑邊、人大 1.36×）與中心 crop（會切到貼邊的目標人，否決）→ 先用 letterbox。相機參數**不需改**：letterbox 的 scale 與 pad_y 已包進 `crop_K`（0px 驗證過）。
- **Debug 產物**：`scratchpad/phase1_verify.py`、`debug/phase1_fullscene/fullscene_overlay_grid.png`、`debug/phase1_fullscene/masks/{view}_mask.png` + `{view}_mask_on_img.png`（per-view mask 逐視角確認，已肉眼驗證 IOI_02 右/IOI_06 左 同一人）

## Phase 2 — 可微分 SMPL-X decoder + param loss ✅ 完成 (2026-06-28)
- [x] **決策變更**：改用 **smplx 套件的完整 SMPL-X layer**（非 clche 手寫 SMPL-72 port）。原因：GT 手/臉 pose 全非零（clche zero-pad 表達不出）、betas 16 維、smplx 套件對得上資料到 0.58mm。
- [x] `training/smplx_utils.py`：`get_smplx`（cache, frozen）、`decode_smplx` / `decode_smplx_from_pose165` / `decode_smplx_grouped`（per-gender）、`project_world_to_image`、rotation helpers（aa/6D/rotmat）、`smplx_param_loss`（pose rotmat Frobenius + beta L1 + world-trans L1 + joints3d/vertices world-space）
- [x] 直接 world translation（無 mesh_translate / gauge 正規化）
- **Debug 關卡 ✅**：
  - decode GT（完整）vs 資料集 `vertices3d`：**0.58mm** 中位數（4 view）
  - param loss：loss(pred=GT)=0；擾動 body+0.2 → loss_smplx 0.515；梯度經 decode 回流（body_pose.grad 0.45 / transl.grad 0.36）
  - 投影疊圖：GT mesh → letterbox K + extrinsics → 4 view 全貼合目標人、100% 在框內
- **重要座標系慣例（給 Phase 8/9）**：
  - `vertices3d` = **camera-space**（需 `cam_ext` 轉換才比對）
  - `joints3d` = **world-space**（直接 = smplx world decode，0.53mm）→ joints3d loss 可直接在 world space 算，免轉換
  - body-only decode 中位數僅差 0.81mm（手/臉 gap 集中在末端 verts，不影響身體）→ body-only 回歸對身體指標可行，但末端有 tail error
- **Debug 產物**：`training/smplx_utils.py`、`debug/phase2_verify_smplx.py`、`debug/phase2_smplx/gt_mesh_projection.png`

## Phase 3 — VGGT encoder 接上 + variable-V ✅ 完成 (2026-06-28)
- [x] 下載 `facebook/VGGT-1B` 權重（5.03GB）到 HF cache（`~/.cache/huggingface/hub/models--facebook--VGGT-1B/.../model.pt`）
- [x] `Aggregator(img_size=518,patch_size=14,embed_dim=1024)` 909M params，`patch_start_idx=5`
- [x] variable-V：S=3/4/8 都輸出 `aggregated_tokens_list[-1]=[B,S,1374,2048]`、psi=5、len(list)=24（每 block 一個）
- [x] pretrained 載入：`missing=0`（encoder+camera head 齊全）、`unexpected=518`（停用的 depth/point/track head，預期內）；camera `pose_enc=[1,4,9]`
- **Debug 關卡 ✅**：permutation-equivariance max diff 2.4e-5（Invariant #7）；patch grid **37×37=1369**（Phase 4 mask fusion 對齊用）
- **載入方式**：`VGGT(enable_camera=True, enable_depth/point/track=False)` + `load_state_dict(torch.load(cached_model_pt), strict=False)`
- **Debug 產物**：`debug/phase3_vggt_encoder.py`

## Phase 4 — Mask fusion（per-view local add）✅ 完成 (2026-06-28)
- [x] `training/smplx_model.py`：`LayerNorm2d` + `MaskFusion`（35K params）。mask resize 到 2×grid(74) → VGGT-S 式 `Conv2d(1,16,k2s2)→LayerNorm2d→GELU→Conv2d(16,2048,k1)` → 37×37 → `F' = F + conv(M)`，per-view 本地相加，**無 cross-view attn**；special tokens(0:5) 不動；可選 `zero_init`（residual-safe，Phase 9 訓練可開）
- **Debug 關卡 ✅**：shape 保留 `[1,4,1374,2048]`、special tokens 不動；mask 全1 vs 全0 patch 差 0.24（mask 有作用）；梯度回流 conv（first 0.13 / last 0.057）；真實 mask 的 37×37 conv response **精準集中在目標人輪廓、跨視角跟著移動**
- **Debug 產物**：`debug/phase4_mask_fusion.py`、`debug/phase4_mask_fusion/mask_conv_response.png`

## Phase 5 — Camera head（mask 前讀乾淨 token）✅ 完成 (2026-06-28)
- [x] 用 pretrained VGGT camera head 在真實 8-view Mamma 影像上 forward → decode K,R,t
- [x] **Invariant #2 結構保證**：camera head 只讀 camera token(index 0)，MaskFusion 只改 patch tokens(index 5+) → 相機與 mask 完全隔離（`|pose(clean)-pose(mask-fused)|=0`）
- **Debug 關卡 ✅**（gauge-free，VGGT 相機在正規化 gauge）：
  - 相機中心 Umeyama 相似對齊殘差 = 3.7% scene scale
  - 相對旋轉測地誤差 median **1.74°**（多視角幾何良好）
  - focal pred 465.6 vs GT 452.0 px@518（~3%）
- **Debug 產物**：`debug/phase5_camera_head.py`

## Phase 6 — Dense aux head（MAMMA）+ landmark loss ✅ 完成 (2026-06-28)
- [x] `smplx_model.py`：`MLP`、`DenseLandmarkHead`（512 learnable query + 6-layer DETR-style `nn.TransformerDecoder` 讀 mask-fused patch feature，輸出**正規化 [0,1]** joints2d+log_sigma + visibility logit）、`gnll_landmark_loss`（MAMMA GNLL）、`visibility_bce_loss`
- [x] **踩雷修正**：head 原本輸出 pixel 座標 → GNLL 的 `kpts_loss_thresh=25` 一開始就 clip 掉所有座標梯度 → sigma 崩潰。改為**正規化座標** + coord-MSE warmup(100) + cosine lr decay
- **Debug 關卡 ✅**（frozen encoder，只訓 fusion+head，2 sample overfit）：valid-lm px error **57.4 → 15.8**（單調收斂）；疊圖確認 512 landmark 從中央一團擴散到目標人身體、跨 4 view 貼合 GT、只落在 masked 目標人
- **註**：~15.8px ≈ frozen 37×37 特徵解析度地板（1 patch=14px），Phase 9 解凍 encoder 會改善
- **Debug 產物**：`debug/phase6_dense_head.py`、`debug/phase6_dense_head/{loss_curve,landmarks_before_after}.png`

## Phase 7 — Per-view SMPL token + camera pose embedding ✅ 完成 (2026-06-28)
- [x] `smplx_model.py`：`SMPLViewEncoder`（每 view 1 個 learnable SMPL token，2-layer cross-attn 讀 mask-fused patch feature → per-view embedding [B,S,512]，先不回歸參數）、`CameraPoseEmbedding`（extrinsic 6D rot + t + focal → MLP → d_model，fusion 前加到 per-view token）
- **Debug 關卡 ✅**：per-view token `[1,4,512]`；per-view 獨立性 max diff **0**（每 view 只依賴自己特徵）；camera embedding 跨視角不同（‖v0-v1‖=1.44）、注入改變 token（0.31）
- **Debug 產物**：`debug/phase7_view_token.py`

## Phase 8 — Reference-query fusion + SMPL-X 回歸 head ✅ 完成 (2026-06-28)
- [x] `smplx_model.py`：`ReferenceFusion`（1 learnable reference token cross-attend V 個 per-view token → world-frame embedding，view-count 無關）、`SMPLXRegressionHead`（分開 MLP：global 6D / body 21×6D / betas16 / transl3，**6D 旋轉** Invariant #8）
- [x] `smplx_utils.py` 補 `matrix_to_axis_angle` / `rotation_6d_to_axis_angle`（6D→decode）
- [x] **回歸範圍定案**（取代早期 beta10/SMPL-72）：用 SMPL-X 原生 **global+body(21 joints)+betas16+transl**，6D 旋轉，手/臉不回歸（decode 補 0）。原因：decoder 是完整 SMPL-X、betas[10:16] 含大值不可省
- **Debug 關卡 ✅**：單 sample overfit loss 7.19→0.0001、body MPJPE/PVE **0.0mm**；投影疊圖 pred mesh 完全覆蓋 GT；**variable-V** V=3(0.2mm)/V=2(0.4mm) 免重訓正常（Invariant #7）
- **Debug 產物**：`debug/phase8_fusion_regress.py`、`debug/phase8_fusion_regress/{loss_curve,mesh_proj_pred_vs_gt}.png`

## Phase 9 — 完整 loss 組裝 + 訓練配方 ✅ 完成 (2026-06-28)
- [x] `training/smplx_pipeline.py`：`VGGTSMPLX`（組裝 aggregator→camera_head(clean)→MaskFusion→{DenseLandmarkHead || SMPLViewEncoder+CameraPoseEmbedding}→ReferenceFusion→SMPLXRegressionHead；`set_encoder_trainable()` / `load_pretrained_vggt()`）、`SMPLXMultitaskLoss`（SMPL-X param + aux landmark GNLL + visibility BCE + camera pose_enc L1）
- [x] **整合 overfit gate（Stage A：凍結 aggregator，150 步）所有 loss 一起降**：SMPL-X j3d **475→10mm**、landmark **91→9px**、camera **0.86→0.1**、vis 穩定
- [x] **兩階段配方**：Stage B 解凍 aggregator + 差分 lr（enc 1e-5 / heads 1e-4）跑通
- **踩雷/環境**：(1) AdamW 為 909M encoder 建 momentum buffer ~7GB → 16GB 卡 OOM；改 **encoder 用 SGD(no momentum) + heads AdamW + bf16 + B=1** 解決。(2) 解凍瞬間有暫態抖動後回穩 → 印證 §4.1 catastrophic-forgetting，需小 lr / warmup
- **Debug 產物**：`training/smplx_pipeline.py`、`debug/phase9_integrated_train.py`、`debug/phase9_integrated/all_losses.png`

### 接 hydra trainer（已串起來 ✅ 2026-06-28）
- [x] `config`：`model._target_=smplx_pipeline.VGGTSMPLX`、`loss._target_=smplx_pipeline.SMPLXMultitaskLoss`（含權重、betas16、`load_pretrained:True`、`freeze_encoder:True`）
- [x] `trainer._step`：依 batch 有無 `person_masks` 傳 `model(images, person_masks, gt_extrinsics=full_extrinsics, gt_intrinsics=intrinsics)`（vanilla VGGT 路徑不受影響）
- [x] `trainer._process_batch`：**無 dense 幾何 GT（point_masks 全 0）時跳過 gauge 正規化** → `extrinsics` 保持 RAW（且 SMPL-X 路徑統一讀 `full_extrinsics`）
- [x] model 內建載入 `facebook/VGGT-1B`（missing=0）；loss 輸出加 `objective` 給 trainer
- [x] **驗證**：`debug/phase9_wire_check.py` 從 config instantiate → model(243.6M 可訓/909M 凍結)→loss→backward，objective 8.3→3.3、extrinsics 確認 RAW
- [x] **config 重構**：新增自包含 `config/default_smplx.yaml`（model/loss/optim/logging/checkpoint 全 SMPL-X 化），`mamma_smplx.yaml` 改 `defaults: - default_smplx.yaml` 只留 data+scalar。修掉：gradient_clip 改 catch-all（涵蓋所有 head 參數，否則 `setup_clipping` raise）、logging scalar keys 換成我們的 loss、checkpoint resume=null、`find_unused_parameters:True`
- [x] 裝 `wcmatch`（`freeze.py` 需要）；`phase9_wire_check.py` 驗 optim+freeze+gradient_clip 全部從 config instantiate OK
- [x] **真實訓練啟動 smoke test 通過 (2026-06-28)**：`cd training && torchrun --master_port=29200 --nproc_per_node=1 launch.py --config mamma_smplx`（單卡）。epoch 0 跑完 20 batch、loss 全降（objective 8.25→4.94, pose 5.9→3.1, landmark 轉負）、checkpoint 存檔成功
  - 修掉 3 個整合 bug：(1) `launch.py` 加 sys.path（vggt import，免 PYTHONPATH）；(2) `SMPLXMultitaskLoss` 改 fp32 計算（bf16 autocast 下 smplx decode dtype 不符）；(3) dataloader 只選真的有目標人的 view + 快取 person（某 view 缺人會 KeyError）；裝 `wcmatch`
- **尚待（真正啟動大規模訓練）**：用 `launch.py` 走 DDP/torchrun + val；要解凍 encoder 時把 `freeze_encoder:False` + 加差分 lr param groups（16GB 卡用 SGD/8bit-adam 或只解凍頂層 blocks）+ 開 `amp.enabled`(bf16)+梯度累積

## Phase 10（可選）— 強化
- [ ] 重疊嚴重才加 VGGT-S cross-view bottleneck
- [ ] contact / SDF、decode-camera reproj（未來延伸，§4.10）
