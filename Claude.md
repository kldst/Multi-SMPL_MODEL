# 多視角多人 Feed-Forward SMPL-X 架構設計

> 一個從多視角 scene image + 「指定單一目標人」的 per-view mask，前饋回歸出**該目標人**單一 world-frame SMPL-X 的系統。
> **資料設定**：場景本身是多人的，但每一筆訓練鎖定其中一個人——給的 V 張 mask 都只框那一個目標人，模型 decode 的就是被框那個人的 SMPL-X。多人靠「同一場景換 mask 對不同目標人各跑一次」達成（人與人獨立，見 §4.10）。
> 核心定位：把 MAMMA 的「per-view landmark + 離線 L-BFGS fitting」中的 fitting 部分，用 VGGT 式 reference-query 前饋取代，達到 **view-count 無關、免重訓、無需 optimization**。

[TOC]

---

## 1. 任務定義（Task）

| 項目 | 內容 |
|---|---|
| **Input** | 多視角同步 scene image（×V），場景為多人 |
| **額外 input** | 指定目標人的 per-view mask：每個 view 一張，**V 張都框同一個目標人**（從多人場景中指認要抽哪一個） |
| **相機** | 已知 K, R, t（用於 loss 投影；同時 camera head 也會 decode） |
| **Output** | 該目標人一組 world-frame SMPL-X（global orient, body pose, betas, expression, translation），以及 per-view camera |
| **架構基底** | VGGT alternating-attention encoder（feed-forward、permutation-equivariant、variable-V） |

---

## 2. 定位：相對 MAMMA 與 VGGT-S（Why this is novel）

### 2.1 MAMMA 不是 feed-forward 系統

MAMMA（CVPR 2026）其實是三段式管線：

1. **Feed-forward** 預測 per-view dense landmark（512 點 + visibility + uncertainty + contact）
2. **離線 optimization**（L-BFGS）把 SMPL-X fit 到所有 view 的 landmark
3. **幾何後處理** 做跨視角關聯（epipolar + Hungarian + cycle graph）

它的網路**從頭到尾沒有做跨視角特徵融合**——融合只發生在最後的 fitting。論文第 8 頁親口承認：

> *"We do not compare with pure feed-forward approaches because they need to be retrained for each multi-view configuration."*

這正是本研究的 gap：**做出那個「不需為每組相機配置重訓」的 feed-forward 系統**，把 L-BFGS fitting 換成 reference-query 前饋。

### 2.2 從 MAMMA 吸收 vs 取代

| 從 MAMMA 吸收（前饋零件） | 取代掉（離線零件） |
|---|---|
| Per-landmark learnable query（512 點） | L-BFGS optimization fitting → **改為 reference-query 前饋** |
| Mask 條件化（conv(mask) + element-wise add） | — |
| Visibility + uncertainty 輸出（occlusion 命脈） | — |
| 跨視角關聯（epipolar + Hungarian，MAMMA 實測 100%） | **本系統移除**：V 張 mask 同指一個目標人 → 不需關聯（見 §4.9） |

### 2.3 VGGT-S 提供的三個依據

VGGT-S（geometry-enhanced cross-view segmentation）驗證了三件事：

1. **Mask 在 VGGT encoder 之後注入是對的**：它的 Mask Prompt Fusion 就是 `F' = F + conv(mask)`，與 MAMMA 獨立地用了同一招 → 引這兩篇即可，不需要其他「VGGT + mask」先例。
2. **凍結 backbone 可行**（但我們選擇不凍結，見 §4.1）。
3. **VGGT 的 point projection 會 drift，但 feature-level alignment 可靠** → **不要用 VGGT 的 point/track head 定位 landmark**，要在 feature 層級用 learnable query 解碼（與 MAMMA 同源）。

> :::info
> **本研究的一句話 pitch**：MAMMA 證明 dense landmark + 幾何夠強，但它停在 per-view + 離線 optimization。我把跨視角融合與 SMPL-X 回歸做成單一前饋網路，因此 view-count 無關、免重訓、無需 fitting。
> :::

---

## 3. 整體資料流（Pipeline）

```
多人 scene images (×V) + 指定目標人的 per-view mask (×V，都框同一人)
        │
        ▼
VGGT encoder (trainable)
        │
   ┌────┴───────────────┐
   ▼                     ▼
Camera head        Local mask add ⊕  ← 目標人 per-view mask (conv)
(讀乾淨特徵,         (no cross-view attn)
 per-view,                │
 mask-free)               ▼
   │              Mask-fused features (已鎖定目標人 · 共享地基)
   │                      │
   │            ┌─────────┴──────────┐
   │            ▼                    ▼
   │   Per-view SMPL tokens   Dense aux head
   │   (目標人 · × V)          (512 queries · vis · unc)
   │            │                    │
   │   + camera pose embedding       │ aux 2D reproj loss
   │            │                    │ (梯度回流打地基)
   │            ▼                    │
   │   Reference-query fusion        │
   │   (1 reference token · var V)   │
   │            │                    │
   │            ▼                    │
   │   SMPL-X 參數 (目標人 1 組)       │
   │   6D orient/pose · β · expr · t │
   │            │                    │
   │            ▼                    │
   │   SMPL-X model → mesh / joints  │
   │            │                    │
   └────────────┴────────────────────┴──► Training losses
        (camera · 3D joint/vertex · 2D reproj · aux landmark/vis)
```

> :::info
> **一筆訓練 = 一個目標人**：場景多人，但 mask 只框目標人，整條管線只 decode 該人的 SMPL-X。沒有 P 維度在單筆 batch 裡——多人靠換 mask 多次 forward。
> :::

> :::info
> **關聯（association）已移除**：V 張 mask 都指同一個目標人（資料構造天然保證），per-view token 本來就是同一人的不同視角，reference token 直接 attend 這 V 個 token 即可。**不需顯式 epipolar + Hungarian 幾何模組。**
> :::

---

## 4. 各組件設計與理由（Design & Why）

### 4.1 VGGT encoder — 可訓練（不 frozen）

- **決定**：不凍結 encoder。
- **理由**：VGGT-S 能凍結是因為它的任務（cross-view segmentation）與 VGGT 預訓練目標（幾何）很接近；本研究是 SMPL-X 人體參數回歸，與預訓練目標有 gap，unfreeze 讓 backbone 往人體幾何 adapt，上限更高。
- **代價 / 風險**：顯存算力大增、需更多資料、可能洗掉 VGGT 學好的跨視角先驗（catastrophic forgetting）。
- **訓練策略**：用**差分學習率**（encoder 小 lr、heads 正常 lr），或**先凍結暖身、再解凍 fine-tune** 的兩階段。

### 4.2 Camera head — 在 mask fusion **之前**、讀乾淨特徵、per-view 唯一

- **決定**：camera head 接在 encoder 輸出後、mask fusion **之前**，吃未被 mask 污染的特徵；每個 view 一組 K, R, t，**不與 person 維度耦合**。
- **理由**：camera 是 **scene-level、person-agnostic** 的量，描述相機相對世界的位置，跟「圖裡有誰、抽哪個人」無關。把 person-specific 的 mask 訊號灌進 camera 預測，會讓同一 view、不同 person 的 mask 跑出 P 個不一致的 camera，破壞 camera 唯一性。
- **注意**：encoder 不凍結時，人體任務的梯度會透過共享 encoder 間接影響 camera head 學到的特徵；若 camera 被人體任務帶歪，用差分學習率或對 camera loss 加權。

### 4.3 Mask 注入 — per-view 本地相加，**不做 cross-view attention**

- **決定**：每個 view 各自 `F'_v = F_v + conv(M_v)`，本地相加即可。
- **理由（與 VGGT-S 的關鍵差異）**：VGGT-S 只有單一 source-view mask，需靠 cross-view attention 把 mask「傳播」到沒 mask 的 target view。**我們每個 view 都有自己的 mask，person-specific 訊號本地自足**，那個傳播 attention 解決的問題在這裡不存在。
- **attention 放哪**：相加之後的 attention 交互，發生在下游的 SMPL token / landmark query 的 decoder（MAMMA 式），不需獨立 fusion 模組。
- **選擇性強化**：若重疊嚴重、單一 view mask 切不乾淨（SAM2 在貼身時常漏切），可在 local add 後補一個 VGGT-S 式輕量 cross-view bottleneck attention 互相補洞。**第一版不加，實測掉點才加。**

### 4.4 Per-view SMPL token — 一個 (person, view) 一個 token

- **決定**：目標人的每個 view 配一個 learnable SMPL token（一筆資料共 V 個），cross-attend 該 view 的 mask-fused feature map，輸出一個 per-view 姿態 embedding。
- **理由**：每台相機看到的是該目標人局部、帶角度偏見的證據（正面看不到背、側面看不到正面），所以需要 V 個各自的猜測互補。單一 token 足以承載「一個人的一組 SMPL-X 參數」這種低維全域量；它透過 cross-attention 掃描整張 feature map，是「一個 token 但濃縮全圖證據」。
- **重要**：此時**先不要**回歸 SMPL-X 參數，停在 embedding，融合後再回歸 → world-frame 一致性由 fusion 保證。

### 4.5 Dense aux head — 與 SMPL token **並列**、共享 feature map、**非串接**

- **決定**：512 個 per-landmark query（MAMMA 機制）+ visibility + uncertainty，與 SMPL token **並列**從**同一份** mask-fused feature map 各自 cross-attend；兩者**手足關係、不互相連接**。
- **dense head 不對 SMPL token 做 fusion**（這是常見誤解）：
  - SMPL head：1 query → 1 pose embedding。
  - Dense head：512 queries → 512 個 2D 點。
- **怎麼提升精度（間接路徑）**：dense head 有自己的 2D reprojection loss，**梯度回流到共享的 feature map 與 backbone**，逼地基在 2D 幾何上更精確；SMPL token 因為讀同一份變好的地基而間接受益。這就是 "auxiliary regularizer" 的真正意思。
- **不要用 VGGT 的 point/track head**：VGGT-S 證明那條會 projection drift，要在 feature 層級用 query 解碼。

### 4.6 Camera pose embedding — 在 fusion **之前**注入每個 per-view token

- **決定**：per-view SMPL embedding 出來後、進 fusion 前，把該 view 的相機 pose 編成 embedding 加上去。
- **理由（最容易漏、漏了會壞）**：V 個 per-view token 各自是「在自己相機座標系下」的證據。要融成單一 world-frame，reference token 必須知道每個 token 來自哪個相機、朝哪個方向，才能把各座標系的猜測翻譯到同一世界座標。**少了它，fusion 不知道各 view 相對幾何，world-frame 對齊學不起來。**

### 4.7 Reference-query fusion — 核心新意（取代 L-BFGS）

- **決定**：這筆資料配一個 learnable reference token（代表目標人）；它 cross-attend 那 V 個（帶 camera embedding 的）per-view SMPL token（key/value），堆疊 2–4 個 decoder block（cross-attn + FFN），輸出單一 world-frame 結果。
- **為何 view-count 無關**：attention 本質是對一組 key/value 的加權平均，3 個或 8 個 view 都用同一組權重，不需固定數量、不需重訓。這正是 MAMMA 自承 feed-forward 做不到、而我們做得到的點。
- **person 之間的 self-attention**：在目前「一次一個目標人」的 setup 下**不存在**（一筆只有一個 reference token）。若未來要讓同場景的人互相感知（貼身互動），需改成一次餵多個目標人的 reference token 並開 self-attn——屬於 §4.10 的未來延伸。

### 4.8 回歸 head + SMPL-X model — 先 decode 參數，再出 mesh 才算 loss

- **決定**：從 fused person token 接**獨立**的小 MLP 分別 decode 出 SMPL-X **參數**：global orientation（6D）、body pose（per-joint 6D）、betas、expression、world translation。不要共用一個大 MLP 硬輸出整個向量。
- **關鍵資料流（容易講錯的地方）**：network 直接輸出的是**參數**，不是 mesh。要把這組參數餵進可微分的 **SMPL-X model**，跑出 posed mesh（vertices）與 joints，**才能計算 loss**：
  - 3D joint / vertex loss：直接比對 SMPL-X model 輸出的 joints / vertices。
  - 2D reprojection loss：把 joints / vertices 用相機投影到各 view 再比對。
- 換句話說：`token → MLP → SMPL-X 參數 → SMPL-X(參數) → mesh/joints → {3D loss, 2D 投影 loss}`。參數本身也可加 prior 正則（β、pose 的 L2 等）。

### 4.9 跨視角關聯 — 已移除（V 張 mask 同指一個目標人）

- **決定**：**不設顯式關聯模組**（原本的 epipolar + Hungarian + cycle graph 拿掉）。
- **理由**：一筆資料只鎖定一個目標人，給的 V 張 mask **本來就都框同一個人**（資料構造保證），所以 V 個 per-view token 必然是同一人的不同視角，reference token 直接 attend 即可，根本沒有「哪個 view 的誰對應哪個 view 的誰」要解。
- **前提警告（寫進 invariant）**：此簡化依賴「V 張 mask 同指一個目標人」。若未來改成一次輸入多人、且 mask 未標明跨視角對應，這條失效，須補回關聯機制。

### 4.10 多人推論 與 互動（Contact）— 目前獨立、互動為未來延伸

- **推論時的多人**：同一個多人場景，**換 mask、對每個目標人各跑一次 forward**，各自獨立得到 world-frame SMPL-X。人與人之間模型不互相感知。這跟訓練方式（一次一個目標人）一致。
- **Contact / 人際互動（超出當前 setup）**：因為一次只看一個人，模型本身無法建模人跟人的接觸或互穿。若要支援貼身互動，需改成一次輸入多個目標人的 reference token + 開 person-token self-attn（§4.7）+ 加 per-vertex contact head 與 SDF repulsion/attraction（MAMMA 的 close-interaction 主貢獻）。**列為未來延伸，第一版不做。**
- 注意 MAMMA ablation：contact 主要改善 penetration 指標，對 MPJPE/PVE 幾乎無影響——即使未來要加，優先級也不高。

---

## 5. 架構 Invariants（開發全程必須守住）

1. **輸出唯一性（每筆一個目標人）**：一筆資料只 decode 一個 world-frame SMPL-X，即 mask 鎖定的那個目標人；多人靠多次 forward。
2. **Camera 唯一性 + 乾淨輸入**：camera head 在 mask fusion 前讀乾淨特徵，per-view 一組，不被目標人 mask 污染（camera 是 scene-level、與選誰無關）。
3. **Mask 不進 AA backbone**：mask 在 encoder 之後注入，per-view 本地相加，不做 cross-view attention。
4. **SMPL token 與 dense head 並列**：兩者共享 feature map、各自 cross-attend、互不串接。
5. **Camera pose embedding 在 fusion 前注入**：每個 per-view token 必須帶 camera 幾何才能對齊 world-frame。
6. **Dense head 不依賴 VGGT point/track head**：避開 projection drift，用 feature-level query 解碼。
7. **AA permutation-equivariance / variable-V**：fusion 對 view 數量不敏感（view-count 無關的數學根源）。
8. **6D rotation 表示**：所有旋轉用 6D。
9. **座標系一致性**：dense head 的 2D 預測與 reprojection loss 在一致的座標約定下。
10. **參數 → SMPL-X model → loss**：network 輸出的是 SMPL-X 參數，必須經可微分 SMPL-X model 出 mesh/joints 才計算 3D / 2D loss（不可直接拿參數比 vertex）。
11. **V 張 mask 同指一個目標人（前提依賴）**：移除顯式關聯依賴此前提（一筆資料的 V 張 mask 都框同一個目標人）。若未來改成一次多人輸入，須補回關聯機制。

---

## 6. Loss 設計

| Loss | 對象 | 說明 |
|---|---|---|
| 3D joint / vertex | SMPL-X model 輸出的 joints / vertices | 主任務監督，world frame |
| SMPL-X 參數 prior | β / pose | L2 等正則（可選） |
| Camera | per-view K, R, t | camera head 輸出監督 |
| 2D reprojection | joints / vertices 投回各 view | **用 GT camera 投影**（較穩） |
| Aux landmark | 512 點 | Gaussian NLL（MAMMA 式） |
| Aux visibility | per-landmark | BCE |
| （可選）Contact / SDF | per-vertex | 僅貼身互動 |

> **資料流提醒**：3D 與 2D loss 都作用在「參數經 SMPL-X model 還原出的 mesh/joints」上，不是直接作用在 network 輸出的參數向量。

> **設計決定待 commit**：2D reprojection 用 GT camera（建議、穩）還是用 camera head decode 出的 camera（全 self-contained、但訓練更難）。早期建議 GT camera。

---

## 7. 要做的事（To-Do / Pre-flight Checklist）

### 7.1 最高優先 — 訓練前必做的 sanity check
- [ ] **Crop / virtual camera 推導驗證**：寫一個 standalone 檢查，確認 GT SMPL-X mesh vertices 能正確投影到各 view 影像上。**這是訓練啟動前的最高優先驗證項**，座標系錯了後面全錯。
- [ ] 驗證 camera pose embedding 確實有助 world-frame 對齊（用「有/無 embedding」做小規模 ablation）。

### 7.2 架構實作
- [ ] VGGT encoder 接上 + 確認 variable-V forward 正常。
- [ ] Camera head 接在 encoder 後、mask 前，輸出 per-view K, R, t。
- [ ] Mask CNN + per-view local add，產生 mask-fused features。
- [ ] Per-view SMPL token decoder（先 2 層）。
- [ ] Dense aux head（512 query + vis + unc），與 SMPL token 並列讀同一 feature map。
- [ ] Camera pose embedding 注入 per-view token。
- [ ] Reference-query fusion（先 2 層；目前單一 reference token，無 person-token self-attn）。
- [ ] 分開的 MLP decode SMPL-X 參數（6D orient / per-joint 6D / betas / expr / translation），再接可微分 SMPL-X model 出 mesh/joints。
- [ ] ~~幾何關聯模組~~ → 已移除（V 張 mask 同指一個目標人，§4.9）。確認資料管線確實對同一目標人輸出跨視角一致的 mask。

### 7.3 訓練配方
- [ ] 差分學習率（encoder 小 lr）或兩階段（凍結暖身 → 解凍）。
- [ ] Loss 權重策略（先讓 aux landmark/vis 有足夠權重把地基打好）。
- [ ] 資料：可沿用 BEDLAM / MammaSyn 式多視角合成資料（含 SMPL-X GT + per-view mask + visibility）。

### 7.4 待 commit 的設計選擇
- [ ] SMPL token decoder 與 fusion block 各幾層（建議都從 2 起）。
- [ ] Fusion 的 person-token self-attention 開或關（貼身互動開）。
- [ ] 2D reprojection 用 GT camera 還是 decode camera（早期 GT）。
- [ ] 是否支援人際互動 / contact（需改多人輸入，目前超出 setup，預設不做）。
- [ ] 是否補 VGGT-S 式 cross-view bottleneck（僅重疊嚴重時）。
- [ ] SMPL head 與 dense head 是否各加一層 head-specific adapter（第一版共用）。

---

## 8. 靈感來源對照

| 組件 | 來源 |
|---|---|
| 變視角 AA backbone、perm-equivariance、免重訓 | VGGT (CVPR 2025) |
| Mask 在 encoder 後 conv + add 注入 | VGGT-S；MAMMA（兩篇獨立印證） |
| 「VGGT point/track 會 drift、feature alignment 可靠」 | VGGT-S |
| 512 per-landmark query + visibility + uncertainty | MAMMA / MammaNet (CVPR 2026) |
| Reference-query 前饋融合（取代 L-BFGS） | VGGT camera token + 本研究設計 |
| Contact / SDF（可選） | MAMMA；Müller et al. |
| 6D rotation / SMPL-X | Zhou et al. (CVPR 2019) / SMPL-X (CVPR 2019) |
