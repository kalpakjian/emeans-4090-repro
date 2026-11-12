# 這批結果說明什麼

對應論文 [arXiv:2611.09501](https://arxiv.org/abs/2611.09501)。數字來自 `results/`，`RESULTS.md` 是自動產生的原表，
這份是**對照論文主張**的判讀。硬體：RTX 4090（SM89, 24 GB）、torch 2.15.0+cu132、triton 3.9.0、e-means backend = Triton。

## 論文自己宣稱的兩個數字

摘要原文：「beats the loss of Lloyd k-means at 25 epochs in **1.4 to 2.6 epochs**, which translates to
**10-45x GPU wall-clock speedups**」。

## 我們量到的（70 個 (dataset, seed) 全部 cross）

| dataset | seeds | crossing（中位） | crossing 範圍 | speedup（中位） |
|---|---|---|---|---|
| deep1b-1M | 10/10 | 1.8 | 1.6-2.2 | 13.9x |
| dino10b-1M-div4096 | 10/10 | 2.0 | 1.8-2.2 | 42.2x |
| spacev-1M | 10/10 | 2.2 | 2.0-2.6 | 19.7x |
| fb-ssnpp-1M | 10/10 | 1.4 | 1.4-1.4 | 23.8x |
| deep1b-10M | 10/10 | 2.2 | 1.4-2.8 | 13.8x |
| spacev-10M | 10/10 | 2.2 | 2.0-2.4 | 21.8x |
| fb-ssnpp-10M | 10/10 | 1.6 | 1.4-1.8 | 23.5x |

逐 seed 的極值：crossing **1.4–2.8** epochs、speedup **10.28–46.85×**（整體中位 21.8×）。

- 落在論文區間內：70/70 個 seed 的 speedup ≥ 10×；69/70 的 crossing ≤ 2.6。
- 擦邊超出：`deep1b-10M` seed 18 的 2.8 epochs；`dino10b-1M` 三個 seed 的 46.5–46.9×（上限 45×）。

**結論：兩個 headline 都重現，超出量在單一 seed 的抖動範圍。**

## 這些數字的正確讀法（條件，不是保證）

- 「10–45×」是 Meta FAIR **自報**、preprint，分母是 **Flash-KMeans 0.3.1 在 H100 上的端到端 wall-clock**。
  它不是「e-means 對任何 k-means 都快 10–45 倍」。同一張卡、同一批資料集，中位倍率就從 13.8× 到 42.2×
  （差 3 倍）；最高的那格（DINO 1M、d=4096、42.2×，單 seed 46.9×）不是通用值。
- 「1.4–2.6 epochs」是「追平 Lloyd@25 的 loss」所需輪數，不是固定預算，也不是「永遠 1.4–2.6」：
  我們 70 個 seed 裡有 1 個到 2.8。
- **規模與資料分布會反轉結論。** 同一個 API、同一台機器（Windows、CPU、`NumpyEmeans` vs `faiss.Kmeans`、
  都不做 subsample）：

| n | d | k | e-means | Lloyd | 誰好 |
|---|---|---|---|---|---|
| 50k | 64 | 256 | 2 epochs → loss 132.07（0.13 s） | 25 iters → 125.58（0.44 s） | **Lloyd 較好** |
| 500k | 96 | 256 | 2 epochs → **127.29**（1.36 s） | 25 iters → 128.00（5.83 s） | e-means 較好，且少 4.3× 時間 |

  第一列就是「數十萬筆、群集不多、分離良好」的反例：blob 分得很開時 Lloyd 幾乎一步到位，e-means 的 EMA 反而落後。
  這與第三方在類似設定只看到約 1.5×、loss 略差的報告一致；也因此不要把 1.4–2.6 當成任何資料都成立的常數。
- 適用範圍：大規模 embedding、高 `k`、全量掃描成本高的情境（向量檢索、去重、RAG 索引）。
- 論文是 preprint、未經同儕審查；本文所有倍率都是「作者實作 vs 作者選定的 baseline」，不是對 k-means 生態的橫向評比。
- 我們**沒有**驗證「CPU 相對 Faiss 8–15×」這條：我們量的是 GPU e-means 對上 CPU Faiss（Figure 3 那一臂），
  兩者不能換算成 CPU-vs-CPU。

### 二手來源裡有兩個數字與官方程式碼衝突

- 「DINO10B＝10 億筆、約 1.05M 群集」：官方是 **100 億筆、1024 維、k ≈ 1,670 萬**（`--n 10_000_000_000 --k $((1<<24))`），
  不是 10 億；「10 億」是 Figure 1 scaling plot 的另一個點。
- 「779 GPU-hours、跨 32 台機器」：官方重現 README 的 10B 指令是 `--launch 8`、實測 **9762.65 s**（3 epochs），
  約 **22 GPU-hours**；README 同時記論文 Table 20 是 23.2 h/machine（= 10440 s/機器）。
  用我們的 4090 數字外推也落在同一個量級（10M/2 epochs = 2.92 s，×1000 筆 × 10.7 維 × 1.25 輪 ≈ 10.9 GPU-hours，
  還沒算 IVF 省下的部分）。779 差了約 35 倍，**這兩條不要引用**。

## baseline 忠實度（決定這批數字能不能當「重現」）

用論文自己的 per-seed 基線（`paper_targets.json`，H100 上量的）對照我們的 Flash-KMeans@25：

| dataset | test MSE 相對差（中位） | 相對差（最大） | Flash s 4090 | Flash s H100 |
|---|---|---|---|---|
| deep1b-1M | 5.9e-04 | 4.3e-03 | 1.00 | 0.41 |
| dino10b-1M-div4096 | 2.4e-04 | 8.5e-04 | 23.90 | 1.70 |
| spacev-1M | 6.8e-04 | 2.4e-03 | 1.76 | 0.33 |
| fb-ssnpp-1M | 1.8e-04 | 7.8e-04 | 2.57 | 0.48 |
| deep1b-10M | 1.0e-03 | 3.3e-03 | 40.33 | 15.84 |
| spacev-10M | 7.0e-04 | 1.1e-03 | 68.62 | 11.21 |
| fb-ssnpp-10M | 4.7e-05 | 6.0e-04 | 106.57 | 21.78 |

- **不是 bit-exact，也不需要是**：Flash-KMeans 是 Triton + fp16 累加，SM89 的 tiling／累加順序與 H100 不同，
  torch 2.15+cu132 也與論文環境（2.14+cu130）不同。目標誤差因此有 0.005%–0.43% 的抖動。
  crossing 判定有 10 倍以上餘裕（1.4–2.8 對 25 iterations），所以結論對這點抖動不敏感。
- **Faiss CPU 那一臂幾乎完全可重複**（決定性運算）：Figure 3 的 Faiss@25 test MSE 與論文差
  1.4e-4（DEEP\* 1M）、1.2e-4（ImageNet-CLIP 1M）。這臂可以用來檢驗資料切分與 `k` 是否正確。
- SPACEV／SimSearchNet++ 的 baseline 用的是論文規則的 float32-平方範數副本
  （`flash_norms: float32`，`cache/flash_matched`），不是預設 float16 版；否則 6.2e4／4.3e6 的範數會溢位。

## Figure 3：不只更快，同預算下 loss 更低

3 seeds 中位數，test MSE：

| dataset | budget | e-means | Faiss CPU | e-means 低多少 |
|---|---|---|---|---|
| deep1b-1M | 1 | 0.393692 | 0.405458 | 2.90% |
| deep1b-1M | 25 | 0.385845 | 0.389158 | 0.85% |
| imagenet-clip-512-normalized | 1 | 0.216109 | 0.224004 | 3.52% |
| imagenet-clip-512-normalized | 25 | 0.210842 | 0.213139 | 1.08% |

八個預算（1/2/3/5/8/12/17/25）**每一個**都是 e-means 較低，與論文 "converges to lower losses" 一致；
差距隨預算縮小但仍不反轉。

## 沒重現的部分

| 項目 | 狀態 | 原因 |
|---|---|---|
| `dino10b-10M-div4096` | 沒跑 | 10M × 1024 維轉 bf16 就 20.5 GB，24 GB 在 assignment 前 OOM。論文這格要 H100 |
| 100M 三組 | 沒跑 | 下載 100.8 GB；論文自身在 100M 要用 20–51 GB VRAM |
| Figure 4（全部） | 沒跑 | 每 seed 24 個新 process，成本最高的一項 |
| Figure 3 的 10M 面板（SPACEV／SIFT） | 沒跑 | Faiss CPU 臂每面板每 seed 約 7 分鐘起跳；不影響 Table 3 的結論 |

## 為什麼倍率要寫「在這張卡上」

倍率 = e-means 時間 ÷ Flash 時間，而**兩個方法在 SM89 上的退化幅度不一樣**：
Flash-KMeans@25 相對 H100 慢 2.4×（DEEP\* 1M）到 **14×**（DINO 1M，d=4096），差 6 倍。

推論（未量測）：DINO 那格 42× 偏高，是因為 Flash-KMeans 的 Triton kernel 在高維、低 SM 數的
SM89 上效率掉得比 e-means 的 kernel 多。要驗證得在 H100 上量同一份 e-means 時間，這裡沒有。

## 可信度分級

| 等級 | 項目 |
|---|---|
| 高 | epochs-to-cross、e-means vs Faiss 的 loss 排序、Faiss CPU 臂的絕對誤差 |
| 中 | speedup 的絕對倍率（硬體相依，換卡就變；趨勢可信） |
| 不可用 | 絕對秒數與論文比較；`dino10b-10M`、100M 的結論 |

## 成本

1M 全表（4 資料集 × 10 seeds，共 20 + 最多 130 個 process）**實測約 15 分鐘**，不是原本估的「數小時」。
真正的成本在 Faiss CPU 的 Figure 3。單次 run 的 GPU 計算只有零點幾秒，時間幾乎全花在每個 run 開新 process。
