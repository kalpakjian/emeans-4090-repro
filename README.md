# e-means on an RTX 4090 (Omarchy 端)

重現 FAISS 官方 e-means 論文（[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）的 **Table 3、1M 規模**，
在 4090（SM89, 24 GB）上以 **Triton backend** 執行官方重現腳本，並把結果寫回這個 repo。

## 為什麼不是照論文 README 原封不動跑

官方 `benchs/e_means/paper_xb.py` 把 backend 寫死成 `assignment_backend="cute"`，而 CuTe kernel 只針對
**SM90（H100/H200）**。4090 是 SM89，會在第一次 assignment 就失敗。
`patches/0001-xb-backend-env.patch` 把它改成讀 `XB_BACKEND`（預設 `auto`）：
在 H100 上仍解析成 CuTe，在 4090 上解析成 Triton。其餘計時協定完全沒動
（每次 run 獨立 process、未計時 warm-up、同一 job 同一張卡、同一個 float64 scorer）。

## 在 Omarchy 上跑（四步）

```bash
git clone <this repo> && cd emeans-4090-repro
./setup.sh                       # venv + torch 2.15.0 + faiss-cpu 1.16.0 + flash-kmeans 0.3.1
                                 # + faiss v1.16.0 checkout 並套用 patch
./run.sh download                # 1M 資料 4.5 GB，含 fingerprint 校驗
.venv/bin/python preflight.py    # 全部 ok 才繼續（十秒內就會告訴你哪裡壞）
./run.sh quick                   # DEEP* 1M + SPACEV 1M，seeds 10/11/12
```

`preflight.py` 會實際在 GPU 上跑一次 e-means，並印出**解析到的 backend**
（4090 上應該是 `triton`；若看到 `cute` 就是環境有問題）。它也會載入四個 1M 資料集、
檢查 flash-kmeans 版本必須是 0.3.1、faiss 必須是 1.16.0、磁碟至少 30 GB。

接著：

```bash
./run.sh full        # Table 3 全部四組資料 × 10 seeds（放過夜）
./run.sh figure3     # 可選：Figure 3（e-means 之後換 Faiss CPU k-means）
./run.sh push        # 重算 RESULTS.md，commit 並推回 repo
```

`push` 需要那台機器對 repo 有寫入權限（`gh auth login` 一次，或設好 SSH key / PAT），
並先做過 `git config user.name` 與 `user.email`。不想設權限也行：`run.sh` 跑完後
`results/`、`RESULTS.md` 都已在本地 commit 好，之後在任何地方 `git push` 即可。

結果檔案：`RESULTS.md`（人看）、`results/**.json`（機器看，官方腳本原生產物）、
`results/summary.txt`（官方 `table3 --print` 的輸出，用來交叉核對 `report.py`）。

## 時間預估（4090）

| tier | 內容 | 估計 |
|---|---|---|
| setup + download | 安裝、4.5 GB 下載 | 3–10 分鐘（看網路） |
| quick | 2 資料集 × 3 seeds，每個 seed 1 次 Flash@25 + 最多 13 次 e-means | 約 20 分鐘 |
| full | 4 資料集 × 10 seeds | 數小時，建議過夜 |

單次 run 的 GPU 計算其實只有零點幾秒（論文在 H100 上 10M×2 epochs 是 0.52 s），
**時間幾乎都花在「每次 run 開一個新 process + 載入 torch」**，這是官方協定刻意的。

## 這些數字能說什麼、不能說什麼

- **能**：相對趨勢。e-means 需要幾個 epoch 才追上 Flash-KMeans@25、速度倍率的大致量級。
- **不能**：絕對秒數。論文是 H100 + CuTe，這裡是 4090 + Triton；README 原話是
  「timings are only comparable on an H100」。
- test error 本身可跨硬體驗證（同資料、同 k、同初始化種子）；`preflight`/報告會記錄
  `faiss.contrib.e_means` 的原始碼 hash，方便日後追。
- `k` 是論文在 1M 規模用的 **16384**（`paper_xb.k_for`），不是手寫 benchmark 常見的 1024/4096。

## 已知風險

1. **Triton backend 沒在 4090 上被 Meta 驗證過**；CuTe 路線才是論文調過的。若 preflight 的
   e-means 冒煙測試就失敗，先看 `XB_BACKEND=native` 是否能跑（慢但純 PyTorch）。
2. **Flash-KMeans 0.3.1 在 Arch/Python 3.13+ 的 wheel 是 `py3-none-any`**，靠 Triton 編譯；
   若 import 失敗，多半是 Triton 與 torch 版本不匹配，別自己升級其中一個。
3. 4090 只有 24 GB：1M 全部資料集沒問題，**10M 的 DINOv3/SimSearchNet++ 可能吃不下**
   （論文在 100M 會用到 51 GB VRAM）。要放大到 10M 前先看 `preflight` 的記憶體估算。

## Windows 端的預先驗證（已完成）

無法在沒開機的 Omarchy 上跑 GPU 部分，但以下已在 Windows 的 Python 3.14 + faiss-cpu 1.16.0 上驗證：

- `faiss-cpu 1.16.0` wheel **已內含** `faiss.contrib.e_means`，不需要再從 checkout 複製。
- `report.py` 以合成 JSON fixture 驗證過表格/中位數/crossing 邏輯，輸出格式見 `RESULTS.md`。
- `patches/0001-xb-backend-env.patch` 以 `git apply --check` 對 v1.16.0 的 `paper_xb.py` 驗證通過。
- CPU 上的 e-means 趨勢（500k×96、k=256、無 subsample）：e-means 2 epochs 1.36 s / loss 127.29
  vs Lloyd 25 iters 5.83 s / loss 128.00。
