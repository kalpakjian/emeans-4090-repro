# e-means on an RTX 4090 (Omarchy 端)

重現 FAISS 官方 e-means 論文（[arXiv:2611.09501](https://arxiv.org/abs/2611.09501)）的 **Table 3、1M 規模**，
在 4090（SM89, 24 GB）上以 **Triton backend** 執行官方重現腳本，並把結果寫回這個 repo。

## 為什麼不是照論文 README 原封不動跑

官方有 **兩個** 地方把 backend 寫死成 `assignment_backend="cute"`：
`benchs/e_means/paper_xb.py`（`run_emeans`，Table 3）與 `benchs/e_means/paper_bench.py`
（`fig3_emeans`，Figure 3）。而 CuTe kernel 只針對 **SM90（H100/H200）**，
4090 是 SM89，會在第一次 assignment 就失敗（本機實測：`CuTe assignment requires
quack-kernels==0.6.5 ... (optional CuTe stack is unavailable)`）。
`patches/0001-xb-backend-env.patch` 與 `patches/0002-fig3-backend-env.patch` 各修一個，改成讀
`XB_BACKEND`（預設 `auto`）：在 H100 上仍解析成 CuTe，在 4090 上解析成 Triton。其餘計時協定完全沒動
（每次 run 獨立 process、未計時 warm-up、同一 job 同一張卡、同一個 float64 scorer）。
`benchs/e_means/demo_deep10m.py` 裡的兩處 `cute` 沒有動：那是示範腳本，重現流程不會執行它。

## 在 Omarchy 上跑（四步）

```bash
git clone https://github.com/kalpakjian/emeans-4090-repro.git
cd emeans-4090-repro
./setup.sh                       # venv + torch 2.15.0 + faiss-cpu 1.16.0 + flash-kmeans 0.3.1
                                 # + faiss v1.16.0 checkout 並套用 patch
                                 # + cache/flash_matched（SPACEV/SimSearchNet++ 的 float32-norms 版）
./run.sh download                # 1M 資料 4.5 GB，含 fingerprint 校驗
.venv/bin/python preflight.py    # 全部 ok 才繼續（十秒內就會告訴你哪裡壞）
./run.sh quick                   # DEEP* 1M + SPACEV 1M，seeds 10/11/12
```

`preflight.py` 會實際在 GPU 上跑一次 e-means，並印出**解析到的 backend**
（4090 上應該是 `triton`；若看到 `cute` 就是環境有問題）。它也會載入四個 1M 資料集、
檢查 flash-kmeans 版本必須是 0.3.1、faiss 必須是 1.16.0、磁碟至少 30 GB。

接著：

```bash
./run.sh full        # Table 3 全部四組資料 × 10 seeds（1M 實測約 15 分鐘，見下表）
./run.sh figure3-1m  # 可選：Figure 3 的兩個 1M 面板（e-means 之後換 Faiss CPU k-means）
./run.sh download10m # 可選：補到 21.7 GB，10M 的前綴檔
./run.sh full10m     # 可選：Table 3 at 10M 的 DEEP*/SPACEV/SimSearchNet++
./run.sh figure4     # 可選：Figure 4（每個資料集 12×12 組，很重，見下表）
./run.sh test        # 可選：report.py 的單元測試（純標準庫，不需要 GPU，幾毫秒）
./run.sh push        # 重算 RESULTS.md，commit 並推回 repo
```

`full10m` **刻意不含 `dino10b-10M-div4096`**：10M × 1024 維的 bf16 資料本身就要 20.5 GB，
24 GB 的 4090 在 assignment 前就會 OOM。要親自確認就 `bench table3 --dataset dino10b-10M-div4096`，
失敗是預期結果。`figure3`（不帶 `-1m`）會連 10M 的 SPACEV/SIFT 面板一起畫，需要先 `download10m`。

`push` 需要那台機器對 repo 有寫入權限（`gh auth login` 一次，或設好 SSH key / PAT），
並先做過 `git config user.name` 與 `user.email`。不想設權限也行：`run.sh` 跑完後
`results/`、`RESULTS.md` 都已在本地 commit 好，之後在任何地方 `git push` 即可。

結果檔案：`RESULTS.md`（人看）、`results/**.json`（機器看，官方腳本原生產物）、
`results/summary.txt`（官方 `table3 --print` 的輸出，用來交叉核對 `report.py`）。

## 時間預估（4090）

| tier | 內容 | 估計 |
|---|---|---|
| setup + download | 安裝、4.5 GB 下載 | 實測 setup 約 2 分鐘、下載約 70 秒（69 MB/s） |
| quick | 2 資料集 × 3 seeds，每個 seed 1 次 Flash@25 + 最多 13 次 e-means | 實測約 2 分鐘 |
| full | 4 資料集 × 10 seeds | **實測約 15 分鐘**（不是「放過夜」） |
| figure3-1m | 2 個 1M 面板 × 1 seed | e-means 臂幾十秒；Faiss CPU 臂才是大宗（25 次 Lloyd @ k=16384） |
| figure4 | 每個資料集 12 個 Flash 迭代上限 × 12 個 epoch 上限 × seeds | 最重的一項：每個 seed 24 個新 process，10 seeds 就是 240 個；還沒實測過 |

單次 run 的 GPU 計算其實只有零點幾秒（論文在 H100 上 10M×2 epochs 是 0.52 s），
**時間幾乎都花在「每次 run 開一個新 process + 載入 torch」**，這是官方協定刻意的。
所以 1M 全表比原本估的「數小時」快一個量級：真正的成本在 Faiss CPU 的 Figure 3。

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
   實測：`dino10b-10M-div4096` 光是把 10M × 1024 維轉 bf16 就 20.5 GB，assignment 之前就 OOM，
   所以 `./run.sh full10m` 刻意只跑 DEEP*／SPACEV／SimSearchNet++ 三組。
4. **SPACEV / SimSearchNet++** 的 baseline 用的是「平方範數算在 float32」的 Flash-KMeans 副本
   （`cache/flash_matched`），因為 float16 的最大值 65,504 撐不住這兩個語料的範數（分別達 6.2e4 / 4.3e6）。
   `setup.sh` 會自動建好，`preflight` 也會檢查；若缺這個副本，`paper_xb.py` 會直接 `sys.exit`，
   而不是給你一個看起來對、其實不是論文的 baseline。
5. **`dl.fbaipublicfiles.com` 的憑證在 2026-11-11 到期**（`notAfter=Nov 11 23:59:59 2026 GMT`），
   而 10M 的 DINOv3、SIFT、SimSearchNet++ 前綴都放在那裡，所以 `download10m` 會以
   `CERTIFICATE_VERIFY_FAILED: certificate has expired` 收場。這是上游的事，不是本機時鐘。
   `patches/0003-download-insecure-tls.patch` 提供 `EMEANS_INSECURE_TLS=1` 這個**要自己開**的開關
   （預設仍是驗證），真正的完整性保證是腳本結尾的 fingerprint 校驗：
   `EMEANS_INSECURE_TLS=1 ./run.sh download10m`。等 Meta 換憑證就可以拿掉這個環境變數。

## Windows 端的預先驗證（已被取代）

這一節原本是「Omarchy 沒開機，所以先在 Windows 上做 CPU 部分的預檢」。**現在已經不需要了**：
Table 3 的 1M（4 資料集 × 10 seeds）與 10M（3 資料集 × 10 seeds）、Figure 3 的 1M 面板，
都已在 Omarchy 的 4090 上實跑，`RESULTS.md` 就是產物。

而且 Windows 做不到這裡真正重要的事：`flash-kmeans` 0.3.1 是 Triton 套件，官方沒有 Windows
wheel，少了它就沒有 Flash-KMeans baseline，也就沒有 crossing／speedup 這兩個數字。
當時在 Windows 上驗證過、現在仍然成立的三件事：

- `faiss-cpu 1.16.0` wheel **已內含** `faiss.contrib.e_means`，不需要再從 checkout 複製。
- `report.py` 是純標準庫，任何有 Python 的機器都能跑（本機已重跑多次）。
- `patches/*.patch` 對 v1.16.0 的三個檔案都能 `git apply --check` 通過。
