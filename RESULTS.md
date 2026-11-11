# e-means reproduction on an RTX 4090

Generated 2026-11-12T07:47:03 by `report.py` from 250 run records and 40 crossing records.

## Hardware and software

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| torch | 2.15.0+cu132 |
| triton | 3.9.0 |
| e-means assignment backend | triton (210 runs) |
| `faiss.contrib.e_means` source hash | 7a4c8489e2410777 |

## Table 3, 1M rows: epochs of e-means to reach Flash-KMeans' error after 25 iterations

| dataset | seeds | crossed | crossing (epochs) | speedup | Flash@25 test MSE | Flash@25 s | e-means s | input | batch |
|---|---|---|---|---|---|---|---|---|---|
| deep1b-1M | 10 | 10 | 1.8 | 13.94x | 0.389309 | 1.00 | 0.07 | cast | 32768 |
| dino10b-1M-div4096 | 10 | 10 | 2.0 | 42.19x | 0.0122802 | 23.90 | 0.57 | cast | 32768 |
| spacev-1M | 10 | 10 | 2.2 | 19.74x | 7231.73 | 1.76 | 0.09 | cast | 32768 |
| fb-ssnpp-1M | 10 | 10 | 1.4 | 23.78x | 113564 | 2.57 | 0.11 | cast | 32768 |

## Per seed

| dataset | seed | crossing | speedup | Flash test MSE | Flash s | e-means test MSE | e-means s |
|---|---|---|---|---|---|---|---|
| deep1b-1M | 10 | 1.8 | 13.94x | 0.389308 | 1.00 | 0.389091 | 0.07 |
| deep1b-1M | 11 | 2.2 | 11.43x | 0.388473 | 1.00 | 0.388126 | 0.09 |
| deep1b-1M | 12 | 2.0 | 12.54x | 0.389311 | 1.00 | 0.38794 | 0.08 |
| deep1b-1M | 13 | 1.6 | 15.54x | 0.390261 | 1.00 | 0.389636 | 0.06 |
| deep1b-1M | 14 | 1.8 | 13.94x | 0.389706 | 1.00 | 0.389387 | 0.07 |
| deep1b-1M | 15 | 1.8 | 13.94x | 0.389294 | 1.00 | 0.38918 | 0.07 |
| deep1b-1M | 16 | 1.8 | 13.88x | 0.389275 | 1.00 | 0.389242 | 0.07 |
| deep1b-1M | 17 | 1.6 | 15.60x | 0.390116 | 1.00 | 0.389264 | 0.06 |
| deep1b-1M | 18 | 1.8 | 13.97x | 0.388775 | 1.00 | 0.388742 | 0.07 |
| deep1b-1M | 19 | 1.8 | 13.91x | 0.38978 | 1.00 | 0.389744 | 0.07 |
| dino10b-1M-div4096 | 10 | 2.0 | 42.05x | 0.012277 | 23.88 | 0.0122692 | 0.57 |
| dino10b-1M-div4096 | 11 | 2.0 | 42.25x | 0.0122822 | 24.01 | 0.0122767 | 0.57 |
| dino10b-1M-div4096 | 12 | 2.0 | 42.14x | 0.0122781 | 23.95 | 0.0122725 | 0.57 |
| dino10b-1M-div4096 | 13 | 2.0 | 42.39x | 0.0122735 | 24.07 | 0.0122729 | 0.57 |
| dino10b-1M-div4096 | 14 | 1.8 | 46.85x | 0.0122883 | 23.90 | 0.0122836 | 0.51 |
| dino10b-1M-div4096 | 15 | 2.0 | 42.13x | 0.0122752 | 23.96 | 0.0122661 | 0.57 |
| dino10b-1M-div4096 | 16 | 1.8 | 46.52x | 0.0122858 | 23.76 | 0.0122833 | 0.51 |
| dino10b-1M-div4096 | 17 | 1.8 | 46.81x | 0.0122854 | 23.90 | 0.0122849 | 0.51 |
| dino10b-1M-div4096 | 18 | 2.2 | 38.13x | 0.0122737 | 23.82 | 0.0122703 | 0.62 |
| dino10b-1M-div4096 | 19 | 2.0 | 41.99x | 0.012287 | 23.67 | 0.0122717 | 0.56 |
| fb-ssnpp-1M | 10 | 1.4 | 23.78x | 113618 | 2.57 | 113483 | 0.11 |
| fb-ssnpp-1M | 11 | 1.4 | 23.82x | 113563 | 2.57 | 113448 | 0.11 |
| fb-ssnpp-1M | 12 | 1.4 | 23.79x | 113564 | 2.57 | 113449 | 0.11 |
| fb-ssnpp-1M | 13 | 1.4 | 23.81x | 113577 | 2.57 | 113422 | 0.11 |
| fb-ssnpp-1M | 14 | 1.4 | 23.79x | 113582 | 2.57 | 113445 | 0.11 |
| fb-ssnpp-1M | 15 | 1.4 | 23.72x | 113519 | 2.57 | 113421 | 0.11 |
| fb-ssnpp-1M | 16 | 1.4 | 23.73x | 113492 | 2.57 | 113446 | 0.11 |
| fb-ssnpp-1M | 17 | 1.4 | 23.75x | 113577 | 2.57 | 113431 | 0.11 |
| fb-ssnpp-1M | 18 | 1.4 | 23.76x | 113564 | 2.57 | 113464 | 0.11 |
| fb-ssnpp-1M | 19 | 1.4 | 23.78x | 113550 | 2.57 | 113411 | 0.11 |
| spacev-1M | 10 | 2.0 | 21.73x | 7236.31 | 1.77 | 7231.92 | 0.08 |
| spacev-1M | 11 | 2.4 | 18.86x | 7229.7 | 1.76 | 7220.98 | 0.09 |
| spacev-1M | 12 | 2.0 | 21.73x | 7242.22 | 1.77 | 7227.81 | 0.08 |
| spacev-1M | 13 | 2.4 | 18.96x | 7223.22 | 1.76 | 7216.95 | 0.09 |
| spacev-1M | 14 | 2.2 | 19.81x | 7233.97 | 1.76 | 7230.22 | 0.09 |
| spacev-1M | 15 | 2.0 | 21.82x | 7230.52 | 1.76 | 7229.8 | 0.08 |
| spacev-1M | 16 | 2.4 | 18.91x | 7232.94 | 1.76 | 7212.99 | 0.09 |
| spacev-1M | 17 | 2.6 | 17.45x | 7227.09 | 1.76 | 7219.4 | 0.10 |
| spacev-1M | 18 | 2.0 | 21.62x | 7241.13 | 1.76 | 7232.01 | 0.08 |
| spacev-1M | 19 | 2.2 | 19.67x | 7229.71 | 1.76 | 7222.18 | 0.09 |

## How to read this

- Test error transfers across hardware; **wall-clock does not**. The paper's timings are H100 (SM90, CuTe kernels); these are a 4090 (SM89, Triton backend), so compare the trend, never the absolute seconds.
- `k` is the paper's 16384 at 1M rows (`paper_xb.k_for`), not the 1024/4096 used in hand-rolled benchmarks.
- Timings follow the paper's protocol: a fresh process per run, an untimed warm-up, both methods in the same job on one GPU, one exact float64 scorer.
- Raw JSON under `results/` is the source of truth; this file is generated from it.
