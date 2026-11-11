# e-means reproduction on an RTX 4090

Generated 2026-11-12T07:29:13 by `report.py` from 44 run records and 6 crossing records.

## Hardware and software

| field | value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| torch | 2.15.0+cu132 |
| triton | 3.9.0 |
| e-means assignment backend | triton (38 runs) |
| `faiss.contrib.e_means` source hash | 7a4c8489e2410777 |

## Table 3, 1M rows: epochs of e-means to reach Flash-KMeans' error after 25 iterations

| dataset | seeds | crossed | crossing (epochs) | speedup | Flash@25 test MSE | Flash@25 s | e-means s | input | batch |
|---|---|---|---|---|---|---|---|---|---|
| deep1b-1M | 3 | 3 | 2.0 | 12.54x | 0.389308 | 1.00 | 0.08 | cast | 32768 |
| spacev-1M | 3 | 3 | 2.0 | 21.73x | 7236.31 | 1.77 | 0.08 | cast | 32768 |

## Per seed

| dataset | seed | crossing | speedup | Flash test MSE | Flash s | e-means test MSE | e-means s |
|---|---|---|---|---|---|---|---|
| deep1b-1M | 10 | 1.8 | 13.94x | 0.389308 | 1.00 | 0.389091 | 0.07 |
| deep1b-1M | 11 | 2.2 | 11.43x | 0.388473 | 1.00 | 0.388126 | 0.09 |
| deep1b-1M | 12 | 2.0 | 12.54x | 0.389311 | 1.00 | 0.38794 | 0.08 |
| spacev-1M | 10 | 2.0 | 21.73x | 7236.31 | 1.77 | 7231.92 | 0.08 |
| spacev-1M | 11 | 2.4 | 18.86x | 7229.7 | 1.76 | 7220.98 | 0.09 |
| spacev-1M | 12 | 2.0 | 21.73x | 7242.22 | 1.77 | 7227.81 | 0.08 |

## How to read this

- Test error transfers across hardware; **wall-clock does not**. The paper's timings are H100 (SM90, CuTe kernels); these are a 4090 (SM89, Triton backend), so compare the trend, never the absolute seconds.
- `k` is the paper's 16384 at 1M rows (`paper_xb.k_for`), not the 1024/4096 used in hand-rolled benchmarks.
- Timings follow the paper's protocol: a fresh process per run, an untimed warm-up, both methods in the same job on one GPU, one exact float64 scorer.
- Raw JSON under `results/` is the source of truth; this file is generated from it.
