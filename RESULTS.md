# e-means reproduction on an RTX 4090 — not run yet

`report.py` overwrites this file as soon as there are runs under `results/`.

Expected shape once `./run.sh quick` finishes:

| dataset | seeds | crossed | crossing (epochs) | speedup | Flash@25 test MSE | Flash@25 s | e-means s |
|---|---|---|---|---|---|---|---|
| deep1b-1M | 3 | ? | ? | ? | ? | ? | ? |
| spacev-1M | 3 | ? | ? | ? | ? | ? | ? |

Reminder: on a 4090 this compares the Triton e-means backend with Flash-KMeans 0.3.1 on the same
card. Test errors are comparable with the paper; wall-clock is not (paper timings are H100 + CuTe).
