"""Turn the JSON the official scripts leave in results/ into RESULTS.md.

Pure stdlib, no torch: runs anywhere, including here on Windows.
"""
import collections
import datetime
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "results")
ORDER = ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M"]


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, spec="%.4g"):
    return "-" if x is None else spec % x


def cell(recs):
    """Same reduction as paper_bench.cell: upper median crossing, median speedup."""
    hits = [r for r in recs if r.get("crossing") is not None]
    crossings = sorted(r["crossing"] for r in hits)
    return dict(
        seeds=len(recs),
        crossed=len(hits),
        crossing=crossings[len(crossings) // 2] if crossings else None,
        speedup=med([r.get("speedup") for r in hits]),
        flash_mse=med([(r.get("flash") or {}).get("test_mse") for r in recs]),
        flash_seconds=med([(r.get("flash") or {}).get("seconds") for r in recs]),
        emeans_seconds=med([r["emeans"][-1]["seconds"] for r in hits]),
        input=collections.Counter(r["emeans"][-1].get("input") for r in hits).most_common(1),
        batch=med([r["emeans"][-1].get("batch") for r in hits]),
    )



def main():
    run_paths = sorted(glob.glob(os.path.join(OUT, "runs", "*", "*.json")))
    cross_paths = sorted(glob.glob(os.path.join(OUT, "crossing", "*.json")))
    if not run_paths and not cross_paths:
        print("nothing under %s yet: run ./run.sh download, then ./run.sh quick" % OUT)
        return 0

    runs = [jload(p) for p in run_paths]
    recs = [jload(p) for p in cross_paths]
    by_dataset = collections.defaultdict(list)
    for r in recs:
        by_dataset[r["dataset"]].append(r)
    names = [n for n in ORDER if n in by_dataset] + sorted(set(by_dataset) - set(ORDER))

    gpu = collections.Counter(r.get("gpu") for r in runs).most_common(1)
    backend = collections.Counter(r.get("assignment_backend") for r in runs
                                  if r.get("assignment_backend")).most_common()
    torch_v = collections.Counter(r.get("torch") for r in runs).most_common(1)
    triton_v = collections.Counter(r.get("triton") for r in runs).most_common(1)
    sha = collections.Counter(r.get("emeans_source_sha256") for r in runs).most_common(1)

    lines = [
        "# e-means reproduction on an RTX 4090",
        "",
        "Generated %s by `report.py` from %d run records and %d crossing records."
        % (datetime.datetime.now().isoformat(timespec="seconds"), len(runs), len(recs)),
        "",
        "## Hardware and software",
        "",
        "| field | value |",
        "|---|---|",
        "| GPU | %s |" % (gpu[0][0] if gpu else "?"),
        "| torch | %s |" % (torch_v[0][0] if torch_v else "?"),
        "| triton | %s |" % (triton_v[0][0] if triton_v else "?"),
        "| e-means assignment backend | %s |"
        % ", ".join("%s (%d runs)" % (k, v) for k, v in backend),
        "| `faiss.contrib.e_means` source hash | %s |" % (sha[0][0] if sha else "?"),
        "",
        "## Table 3, 1M rows: epochs of e-means to reach Flash-KMeans' error after 25 iterations",
        "",
        "| dataset | seeds | crossed | crossing (epochs) | speedup | Flash@25 test MSE "
        "| Flash@25 s | e-means s | input | batch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        c = cell(by_dataset[name])
        lines.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            name, c["seeds"], c["crossed"], fmt(c["crossing"], "%.1f"),
            fmt(c["speedup"], "%.2fx"), fmt(c["flash_mse"], "%.6g"),
            fmt(c["flash_seconds"], "%.2f"), fmt(c["emeans_seconds"], "%.2f"),
            c["input"][0][0] if c["input"] else "?", fmt(c["batch"], "%.0f")))

    missed = [(r["dataset"], r["seed"]) for r in recs if r.get("crossing") is None]
    if missed:
        lines += ["", "## Seeds that never crossed", ""]
        lines += ["- %s seed %d" % (d, s) for d, s in missed]

    if len(recs) <= 200:
        lines += ["", "## Per seed", "",
                  "| dataset | seed | crossing | speedup | Flash test MSE | Flash s "
                  "| e-means test MSE | e-means s |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in recs:
            last = r["emeans"][-1]
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                r["dataset"], r["seed"], fmt(r.get("crossing"), "%.1f"),
                fmt(r.get("speedup"), "%.2fx"), fmt((r.get("flash") or {}).get("test_mse"), "%.6g"),
                fmt((r.get("flash") or {}).get("seconds"), "%.2f"),
                fmt(last.get("test_mse"), "%.6g"), fmt(last.get("seconds"), "%.2f")))

    lines += [
        "",
        "## How to read this",
        "",
        "- Test error transfers across hardware; **wall-clock does not**. The paper's timings are "
        "H100 (SM90, CuTe kernels); these are a 4090 (SM89, Triton backend), so compare the trend, "
        "never the absolute seconds.",
        "- `k` is the paper's 16384 at 1M rows (`paper_xb.k_for`), not the 1024/4096 used in "
        "hand-rolled benchmarks.",
        "- Timings follow the paper's protocol: a fresh process per run, an untimed warm-up, both "
        "methods in the same job on one GPU, one exact float64 scorer.",
        "- Raw JSON under `results/` is the source of truth; this file is generated from it.",
    ]
    path = os.path.join(REPO, "RESULTS.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s (%d datasets, %d seeds)" % (path, len(names), len(recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
