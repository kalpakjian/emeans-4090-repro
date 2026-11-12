"""Turn the JSON the official scripts leave in results/ into RESULTS.md.

Pure stdlib, no torch: runs anywhere, including here on Windows.
"""
import collections
import glob
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "results")
ORDER = ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M"]
SCALES = ["1M", "10M", "100M"]        # Table 3 is per scale; one table each


def scale_of(name):
    """The scale is a hyphen-delimited token: deep1b-10M, dino10b-1M-div4096 both work.

    Longest first so 100M can never be read as 10M; unrecognised names get their own bucket
    rather than being guessed into a table.
    """
    tokens = name.split("-")
    for scale in ["100M", "10M", "1M"]:
        if scale in tokens:
            return scale
    return "?"


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



def figure3(out):
    """Figure 3 lives beside the Table 3 crossings: one record per (dataset, method, seed).

    Averaged over seeds per budget, so a missing arm shows "-" rather than silently averaging
    over fewer seeds than the other column.
    """
    paths = sorted(glob.glob(os.path.join(out, "figure3", "*.json")))
    if not paths:
        return []

    by_ds = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in paths:
        rec = jload(path)
        by_ds[rec["dataset"]][rec["seed"]][rec["method"]] = {
            p["budget"]: p for p in rec["points"]}

    lines = ["", "## Figure 3: error per budget, e-means (GPU) vs Faiss k-means (CPU)", "",
             "Both arms start from the same centroids per seed; the e-means arm draws them on the "
             "GPU and saves them for the Faiss arm, so it must run first.", ""]
    for name in sorted(by_ds):
        seeds = by_ds[name]
        budgets = sorted({b for per_method in seeds.values()
                          for points in per_method.values() for b in points})

        def avg(method, what, budget):
            vals = [per_method[method][budget][what] for per_method in seeds.values()
                    if method in per_method and budget in per_method[method]]
            return fmt(statistics.mean(vals), "%.6g") if vals else "-"

        n_em = sum(1 for per_method in seeds.values() if "emeans" in per_method)
        n_fa = sum(1 for per_method in seeds.values() if "faiss" in per_method)
        lines += ["### %s (%d e-means seeds, %d Faiss seeds)" % (name, n_em, n_fa), "",
                  "| budget | e-means train | e-means test | Faiss train | Faiss test |",
                  "|---|---|---|---|---|"]
        for b in budgets:
            lines.append("| %g | %s | %s | %s | %s |" % (
                b, avg("emeans", "train", b), avg("emeans", "test", b),
                avg("faiss", "train", b), avg("faiss", "test", b)))
        lines.append("")
    return lines


def main():
    run_paths = sorted(glob.glob(os.path.join(OUT, "runs", "*", "*.json")))
    cross_paths = sorted(glob.glob(os.path.join(OUT, "crossing", "*.json")))
    fig_paths = sorted(glob.glob(os.path.join(OUT, "figure3", "*.json")))
    if not run_paths and not cross_paths and not fig_paths:
        print("nothing under %s yet: run ./run.sh download, then ./run.sh quick" % OUT)
        return 0

    runs = [jload(p) for p in run_paths]
    recs = [jload(p) for p in cross_paths]
    by_dataset = collections.defaultdict(list)
    for r in recs:
        by_dataset[r["dataset"]].append(r)
    names = [n for n in ORDER if n in by_dataset] + sorted(set(by_dataset) - set(ORDER))
    by_scale = collections.defaultdict(list)
    for n in names:
        by_scale[scale_of(n)].append(n)

    gpu = collections.Counter(r.get("gpu") for r in runs).most_common(1)
    backend = collections.Counter(r.get("assignment_backend") for r in runs
                                  if r.get("assignment_backend")).most_common()
    torch_v = collections.Counter(r.get("torch") for r in runs).most_common(1)
    triton_v = collections.Counter(r.get("triton") for r in runs).most_common(1)
    sha = collections.Counter(r.get("emeans_source_sha256") for r in runs).most_common(1)

    lines = [
        "# e-means reproduction on an RTX 4090",
        "",
        # No wall-clock timestamp: it made every run of report.py produce a diff, so ./run.sh push
        # committed a meaningless RESULTS.md every time. Provenance of *when* is the git commit date,
        # and "shape of the data" is these counts, which do change when results change.
        "Generated by `report.py` from %d run records and %d crossing records."
        % (len(runs), len(recs)),
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
        "## Table 3: epochs of e-means to reach Flash-KMeans' error after 25 iterations",
        "",
    ]
    for scale in SCALES + ["?"]:
        if scale not in by_scale:
            continue
        lines += [
            "### %s rows" % scale if scale != "?" else "### scale not in the name",
            "",
            "| dataset | seeds | crossed | crossing (epochs) | speedup | Flash@25 test MSE "
            "| Flash@25 s | e-means s | input | batch |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in by_scale[scale]:
            c = cell(by_dataset[name])
            lines.append("| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                name, c["seeds"], c["crossed"], fmt(c["crossing"], "%.1f"),
                fmt(c["speedup"], "%.2fx"), fmt(c["flash_mse"], "%.6g"),
                fmt(c["flash_seconds"], "%.2f"), fmt(c["emeans_seconds"], "%.2f"),
                c["input"][0][0] if c["input"] else "?", fmt(c["batch"], "%.0f")))
        lines.append("")

    missed = [(r["dataset"], r["seed"]) for r in recs if r.get("crossing") is None]
    if missed:
        lines += ["", "## Seeds that never crossed", ""]
        lines += ["- %s seed %d" % (d, s) for d, s in missed]

    if 0 < len(recs) <= 200:
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

    lines += figure3(OUT)

    lines += [
        "",
        "## How to read this",
        "",
        "- Test error transfers across hardware; **wall-clock does not**. The paper's timings are "
        "H100 (SM90, CuTe kernels); these are a 4090 (SM89, Triton backend), so compare the trend, "
        "never the absolute seconds.",
        "- `k` is the paper's `paper_xb.k_for`: 16384 at 1M rows, 65536 at 10M — not the "
        "1024/4096 used in hand-rolled benchmarks.",
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
