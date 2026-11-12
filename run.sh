#!/usr/bin/env bash
# Tiers, cheapest first. Every tier appends JSON to results/ and rewrites RESULTS.md.
#   ./run.sh download    4.5 GB of the 1M data (DNS: run once)
#   ./run.sh download10m 21.7 GB in total: adds the 10M prefixes (~17 GB more)
#   ./run.sh quick       DEEP* 1M + SPACEV 1M, 3 seeds   (~20 min on a 4090)
#   ./run.sh full        Table 3 at 1M: 4 datasets, 10 seeds (about 15 min, not overnight)
#   ./run.sh full10m     Table 3 at 10M: DEEP*/SPACEV/SimSearchNet++ (dino10b needs >24 GB)
#   ./run.sh figure3     Figure 3: e-means then Faiss CPU k-means
#   ./run.sh figure3-1m  the two 1M Figure 3 panels only, no 10M data needed
#   ./run.sh figure4     Figure 4: 12 Flash iteration budgets x 12 e-means epoch budgets per seed
#                        per dataset (heavy: ~240 fresh processes per dataset at 10 seeds)
#   ./run.sh figure4-plot draw results/figure4.pdf from whatever figure4 runs exist
#   ./run.sh test        unit tests for report.py (stdlib only, no GPU, seconds)
#   ./run.sh report      regenerate RESULTS.md from whatever is in results/
#   ./run.sh push        report + commit + push
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"
. .venv/bin/activate
export EMEANS_DATA=${EMEANS_DATA:-$REPO/data}
export EMEANS_CACHE=${EMEANS_CACHE:-$REPO/cache}
export XB_BACKEND=${XB_BACKEND:-auto}
OUT=$REPO/results
BENCH=$REPO/.deps/faiss/benchs/e_means

bench() { ( cd "$BENCH" && python paper_bench.py "$@" --out "$OUT" ); }

case "${1:-}" in
    download) ( cd "$BENCH" && python paper_download.py --results 1M --data "$EMEANS_DATA" --yes ) ;;
    download10m) ( cd "$BENCH" && python paper_download.py --results 10M --data "$EMEANS_DATA" --yes ) ;;
    quick)    for d in deep1b-1M spacev-1M; do bench table3 --dataset "$d" --seeds 10 11 12; done ;;
    full)     bench table3 --scale 1M ;;
    # 10M on a 24 GB card: dino10b-10M is 10M x 1024 bf16 rows = 20.5 GB before any workspace, so
    # it is not in this tier. Run it by hand to see the OOM:
    #   bench table3 --dataset dino10b-10M-div4096 --seeds 10
    full10m)  for d in deep1b-10M spacev-10M fb-ssnpp-10M; do bench table3 --dataset "$d"; done ;;
    figure3)  bench figure3 --method emeans; bench figure3 --method faiss ;;
    figure3-1m) bench figure3 --dataset deep1b-1M --method emeans
                bench figure3 --dataset deep1b-1M --method faiss
                bench figure3 --dataset imagenet-clip-512-normalized --method emeans
                bench figure3 --dataset imagenet-clip-512-normalized --method faiss ;;
    figure4)  bench figure4 --scale 10M ;;
    figure4-plot)
        # Figure 4's iteration sweep is the only thing that writes flash_i100; Table 3 stops at 25.
        # Without this check plot_figure4 asks matplotlib for 0 rows and dies in a traceback.
        if ! ls "$OUT"/runs/*/flash_i100_s*.json >/dev/null 2>&1; then
            echo "figure4-plot: no Figure 4 runs under $OUT yet (no flash_i100 records):" \
                 "run ./run.sh figure4 first" >&2
            exit 2
        fi
        ( cd "$BENCH" && python paper_bench.py figure4 --plot --out "$OUT" ) ;;
    test)     python -m unittest discover -s tests -v || exit 1; exit 0 ;;
    report)   python "$REPO/report.py"; exit 0 ;;
    push)
        python "$REPO/report.py"
        git -C "$REPO" add -A
        git -C "$REPO" diff --cached --quiet || git -C "$REPO" commit -q \
            -m "results: $(date -Is) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        git -C "$REPO" push
        exit 0 ;;
    *)  sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 2 ;;
esac

# The official table view, as a second opinion on report.py's numbers.
( cd "$BENCH" && python paper_bench.py table3 --print --out "$OUT" ) 2>/dev/null | tee "$OUT/summary.txt" || true
python "$REPO/report.py"
