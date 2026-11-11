#!/usr/bin/env bash
# Tiers, cheapest first. Every tier appends JSON to results/ and rewrites RESULTS.md.
#   ./run.sh download   4.5 GB of the 1M data (DNS: run once)
#   ./run.sh quick      DEEP* 1M + SPACEV 1M, 3 seeds   (~20 min on a 4090)
#   ./run.sh full       Table 3 at 1M: 4 datasets, 10 seeds (several hours, run overnight)
#   ./run.sh figure3    Figure 3: e-means then Faiss CPU k-means
#   ./run.sh report     regenerate RESULTS.md from whatever is in results/
#   ./run.sh push       report + commit + push
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
    quick)    for d in deep1b-1M spacev-1M; do bench table3 --dataset "$d" --seeds 10 11 12; done ;;
    full)     bench table3 --scale 1M ;;
    figure3)  bench figure3 --method emeans; bench figure3 --method faiss ;;
    report)   python "$REPO/report.py"; exit 0 ;;
    push)
        python "$REPO/report.py"
        git -C "$REPO" add -A
        git -C "$REPO" diff --cached --quiet || git -C "$REPO" commit -q \
            -m "results: $(date -Is) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        git -C "$REPO" push
        exit 0 ;;
    *)  sed -n '2,11p' "$0"; exit 2 ;;
esac

# The official table view, as a second opinion on report.py's numbers.
( cd "$BENCH" && python paper_bench.py table3 --print --out "$OUT" ) 2>/dev/null | tee "$OUT/summary.txt" || true
python "$REPO/report.py"
