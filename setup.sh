#!/usr/bin/env bash
# One-time setup (Omarchy/Arch): venv + pinned deps + faiss v1.16.0 checkout, patched so the
# official scripts do not force the H100-only CuTe backend. Safe to re-run.
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"
FAISS_TAG=v1.16.0
PYTHON=${PYTHON:-python3}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: install the NVIDIA driver first"; exit 1; }

[ -d .venv ] || "$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install -q -U pip

# torch brings Triton on Linux; both are the versions the paper's scripts were validated with.
python -m pip install -q "torch==2.15.0" "faiss-cpu==1.16.0" "flash-kmeans==0.3.1" h5py matplotlib

if [ ! -d .deps/faiss ]; then
    mkdir -p .deps
    git clone -q --depth 1 --branch "$FAISS_TAG" https://github.com/facebookresearch/faiss.git .deps/faiss
fi

# Idempotent: a patch whose reverse applies cleanly is already in.
for p in "$REPO"/patches/*.patch; do
    if git -C .deps/faiss apply --reverse --check "$p" 2>/dev/null; then
        echo "patch already applied: $(basename "$p")"
    else
        git -C .deps/faiss apply "$p"
        echo "patch applied: $(basename "$p")"
    fi
done

# SPACEV and SimSearchNet++ time Flash-KMeans with squared norms in float32 (see paper_xb.MATCHED):
# float16 norms overflow on those two corpora. paper_xb.py refuses to run them without this copy,
# so build it here rather than five minutes into ./run.sh quick.
export EMEANS_CACHE=${EMEANS_CACHE:-$REPO/cache}
python "$REPO/.deps/faiss/benchs/e_means/paper_flash_matched.py"

echo "setup ok: $(python -V 2>&1), torch $(python -c 'import torch; print(torch.__version__)')"
echo "next: ./.venv/bin/python preflight.py"
