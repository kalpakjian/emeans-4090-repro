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

# Idempotent: skip if already applied.
if git -C .deps/faiss apply --reverse --check "$REPO/patches/0001-xb-backend-env.patch" 2>/dev/null; then
    echo "patch already applied"
else
    git -C .deps/faiss apply "$REPO/patches/0001-xb-backend-env.patch"
    echo "patch applied"
fi

echo "setup ok: $(python -V 2>&1), torch $(python -c 'import torch; print(torch.__version__)')"
echo "next: ./.venv/bin/python preflight.py"
