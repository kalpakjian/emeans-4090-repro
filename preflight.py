"""Pre-flight check: fail here, in seconds, instead of six hours into a run.

Run with the venv's python:  .venv/bin/python preflight.py
Exit code 0 only if every check passed.
"""
import importlib.metadata
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(REPO, ".deps", "faiss", "benchs", "e_means")
DATA = os.environ.get("EMEANS_DATA") or os.path.join(REPO, "data")
BACKEND = os.environ.get("XB_BACKEND", "auto")
DATASETS_1M = ["deep1b-1M", "dino10b-1M-div4096", "spacev-1M", "fb-ssnpp-1M"]

fails = []


def check(name, fn):
    try:
        detail = fn()
    except Exception as exc:  # a check failing is data, not a crash
        fails.append(name)
        print("FAIL  %-26s %s: %s" % (name, type(exc).__name__, str(exc)[:300]))
    else:
        print("ok    %-26s %s" % (name, "" if detail is None else detail))


def driver():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    return out


def gpu():
    import torch
    assert torch.cuda.is_available(), "torch sees no CUDA device"
    cap = torch.cuda.get_device_capability(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    assert cap >= (8, 0), "e-means needs SM80 or newer, this is SM%d%d" % cap
    return "%s sm_%d%d, %.0f GB, torch %s cuda %s" % (
        torch.cuda.get_device_name(0), cap[0], cap[1], gb,
        torch.__version__, torch.version.cuda)


def triton_version():
    import triton
    return triton.__version__


def faiss_pkg():
    import faiss
    from faiss.contrib.e_means import NumpyEmeans  # noqa: F401
    assert faiss.__version__ == "1.16.0", "need faiss-cpu 1.16.0, found %s" % faiss.__version__
    return "1.16.0 with contrib.e_means"


def checkout():
    """Both scripts that pick an assignment backend must honour XB_BACKEND.

    paper_xb.run_emeans drives Table 3; paper_bench.fig3_emeans drives Figure 3. 0001 patches the
    first, 0002 the second; a bare `assignment_backend="cute"` in either is an SM89 dead end.
    """
    for name, what in (("paper_xb.py", "Table 3"), ("paper_bench.py", "Figure 3")):
        path = os.path.join(BENCH, name)
        assert os.path.exists(path), "missing %s: run ./setup.sh" % path
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert 'assignment_backend="cute"' not in text, \
            "%s still hardcodes the CuTe backend (%s): run ./setup.sh" % (name, what)
        assert "XB_BACKEND" in text, \
            "%s does not read XB_BACKEND (%s): run ./setup.sh" % (name, what)
    return "patched, backend=%s" % BACKEND


def emeans_smoke():
    """One tiny GPU train: proves the selected assignment backend compiles and runs."""
    import torch
    from faiss.contrib.e_means import Emeans

    d, k, n = 64, 256, 4096
    x = torch.randn(n, d, device="cuda", dtype=torch.bfloat16)
    model = Emeans(d, k, nepochs=1, seed=0, assignment_backend=BACKEND)
    model.train(x)
    recipe = model.recipe
    used = recipe["assignment_backend"]
    if BACKEND == "auto" and used == "cute":
        raise AssertionError("resolved to cute on a non-SM90 GPU")
    return "resolved %s, batch %d, alpha0 %.4g" % (used, recipe["batch_size"], recipe["alpha0"])


def flash():
    version = importlib.metadata.version("flash-kmeans")
    assert version == "0.3.1", "the paper timed 0.3.1, found %s" % version
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid  # noqa: F401
    return "0.3.1, batch_kmeans_Euclid importable"


def flash_matched():
    """SPACEV/SimSearchNet++ time a float32-norms Flash copy; paper_xb.py exits without it."""
    cache = os.environ.get("EMEANS_CACHE") or os.path.join(REPO, "cache")
    marker = os.path.join(cache, "flash_matched", "VERSION")
    if not os.path.exists(marker):
        raise FileNotFoundError("%s missing: run ./setup.sh (paper_flash_matched.py)" % marker)
    with open(marker, encoding="utf-8") as fh:
        version = fh.read().strip()
    assert version == "0.3.1", "flash_matched built from %s, need 0.3.1: re-run ./setup.sh" % version
    return "0.3.1, float32 norms, %s" % os.path.dirname(marker)


def datasets():
    """Load every 1M dataset the way the scripts will (memory maps, no copy)."""
    os.environ["EMEANS_DATA"] = DATA
    sys.path.insert(0, BENCH)
    import paper_data as data

    seen = []
    for name in DATASETS_1M:
        train, test = data.load(name)
        seen.append("%s %dx%d/test %d" % (name, train.shape[0], train.shape[1], test.shape[0]))
    return "; ".join(seen)


def disk():
    want = 30 * 2 ** 30  # 1M download 4.5 GB, runs and caches on top
    free = shutil.disk_usage(DATA if os.path.isdir(DATA) else REPO).free
    assert free > want, "%.1f GB free under %s, need ~30 GB" % (free / 2 ** 30, DATA)
    return "%.0f GB free" % (free / 2 ** 30)


def main():
    print("repo %s\nDATA %s\n" % (REPO, DATA))
    check("nvidia driver", driver)
    check("torch cuda", gpu)
    check("triton", triton_version)
    check("faiss-cpu", faiss_pkg)
    check("faiss checkout", checkout)
    check("flash-kmeans", flash)
    check("flash float32 norms", flash_matched)
    check("e-means gpu run", emeans_smoke)
    check("1M datasets", datasets)
    check("disk", disk)
    if fails:
        print("\n%d check(s) FAILED: %s" % (len(fails), ", ".join(fails)))
        return 1
    print("\nall checks passed: ./run.sh download, then ./run.sh quick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
