"""Simplicity measurements for the five frameworks.

Everything here is measured, nothing is typed in:

- packages, install_mb: the framework's full dependency closure as installed
  in the current environment (read from package metadata), and the size of
  those packages on disk. fastai, TabNet and Lightning all pull in PyTorch,
  so PyTorch counts toward each of them. GPU-only runtime packages (nvidia-*,
  cuda-*, triton) are left out, because the PyTorch build here is the CUDA one
  and TensorFlow is the CPU one; without them every framework is measured as
  a CPU install.
- import_seconds: median time of a cold `import` in a fresh Python process
  (5 runs). This is what you wait for before any code runs.
- fraud/german train seconds: mean training time per model, read from each
  framework's result CSV. All runs were done on the same 2-core CPU machine.

Lines of code are not reported: each person wrote their training code in a
different style (script vs notebook, logging, plots), so a line count would
measure the author more than the framework.

Writes results/person_c/simplicity.csv
"""
import re
import statistics
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "person_c"

FRAMEWORKS = {
    # name: (pip distribution, import statement)
    "XGBoost": ("xgboost", "import xgboost"),
    "fastai": ("fastai", "from fastai.tabular.all import tabular_learner"),
    "Keras": ("tensorflow-cpu", "from tensorflow import keras"),
    "TabNet": ("pytorch-tabnet", "from pytorch_tabnet.tab_model import TabNetClassifier"),
    "Lightning": ("lightning", "import lightning"),
}


GPU_ONLY = ("nvidia-", "cuda-", "triton")


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed(name):
    try:
        metadata.distribution(name)
        return True
    except metadata.PackageNotFoundError:
        return False


def dependency_closure(root):
    """Installed distributions reachable from root through required
    (non-extra) dependencies."""
    seen, stack = set(), [_norm(root)]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        if name.startswith(GPU_ONLY):
            continue
        seen.add(name)
        for req in dist.requires or []:
            if "extra ==" in req or "extra==" in req:
                continue
            dep = re.split(r"[ ;<>=!~\[(]", req.strip(), maxsplit=1)[0]
            if dep:
                stack.append(_norm(dep))
    return seen


def installed_mb(names):
    total = 0
    for name in names:
        dist = metadata.distribution(name)
        for f in dist.files or []:
            p = Path(dist.locate_file(f))
            if p.is_file():
                total += p.stat().st_size
    return total / 1e6


def import_seconds(stmt, runs=5):
    code = f"import time; t=time.perf_counter(); {stmt}; print(time.perf_counter()-t)"
    times = []
    for _ in range(runs):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": ""})
        times.append(float(r.stdout.strip().splitlines()[-1]))
    return statistics.median(times)


def train_times():
    a_f = pd.read_csv(RES / "person_a" / "fraud_runs.csv")
    a_f = a_f[(a_f.split == "temporal") & (a_f.features == "base")]
    a_g = pd.read_csv(RES / "person_a" / "german_runs.csv")
    b = RES / "person_b"
    c = RES / "person_c"
    tab_g = pd.read_csv(b / "tabnet_german_25fold_metrics.csv")
    return {
        "XGBoost": (a_f[a_f.model == "xgboost"].train_seconds.mean(),
                    a_g[a_g.model == "xgboost"].train_seconds.mean()),
        "fastai": (a_f[a_f.model == "fastai"].train_seconds.mean(),
                   a_g[a_g.model == "fastai"].train_seconds.mean()),
        "Keras": (pd.read_csv(b / "keras_fraud_runs.csv").train_seconds.mean(),
                  pd.read_csv(b / "keras_german_25fold_metrics.csv").train_seconds.mean()),
        "TabNet": (pd.read_csv(b / "tabnet_fraud_runs.csv").train_seconds.mean(),
                   tab_g.train_seconds.mean()),
        "Lightning": (pd.read_csv(c / "lightning_fraud_runs.csv").train_seconds.mean(),
                      pd.read_csv(c / "lightning_german_runs.csv").train_seconds.mean()),
    }


def main():
    times = train_times()
    rows = []
    for name, (dist, stmt) in FRAMEWORKS.items():
        if name == "Keras" and not _installed(dist):
            dist = "tensorflow"  # Colab and most installs use the full package
        closure = dependency_closure(dist)
        rows.append(dict(
            framework=name,
            pip_package=dist,
            packages=len(closure),
            install_mb=round(installed_mb(closure), 1),
            import_seconds=round(import_seconds(stmt), 2),
            fraud_train_seconds=round(times[name][0], 2),
            german_train_seconds=round(times[name][1], 2),
        ))
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "simplicity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
