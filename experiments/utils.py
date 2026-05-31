"""
utils.py — Shared utilities for the ML-MSVM experiment suite.

Repository structure assumed:
  TFM/
  ├── ml_msvm/
  │   └── ml_msvm.py          ← ML_MSVMClassifier lives here
  └── experiments/
      ├── utils.py             ← this file
      ├── exp1_width_analysis.py
      └── ...

All experiment scripts import from this module.
"""
from __future__ import annotations
import csv, datetime, os, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

# ── Published baselines for inline comparison ─────────────────────────────────
# Source: Acero & Belanche (2025) Table 3. Values are test ACCURACY (1 − error).
PUBLISHED = {
    "acero2025": {
        "name": "ML-SVM (Acero & Belanche 2025)",
        "Glass":         {"acc": 0.820},
        "Breast Cancer": {"acc": 0.992},
        "Ionosphere":    {"acc": 0.950},
        "Magic":         {"acc": 0.850},
        "Spambase":      {"acc": 0.850},
        "Cover Type":    {"acc": 0.790},
    },
    # Mehrkanoon 2018 (Neurocomputing) — only MNIST with 60k/10k protocol.
    # Not directly comparable to our 10k/2k protocol; noted for reference only.
    "mehrkanoon2018": {
        "name": "DHNKN (Mehrkanoon & Suykens 2018)",
        "MNIST": {"acc": 0.9756, "note": "60k train / 10k test (protocol differs from ours)"},
    },
}

# Exact RBF SVM is skipped for n_train above this threshold
RBF_N_LIMIT = 20_000


# ── Import ML_MSVMClassifier ──────────────────────────────────────────────────
def import_ml_msvm():
    """
    Import ML_MSVMClassifier from mlsvm/ml_msvm.py.
    Adds TFM/ (parent of experiments/) to sys.path so that
    'from mlsvm.ml_msvm import ML_MSVMClassifier' resolves correctly.
    """
    tfm_root = Path(__file__).parent.parent   # TFM/
    sys.path.insert(0, str(tfm_root))
    try:
        from ml_msvm.ml_msvm import ML_MSVMClassifier
    except ImportError:
        raise ImportError(
            "Cannot import mlsvm.ml_msvm.ML_MSVMClassifier.\n"
            f"Expected: {tfm_root / 'mlsvm' / 'ml_msvm.py'}\n"
            "Check that the folder structure matches: TFM/mlsvm/ml_msvm.py"
        )
    # Verify dual=False is in _make_svm (critical for performance)
    import inspect
    if "dual=False" not in inspect.getsource(ML_MSVMClassifier._make_svm):
        print("[WARNING] ml_msvm._make_svm uses dual='auto' or 'True'.")
        print("          Apply the dual=False fix before running experiments.")
    return ML_MSVMClassifier


# ── Tee: mirror stdout to a log file ─────────────────────────────────────────
class Tee:
    def __init__(self, stream, filepath: str):
        self._stream = stream
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")
        self._file.write(
            f"\n{'='*70}\n  Session: {datetime.datetime.now()}\n{'='*70}\n"
        )
        self._file.flush()

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def isatty(self):
        return False

    def close(self):
        self._file.close()


# ── CSV result writer ─────────────────────────────────────────────────────────
class CSVWriter:
    """Writes one row per (model, split) immediately after each evaluation."""
    FIELDS = [
        "exp_id", "dataset", "n_total", "n_train", "n_test", "d", "n_classes",
        "model", "kernel", "L", "m", "P", "split_id", "acc", "time_s", "timestamp",
    ]

    def __init__(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        new_file = not os.path.exists(filepath)
        self._f = open(filepath, "a", newline="", encoding="utf-8", buffering=1)
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        if new_file:
            self._w.writeheader()
            self._f.flush()

    def write(self, row: dict):
        row.setdefault("timestamp", datetime.datetime.now().isoformat())
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()


# ── Dataset loaders ───────────────────────────────────────────────────────────
_CACHE: dict = {}


def _openml(name=None, version=1, data_id=None) -> tuple:
    kw = dict(as_frame=False, return_X_y=True)
    if data_id is not None:
        X, y = fetch_openml(data_id=data_id, **kw)
    else:
        X, y = fetch_openml(name=name, version=version, **kw)
    X = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)


def load(tag: str, verbose: bool = True, data_dir: str = "data") -> tuple:
    """
    Load a dataset by tag. Results are cached in memory.
    For large datasets (SUSY, HIGGS), a .npz file is saved in data_dir
    to avoid re-downloading the full dataset on subsequent runs.
    """
    if tag in _CACHE:
        return _CACHE[tag]

    if verbose:
        print(f"  Loading {tag}...", end=" ", flush=True)
    t0 = time.perf_counter()

    if tag == "wine":
        X, y = load_wine(return_X_y=True)
        X = X.astype(np.float64)
    elif tag == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True)
        X = X.astype(np.float64)
    elif tag == "ionosphere":
        X, y = _openml("ionosphere", version=1)
    elif tag == "sonar":
        X, y = _openml("sonar", version=1)
    elif tag == "glass":
        X, y = _openml(data_id=41)        # data_id=41 is reliable for Glass
    elif tag == "magic":
        X, y = _openml("MagicTelescope", version=1)
    elif tag == "spambase":
        X, y = _openml(data_id=44)
        y = y.astype(int)
    elif tag == "covertype_sub":
        X, y = _load_covertype_sub(10_000)
    elif tag == "covertype":
        X, y = _openml("covertype", version=3)
    elif tag == "mnist":
        X, y = _openml("mnist_784", version=1)
    elif tag == "fashion":
        X, y = _openml("Fashion-MNIST", version=1)
    elif tag in ("susy", "higgs"):
        X, y = _load_large_cached(tag, data_dir)
    else:
        raise ValueError(f"Unknown dataset tag: '{tag}'")

    if verbose:
        print(f"done ({time.perf_counter() - t0:.1f}s, shape={X.shape})", flush=True)
    _CACHE[tag] = (X, y)
    return X, y


def _load_covertype_sub(n: int) -> tuple:
    X, y = _openml("covertype", version=3)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=0)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _load_large_cached(tag: str, data_dir: str, n_max: int = 500_000) -> tuple:
    """
    Load SUSY or HIGGS. Caches a 500k subsample as .npz to avoid
    re-downloading the full 5M/11M dataset. The .npz is loaded on
    subsequent runs (~1s) instead of the full download (~minutes + GBs RAM).
    """
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, f"{tag}_{n_max}.npz")

    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"].astype(np.float64), d["y"].astype(int)

    ids = {"susy": 4135, "higgs": 23512}
    print(f"\n  First-time download of {tag} (may take several minutes)...",
          flush=True)
    X, y = fetch_openml(data_id=ids[tag], as_frame=False, return_X_y=True)
    X = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    y = y.astype(int)

    # Subsample and cache immediately
    if len(y) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), n_max, replace=False)
        X, y = X[idx], y[idx]

    np.savez_compressed(cache, X=X, y=y)
    print(f"  Cached {len(y):,} rows to {cache}")
    return X, y


# ── Split strategies ──────────────────────────────────────────────────────────
def make_splits(X: np.ndarray, y: np.ndarray,
                regime: int, n_repeats: int, seed: int = 0) -> list:
    """
    Regime 1 / 2 : n_repeats × stratified 90/10 splits.
    Regime 3      : n_repeats × fixed 10 000 train / 2 000 test splits.
    Regime 4      : n_repeats × stratified 80/20 splits (large-scale datasets).
    """
    if regime in (1, 2):
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.1,
                                     random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]

    elif regime == 3:
        splits = []
        for s in range(n_repeats):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, train_size=10_000, test_size=2_000,
                stratify=y, random_state=s)
            splits.append((Xtr, ytr, Xte, yte))
        return splits

    elif regime == 4:
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.2,
                                     random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]

    raise ValueError(f"Unknown regime: {regime}")


# ── Model factories ───────────────────────────────────────────────────────────
def pipe(*steps) -> Pipeline:
    return Pipeline([("scaler", StandardScaler())] +
                    [(f"s{i}", s) for i, s in enumerate(steps)])


def make_linear_svm() -> Pipeline:
    """Linear SVM with primal solver (dual=False). Scales as O(n·d)."""
    return pipe(LinearSVC(C=1.0, dual=False, max_iter=10_000, random_state=0))


def make_rbf_svm() -> Pipeline:
    """Exact RBF kernel SVM. Only feasible for n_train <= RBF_N_LIMIT."""
    return pipe(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))


def make_nystroem_svm(P: int) -> Pipeline:
    """Nystroem approximation + LinearSVC at matched approximation size P."""
    return Pipeline([
        ("scaler",   StandardScaler()),
        ("nystroem", Nystroem(kernel="rbf", n_components=P,
                              gamma=None, random_state=0)),
        ("svm",      LinearSVC(C=1.0, dual=False, max_iter=10_000, random_state=0)),
    ])


def make_mlmsvm(ML_MSVM, L: int, m: int, P: int, kernel: str,
                C_values=None, final_C: float = 1.0, seed: int = 0) -> Pipeline:
    if C_values is None:
        C_values = list(np.logspace(-2, 1, num=max(m, 2)))
    return pipe(ML_MSVM(
        num_layers=L, svms_per_block=m, C_values=C_values,
        rff_features=P, kernel=kernel, arc_cosine_degree=1,
        final_C=final_C, random_state=seed,
    ))


def make_flat_rff(ML_MSVM, P: int, kernel: str = "rbf") -> Pipeline:
    """L=0: single RFF map + linear SVM head. Direct ablation baseline."""
    return make_mlmsvm(ML_MSVM, L=0, m=1, P=P, kernel=kernel)


# ── m-value sweep schedule ────────────────────────────────────────────────────
def ms_for(d: int) -> list[int]:
    """
    Returns a sorted list of m values to sweep for a dataset with d features.
    Always includes 1, 2, 3. High-d datasets are capped to avoid explosion
    of W when combined with K classes in OvR multiclass.
    """
    if d <= 20:
        base = [1, 2, 3, max(4, d // 4), d // 2, d, 2 * d]
    elif d <= 100:
        base = [1, 2, 3, d // 4, d // 2, d]
    else:                  # d > 100 (e.g. MNIST d=784): cap at d//8
        base = [1, 2, 3, d // 16, d // 8]
    return sorted(set(max(1, v) for v in base))


# ── Utilities ─────────────────────────────────────────────────────────────────
def hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def banner(title: str, *lines):
    width = max(len(title), max((len(l) for l in lines), default=0)) + 4
    bar = "█" * width
    print(f"\n{bar}\n  {title}")
    for l in lines:
        print(f"  {l}")
    print(f"{bar}\n", flush=True)
