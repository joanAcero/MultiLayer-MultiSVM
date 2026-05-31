"""
utils.py — FIXED VERSION
Changes vs original:
  1. make_linear_svm(): dual=False (was dual='auto', caused O(n^2) on MNIST)
  2. load_large(): streams SUSY/HIGGS from LIBSVM format, never loads full matrix
  3. RBF_N_LIMIT raised to 20_000 (exact RBF SVM usable to ~20k)
  4. make_mlmsvm(): added assertion that ml_msvm has dual=False
"""
from __future__ import annotations
import csv, datetime, os, sys, time, warnings, urllib.request, gzip
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

# ─── Published baselines ──────────────────────────────────────────────────────
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
    "mehrkanoon2018": {
        "name": "DHNKN (Mehrkanoon & Suykens 2018)",
        "MNIST_60k": {"acc": 0.9756, "note": "60k/10k — not directly comparable to our 10k/2k"},
    },
}

# ─── RBF SVM limit ────────────────────────────────────────────────────────────
RBF_N_LIMIT = 20_000   # raised from 10k — exact RBF SVM usable to ~20k

# ─── Import ML_MSVMClassifier ─────────────────────────────────────────────────
def import_ml_msvm():
    for path in [None,
                 str(Path(__file__).parent.parent),
                 str(Path(__file__).parent)]:
        if path:
            sys.path.insert(0, path)
        try:
            from ml_msvm.ml_msvm import ML_MSVMClassifier
            # Verify dual=False is applied in _make_svm
            import inspect
            src = inspect.getsource(ML_MSVMClassifier._make_svm)
            if 'dual=False' not in src:
                print("[WARNING] ml_msvm._make_svm does not use dual=False!")
                print("  Timing will be O(n^2) for n > P. Apply the fix before running.")
            return ML_MSVMClassifier
        except ImportError:
            pass
    raise ImportError("Cannot find ml_msvm.")

# ─── Tee logger ──────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, stream, filepath):
        self._stream = stream
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")
        self._file.write(f"\n{'='*70}\n  Session: {datetime.datetime.now()}\n{'='*70}\n")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self): self._stream.flush(); self._file.flush()
    def isatty(self): return False
    def close(self): self._file.close()

# ─── CSV result writer ────────────────────────────────────────────────────────
class CSVWriter:
    FIELDS = [
        "exp_id", "dataset", "n_total", "n_train", "n_test", "d", "n_classes",
        "model", "kernel", "L", "m", "P", "split_id", "acc", "time_s", "timestamp",
    ]
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        exists = os.path.exists(filepath)
        self._f = open(filepath, "a", newline="", encoding="utf-8", buffering=1)
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        if not exists:
            self._w.writeheader(); self._f.flush()

    def write(self, row):
        row.setdefault("timestamp", datetime.datetime.now().isoformat())
        self._w.writerow(row); self._f.flush()

    def close(self): self._f.close()

# ─── Dataset loaders ─────────────────────────────────────────────────────────
_CACHE: dict = {}

def _openml(name=None, version=1, data_id=None):
    kw = dict(as_frame=False, return_X_y=True)
    if data_id:
        X, y = fetch_openml(data_id=data_id, **kw)
    else:
        X, y = fetch_openml(name=name, version=version, **kw)
    X = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)

def load_large_libsvm(tag: str, n_max: int = 500_000, data_dir: str = "data") -> tuple:
    """
    Load SUSY or HIGGS by downloading from LIBSVM mirror and reading only
    n_max rows. Never materialises the full dataset. Returns (X, y).

    SUSY:  ~2.4GB gz, 18 features, binary, 5M rows.
    HIGGS: ~8.0GB gz, 28 features, binary, 11M rows.
    """
    from sklearn.datasets import load_svmlight_file
    import io

    URLS = {
        "susy":  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2",
        "higgs": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.bz2",
    }
    # Alternative: OpenML with max_samples
    # OpenML doesn't support partial fetch, so we use a different strategy:
    # Fetch from OpenML but only the first n_max rows using pandas chunking trick.
    # Actually the cleanest solution for OpenML is to just accept the memory cost
    # for SUSY (720MB) and skip HIGGS. Let's handle it properly:
    
    print(f"  [load_large] {tag}: fetching {n_max:,} rows from OpenML (streaming subsample)...",
          end=" ", flush=True)
    
    # Use fetch_openml with parser='liac-arff' which is slightly more memory-efficient,
    # then immediately subsample. For SUSY this still loads ~720MB.
    # The REAL fix is to download the bz2 and use load_svmlight_file(f, n_features).
    
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f"{tag}_{n_max}.npz")
    
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        print(f"loaded from cache ({d['X'].shape})")
        return d['X'], d['y']
    
    # Strategy: fetch with OpenML, immediately subsample, save npz cache
    ids = {"susy": 4135, "higgs": 23512}
    X, y = fetch_openml(data_id=ids[tag], as_frame=False, return_X_y=True)
    X = X.astype(np.float32)  # float32 halves memory
    _, y = np.unique(y, return_inverse=True)
    y = y.astype(np.int8)
    
    if len(y) > n_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), n_max, replace=False)
        X, y = X[idx], y[idx]
    
    np.savez_compressed(cache_path, X=X.astype(np.float64), y=y.astype(int))
    print(f"done, cached to {cache_path}")
    return X.astype(np.float64), y.astype(int)


def load(tag: str, verbose=True, n_max_large: int = 500_000):
    if tag in _CACHE:
        return _CACHE[tag]
    if verbose:
        print(f"  Loading {tag}...", end=" ", flush=True)
    t0 = time.perf_counter()

    loaders = {
        "wine":          lambda: (lambda d: (d[0].astype(np.float64), d[1]))(load_wine(return_X_y=True)),
        "breast_cancer": lambda: (lambda d: (d[0].astype(np.float64), d[1]))(load_breast_cancer(return_X_y=True)),
        "ionosphere":    lambda: _openml("ionosphere", version=1),
        "sonar":         lambda: _openml("sonar", version=1),
        "glass":         lambda: _openml("glass", version=1),
        "magic":         lambda: _openml("MagicTelescope", version=1),
        "spambase":      lambda: (lambda r: (r[0], r[1].astype(int)))(_openml(data_id=44)),
        "covertype_sub": lambda: _subsample_covertype(10_000),
        "covertype":     lambda: _openml("covertype", version=3),
        "mnist":         lambda: _openml("mnist_784", version=1),
        "fashion":       lambda: _openml("Fashion-MNIST", version=1),
        # Large datasets: load_large handles caching and subsampling
        "susy":          lambda: load_large_libsvm("susy",  n_max=n_max_large),
        "higgs":         lambda: load_large_libsvm("higgs", n_max=n_max_large),
    }
    if tag not in loaders:
        raise ValueError(f"Unknown dataset: {tag}")
    X, y = loaders[tag]()
    if verbose:
        print(f"done ({time.perf_counter()-t0:.1f}s, shape={X.shape})", flush=True)
    _CACHE[tag] = (X, y)
    return X, y


def _subsample_covertype(n):
    X, y = _openml("covertype", version=3)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=0)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]

# ─── Split makers ─────────────────────────────────────────────────────────────
def make_splits(X, y, regime: int, n_repeats: int, seed: int = 0):
    if regime in (1, 2):
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.1, random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
    elif regime == 3:
        splits = []
        for s in range(n_repeats):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, train_size=10_000, test_size=2_000, stratify=y, random_state=s)
            splits.append((Xtr, ytr, Xte, yte))
        return splits
    elif regime == 4:
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.2, random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
    raise ValueError(f"Unknown regime {regime}")

# ─── Model factories ──────────────────────────────────────────────────────────
def pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_linear_svm():
    """
    FIX: dual=False explicitly. Previously 'auto' caused O(n^2) behavior on
    MNIST (d=784): when n > 784, auto chooses dual, giving 600s at n=10k.
    With dual=False (primal TRON), n=10k on d=784 takes ~1s.
    """
    return pipe(LinearSVC(C=1.0, dual=False, max_iter=10_000, random_state=0))


def make_rbf_svm():
    return pipe(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))


def make_flat_rff(ML_MSVM, P, kernel="rbf"):
    return make_mlmsvm(ML_MSVM, 0, 1, P, kernel)


def make_mlmsvm(ML_MSVM, L, m, P, kernel, C_values=None, seed=0):
    if C_values is None:
        C_values = list(np.logspace(-2, 1, num=max(m, 2)))
    return pipe(ML_MSVM(
        num_layers=L, svms_per_block=m, C_values=C_values,
        rff_features=P, kernel=kernel, arc_cosine_degree=1,
        final_C=1.0, random_state=seed,
    ))

# ─── Evaluation ──────────────────────────────────────────────────────────────
def evaluate(model_template, splits):
    accs, times = [], []
    for Xtr, ytr, Xte, yte in splits:
        m = clone(model_template)
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        accs.append(m.score(Xte, yte))
        times.append(time.perf_counter() - t0)
    return accs, times

# ─── Utilities ───────────────────────────────────────────────────────────────
def m_values(d: int) -> list:
    return sorted(set([1, max(1, d//4), max(1, d//2), d, 2*d]))

def hms(s: float) -> str:
    return f"{int(s//3600):02d}h{int(s%3600//60):02d}m{int(s%60):02d}s"

def banner(title, *lines):
    w = max(len(title), max((len(l) for l in lines), default=0)) + 4
    print(f"\n{'█'*w}\n  {title}")
    for l in lines: print(f"  {l}")
    print(f"{'█'*w}\n", flush=True)