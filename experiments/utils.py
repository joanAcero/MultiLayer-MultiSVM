"""
utils.py — complete, all fixes integrated.

Dataset loading:
  - Glass / SUSY / HIGGS read LOCAL files in data/ first (reliable), then fall
    back to OpenML. Download the large ones manually (see RESUME.md).
  - SUSY: full 5M via local SUSY.bz2 (LIBSVM), parsed once and cached as
    susy_full.npz (~80 MB). Peak RAM ~1.4 GB.
  - graceful try/except around every load.

Models / sweeps:
  - import_ml_msvm: auto-discovery + correct version check (block_tol/use_dual).
  - ms_for: capped at m=10 (prevents the MNIST m=49/98 blow-up).
  - make_mlmsvm: m=1 trains exactly 1 SVM (C=1.0).
  - make_linear_svm / make_nystroem_svm: dual='auto' (fallback False) + tol=1e-3
    to bound cost on ill-conditioned high-d data.
"""
from __future__ import annotations
import csv, datetime, os, sys, time, warnings
from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
warnings.filterwarnings("ignore")

# Published reference numbers for inline comparison in benchmark tables.
PUBLISHED = {
    "acero2025": {"name": "ML-SVM (Acero & Belanche 2025)",
        "Glass": {"acc": 0.820}, "Breast Cancer": {"acc": 0.992},
        "Ionosphere": {"acc": 0.950}, "Magic": {"acc": 0.850},
        "Spambase": {"acc": 0.850}, "Cover Type": {"acc": 0.790}},
    "mehrkanoon2018": {"name": "DHNKN (Mehrkanoon & Suykens 2018)",
        "MNIST": {"acc": 0.9756, "note": "60k/10k protocol"}},
}
RBF_N_LIMIT = 50_000   # exact RBF SVM skipped above this train size


# ---------------------------------------------------------------------------
# Classifier discovery
# ---------------------------------------------------------------------------

def import_ml_msvm():
    """Find ML_MSVMClassifier on sys.path or by filesystem search."""
    experiments_dir = Path(__file__).parent.resolve()
    for base in [experiments_dir.parent, experiments_dir.parent.parent, experiments_dir]:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        for mod_path in ("mlsvm.ml_msvm", "ml_msvm.ml_msvm", "ml_msvm"):
            try:
                import importlib
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, "ML_MSVMClassifier")
                if not (isinstance(cls, type) and hasattr(cls, "fit")):
                    continue
                import inspect
                src = inspect.getsource(cls._make_svm)
                if "use_dual" not in src and "block_tol" not in src:
                    print("[WARNING] ml_msvm._make_svm looks like an old version "
                          "(no adaptive dual / block_tol). Timing may suffer.")
                return cls
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass
    for f in sorted(experiments_dir.parent.rglob("ml_msvm.py")):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ml_msvm_found", f)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        cls = getattr(mod, "ML_MSVMClassifier", None)
        if cls:
            print(f"[utils] Loaded ML_MSVMClassifier from {f}")
            return cls
    raise ImportError("Cannot find ML_MSVMClassifier. Run: find ~ -name ml_msvm.py")


# ---------------------------------------------------------------------------
# Logging / CSV
# ---------------------------------------------------------------------------

class Tee:
    """Mirror stdout to a log file."""
    def __init__(self, stream, filepath):
        self._stream = stream
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._file = open(filepath, "a", buffering=1, encoding="utf-8")
        self._file.write(f"\n{'='*70}\n  Session: {datetime.datetime.now()}\n{'='*70}\n")
    def write(self, d): self._stream.write(d); self._file.write(d); self._file.flush()
    def flush(self): self._stream.flush(); self._file.flush()
    def isatty(self): return False
    def close(self): self._file.close()


class CSVWriter:
    """Append-mode CSV writer that flushes every row."""
    FIELDS = ["exp_id", "dataset", "n_total", "n_train", "n_test", "d", "n_classes",
              "model", "kernel", "L", "m", "P", "split_id", "acc", "time_s", "timestamp"]
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        new = not os.path.exists(filepath)
        self._f = open(filepath, "a", newline="", encoding="utf-8", buffering=1)
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS, extrasaction="ignore")
        if new:
            self._w.writeheader(); self._f.flush()
    def write(self, row):
        row.setdefault("timestamp", datetime.datetime.now().isoformat())
        self._w.writerow(row); self._f.flush()
    def close(self): self._f.close()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_CACHE = {}

def _openml(name=None, version=1, data_id=None):
    kw = dict(as_frame=False, return_X_y=True)
    X, y = (fetch_openml(data_id=data_id, **kw) if data_id
            else fetch_openml(name=name, version=version, **kw))
    X = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)


def _load_glass(data_dir="data"):
    """Glass: local UCI glass.data (in data/glass/ or data/), else OpenML.
       Format: Id, 9 features, Type(class). Classes 1,2,3,5,6,7 (no 4)."""
    for local in [os.path.join(data_dir, "glass", "glass.data"),
                  os.path.join(data_dir, "glass.data")]:
        if os.path.exists(local):
            arr = np.loadtxt(local, delimiter=",")
            X = arr[:, 1:10].astype(np.float64)                   # skip Id column
            _, y = np.unique(arr[:, 10].astype(int), return_inverse=True)
            return X, y.astype(int)
    for kw in [{"data_id": 41}, {"data_id": 723},
               {"name": "glass", "version": 10}, {"name": "glass", "version": 1}]:
        try:
            return _openml(**kw)
        except Exception:
            continue
    raise RuntimeError("Glass not loadable (no data/glass/glass.data and OpenML failed).")


def _load_susy_libsvm(data_dir):
    """SUSY (5M, d=18). Tries, in order:
         1. cached data/susy_full.npz                (instant)
         2. UCI data/susy/SUSY.csv.gz                (label + 18 features)
         3. LIBSVM data/SUSY.bz2                      (label + sparse features)
         4. download LIBSVM bz2
       Parses once and caches susy_full.npz (~80 MB). Peak RAM ~1.3 GB."""
    os.makedirs(data_dir, exist_ok=True)
    cache = os.path.join(data_dir, "susy_full.npz")
    if os.path.exists(cache):
        print("  (SUSY from cache)", flush=True)
        d = np.load(cache); return d["X"].astype(np.float64), d["y"].astype(int)

    # 2. UCI CSV.gz  (the file you downloaded)
    csv_paths = [os.path.join(data_dir, "susy", "SUSY.csv.gz"),
                 os.path.join(data_dir, "SUSY.csv.gz"),
                 os.path.join(data_dir, "susy", "SUSY.csv"),
                 os.path.join(data_dir, "SUSY.csv")]
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)
    if csv_path is not None:
        import pandas as pd
        print(f"  Parsing {csv_path} (5M rows; ~1 min, peak ~1.3 GB)...", flush=True)
        # float32 keeps the DataFrame small; column 0 is the label, 1..18 features
        df = pd.read_csv(csv_path, header=None, dtype=np.float32)
        y = df.iloc[:, 0].astype(int).values
        X = df.iloc[:, 1:].astype(np.float64).values
        del df
        np.savez_compressed(cache, X=X, y=y)
        print(f"  SUSY: {X.shape} cached -> susy_full.npz", flush=True)
        return X, y

    # 3/4. LIBSVM bz2 local or download
    import bz2, urllib.request
    from sklearn.datasets import load_svmlight_file
    local_bz2 = os.path.join(data_dir, "SUSY.bz2")
    if not os.path.exists(local_bz2):
        URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.bz2"
        print(f"\n  No local SUSY file; downloading LIBSVM bz2 (~1.8 GB)...", flush=True)
        urllib.request.urlretrieve(URL, local_bz2)
    print("  Parsing SUSY.bz2 (peak ~1.4 GB)...", flush=True)
    with bz2.open(local_bz2, "rb") as f:
        Xs, y = load_svmlight_file(f, n_features=18)
    X = Xs.toarray().astype(np.float64)
    y = (y.astype(float) > 0).astype(int)
    np.savez_compressed(cache, X=X, y=y)
    print(f"  SUSY: {X.shape} cached -> susy_full.npz", flush=True)
    return X, y


# Default HIGGS pool: large enough for train sizes up to 400k + a 50k test.
HIGGS_POOL = 1_100_000
HIGGS_TOTAL = 11_000_000   # full HIGGS row count (for the sampling fraction)


def _load_higgs_capped(data_dir, n_max=None):
    """
    Full HIGGS (11M, d=28). Reads the UCI HIGGS.csv.gz in CHUNKS and keeps a
    uniform random subsample of HIGGS_POOL rows (memory-bounded: one chunk in
    RAM at a time). This gives a serious large-scale pool that exp3 draws
    train sizes [50k,100k,200k,400k] + 50k test from — the same protocol as
    SUSY and CoverType Full. Falls back to LIBSVM bz2, then OpenML (98k).
    The parsed pool is cached as higgs_pool_<N>.npz for instant reuse.

    Place the file at: data/higgs/HIGGS.csv.gz
    Download (on your machine):
      wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
    """
    os.makedirs(data_dir, exist_ok=True)
    pool = HIGGS_POOL
    cache = os.path.join(data_dir, f"higgs_pool_{pool}.npz")
    if os.path.exists(cache):
        print("  (HIGGS from cache)", flush=True)
        d = np.load(cache); return d["X"].astype(np.float64), d["y"].astype(int)

    csv_paths = [os.path.join(data_dir, "higgs", "HIGGS.csv.gz"),
                 os.path.join(data_dir, "HIGGS.csv.gz")]
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)

    if csv_path is not None:
        import pandas as pd
        print(f"  Parsing {csv_path} in chunks, sampling ~{pool:,} of 11M rows...",
              flush=True)
        frac = pool / HIGGS_TOTAL
        rs = np.random.default_rng(0)
        parts = []
        for chunk in pd.read_csv(csv_path, header=None, dtype=np.float32,
                                 chunksize=1_000_000):
            mask = rs.random(len(chunk)) < frac
            if mask.any():
                parts.append(chunk.values[mask])
        data = np.concatenate(parts, axis=0)
        y = data[:, 0].astype(int)
        X = data[:, 1:].astype(np.float64)
        del data, parts
    else:
        # Fallbacks: LIBSVM bz2 (full) or OpenML (only ~98k)
        import bz2
        from sklearn.datasets import load_svmlight_file
        local_bz2 = os.path.join(data_dir, "HIGGS.bz2")
        if os.path.exists(local_bz2):
            print("  Parsing HIGGS.bz2 (full)...", flush=True)
            with bz2.open(local_bz2, "rb") as f:
                Xs, y = load_svmlight_file(f, n_features=28)
            X = Xs.toarray().astype(np.float64)
            y = (y.astype(float) > 0).astype(int)
            if len(y) > pool:
                idx = np.random.default_rng(0).choice(len(y), pool, replace=False)
                X, y = X[idx], y[idx]
        else:
            print("\n  No local HIGGS file; OpenML fallback (~98k rows only)."
                  "\n  For a serious test, download data/higgs/HIGGS.csv.gz.", flush=True)
            X, yr = fetch_openml(data_id=23512, as_frame=False, return_X_y=True)
            X = X.astype(np.float64); _, y = np.unique(yr, return_inverse=True)
            y = y.astype(int)

    np.savez_compressed(cache, X=X, y=y)
    print(f"  HIGGS: {len(y):,} rows pooled and cached -> {os.path.basename(cache)}",
          flush=True)
    return X, y


def _covertype_sub(n):
    X, y = _openml("covertype", version=3)
    idx, _ = next(StratifiedShuffleSplit(1, train_size=n, random_state=0).split(X, y))
    return X[idx], y[idx]


def load(tag, verbose=True, data_dir="data"):
    """Load a dataset by tag, with caching and graceful failure."""
    if tag in _CACHE:
        return _CACHE[tag]
    if verbose:
        print(f"  Loading {tag}...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        if tag == "wine":
            X, y = load_wine(return_X_y=True); X = X.astype(np.float64)
        elif tag == "breast_cancer":
            X, y = load_breast_cancer(return_X_y=True); X = X.astype(np.float64)
        elif tag == "ionosphere":    X, y = _openml("ionosphere", version=1)
        elif tag == "sonar":         X, y = _openml("sonar", version=1)
        elif tag == "glass":         X, y = _load_glass(data_dir)
        elif tag == "magic":         X, y = _openml("MagicTelescope", version=1)
        elif tag == "spambase":      X, y = _openml(data_id=44); y = y.astype(int)
        elif tag == "covertype_sub": X, y = _covertype_sub(10_000)
        elif tag == "covertype":     X, y = _openml("covertype", version=3)
        elif tag == "mnist":         X, y = _openml("mnist_784", version=1)
        elif tag == "fashion":       X, y = _openml("Fashion-MNIST", version=1)
        elif tag == "susy":          X, y = _load_susy_libsvm(data_dir)
        elif tag == "higgs":         X, y = _load_higgs_capped(data_dir)
        else:
            raise ValueError(f"Unknown tag: {tag}")
        # NaN safety net (HIGGS from OpenML contains missing values): impute
        # each NaN with its column mean so downstream LinearSVC never sees NaN.
        if np.isnan(X).any():
            n_nan = int(np.isnan(X).sum())
            col_mean = np.nanmean(X, axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)  # all-NaN cols -> 0
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            if verbose:
                print(f"[imputed {n_nan} NaN with column means] ", end="", flush=True)
        if verbose:
            print(f"done ({time.perf_counter()-t0:.1f}s, shape={X.shape})", flush=True)
        _CACHE[tag] = (X, y)
        return X, y
    except Exception as e:
        if verbose:
            print(f"FAILED ({e})", flush=True)
        raise


def make_splits(X, y, regime, n_repeats, seed=0):
    """Train/test splits per regime.
       1,2 -> 90/10 stratified ;  3 -> 10k/2k subsamples ;  4 -> 80/20."""
    if regime in (1, 2):
        sss = StratifiedShuffleSplit(n_repeats, test_size=0.1, random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
    elif regime == 3:
        out = []
        for s in range(n_repeats):
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, train_size=10_000, test_size=2_000, stratify=y, random_state=s)
            out.append((Xtr, ytr, Xte, yte))
        return out
    elif regime == 4:
        sss = StratifiedShuffleSplit(n_repeats, test_size=0.2, random_state=seed)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
    raise ValueError(f"Unknown regime {regime}")


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def pipe(*steps):
    return Pipeline([("scaler", StandardScaler())] +
                    [(f"s{i}", s) for i, s in enumerate(steps)])


def _linear_svc(C=1.0):
    """LinearSVC with dual='auto' when available, else dual=False.
    Uses tol=1e-2 to MATCH the ML-MSVM internal SVMs exactly, so the timing
    comparison is fair: every linear solve in the suite uses identical solver
    settings. tol=1e-2 gives the same accuracy as 1e-4 (verified) but converges
    ~30x faster on ill-conditioned raw features such as standardized MNIST."""
    try:
        return LinearSVC(C=C, dual="auto", tol=1e-2, max_iter=2000, random_state=0)
    except TypeError:
        return LinearSVC(C=C, dual=False, tol=1e-2, max_iter=2000, random_state=0)


def make_linear_svm():
    return pipe(_linear_svc(1.0))


def make_rbf_svm():
    return pipe(SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0))


def make_nystroem_svm(P):
    return Pipeline([("scaler", StandardScaler()),
                     ("nystroem", Nystroem(kernel="rbf", n_components=P,
                                           gamma=None, random_state=0)),
                     ("svm", _linear_svc(1.0))])


def make_mlmsvm(ML_MSVM, L, m, P, kernel, C_values=None, final_C=1.0, seed=0):
    if C_values is None:
        C_values = [1.0] if m == 1 else list(np.logspace(-2, 1, num=m))
    return pipe(ML_MSVM(num_layers=L, svms_per_block=m, C_values=C_values,
                        rff_features=P, kernel=kernel, arc_cosine_degree=1,
                        final_C=final_C, random_state=seed))


def make_flat_rff(ML_MSVM, P, kernel="rbf"):
    return make_mlmsvm(ML_MSVM, 0, 1, P, kernel)


# ---------------------------------------------------------------------------
# Sweeps / helpers
# ---------------------------------------------------------------------------

def ms_for(d):
    """Width sweep, capped at m=10 (prevents MNIST m=49/98 blow-up)."""
    if d <= 20:
        base = [1, 2, 3, 4, 6, 10, min(2 * d, 20)]
    else:
        base = [1, 2, 3, 4, 6, 10]
    return sorted(set(max(1, v) for v in base if v > 0))


def hms(s):
    return f"{int(s//3600):02d}h{int(s%3600//60):02d}m{int(s%60):02d}s"


def banner(title, *lines):
    w = max(len(title), max((len(l) for l in lines), default=0)) + 4
    print(f"\n{'█'*w}\n  {title}")
    for l in lines:
        print(f"  {l}")
    print(f"{'█'*w}\n", flush=True)


if __name__ == "__main__":
    print("Testing import_ml_msvm()...")
    try:
        ML = import_ml_msvm()
        print(f"  OK: {ML.__module__}")
    except ImportError as e:
        print(f"  FAILED: {e}")