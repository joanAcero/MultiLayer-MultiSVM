"""
benchmark_main.py
=================

Head-to-head benchmark of ML-MSVM (RBF and arc-cosine variants) against:
  - Linear SVM
  - RBF SVM          (exact kernel, gold standard; skipped when n is too large)
  - Flat RFF SVM     (single RFF layer + linear SVM; ML-MSVM with L=0)
  - ML-MSVM RBF      (Proposal 2, kernel='rbf')
  - ML-MSVM ArcCos   (Proposal 2, kernel='arc_cosine', degree=1)
  - ML-MSVM ArcCos m=d  (arc-cosine with m set to input dimensionality d,
                          at the depth L that gave best accuracy for ArcCos above)

Datasets are grouped into three regimes following the thesis objectives:

  Regime 1 - SVM-strong (low n):   Ionosphere, Sonar, Breast Cancer, Wine
  Regime 2 - High n, moderate d:   Magic, Spambase, Cover Type (subset)
  Regime 3 - High n, high d:       MNIST (subsampled), Fashion-MNIST (subsampled)

Evaluation protocol (matching Acero & Belanche 2025):
  - Regime 1/2: 10 random 90/10 stratified splits, report mean +/- std accuracy
                and mean training time.
  - Regime 3:   3 random stratified splits of fixed size (10 000 train / 2 000 test).
  - RBF SVM is skipped for n > RBF_N_LIMIT (kernel matrix too large).

The m=d experiment
------------------
For each dataset, after the main sweep, an additional arc-cosine model is run with
m = d (number of SVMs per block = input dimensionality).  The depth L used is the one
that achieved the highest mean accuracy among ML-MSVM ArcCos L=1,2,3 in the main
sweep.  Rationale: m=d avoids the inter-layer bottleneck, the representation
dimension after each block equals the input dimension, so no information capacity
is lost at the projection step.

Run
---
    python benchmark_main.py [--regimes 1 2 3] [--repeats 10] [--rff_features 1000]

Dependencies: numpy, scikit-learn, scipy.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml_msvm"))

from ml_msvm import ML_MSVMClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

RBF_N_LIMIT   = 10_000   # skip exact RBF SVM above this training-set size
REGIME3_TRAIN =  10_000  # training points for Regime 3 datasets
REGIME3_TEST  =   2_000  # test points for Regime 3 datasets
SWEEP_DEPTHS  = [1, 2, 3]
C_SPREAD      = [0.01, 0.1, 1.0, 10.0]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_openml(name=None, version=1, data_id=None):
    if data_id is not None:
        X, y = fetch_openml(data_id=data_id, as_frame=False, return_X_y=True)
    else:
        X, y = fetch_openml(name=name, version=version,
                            as_frame=False, return_X_y=True)
    X = X.astype(np.float64)
    _, y = np.unique(y, return_inverse=True)
    return X, y.astype(int)


def get_datasets() -> dict:
    """Returns {display_name: (X, y, regime_int)}."""
    datasets = {}

    # Regime 1
    X, y = load_wine(return_X_y=True)
    datasets["Wine (13d, 3c, 178n)"] = (X.astype(np.float64), y, 1)

    X, y = load_breast_cancer(return_X_y=True)
    datasets["Breast Cancer (30d, 2c, 569n)"] = (X.astype(np.float64), y, 1)

    X, y = _load_openml("ionosphere", version=1)
    datasets["Ionosphere (34d, 2c, 351n)"] = (X, y, 1)

    X, y = _load_openml("sonar", version=1)
    datasets["Sonar (60d, 2c, 208n)"] = (X, y, 1)

    # Regime 2
    X, y = _load_openml("MagicTelescope", version=1)
    datasets["Magic (10d, 2c, 19k n)"] = (X, y, 2)

    X, y = _load_openml(data_id=44)
    datasets["Spambase (57d, 2c, 4.6k n)"] = (X, y.astype(int), 2)

    try:
        X_ct, y_ct = _load_openml("covertype", version=3)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=10_000, random_state=0)
        idx, _ = next(sss.split(X_ct, y_ct))
        datasets["Cover Type subset (54d, 7c, 10k n)"] = (X_ct[idx], y_ct[idx], 2)
    except Exception as e:
        print(f"  [Cover Type unavailable: {e}]")

    return datasets


def get_regime3_datasets() -> dict:
    datasets = {}
    for name, display in [
        ("mnist_784",     "MNIST (784d, 10c, sub 10k)"),
        ("Fashion-MNIST", "Fashion-MNIST (784d, 10c, sub 10k)"),
    ]:
        try:
            X, y = _load_openml(name, version=1)
            datasets[display] = (X, y, 3)
        except Exception as e:
            print(f"  [{display} unavailable: {e}]")
    return datasets


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _pipe(*steps) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler())] + [(f"s{i}", s) for i, s in enumerate(steps)]
    )


def get_main_models(P: int) -> dict:
    """
    Fixed models for the main sweep.
    P = number of random features per block.
    All models are dataset-independent and can be cloned directly.
    """
    models = {
        "Linear SVM": _pipe(
            LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=0)
        ),
        "RBF SVM": _pipe(
            SVC(kernel="rbf", C=1.0, gamma="scale", random_state=0)
        ),
        "Flat RFF SVM (L=0)": _pipe(
            ML_MSVMClassifier(
                num_layers=0, svms_per_block=4, rff_features=P,
                kernel="rbf", final_C=1.0, random_state=0,
            )
        ),
    }
    for L in SWEEP_DEPTHS:
        models[f"ML-MSVM RBF L={L} m=4"] = _pipe(
            ML_MSVMClassifier(
                num_layers=L, svms_per_block=4, C_values=C_SPREAD,
                rff_features=P, kernel="rbf", final_C=1.0, random_state=0,
            )
        )
    for L in SWEEP_DEPTHS:
        models[f"ML-MSVM ArcCos L={L} m=4"] = _pipe(
            ML_MSVMClassifier(
                num_layers=L, svms_per_block=4, C_values=C_SPREAD,
                rff_features=P, kernel="arc_cosine", arc_cosine_degree=1,
                final_C=1.0, random_state=0,
            )
        )
    return models


def make_arccos_md_model(P: int, d: int, L: int) -> Pipeline:
    """
    Arc-cosine model with m = d (SVMs per block = input dimensionality).

    Setting m = d means the inter-layer representation X^(l+1) = Phi^(l) @ W^(l)
    has dimension m (binary) or m*K (K-class OvR), matching the input dimension
    d of each block. No information capacity is discarded at the projection step.

    C_values is log-spaced over [0.01, 10] with d entries, giving each of the d
    parallel SVMs a different regularisation constant for diverse weight vectors.

    L is the depth that achieved best accuracy among ArcCos L=1,2,3 in the
    main sweep on this dataset (chosen automatically by best_arccos_depth).
    """
    c_vals = list(np.logspace(-2, 1, num=d))
    return _pipe(
        ML_MSVMClassifier(
            num_layers=L, svms_per_block=d, C_values=c_vals,
            rff_features=P, kernel="arc_cosine", arc_cosine_degree=1,
            final_C=1.0, random_state=0,
        )
    )


# ---------------------------------------------------------------------------
# Best-depth selector
# ---------------------------------------------------------------------------

def best_arccos_depth(results: dict) -> int:
    """
    Return the depth L in SWEEP_DEPTHS with the highest mean accuracy among
    ML-MSVM ArcCos L=* m=4 models in the main sweep. Falls back to L=1.
    """
    best_L, best_acc = 1, -1.0
    for L in SWEEP_DEPTHS:
        key = f"ML-MSVM ArcCos L={L} m=4"
        r = results.get(key, {})
        if not r.get("skipped", True) and not np.isnan(r.get("acc_mean", np.nan)):
            if r["acc_mean"] > best_acc:
                best_acc, best_L = r["acc_mean"], L
    return best_L


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

def _run_one(model, X_tr, y_tr, X_te, y_te) -> tuple[float, float]:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    return model.score(X_te, y_te), time.perf_counter() - t0


def _evaluate_model(model_template, splits: list) -> dict:
    """Run one model across pre-computed splits, return result dict."""
    accs, times = [], []
    for X_tr, y_tr, X_te, y_te in splits:
        acc, t = _run_one(clone(model_template), X_tr, y_tr, X_te, y_te)
        accs.append(acc)
        times.append(t)
    return dict(
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        time_mean=float(np.mean(times)),
        skipped=False,
    )


def _make_splits(X, y, regime: int, n_repeats: int) -> list:
    """Pre-compute all train/test splits for a dataset."""
    if regime < 3:
        sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.1, random_state=0)
        return [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
    else:
        splits = []
        for rep in range(n_repeats):
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, train_size=REGIME3_TRAIN, test_size=REGIME3_TEST,
                stratify=y, random_state=rep,
            )
            splits.append((X_tr, y_tr, X_te, y_te))
        return splits


def evaluate_dataset(
    ds_name: str,
    X: np.ndarray,
    y: np.ndarray,
    regime: int,
    main_models: dict,
    P: int,
    n_repeats: int,
) -> dict:
    """
    Run the main model sweep and then the m=d arc-cosine experiment.
    All models share the same pre-computed splits for fair comparison.
    """
    d = X.shape[1]
    n_train = int(0.9 * len(y)) if regime < 3 else REGIME3_TRAIN
    skip_rbf = n_train > RBF_N_LIMIT or regime == 3

    protocol = (f"{n_repeats}x 90/10 splits" if regime < 3
                else f"{n_repeats}x {REGIME3_TRAIN}/{REGIME3_TEST}")
    print(f"\n{'='*65}")
    print(f"  {ds_name}  [Regime {regime} | {protocol} | d={d}]")
    if skip_rbf:
        print(f"  (RBF SVM skipped: n too large or Regime 3)")
    print(f"{'='*65}")

    # Pre-compute splits once -- shared across all models
    splits = _make_splits(X, y, regime, n_repeats)
    results = {}

    # ------------------------------------------------------------------
    # Main sweep
    # ------------------------------------------------------------------
    for model_name, model_template in main_models.items():
        if model_name == "RBF SVM" and skip_rbf:
            results[model_name] = dict(acc_mean=np.nan, acc_std=np.nan,
                                       time_mean=np.nan, skipped=True)
            print(f"  {model_name:<42s}  SKIPPED")
            continue

        results[model_name] = _evaluate_model(model_template, splits)
        r = results[model_name]
        print(f"  {model_name:<42s}  "
              f"acc={r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
              f"({r['time_mean']:.2f}s/run)")

    # ------------------------------------------------------------------
    # m=d arc-cosine experiment
    # Depth chosen as best among ArcCos L=1,2,3 in the sweep above.
    # ------------------------------------------------------------------
    best_L = best_arccos_depth(results)
    md_name = f"ML-MSVM ArcCos m=d={d} L={best_L}"
    print(f"\n  -- m=d experiment: best ArcCos depth was L={best_L} --")

    md_model = make_arccos_md_model(P=P, d=d, L=best_L)
    results[md_name] = _evaluate_model(md_model, splits)
    r = results[md_name]
    print(f"  {md_name:<42s}  "
          f"acc={r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
          f"({r['time_mean']:.2f}s/run)")

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(all_results: dict, main_model_names: list) -> None:
    print(f"\n\n{'='*75}")
    print("  SUMMARY  (sorted by accuracy; * = best; -- = skipped)")
    print(f"  m=d model shown separately below main models")
    print(f"{'='*75}")

    sep = "-" * 72

    for ds_name, results in all_results.items():
        md_keys  = [k for k in results if "m=d" in k]
        main_keys = [k for k in main_model_names if k in results]

        valid_all = {
            k: results[k] for k in list(main_keys) + md_keys
            if not results[k].get("skipped") and
               not np.isnan(results[k].get("acc_mean", np.nan))
        }
        if not valid_all:
            continue

        best_acc = max(v["acc_mean"] for v in valid_all.values())

        print(f"\n  {ds_name}")
        print(f"  {sep}")
        print(f"  {'Model':<42s}  {'Accuracy':>18s}  {'Time/run':>10s}")
        print(f"  {sep}")

        # Main models sorted by accuracy
        sorted_main = sorted(
            main_keys,
            key=lambda m: results[m]["acc_mean"]
            if not results[m].get("skipped") else -1,
            reverse=True,
        )
        for name in sorted_main:
            r = results[name]
            if r.get("skipped") or np.isnan(r["acc_mean"]):
                print(f"  {name:<42s}  {'--':>18s}  {'--':>10s}")
            else:
                marker = " *" if abs(r["acc_mean"] - best_acc) < 1e-9 else "  "
                print(f"  {name:<42s}  "
                      f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
                      f"{r['time_mean']:>8.2f}s{marker}")

        # m=d experiment
        if md_keys:
            print(f"  {'  (m=d experiment)'}")
            for name in md_keys:
                r = results[name]
                marker = " *" if abs(r["acc_mean"] - best_acc) < 1e-9 else "  "
                print(f"  {name:<42s}  "
                      f"{r['acc_mean']:.4f} +/- {r['acc_std']:.4f}  "
                      f"{r['time_mean']:>8.2f}s{marker}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML-MSVM main benchmark")
    p.add_argument("--regimes", nargs="+", type=int, default=[1, 2],
                   choices=[1, 2, 3])
    p.add_argument("--repeats", type=int, default=10,
                   help="Repeats for Regime 1/2 (default: 10)")
    p.add_argument("--repeats3", type=int, default=3,
                   help="Repeats for Regime 3 (default: 3)")
    p.add_argument("--rff_features", type=int, default=1000,
                   help="P: random features per block (default: 1000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_models = get_main_models(P=args.rff_features)
    all_results: dict = {}

    print(f"\n{'#'*65}")
    print(f"  ML-MSVM Benchmark")
    print(f"  Regimes : {args.regimes}")
    print(f"  Repeats : {args.repeats} (R1/R2), {args.repeats3} (R3)")
    print(f"  P (RFFs): {args.rff_features}")
    print(f"{'#'*65}")

    if 1 in args.regimes or 2 in args.regimes:
        print("\nLoading datasets...", flush=True)
        datasets = get_datasets()
        for ds_name, (X, y, regime) in datasets.items():
            if regime not in args.regimes:
                continue
            results = evaluate_dataset(
                ds_name, X, y, regime,
                main_models=main_models,
                P=args.rff_features,
                n_repeats=args.repeats,
            )
            all_results[ds_name] = results

    if 3 in args.regimes:
        print("\nLoading Regime 3 datasets (requires network)...", flush=True)
        for ds_name, (X, y, _) in get_regime3_datasets().items():
            results = evaluate_dataset(
                ds_name, X, y, regime=3,
                main_models=main_models,
                P=args.rff_features,
                n_repeats=args.repeats3,
            )
            all_results[ds_name] = results

    print_summary(all_results, list(main_models.keys()))