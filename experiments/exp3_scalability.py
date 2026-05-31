"""
exp3_scalability.py — Scalability on Large Datasets
====================================================
Hypothesis
----------
ML-MSVM training time scales as O(n·P) with the primal LibLinear solver,
remaining feasible at n > 400k where the exact RBF SVM is intractable
(O(n²) memory, O(n³) time). At these scales ML-MSVM maintains competitive
accuracy over the Linear SVM baseline, demonstrating recovery of nonlinear
structure unavailable to linear models.

Datasets
--------
  SUSY         (d=18, physics binary):  500k pool from 5M total.
  Cover Type   (d=54, 7-class):         full 581k samples.
  HIGGS        (d=28, physics binary):  500k pool from 11M total.

  SUSY and HIGGS are cached as .npz after the first download so that
  subsequent runs do not re-load the full dataset (avoids OOM).

Protocol
--------
  5× stratified 80/20 splits over the pool for each dataset.
  Exact RBF SVM: NOT run — noted explicitly as infeasible.
  Models: Linear SVM, Flat RFF RBF, ML-MSVM RBF m=2 L=2,
          ML-MSVM ArcCos m=1 L=1.
  P = 1000.
  Output: logs/exp3_scalability_<ts>.txt  +  results/exp3_scalability.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_mlmsvm, make_linear_svm,
                   make_flat_rff, import_ml_msvm, hms, banner)

ML_MSVM = import_ml_msvm()
P       = 1000
REPEATS = 5
EXP_ID  = "exp3_scalability"

DATASETS = [
    # tag         display           pool_n  (None = use all)
    ("susy",      "SUSY",           500_000),
    ("covertype", "Cover Type Full", None),
    ("higgs",     "HIGGS",          500_000),
]
MODELS = [
    ("Linear SVM",           lambda: make_linear_svm()),
    ("Flat RFF RBF (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "rbf")),
    ("ML-MSVM RBF m=2 L=2",  lambda: make_mlmsvm(ML_MSVM, 2, 2, P, "rbf")),
    ("ML-MSVM Arc m=1 L=1",  lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine")),
]


def run(log_path: str, csv_path: str, data_dir: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 3 — Scalability on Large Datasets",
               "SUSY (500k/5M, d=18)  |  Cover Type Full (581k, d=54)  |  HIGGS (500k/11M, d=28)",
               "Exact RBF SVM: NOT run — O(n²) memory at this scale.",
               f"P={P}  |  {REPEATS}× stratified 80/20 splits")
        t_start = time.perf_counter()

        for tag, name, pool_n in DATASETS:
            print(f"\n  Loading {name}...", end=" ", flush=True)
            try:
                X, y = load(tag, verbose=False, data_dir=data_dir)
            except Exception as e:
                print(f"FAILED ({e}). Skipping.", flush=True)
                continue
            n_full, d, n_cls = len(y), X.shape[1], len(np.unique(y))
            print(f"done (shape={X.shape})")

            # Apply pool cap if needed
            if pool_n and n_full > pool_n:
                sss = StratifiedShuffleSplit(1, train_size=pool_n, random_state=0)
                idx, _ = next(sss.split(X, y))
                X, y = X[idx], y[idx]
                print(f"  Subsampled to pool={len(y):,}")

            sss = StratifiedShuffleSplit(REPEATS, test_size=0.2, random_state=0)
            splits = [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (full n={n_full:,}, pool={len(y):,}, d={d}, K={n_cls})")
            print(f"  {REPEATS}× 80/20: n_train={n_tr:,}  n_test={n_te:,}")
            print(f"  [Exact RBF SVM: SKIPPED — infeasible at n_train={n_tr:,}]")
            print(f"{'='*72}")
            print(f"  {'model':<38s}  {'acc':>8s}  {'std':>6s}  {'t/run':>9s}  {'t_std':>7s}")
            print(f"  {'─'*72}")

            for lbl, model_fn in MODELS:
                accs, ts = [], []
                model = model_fn()
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    mc = clone(model)
                    t = time.perf_counter()
                    mc.fit(Xtr, ytr)
                    accs.append(mc.score(Xte, yte))
                    ts.append(time.perf_counter() - t)
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                        n_total=n_full, n_train=n_tr, n_test=n_te, d=d,
                        n_classes=n_cls, model=lbl, kernel="varies", L=-1, m=-1, P=P,
                        split_id=i, acc=accs[-1], time_s=ts[-1]))
                print(f"  {lbl:<38s}  {np.mean(accs):.4f}  {np.std(accs):.4f}  "
                      f"{np.mean(ts):>8.1f}s  {np.std(ts):>6.1f}s", flush=True)

        print(f"\n  Experiment 3 complete. {hms(time.perf_counter()-t_start)}")
    finally:
        sys.stdout = tee._stream
        tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",  default="logs")
    p.add_argument("--csv_dir",  default="results")
    p.add_argument("--data_dir", default="data")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for d in (a.log_dir, a.csv_dir, a.data_dir): os.makedirs(d, exist_ok=True)
    run(os.path.join(a.log_dir, f"exp3_scalability_{ts}.txt"),
        os.path.join(a.csv_dir, "exp3_scalability.csv"), a.data_dir)
