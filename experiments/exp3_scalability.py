"""
exp3_scalability_fixed.py — Scalability on Large Datasets (FIXED)
==================================================================
Fixes vs original exp3:
  1. SUSY/HIGGS loaded with n_max=500k cap; result cached as .npz so the
     full 5M/11M array is never in RAM simultaneously with the subsample.
  2. RBF_N_LIMIT raised to 20_000 (exact SVM still usable to ~20k).
  3. Linear SVM uses dual=False explicitly (was causing O(n^2) on high-d).

Hypothesis
----------
ML-MSVM training time scales linearly with n (primal solver O(n·P·iters)),
making it feasible at n > 500k where the exact RBF SVM is intractable.
At these scales, ML-MSVM maintains competitive accuracy over Linear SVM,
demonstrating recovery of nonlinear structure unavailable to linear models.

Datasets: SUSY (500k pool / 5M total, d=18), Cover Type Full (581k, d=54),
          HIGGS (500k pool / 11M total, d=28).
Protocol: 5× stratified 80/20 splits over the pool.
Models:   Linear SVM, Flat RFF (RBF), ML-MSVM RBF m=2 L=2,
          ML-MSVM ArcCos m=1 L=1. Exact RBF SVM: not run.
Output:   logs/exp3_scalability_<ts>.txt  +  results/exp3_scalability.csv
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
N_MAX   = 500_000   # cap for SUSY/HIGGS

DATASETS = [
    # tag        display               pool_n
    ("susy",     "SUSY",               N_MAX),
    ("covertype","Cover Type Full",     None),   # None = use all 581k
    ("higgs",    "HIGGS",              N_MAX),
]

MODELS = [
    ("Linear SVM",           lambda: make_linear_svm()),
    ("Flat RFF RBF (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "rbf")),
    ("ML-MSVM RBF m=2 L=2",  lambda: make_mlmsvm(ML_MSVM, 2, 2, P, "rbf")),
    ("ML-MSVM Arc m=1 L=1",  lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine")),
]


def run(log_path, csv_path, data_dir):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)

    try:
        banner("Experiment 3 — Scalability (FIXED)",
               "SUSY / HIGGS: 500k pool (full dataset never fully loaded simultaneously)",
               "Cover Type: full 581k.",
               "Exact RBF SVM: NOT run — infeasible at this scale.",
               f"P={P}  {REPEATS}× stratified 80/20.")

        t_total = time.perf_counter()

        for tag, name, pool_n in DATASETS:
            print(f"\n  Loading {name}...", end=" ", flush=True)
            t0 = time.perf_counter()
            try:
                X, y = load(tag, verbose=False, n_max_large=pool_n or N_MAX)
            except Exception as e:
                print(f"FAILED: {e}\n  Skipping {name}.", flush=True)
                continue
            print(f"done ({time.perf_counter()-t0:.1f}s, shape={X.shape})")

            n_full = len(y)
            d = X.shape[1]
            n_cls = len(np.unique(y))

            # Optionally further subsample if dataset is still large after load
            pool_size = pool_n if pool_n else n_full
            if n_full > pool_size:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=pool_size, random_state=0)
                idx, _ = next(sss.split(X, y))
                X, y = X[idx], y[idx]
                print(f"  Subsampled to pool={pool_size:,}")

            # 80/20 splits over the pool
            sss = StratifiedShuffleSplit(n_splits=REPEATS, test_size=0.2, random_state=0)
            splits = [(X[tr], y[tr], X[te], y[te]) for tr, te in sss.split(X, y)]
            n_train = len(splits[0][0])
            n_test  = len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (d={d}, classes={n_cls})")
            print(f"  Pool={len(y):,} from full n={n_full:,}")
            print(f"  {REPEATS}× 80/20: n_train={n_train:,}  n_test={n_test:,}")
            print(f"  [Exact RBF SVM: SKIPPED — O(n²) at this scale]")
            print(f"{'='*72}")
            print(f"  {'Model':<36s}  {'acc':>8s}  {'±':>6s}  {'t/run':>8s}  {'t_std':>7s}")
            print(f"  {'─'*72}")

            for model_label, model_fn in MODELS:
                model = model_fn()
                accs, times = [], []
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    mc = clone(model)
                    t0 = time.perf_counter()
                    mc.fit(Xtr, ytr)
                    accs.append(mc.score(Xte, yte))
                    times.append(time.perf_counter() - t0)
                    csv_w.write(dict(
                        exp_id=EXP_ID, dataset=name,
                        n_total=n_full, n_train=n_train, n_test=n_test,
                        d=d, n_classes=n_cls,
                        model=model_label, kernel="varies", L=-1, m=-1, P=P,
                        split_id=i, acc=accs[-1], time_s=times[-1],
                    ))
                print(f"  {model_label:<36s}  {np.mean(accs):.4f}  "
                      f"{np.std(accs):.4f}  {np.mean(times):>7.1f}s  "
                      f"{np.std(times):>6.1f}s", flush=True)

        elapsed = hms(time.perf_counter() - t_total)
        print(f"\n{'█'*60}")
        print(f"  Experiment 3 complete. Total: {elapsed}")
        print(f"{'█'*60}")

    finally:
        sys.stdout = tee._stream
        tee.close()
        csv_w.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",  default="logs")
    p.add_argument("--csv_dir",  default="results")
    p.add_argument("--data_dir", default="data",
                   help="Directory to cache downloaded large datasets (.npz files)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(args.log_dir, f"exp3_scalability_{ts}.txt"),
        os.path.join(args.csv_dir, "exp3_scalability.csv"),
        args.data_dir)