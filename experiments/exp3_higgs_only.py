"""
exp3_higgs_only.py — Large-scale feasibility on FULL HIGGS only.
Same protocol as exp3 (train [50k,100k,200k,400k], 50k test, 3 seeds,
no exact RBF) but HIGGS alone, so you don't re-run SUSY / CoverType.
Writes to the SAME results/exp3_scalability.csv (run dedup_results.py after).

Requires: data/higgs/HIGGS.csv.gz  (full 11M-row UCI file).
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_mlmsvm, make_linear_svm,
                   make_flat_rff, import_ml_msvm, hms, banner)

ML_MSVM  = import_ml_msvm()
P        = 1000
N_SEEDS  = 3
EXP_ID   = "exp3_scalability"          # same id so it merges with exp3
N_TRAINS = [50_000, 100_000, 200_000, 400_000]
N_TEST   = 50_000
MODELS = [
    ("Linear SVM",           lambda: make_linear_svm()),
    ("Flat RFF RBF (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "rbf")),
    ("Flat RFF Arc (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "arc_cosine")),
    ("ML-MSVM RBF m=2 L=2",  lambda: make_mlmsvm(ML_MSVM, 2, 2, P, "rbf")),
    ("ML-MSVM Arc m=1 L=1",  lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine")),
]


def run(log_path, csv_path, data_dir):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 3 (HIGGS only) — Large-scale Feasibility",
               "Full HIGGS (11M -> 1.1M pool), d=28",
               f"n_train ∈ {N_TRAINS}  |  {N_SEEDS} seeds  |  no exact RBF")
        t_start = time.perf_counter()

        try:
            X, y = load("higgs", verbose=True, data_dir=data_dir)
        except Exception as e:
            print(f"\n  HIGGS LOAD FAILED: {e}", flush=True)
            return
        d, n_full, n_cls = X.shape[1], len(y), len(np.unique(y))

        n_test_actual = min(N_TEST, int(0.15 * n_full))
        Xpool, Xte, ypool, yte = train_test_split(
            X, y, test_size=n_test_actual, stratify=y, random_state=42)
        n_pool = len(ypool)

        print(f"\n{'━'*76}")
        print(f"  HIGGS  pool={n_full:,}  d={d}  K={n_cls}  test={n_test_actual:,}")
        print(f"{'━'*76}")
        print(f"  {'n_train':>9s}  {'model':<24s}  {'acc':>8s}  {'std':>6s}  {'time':>8s}")
        print(f"  {'─'*64}")

        for n_tr in N_TRAINS:
            if n_tr >= n_pool:
                print(f"  {n_tr:>9,}  (exceeds pool {n_pool:,}, skipping)", flush=True)
                continue
            for lbl, model_fn in MODELS:
                try:
                    accs, ts = [], []
                    for seed in range(N_SEEDS):
                        sss = StratifiedShuffleSplit(1, train_size=n_tr, random_state=seed)
                        idx, _ = next(sss.split(Xpool, ypool))
                        Xtr, ytr = Xpool[idx], ypool[idx]
                        mc = clone(model_fn())
                        t = time.perf_counter()
                        mc.fit(Xtr, ytr)
                        accs.append(mc.score(Xte, yte))
                        ts.append(time.perf_counter() - t)
                        csv_w.write(dict(exp_id=EXP_ID, dataset="HIGGS",
                            n_total=n_full, n_train=n_tr, n_test=n_test_actual,
                            d=d, n_classes=n_cls, model=lbl, kernel="varies",
                            L=-1, m=-1, P=P, split_id=seed,
                            acc=accs[-1], time_s=ts[-1]))
                    print(f"  {n_tr:>9,}  {lbl:<24s}  {np.mean(accs):.4f}  "
                          f"{np.std(accs):.4f}  {np.mean(ts):>7.1f}s", flush=True)
                except Exception as e:
                    print(f"  {n_tr:>9,}  {lbl:<24s}  FAILED: {e}", flush=True)
                    continue

        print(f"\n  HIGGS-only exp3 complete. {hms(time.perf_counter()-t_start)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",  default="logs")
    p.add_argument("--csv_dir",  default="results")
    p.add_argument("--data_dir", default="data")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp3_higgs_only_{ts}.txt"),
        os.path.join(a.csv_dir, "exp3_scalability.csv"), a.data_dir)
