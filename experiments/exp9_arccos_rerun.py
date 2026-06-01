"""
exp9_arccos_rerun.py — Re-run ONLY the ArcCos timing for MNIST (exp9).

All other exp9 data (Linear SVM, Exact RBF, Flat RFF, ML-MSVM RBF) is valid
and already in the CSV. This script appends only ArcCos rows, covering the
4 n_train values where timing was corrupted by the old bug (n ≥ 10k).

It also covers n=500/1k/2k/5k for completeness (these were fast and likely
correct, but re-running them takes only seconds and ensures a clean dataset).

Run: python exp9_arccos_rerun.py
Output appends to: results/exp9_scalability_timing.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, make_mlmsvm, import_ml_msvm, hms, banner

ML_MSVM  = import_ml_msvm()
P        = 1000
N_SEEDS  = 3
EXP_ID   = "exp9_scalability_timing"

N_TRAINS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000]
N_TEST   = 5_000


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp9 ArcCos re-run (timing fix verification)",
               "Only ML-MSVM Arc m=1 L=1 on MNIST.",
               f"Appends to {csv_path}",
               "After fix, n=40k should take ~5s not 1000s.")
        t_start = time.perf_counter()

        print("  Loading MNIST...", end=" ", flush=True)
        X, y = load("mnist", verbose=False)
        d, n_full = X.shape[1], len(y)
        print(f"done (shape={X.shape})")

        n_test_actual = min(N_TEST, int(0.15 * n_full))
        Xpool, Xte, ypool, yte = train_test_split(
            X, y, test_size=n_test_actual, stratify=y, random_state=42)
        n_pool = len(ypool)
        n_cls = len(np.unique(y))
        print(f"  Pool: {n_pool:,}  Fixed test: {n_test_actual:,}")

        model_fn = lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine")

        print(f"\n  {'n_train':>9s}  {'seed':>4s}  {'acc':>8s}  {'time':>8s}")
        print(f"  {'─'*40}")

        for n_tr in N_TRAINS:
            if n_tr >= n_pool:
                continue
            for seed in range(N_SEEDS):
                sss = StratifiedShuffleSplit(1, train_size=n_tr, random_state=seed)
                idx, _ = next(sss.split(Xpool, ypool))
                Xtr, ytr = Xpool[idx], ypool[idx]

                mc = clone(model_fn())
                t = time.perf_counter()
                mc.fit(Xtr, ytr)
                acc = mc.score(Xte, yte)
                elapsed = time.perf_counter() - t

                csv_w.write(dict(exp_id=EXP_ID, dataset="MNIST",
                    n_total=n_full, n_train=n_tr, n_test=n_test_actual,
                    d=d, n_classes=n_cls, model="ML-MSVM Arc m=1 L=1",
                    kernel="varies", L=-1, m=-1, P=P,
                    split_id=seed, acc=acc, time_s=elapsed))
                print(f"  {n_tr:>9,}  {seed:>4d}  {acc:.4f}  {elapsed:>7.2f}s",
                      flush=True)

        print(f"\n  Done. {hms(time.perf_counter()-t_start)}")
        print(f"  Results appended to: {csv_path}")
    finally:
        sys.stdout = tee._stream
        tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.log_dir, exist_ok=True); os.makedirs(a.csv_dir, exist_ok=True)
    run(os.path.join(a.log_dir, f"exp9_arccos_rerun_{ts}.txt"),
        os.path.join(a.csv_dir, "exp9_scalability_timing.csv"))
