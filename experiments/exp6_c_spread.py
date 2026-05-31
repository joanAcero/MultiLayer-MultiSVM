"""
exp6_c_spread.py — C-Spread Ablation
=====================================
Hypothesis
----------
The logspace spread of regularisation constants C1,...,Cm across the m
parallel block SVMs is essential for making m > 1 useful. Without spread
(all identical C), the m weight vectors point in nearly the same direction
and the architecture degenerates toward m=1. A wide logspace spread allows
different SVMs to capture structure at different regularisation scales,
producing complementary weight vectors that enrich the inter-layer representation.

Five spread strategies are tested with m=4 SVMs per block at L=2:
  same-1.0 : [1.0, 1.0, 1.0, 1.0]          — no diversity
  same-0.1 : [0.1, 0.1, 0.1, 0.1]          — no diversity, small C
  narrow   : logspace(−1, +1, 4)            — [0.1, 0.46, 2.15, 10]
  default  : logspace(−2, +1, 4)            — [0.01, 0.10, 1.0, 10]   (design choice)
  wide     : logspace(−3, +2, 4)            — [0.001, 0.046, 2.15, 100]

Protocol: 7× 90/10 for Magic/Spambase; 3× 10k/2k for MNIST.
Fixed: L=2, m=4, P=1000, kernel=RBF.
Output: logs/exp6_c_spread_<ts>.txt  +  results/exp6_c_spread.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, import_ml_msvm, hms, banner)

ML_MSVM = import_ml_msvm()
EXP_ID  = "exp6_c_spread"
P, L, M = 1000, 2, 4

SPREADS = [
    ("same-1.0", [1.0, 1.0, 1.0, 1.0]),
    ("same-0.1", [0.1, 0.1, 0.1, 0.1]),
    ("narrow",   list(np.logspace(-1, 1, 4))),
    ("default",  list(np.logspace(-2, 1, 4))),
    ("wide",     list(np.logspace(-3, 2, 4))),
]
DATASETS = [
    ("magic",    "Magic",    2,  7),
    ("spambase", "Spambase", 2,  7),
    ("mnist",    "MNIST",    3,  3),
]


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 6 — C-Spread Ablation",
               "Hypothesis: logspace C diversity is essential for m > 1 to help",
               f"Fixed: L={L}, m={M}, P={P}, kernel=RBF",
               "Spread strategies: same-1.0, same-0.1, narrow, default, wide")
        t_start = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*70}")
            print(f"  {name}  (n={n}, d={d})  |  {reps}× {n_tr}/{n_te}")
            print(f"{'='*70}")
            print(f"  {'spread':<12s}  {'C values':<36s}  {'acc':>8s}  {'std':>6s}  {'t/run':>8s}")
            print(f"  {'─'*72}")

            for spread_name, c_vals in SPREADS:
                accs, ts = [], []
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    clf = ML_MSVM(num_layers=L, svms_per_block=M,
                        C_values=c_vals, rff_features=P, kernel="rbf",
                        final_C=1.0, random_state=0)
                    pp = Pipeline([("sc", StandardScaler()), ("clf", clf)])
                    t = time.perf_counter()
                    pp.fit(Xtr, ytr)
                    accs.append(pp.score(Xte, yte))
                    ts.append(time.perf_counter() - t)
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                        n_total=n, n_train=n_tr, n_test=n_te, d=d,
                        n_classes=n_cls, model=f"RBF_L{L}_m{M}_{spread_name}",
                        kernel="rbf", L=L, m=M, P=P, split_id=i,
                        acc=accs[-1], time_s=ts[-1]))
                c_str = "[" + ", ".join(f"{c:.3f}" for c in c_vals) + "]"
                print(f"  {spread_name:<12s}  {c_str:<36s}  "
                      f"{np.mean(accs):.4f}  {np.std(accs):.4f}  "
                      f"{np.mean(ts):>7.2f}s", flush=True)

        print(f"\n  Experiment 6 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp6_c_spread_{ts}.txt"),
        os.path.join(a.csv_dir, "exp6_c_spread.csv"))
