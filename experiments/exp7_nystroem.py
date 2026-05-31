"""
exp7_nystroem.py — Nystroem Approximation Comparison
=====================================================
Hypothesis
----------
The Nystroem method (data-dependent landmark approximation) has a tighter
approximation guarantee than RFF (data-independent) at the same approximation
dimension P: it uses actual data points as landmarks, which are better suited
to the data manifold. The question is whether ML-MSVM's architectural
advantage (composing multiple RFF layers) compensates for this approximation
deficit, and whether it does so across all P values.

At matched approximation size P, Nystroem + LinearSVC is the natural
scalable kernel SVM competitor. Nystroem training cost is O(n·P + P²·d)
versus O(n·P) per layer for RFF, so they are comparable in asymptotic cost.

Protocol: 10× 90/10 for R2; 5× 10k/2k for R3.
Tested at P ∈ {500, 1000, 2000} (approximation quality axis).
Datasets: all Regime 2 + MNIST (Regime 3).
Output: logs/exp7_nystroem_<ts>.txt  +  results/exp7_nystroem.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   make_flat_rff, make_nystroem_svm,
                   import_ml_msvm, hms, banner)

ML_MSVM  = import_ml_msvm()
EXP_ID   = "exp7_nystroem"
P_VALUES = [500, 1000, 2000]

DATASETS = [
    ("magic",         "Magic",         2, 10),
    ("spambase",      "Spambase",      2, 10),
    ("covertype_sub", "Cover Type",    2, 10),
    ("mnist",         "MNIST",         3,  5),
]


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 7 — Nystroem vs RFF Comparison",
               "Hypothesis: ML-MSVM depth compensates for Nystroem approximation advantage",
               f"P ∈ {P_VALUES}  |  All Regime 2 + MNIST")
        t_start = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d})  |  {reps}× {n_tr}/{n_te}")
            print(f"{'='*72}")

            for P in P_VALUES:
                print(f"\n  ── P = {P} ───────────────────────────────────────────────")
                print(f"  {'model':<38s}  {'acc':>8s}  {'std':>6s}  {'t/run':>8s}")
                print(f"  {'─'*62}")

                configs = [
                    ("Nystroem + LinearSVC",   make_nystroem_svm(P)),
                    ("Flat RFF RBF (L=0)",      make_flat_rff(ML_MSVM, P, "rbf")),
                    ("ML-MSVM RBF m=2 L=2",    make_mlmsvm(ML_MSVM, 2, 2, P, "rbf")),
                    ("ML-MSVM Arc m=1 L=1",    make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine")),
                ]
                for lbl, model in configs:
                    accs, ts = [], []
                    for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                        mc = clone(model)
                        t = time.perf_counter()
                        mc.fit(Xtr, ytr)
                        accs.append(mc.score(Xte, yte))
                        ts.append(time.perf_counter() - t)
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                            n_total=n, n_train=n_tr, n_test=n_te, d=d,
                            n_classes=n_cls, model=f"{lbl}_P{P}",
                            kernel="varies", L=-1, m=-1, P=P, split_id=i,
                            acc=accs[-1], time_s=ts[-1]))
                    print(f"  {lbl:<38s}  {np.mean(accs):.4f}  {np.std(accs):.4f}  "
                          f"{np.mean(ts):>7.2f}s", flush=True)

        print(f"\n  Experiment 7 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp7_nystroem_{ts}.txt"),
        os.path.join(a.csv_dir, "exp7_nystroem.csv"))
