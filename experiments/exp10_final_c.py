"""
exp10_final_c.py — Head SVM Regularisation Sensitivity
========================================================
Hypothesis
----------
The default final_C=1.0 for the head SVM is unjustified in theory. The head
trains on an m-dimensional inter-layer representation (m=2 for binary, m·K
for K-class), which is a very different space from the P-dimensional RFF
input to block SVMs. The head SVM with small m may behave differently under
regularisation: too-small C under-regularises a nearly-linearly-separable
space; too-large C overfits on the compressed representation. This experiment
tests whether the architecture is robust to this choice or whether final_C
should be tuned as a hyperparameter.

Varied: final_C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}
Fixed:  RBF m=2 L=2 P=1000 (best established config)
Protocol: 5× 90/10 for Magic/Spambase; 3× 10k/2k for MNIST.
Output: logs/exp10_final_c_<ts>.txt  +  results/exp10_final_c.csv
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

ML_MSVM   = import_ml_msvm()
EXP_ID    = "exp10_final_c"
P, L, M   = 1000, 2, 2
FINAL_CS  = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

DATASETS = [
    ("magic",    "Magic",    2, 5),
    ("spambase", "Spambase", 2, 5),
    ("mnist",    "MNIST",    3, 3),
]


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 10 — Head SVM final_C Sensitivity",
               f"Fixed: RBF m={M} L={L} P={P}",
               f"Varied: final_C ∈ {FINAL_CS}")
        t_start = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*58}")
            print(f"  {name}  (n={n}, d={d})  |  {reps}× {n_tr}/{n_te}")
            print(f"{'='*58}")
            print(f"  {'final_C':>10s}  {'acc':>8s}  {'std':>6s}  {'t/run':>8s}")
            print(f"  {'─'*38}")

            for fc in FINAL_CS:
                accs, ts = [], []
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    clf = ML_MSVM(num_layers=L, svms_per_block=M,
                        C_values=list(np.logspace(-2, 1, M)),
                        rff_features=P, kernel="rbf", final_C=fc, random_state=0)
                    pp = Pipeline([("sc", StandardScaler()), ("clf", clf)])
                    t = time.perf_counter()
                    pp.fit(Xtr, ytr)
                    accs.append(pp.score(Xte, yte))
                    ts.append(time.perf_counter() - t)
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                        n_total=n, n_train=n_tr, n_test=n_te, d=d,
                        n_classes=n_cls, model=f"RBF_L{L}_m{M}_finalC{fc}",
                        kernel="rbf", L=L, m=M, P=P, split_id=i,
                        acc=accs[-1], time_s=ts[-1]))
                print(f"  {fc:>10.3f}  {np.mean(accs):.4f}  {np.std(accs):.4f}  "
                      f"{np.mean(ts):>7.2f}s", flush=True)

        print(f"\n  Experiment 10 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp10_final_c_{ts}.txt"),
        os.path.join(a.csv_dir, "exp10_final_c.csv"))
