"""
exp8_arccosine_degree.py — Arc-Cosine Kernel Degree Ablation
=============================================================
Hypothesis
----------
The arc-cosine kernel of degree n corresponds to the inner product induced
by an infinite neural network with activation x^n (Cho & Saul 2010):
  degree 0: step function  — captures sign-based angular similarity
  degree 1: ReLU           — standard, linear in activations
  degree 2: quadratic      — smoother, amplifies large activations
Degree 1 is expected to be optimal for tabular data (matches deep learning
priors). Degree 0 may be more robust in low-n high-d settings by discarding
magnitude information. Degree 2 risks over-smoothing on noisy data.

Protocol: 10× 90/10 for R1/R2; 5× 10k/2k for R3.
Fixed: m=1, P=1000. L ∈ {1, 2}. Degrees ∈ {0, 1, 2}.
Datasets: Wine, Sonar (R1), Magic, Spambase (R2), MNIST (R3).
Output: logs/exp8_arccosine_degree_<ts>.txt  +  results/exp8_arccosine_degree.csv
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
EXP_ID  = "exp8_arccosine_degree"
P, M    = 1000, 1
DEGREES = [0, 1, 2]
DEPTHS  = [1, 2]

DATASETS = [
    ("wine",     "Wine",     1, 10),
    ("sonar",    "Sonar",    1, 10),
    ("magic",    "Magic",    2, 10),
    ("spambase", "Spambase", 2, 10),
    ("mnist",    "MNIST",    3,  5),
]


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 8 — Arc-Cosine Degree Ablation",
               "Hypothesis: degree=1 (ReLU) optimal on tabular; degree=0 on high-d low-n",
               f"Degrees: {DEGREES}  |  L={DEPTHS}  |  m={M}  |  P={P}")
        t_start = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*65}")
            print(f"  {name}  (n={n}, d={d})  |  {reps}× {n_tr}/{n_te}")
            print(f"{'='*65}")
            print(f"  {'model':<28s}  {'deg':>4s}  {'L':>2s}  "
                  f"{'acc':>8s}  {'std':>6s}  {'t/run':>8s}")
            print(f"  {'─'*62}")

            for degree in DEGREES:
                for L in DEPTHS:
                    accs, ts = [], []
                    for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                        clf = ML_MSVM(num_layers=L, svms_per_block=M,
                            C_values=[1.0], rff_features=P,
                            kernel="arc_cosine", arc_cosine_degree=degree,
                            final_C=1.0, random_state=0)
                        pp = Pipeline([("sc", StandardScaler()), ("clf", clf)])
                        t = time.perf_counter()
                        pp.fit(Xtr, ytr)
                        accs.append(pp.score(Xte, yte))
                        ts.append(time.perf_counter() - t)
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                            n_total=n, n_train=n_tr, n_test=n_te, d=d,
                            n_classes=n_cls, model=f"ArcCos_deg{degree}_L{L}_m{M}",
                            kernel=f"arc_cosine_deg{degree}", L=L, m=M, P=P,
                            split_id=i, acc=accs[-1], time_s=ts[-1]))
                    print(f"  {'ArcCos deg=' + str(degree) + ' L=' + str(L):<28s}  "
                          f"{degree:>4d}  {L:>2d}  "
                          f"{np.mean(accs):.4f}  {np.std(accs):.4f}  "
                          f"{np.mean(ts):>7.2f}s", flush=True)

        print(f"\n  Experiment 8 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp8_arccosine_degree_{ts}.txt"),
        os.path.join(a.csv_dir, "exp8_arccosine_degree.csv"))
