"""
exp5_depth_p_analysis.py — Depth and P Analysis
================================================
Hypothesis A (depth)
--------------------
On low-complexity data (Magic, d=10), L=1 is already optimal; extra layers
add no representational benefit and may hurt due to compounding approximation
error. On high-complexity data (MNIST, d=784), deeper architectures with
L=3 or L=4 provide measurable gains, provided P is large enough to support
them. This produces an interaction effect: depth benefit is conditional on P.

Hypothesis B (P — approximation quality)
-----------------------------------------
P saturation scales with the complexity of the classification boundary, not
simply with d. Low-d datasets (Magic) saturate at P≈500. High-d datasets
(Spambase, MNIST) require P≈2000 before the approximation is adequate.
Below the saturation threshold, increasing L cannot compensate for poor
feature approximation; this gives a P × L interaction.

Protocol
--------
  Datasets : Magic (d=10, regime 2), Spambase (d=57, regime 2),
             MNIST (d=784, regime 3).
  Splits   : 5× 90/10 for Magic/Spambase; 3× 10k/2k for MNIST.
  Varied   : L ∈ {1,2,3,4}  ×  P ∈ {250,500,1000,2000,3000,5000}
             × kernel ∈ {rbf, arc_cosine} (best m per kernel from Exp 1).
  Best m   : RBF m=2, ArcCos m=1 (established from Exp 1).

Output: logs/exp5_depth_p_<ts>.txt  +  results/exp5_depth_p.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   import_ml_msvm, hms, banner)

ML_MSVM  = import_ml_msvm()
EXP_ID   = "exp5_depth_p"
DEPTHS   = [1, 2, 3, 4]
P_VALUES = [250, 500, 1000, 2000, 3000, 5000]

DATASETS = [
    # tag       display    regime  repeats  m_rbf  m_arc
    ("magic",    "Magic",    2,     5,       2,     1),
    ("spambase", "Spambase", 2,     5,       2,     1),
    ("mnist",    "MNIST",    3,     3,       2,     1),
]


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)

    try:
        banner("Experiment 5 — Depth × P Interaction",
               "Hypothesis A: Depth benefit conditional on data complexity",
               "Hypothesis B: P saturation scales with boundary complexity, not d",
               f"L ∈ {DEPTHS}  ×  P ∈ {P_VALUES}  ×  {{RBF, ArcCos}}")

        t_total = time.perf_counter()

        for tag, name, regime, repeats, m_rbf, m_arc in DATASETS:
            X, y = load(tag)
            d, n = X.shape[1], len(y)
            splits = make_splits(X, y, regime, repeats)
            n_train = len(splits[0][0])
            n_test  = len(splits[0][2])
            n_cls   = len(np.unique(y))

            for kernel, klabel, m in [("rbf", "RBF", m_rbf),
                                       ("arc_cosine", "ArcCos", m_arc)]:
                print(f"\n{'='*72}")
                print(f"  {name}  kernel={klabel}  m={m}  "
                      f"(n={n}, d={d})  {repeats}× {n_train}/{n_test}")
                print(f"{'='*72}")

                # Print P as columns, L as rows
                print(f"  {'':>6s}", end="")
                for P in P_VALUES:
                    print(f"  P={P:<5d} acc   ±    time", end="")
                print()
                print(f"  {'─'*72}")

                for L in DEPTHS:
                    print(f"  L={L:<4d}", end="")
                    for P in P_VALUES:
                        model = make_mlmsvm(ML_MSVM, L, m, P, kernel)
                        accs, times = [], []
                        for Xtr, ytr, Xte, yte in splits:
                            mc = clone(model)
                            t0 = time.perf_counter()
                            mc.fit(Xtr, ytr)
                            accs.append(mc.score(Xte, yte))
                            times.append(time.perf_counter() - t0)
                            csv_w.write(dict(
                                exp_id=EXP_ID, dataset=name,
                                n_total=n, n_train=n_train, n_test=n_test,
                                d=d, n_classes=n_cls,
                                model=f"{klabel}_L{L}_m{m}_P{P}",
                                kernel=kernel, L=L, m=m, P=P,
                                split_id=len(accs)-1,
                                acc=accs[-1], time_s=times[-1],
                            ))
                        mu  = np.mean(accs)
                        std = np.std(accs)
                        tm  = np.mean(times)
                        print(f"  {mu:.4f} ±{std:.4f} {tm:4.1f}s", end="", flush=True)
                    print()

                # Summary: best (L, P) combo
                # (Can't easily do in-loop; reader will extract from CSV)
                print(f"  => See CSV for full (L, P) grid.")

        elapsed = hms(time.perf_counter() - t_total)
        print(f"\n{'█'*60}")
        print(f"  Experiment 5 complete. Total: {elapsed}")
        print(f"{'█'*60}")

    finally:
        sys.stdout = tee._stream
        tee.close()
        csv_w.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(args.log_dir, f"exp5_depth_p_{ts}.txt"),
        os.path.join(args.csv_dir, "exp5_depth_p.csv"))
