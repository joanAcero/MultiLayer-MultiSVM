"""
exp5_depth_p_analysis.py — Depth × P Interaction
=================================================
Hypothesis A (depth)
--------------------
On low-complexity tabular data (Magic, d=10), L=1 is already optimal; extra
layers compound approximation error without adding representational benefit.
On high-complexity image data (MNIST, d=784), deeper architectures (L=3, 4)
with sufficient P provide measurable accuracy gains. The interaction effect
is: depth benefit is conditional on P being large enough to support it.

Hypothesis B (P — approximation quality)
-----------------------------------------
P saturation scales with the complexity of the decision boundary, not simply
with d. Magic saturates near P=500; Spambase and MNIST require P ≈ 2000.
Below saturation, increasing L cannot compensate for poor approximation.

Protocol: 5× 90/10 for Magic/Spambase; 3× 10k/2k for MNIST.
Varied: L ∈ {1,2,3,4}  ×  P ∈ {250,500,1000,2000,3000,5000}
        × kernel ∈ {RBF (m=2), ArcCos (m=1)}.
Output: logs/exp5_depth_p_analysis_<ts>.txt  +  results/exp5_depth_p_analysis.csv
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
EXP_ID   = "exp5_depth_p_analysis"
DEPTHS   = [1, 2, 3, 4]
P_VALUES = [250, 500, 1000, 2000, 3000, 5000]

DATASETS = [
    # tag       display    regime  reps  m_rbf  m_arc
    ("magic",    "Magic",    2,     5,    2,     1),
    ("spambase", "Spambase", 2,     5,    2,     1),
    ("mnist",    "MNIST",    3,     3,    2,     1),
]


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 5 — Depth × P Interaction",
               "Hyp A: depth benefit conditional on data complexity",
               "Hyp B: P saturation driven by boundary complexity, not d",
               f"L={DEPTHS}  ×  P={P_VALUES}  ×  {{RBF m=2, ArcCos m=1}}")
        t_start = time.perf_counter()

        for tag, name, regime, reps, m_rbf, m_arc in DATASETS:
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            for kernel, kl, m in [("rbf", "RBF", m_rbf),
                                   ("arc_cosine", "ArcCos", m_arc)]:
                print(f"\n{'='*72}")
                print(f"  {name}  kernel={kl}  m={m}  (n={n}, d={d})  "
                      f"{reps}× {n_tr}/{n_te}")
                print(f"{'='*72}")
                print(f"  {'':6s}", end="")
                for P in P_VALUES:
                    print(f"   P={P:<5d}  acc   ±   time", end="")
                print()
                print(f"  {'─'*74}")

                for L in DEPTHS:
                    print(f"  L={L:<4d}", end="")
                    for P in P_VALUES:
                        accs, ts = [], []
                        model = make_mlmsvm(ML_MSVM, L, m, P, kernel)
                        for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                            mc = clone(model)
                            t = time.perf_counter()
                            mc.fit(Xtr, ytr)
                            accs.append(mc.score(Xte, yte))
                            ts.append(time.perf_counter() - t)
                            csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                                n_total=n, n_train=n_tr, n_test=n_te, d=d,
                                n_classes=n_cls, model=f"{kl}_L{L}_m{m}_P{P}",
                                kernel=kernel, L=L, m=m, P=P, split_id=i,
                                acc=accs[-1], time_s=ts[-1]))
                        print(f"   {np.mean(accs):.4f} ±{np.std(accs):.4f} "
                              f"{np.mean(ts):4.1f}s", end="", flush=True)
                    print()

        print(f"\n  Experiment 5 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp5_depth_p_analysis_{ts}.txt"),
        os.path.join(a.csv_dir, "exp5_depth_p_analysis.csv"))
