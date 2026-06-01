"""
exp1_width_analysis.py
Width analysis — RECOVERY VERSION
Skips datasets already completed (Wine, Breast Cancer, Ionosphere, Sonar).
Adds graceful try/except around each dataset load so one failure doesn't abort.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   import_ml_msvm, ms_for, hms, banner)

ML_MSVM = import_ml_msvm()
P       = 1000
EXP_ID  = "exp1_width_analysis"
DEPTHS  = [1, 2, 3, 4]
KERNELS = [("rbf","RBF"),("arc_cosine","ArcCos")]

DATASETS = [
    ("wine",          "Wine",           1, 10),
    ("breast_cancer", "Breast Cancer",  1, 10),
    ("ionosphere",    "Ionosphere",     1, 10),
    ("sonar",         "Sonar",          1, 10),
    ("glass",         "Glass",          1, 10),
    ("magic",         "Magic",          2, 10),
    ("spambase",      "Spambase",       2, 10),
    ("covertype_sub", "Cover Type",     2, 10),
    ("mnist",         "MNIST",          3,  5),
    ("fashion",       "Fashion-MNIST",  3,  5),
]
# Already completed in previous run — skip to save time
ALREADY_DONE = {"Wine", "Breast Cancer", "Ionosphere", "Sonar"}


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp1 — Width Analysis (recovery: skipping already-done datasets)",
               f"P={P}  L={DEPTHS}  m={{1,2,3,d/4,d/2,d}}",
               f"Skipping: {sorted(ALREADY_DONE)}")
        t_start = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            if name in ALREADY_DONE:
                print(f"  [{name}] already completed — skipping.", flush=True)
                continue

            # ── graceful dataset load ────────────────────────────────────────
            try:
                X, y = load(tag)
            except Exception as e:
                print(f"\n  [{name}] LOAD FAILED: {e}. Skipping.", flush=True)
                continue

            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            ms = ms_for(d)
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, K={n_cls})  m={ms}  {reps}×{n_tr}/{n_te}")
            print(f"{'='*72}")
            print(f"  {'label':<36s} {'ker':>7s} {'L':>2s} {'m':>4s} {'acc':>8s} {'std':>6s} {'t/run':>8s}")
            print(f"  {'─'*72}")

            for kernel, kl in KERNELS:
                for L in DEPTHS:
                    for m in ms:
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
                                n_classes=n_cls, model=f"{kl}_L{L}_m{m}",
                                kernel=kernel, L=L, m=m, P=P, split_id=i,
                                acc=accs[-1], time_s=ts[-1]))
                        print(f"  {f'{kl} L={L} m={m}':<36s} {kl:>7s} {L:>2d} {m:>4d} "
                              f"{np.mean(accs):.4f} {np.std(accs):.4f} "
                              f"{np.mean(ts):>7.2f}s", flush=True)

                    if kernel == "arc_cosine" and L == 2:
                        accs2, ts2 = [], []
                        for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                            clf = ML_MSVM(num_layers=2, svms_per_block=1,
                                C_values=[1.0], rff_features=P,
                                kernel="arc_cosine", arc_cosine_degree=1,
                                final_C=1.0, normalize_inter_layer=False,
                                random_state=0)
                            pp = Pipeline([("sc", StandardScaler()), ("clf", clf)])
                            t = time.perf_counter()
                            pp.fit(Xtr, ytr)
                            accs2.append(pp.score(Xte, yte))
                            ts2.append(time.perf_counter() - t)
                            csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                                n_total=n, n_train=n_tr, n_test=n_te, d=d,
                                n_classes=n_cls, model="ArcCos_L2_m1_noNorm",
                                kernel="arc_cosine_nonorm", L=2, m=1, P=P,
                                split_id=i, acc=accs2[-1], time_s=ts2[-1]))
                        print(f"  {'ArcCos L=2 m=1 noNorm':<36s} {'arc':>7s} {'2':>2s} {'1':>4s} "
                              f"{np.mean(accs2):.4f} {np.std(accs2):.4f} "
                              f"{np.mean(ts2):>7.2f}s  [ablation]", flush=True)

        print(f"\n  Exp1 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp1_width_analysis_{ts}.txt"),
        os.path.join(a.csv_dir, "exp1_width_analysis.csv"))