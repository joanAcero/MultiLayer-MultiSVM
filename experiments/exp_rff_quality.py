"""
exp_rff_quality.py — Monte Carlo vs Quasi-Monte Carlo random features.

Question: can better spectral sampling close the accuracy gap to the exact RBF SVM,
and does it help arc-cosine features too?

RFF modes compared:
  standard    i.i.d. Gaussian Ω — current baseline.
  orf         Orthogonal Random Features (Yu et al. 2016). Same marginal
              distribution as Gaussian but inter-feature orthogonality reduces
              approximation variance by ~30-50 % for same P.
  qmc         Quasi-Monte Carlo via Sobol low-discrepancy sequence. Provides
              better spectral coverage; equivalent to ~3× more standard features.

Both kernels, varying P from 250 to 5000. L=1 (depth is not the variable here).
Exact RBF SVM included as oracle (skipped above RBF_N_LIMIT).
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, make_splits, hms, banner, RBF_N_LIMIT
sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import QMC_MLMSVMClassifier

EXP_ID   = "exp_rff_quality"
P_VALUES = [250, 500, 1000, 2000, 5000]
RFF_MODES = ["standard", "orf", "qmc"]
KERNELS   = [("rbf", "RBF"), ("arc_cosine", "Arc")]
DATASETS  = [
    ("mnist",    "MNIST",    3, 5),
    ("spambase", "Spambase", 2, 7),
]


def make_model(kernel, P, mode):
    clf = QMC_MLMSVMClassifier(
        num_layers=1, svms_per_block=1, rff_features=P,
        kernel=kernel, arc_cosine_degree=1,
        final_C=1.0, C_values=[1.0], random_state=0,
        rff_mode=mode,
        normalize_inter_layer=(kernel == "arc_cosine"),
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Monte Carlo vs Quasi-Monte Carlo RFF",
               "Q: does better spectral sampling close the gap to exact RBF SVM?",
               f"Modes: {RFF_MODES}",
               f"P={P_VALUES}  L=1  m=1  Both kernels")
        t0 = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            try:
                X, y = load(tag)
            except Exception as e:
                print(f"\n  [{name}] LOAD FAILED: {e}", flush=True); continue
            d, n, K = X.shape[1], len(y), len(np.unique(y))
            splits  = make_splits(X, y, regime, reps)
            n_tr    = len(splits[0][0])
            n_te    = len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, K={K})  {reps}×{n_tr}/{n_te}")
            print(f"{'='*72}")
            print(f"  {'model':<32s} {'P':>5s} {'mode':>8s} {'acc':>8s} {'std':>7s} {'t/run':>8s}")
            print("  " + "─"*68)

            # ── Exact RBF SVM oracle ─────────────────────────────────────
            if n_tr <= RBF_N_LIMIT:
                try:
                    exact = Pipeline([("scaler", StandardScaler()),
                                      ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))])
                    accs, ts = [], []
                    for Xtr, ytr, Xte, yte in splits:
                        mc = clone(exact); t = time.perf_counter()
                        mc.fit(Xtr, ytr); accs.append(mc.score(Xte, yte))
                        ts.append(time.perf_counter()-t)
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                            n_total=n, n_train=n_tr, n_test=n_te, d=d, n_classes=K,
                            model="RBF_exact", kernel="rbf", L=-1, m=-1, P=-1,
                            split_id=len(accs)-1, acc=accs[-1], time_s=ts[-1]))
                    print(f"  {'RBF SVM exact':<32s} {'-':>5s} {'oracle':>8s}"
                          f" {np.mean(accs):.4f} {np.std(accs):.4f}"
                          f" {np.mean(ts):>7.1f}s  ← oracle")
                except Exception as e:
                    print(f"  RBF exact FAILED: {e}")
            else:
                print(f"  RBF SVM exact                    (n_train={n_tr} > {RBF_N_LIMIT}: SKIP)")

            # ── RFF variants ─────────────────────────────────────────────
            for kernel, kl in KERNELS:
                for P in P_VALUES:
                    for mode in RFF_MODES:
                        label = f"{kl}_L1_m1_P{P}_{mode}"
                        try:
                            model = make_model(kernel, P, mode)
                            accs, ts = [], []
                            for Xtr, ytr, Xte, yte in splits:
                                mc = clone(model); t = time.perf_counter()
                                mc.fit(Xtr, ytr); accs.append(mc.score(Xte, yte))
                                ts.append(time.perf_counter()-t)
                                csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                                    n_total=n, n_train=n_tr, n_test=n_te, d=d, n_classes=K,
                                    model=label, kernel=kernel, L=1, m=1, P=P,
                                    split_id=len(accs)-1, acc=accs[-1], time_s=ts[-1]))
                            print(f"  {label:<32s} {P:>5d} {mode:>8s}"
                                  f" {np.mean(accs):.4f} {np.std(accs):.4f}"
                                  f" {np.mean(ts):>7.1f}s", flush=True)
                        except Exception as e:
                            print(f"  {label:<32s} FAILED: {e}", flush=True)

        print(f"\n  exp_rff_quality complete. {hms(time.perf_counter()-t0)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_rff_quality_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_rff_quality.csv"))
