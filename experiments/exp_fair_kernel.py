"""
exp_fair_kernel.py — Unbiased Arc-cosine vs RBF kernel comparison.

Previous experiments used DIFFERENT canonical configs for the two kernels
(Arc m=1 L=1 vs RBF m=2 L=2) based on exp1 optima — which conflates the
config choice with the kernel choice.

This experiment tests ALL (kernel × m × L) combinations with:
  - Identical C=1.0 for all block SVMs (post-exp6 fix)
  - Identical m, L, P, solver settings for both kernels
  - Baselines: Linear SVM, Flat RFF (L=0) for both kernels

Research questions:
  1. Is Arc > RBF consistent across all (m, L) configurations?
  2. Does the optimal (m, L) differ by kernel?
  3. Is there a configuration × kernel interaction?
  4. Why is RBF slower? (Report time per fit alongside accuracy)
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, import_ml_msvm,
                   make_linear_svm, hms, banner, RBF_N_LIMIT)

ML_MSVM = import_ml_msvm()
EXP_ID  = "exp_fair_kernel"
P       = 1000

# All (m, L) configs — tested identically for both kernels
CONFIGS = [
    (1, 1, "m1L1"),
    (1, 2, "m1L2"),
    (1, 3, "m1L3"),
    (2, 1, "m2L1"),
    (2, 2, "m2L2"),
]
KERNELS = [("arc_cosine", "Arc"), ("rbf", "RBF")]

DATASETS = [
    ("magic",         "Magic",       2, 10),
    ("spambase",      "Spambase",    2, 10),
    ("covertype_sub", "Cover Type",  2, 10),
    ("mnist",         "MNIST",       3,  5),
    ("fashion",       "Fashion",     3,  5),
]


def make_mlmsvm(kernel, m, L):
    """C=1.0 for all block SVMs (post-exp6 fix: uniform C is best)."""
    clf = ML_MSVM(
        num_layers=L, svms_per_block=m, C_values=[1.0] * m,
        rff_features=P, final_C=1.0,
        kernel=kernel, arc_cosine_degree=1, random_state=0,
        normalize_inter_layer=(kernel == "arc_cosine"),
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_flat(kernel):
    """L=0 flat RFF baseline."""
    clf = ML_MSVM(
        num_layers=0, svms_per_block=1, C_values=[1.0],
        rff_features=P, final_C=1.0,
        kernel=kernel, arc_cosine_degree=1, random_state=0,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Fair Kernel Comparison (Arc vs RBF, equal configs)",
               "All block SVMs at C=1.0. Same (m, L) for both kernels.",
               f"Configs: {[c[2] for c in CONFIGS]}  |  P={P}",
               "Q: is Arc > RBF consistent? Does optimal config differ by kernel?")
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
            print(f"  {'model':<30s} {'ker':>3s} {'m':>2s} {'L':>2s} {'acc':>8s} {'std':>7s} {'t/run':>8s}")
            print("  " + "─"*65)

            def eval_model(model, label, kernel, m, L):
                accs, ts = [], []
                for Xtr, ytr, Xte, yte in splits:
                    mc = clone(model); t = time.perf_counter()
                    mc.fit(Xtr, ytr); accs.append(mc.score(Xte, yte))
                    ts.append(time.perf_counter()-t)
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                        n_total=n, n_train=n_tr, n_test=n_te, d=d, n_classes=K,
                        model=label, kernel=kernel, L=L, m=m, P=P,
                        split_id=len(accs)-1, acc=accs[-1], time_s=ts[-1]))
                kl = "Arc" if kernel == "arc_cosine" else "RBF"
                print(f"  {label:<30s} {kl:>3s} {m:>2s} {L:>2s}"
                      f" {np.mean(accs):.4f} {np.std(accs):.4f}"
                      f" {np.mean(ts):>7.2f}s", flush=True)

            # ── Linear SVM baseline ──────────────────────────────────────
            try:
                eval_model(make_linear_svm(), "Linear SVM", "linear", "-", "-")
            except Exception as e:
                print(f"  Linear SVM FAILED: {e}")

            # ── Exact RBF oracle (if feasible) ───────────────────────────
            if n_tr <= RBF_N_LIMIT:
                try:
                    exact = Pipeline([("scaler", StandardScaler()),
                                      ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))])
                    eval_model(exact, "RBF SVM exact", "rbf", "-", "-")
                except Exception as e:
                    print(f"  RBF exact FAILED: {e}")

            # ── Flat baselines for both kernels ──────────────────────────
            for kernel, kl in KERNELS:
                lbl = f"Flat {kl} (L=0)"
                try:
                    eval_model(make_flat(kernel), lbl, kernel, "0", "0")
                except Exception as e:
                    print(f"  {lbl} FAILED: {e}")

            # ── All (m, L) × kernel combinations ─────────────────────────
            for m, L, cfg_lbl in CONFIGS:
                for kernel, kl in KERNELS:
                    lbl = f"{kl}_{cfg_lbl}_C1"
                    try:
                        eval_model(make_mlmsvm(kernel, m, L), lbl, kernel,
                                   str(m), str(L))
                    except Exception as e:
                        print(f"  {lbl} FAILED: {e}")

        print(f"\n  exp_fair_kernel complete. {hms(time.perf_counter()-t0)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_fair_kernel_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_fair_kernel.csv"))
