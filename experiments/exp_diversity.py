"""
exp_diversity.py — Block diversity mechanism ablation.

Question: when m > 1, which diversity strategy actually helps?
Four strategies:
  c_spread      Original logspace-C scheme (exposed as harmful by exp6).
  same_c        All SVMs at C=1.0, same data. Predicted rank-1 collapse.
  bootstrap     Each SVM trains on 80 % bootstrap subsample; C=1.0.
  feature_subset Each SVM uses floor(P/m) random features; C=1.0.

Both kernels tested across m={1,2,3,4,6}, L=1 and L=2.
Datasets that showed non-trivial width effects: Spambase, CoverType-sub, MNIST.
Magic included as a simple baseline where nothing helps.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, make_splits, import_ml_msvm, hms, banner

# Import extension — expects mlsvm_extensions.py next to this script
sys.path.insert(0, str(Path(__file__).parent))
from mlsvm_extensions import DiverseMLMSVM

EXP_ID  = "exp_diversity"
P       = 1000
DEPTHS  = [1, 2,3]
MS      = [1, 2, 3, 4, 6]
KERNELS = [("arc_cosine", "Arc"), ("rbf", "RBF")]
MODES   = ["c_spread", "same_c", "bootstrap", "feature_subset"]
DATASETS = [
    ("spambase",      "Spambase",    2, 10),
    ("covertype_sub", "Cover Type",  2, 10),
    ("mnist",         "MNIST",       3,  5),
]


def make_model(kernel, L, m, mode):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=kernel, arc_cosine_degree=1,
        final_C=1.0, random_state=0,
        diversity_mode=mode,
        normalize_inter_layer=(kernel == "arc_cosine"),
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Diversity Mechanism Ablation",
               "Q: does m>1 help when diversity comes from bootstrap/feature-subset?",
               f"Modes: {MODES}",
               f"m={MS}  L={DEPTHS}  P={P}  Kernels: RBF, Arc-cosine")
        t0 = time.perf_counter()

        for tag, name, regime, reps in DATASETS:
            try:
                X, y = load(tag)
            except Exception as e:
                print(f"\n  [{name}] LOAD FAILED: {e}. Skipping.", flush=True); continue
            d, n, K = X.shape[1], len(y), len(np.unique(y))
            splits  = make_splits(X, y, regime, reps)
            n_tr    = len(splits[0][0])
            n_te    = len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, K={K})  {reps}×{n_tr}/{n_te}")
            print(f"{'='*72}")
            header = f"  {'label':<34s} {'ker':>3s} {'L':>2s} {'m':>2s} {'mode':>14s} {'acc':>8s} {'std':>6s} {'t/run':>8s}"
            print(header); print("  " + "─"*(len(header)-2))

            for kernel, kl in KERNELS:
                for L in DEPTHS:
                    for m in MS:
                        for mode in MODES:
                            if m == 1 and mode in ("same_c","bootstrap","feature_subset"):
                                # For m=1 all modes collapse to the same single SVM
                                if mode != "same_c":
                                    continue   # only run once when m=1
                            label = f"{kl}_L{L}_m{m}_{mode}"
                            try:
                                accs, ts = [], []
                                model = make_model(kernel, L, m, mode)
                                for Xtr, ytr, Xte, yte in splits:
                                    mc = clone(model)
                                    t = time.perf_counter()
                                    mc.fit(Xtr, ytr); accs.append(mc.score(Xte, yte))
                                    ts.append(time.perf_counter() - t)
                                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                                        n_total=n, n_train=n_tr, n_test=n_te,
                                        d=d, n_classes=K, model=label, kernel=kernel,
                                        L=L, m=m, P=P, split_id=len(accs)-1,
                                        acc=accs[-1], time_s=ts[-1]))
                                print(f"  {label:<34s} {kl:>3s} {L:>2d} {m:>2d} {mode:>14s}"
                                      f" {np.mean(accs):.4f} {np.std(accs):.4f}"
                                      f" {np.mean(ts):>7.2f}s", flush=True)
                            except Exception as e:
                                print(f"  {label:<34s} FAILED: {e}", flush=True)

        print(f"\n  exp_diversity complete. {hms(time.perf_counter()-t0)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_diversity_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_diversity.csv"))
