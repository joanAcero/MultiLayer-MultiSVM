"""
exp4_learning_curves.py — Learning Curves (n-Scaling)
======================================================
Hypothesis
----------
ML-MSVM accuracy improves monotonically with training set size and retains
its advantage over the Linear SVM at all scales. The exact RBF SVM is
competitive at small n but becomes infeasible above n≈10k; ML-MSVM fills
this gap. The flat RFF SVM (depth-0 ablation) provides a ceiling check:
any gain of ML-MSVM over it is attributable purely to architectural depth.
Training time for ML-MSVM scales linearly with n (primal solver), directly
supporting the scalability claim.

Protocol
--------
  Datasets : MNIST (d=784, regime 3) and SUSY (d=18, large-scale).
  Fixed test set: 5k points for MNIST, 50k for SUSY. Same test set for all n.
  n_train  : MNIST: [500, 1k, 2k, 5k, 10k, 20k, 40k, 60k]
             SUSY:  [1k, 2k, 5k, 10k, 25k, 50k, 100k, 200k, 400k]
  3 independent random subsamples per n_train point (different train seeds).
  Models   : Linear SVM, Exact RBF SVM (n ≤ 10k only), Flat RFF,
             ML-MSVM RBF m=2 L=2, ML-MSVM ArcCos m=1 L=1.

Output: logs/exp4_lcurves_<ts>.txt  +  results/exp4_learning_curves.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_mlmsvm, make_linear_svm,
                   make_rbf_svm, make_flat_rff, import_ml_msvm, hms, banner)

ML_MSVM   = import_ml_msvm()
P         = 1000
EXP_ID    = "exp4_lcurves"
N_SEEDS   = 3   # random subsamples per n_train point
RBF_LIMIT = 10_000

DATASETS = {
    "mnist": {
        "display":  "MNIST",
        "n_trains": [500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000],
        "n_test":   5_000,
    },
    "susy": {
        "display":  "SUSY",
        "n_trains": [1_000, 2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 400_000],
        "n_test":   50_000,
    },
}

MODELS = [
    ("Linear SVM",          lambda: make_linear_svm(), False),
    ("RBF SVM (exact)",     lambda: make_rbf_svm(),    True),   # True = skip if n>limit
    ("Flat RFF RBF (L=0)",  lambda: make_flat_rff(ML_MSVM, P), False),
    ("ML-MSVM RBF m=2 L=2", lambda: make_mlmsvm(ML_MSVM, 2, 2, P, "rbf"), False),
    ("ML-MSVM Arc m=1 L=1", lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine"), False),
]


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)

    try:
        banner("Experiment 4 — Learning Curves (n-scaling)",
               "Datasets: MNIST (d=784), SUSY (d=18)",
               "Fixed test set; accuracy and time reported as function of n_train.",
               f"P={P}  {N_SEEDS} random subsamples per n_train")

        t_total = time.perf_counter()

        for tag, cfg in DATASETS.items():
            name = cfg["display"]
            print(f"\n  Loading {name}...", end=" ", flush=True)
            try:
                X, y = load(tag, verbose=False)
                print(f"done (shape={X.shape})")
            except Exception as e:
                print(f"FAILED ({e}). Skipping.")
                continue

            d, n_full = X.shape[1], len(y)
            n_test    = cfg["n_test"]
            n_cls     = len(np.unique(y))

            # Fixed held-out test set (same for all n_train)
            Xtr_pool, Xte, ytr_pool, yte = train_test_split(
                X, y, test_size=n_test, stratify=y, random_state=42)

            print(f"\n{'='*80}")
            print(f"  {name}  (n={n_full:,}, d={d})  |  test set fixed: {n_test:,}")
            print(f"{'='*80}")

            # Column header
            model_names = [lbl for lbl, _, _ in MODELS]
            print(f"  {'n_train':>8s}", end="")
            for lbl, _, skip in MODELS:
                short = lbl[:16]
                print(f"  {short:>16s}", end="")
            print()
            print(f"  {'─'*80}")

            for n_train in cfg["n_trains"]:
                if n_train > len(ytr_pool):
                    continue
                print(f"  {n_train:>8,}", end="", flush=True)

                for model_label, model_fn, is_rbf_exact in MODELS:
                    if is_rbf_exact and n_train > RBF_LIMIT:
                        print(f"  {'SKIP':>16s}", end="", flush=True)
                        continue

                    seed_accs, seed_times = [], []
                    for seed in range(N_SEEDS):
                        # Draw n_train stratified from pool
                        if n_train == len(ytr_pool):
                            Xtr, ytr = Xtr_pool, ytr_pool
                        else:
                            sss = StratifiedShuffleSplit(n_splits=1,
                                train_size=n_train, random_state=seed)
                            idx, _ = next(sss.split(Xtr_pool, ytr_pool))
                            Xtr, ytr = Xtr_pool[idx], ytr_pool[idx]

                        mc = clone(model_fn())
                        t0 = time.perf_counter()
                        mc.fit(Xtr, ytr)
                        acc = mc.score(Xte, yte)
                        t = time.perf_counter() - t0
                        seed_accs.append(acc)
                        seed_times.append(t)
                        csv_w.write(dict(
                            exp_id=EXP_ID, dataset=name,
                            n_total=n_full, n_train=n_train, n_test=n_test,
                            d=d, n_classes=n_cls,
                            model=model_label, kernel="", L=-1, m=-1, P=P,
                            split_id=seed, acc=acc, time_s=t,
                        ))

                    mu  = np.mean(seed_accs)
                    std = np.std(seed_accs)
                    tm  = np.mean(seed_times)
                    cell = f"{mu:.4f}({tm:.1f}s)"
                    print(f"  {cell:>16s}", end="", flush=True)

                print(flush=True)

        elapsed = hms(time.perf_counter() - t_total)
        print(f"\n{'█'*60}")
        print(f"  Experiment 4 complete. Total: {elapsed}")
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
    run(os.path.join(args.log_dir, f"exp4_lcurves_{ts}.txt"),
        os.path.join(args.csv_dir, "exp4_learning_curves.csv"))
