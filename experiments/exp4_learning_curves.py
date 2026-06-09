"""
exp4_learning_curves.py — Learning Curves (n-scaling). MNIST + SUSY.
Accuracy(n) and time(n) for all models. Per-model try/except.
Models include BOTH flat baselines (RBF & Arc) so depth vs kernel can be
separated. Exact RBF SVM runs up to RBF_N_LIMIT.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_mlmsvm, make_linear_svm,
                   make_rbf_svm, make_flat_rff, import_ml_msvm,
                   RBF_N_LIMIT, hms, banner)

ML_MSVM  = import_ml_msvm()
P        = 1000
N_SEEDS  = 3
EXP_ID   = "exp4_learning_curves"

DATASETS = {
    "mnist": {"display": "MNIST",
              "n_trains": [500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000, 60_000],
              "n_test": 5_000},
    "susy":  {"display": "SUSY",
              "n_trains": [1_000, 2_000, 5_000, 10_000, 25_000,
                           50_000, 100_000, 200_000, 400_000],
              "n_test": 50_000},
}
MODELS = [
    ("Linear SVM",           lambda: make_linear_svm(),                            False),
    ("RBF SVM (exact)",      lambda: make_rbf_svm(),                               True),
    ("Flat RFF RBF (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "rbf"),            False),
    ("Flat RFF Arc (L=0)",   lambda: make_flat_rff(ML_MSVM, P, "arc_cosine"),     False),
    ("ML-MSVM RBF m=2 L=2",  lambda: make_mlmsvm(ML_MSVM, 2, 2, P, "rbf"),       False),
    ("ML-MSVM Arc m=1 L=1",  lambda: make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine"), False),
]


def run(log_path, csv_path, data_dir):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 4 — Learning Curves (n-scaling)",
               "MNIST (d=784) | SUSY (d=18). Accuracy AND time vs n_train.",
               f"Exact RBF SVM: n_train ≤ {RBF_N_LIMIT}  |  {N_SEEDS} seeds")
        t_start = time.perf_counter()

        for tag, cfg in DATASETS.items():
            name, n_test = cfg["display"], cfg["n_test"]
            try:
                X, y = load(tag, verbose=True, data_dir=data_dir)
            except Exception as e:
                print(f"\n  [{name}] LOAD FAILED: {e}. Skipping.", flush=True); continue
            d, n_full, n_cls = X.shape[1], len(y), len(np.unique(y))

            n_test_actual = min(n_test, int(0.15 * n_full))
            Xpool, Xte, ypool, yte = train_test_split(
                X, y, test_size=n_test_actual, stratify=y, random_state=42)
            n_pool = len(ypool)

            print(f"\n{'━'*76}")
            print(f"  {name}  d={d}  K={n_cls}  test={n_test_actual:,}")
            print(f"{'━'*76}")
            print(f"  {'n_train':>9s}  {'model':<24s}  {'acc':>8s}  {'std':>6s}  {'time':>8s}")
            print(f"  {'─'*64}")

            for n_tr in cfg["n_trains"]:
                if n_tr >= n_pool:
                    continue
                for lbl, model_fn, is_exact in MODELS:
                    if is_exact and n_tr > RBF_N_LIMIT:
                        print(f"  {n_tr:>9,}  {lbl:<24s}  {'SKIP':>8s}", flush=True)
                        continue
                    try:
                        accs, ts = [], []
                        for seed in range(N_SEEDS):
                            sss = StratifiedShuffleSplit(1, train_size=n_tr, random_state=seed)
                            idx, _ = next(sss.split(Xpool, ypool))
                            Xtr, ytr = Xpool[idx], ypool[idx]
                            mc = clone(model_fn())
                            t = time.perf_counter()
                            mc.fit(Xtr, ytr)
                            accs.append(mc.score(Xte, yte))
                            ts.append(time.perf_counter() - t)
                            csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                                n_total=n_full, n_train=n_tr, n_test=n_test_actual,
                                d=d, n_classes=n_cls, model=lbl, kernel="varies",
                                L=-1, m=-1, P=P, split_id=seed,
                                acc=accs[-1], time_s=ts[-1]))
                        print(f"  {n_tr:>9,}  {lbl:<24s}  {np.mean(accs):.4f}  "
                              f"{np.std(accs):.4f}  {np.mean(ts):>7.1f}s", flush=True)
                    except Exception as e:
                        print(f"  {n_tr:>9,}  {lbl:<24s}  FAILED: {e}", flush=True)
                        continue

        print(f"\n  Exp4 complete. {hms(time.perf_counter()-t_start)}")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir",  default="logs")
    p.add_argument("--csv_dir",  default="results")
    p.add_argument("--data_dir", default="data")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp4_learning_curves_{ts}.txt"),
        os.path.join(a.csv_dir, "exp4_learning_curves.csv"), a.data_dir)