"""
exp2_main_benchmark.py — Main Head-to-Head Benchmark
=====================================================
Hypothesis
----------
ML-MSVM with best configurations matches or exceeds the published accuracy of
ML-SVM (Acero & Belanche 2025) across all regimes, while training without
backpropagation and remaining competitive with Mehrkanoon DHNKN on Regime 3.
The flat RFF SVM (L=0) serves as the direct ablation baseline: every gain
of the layered architecture over it is attributable to the depth alone.

Protocol
--------
  Regime 1/2 : 10× stratified 90/10 splits
  Regime 3   : 5× fixed 10k train / 2k test splits
  Best configs established from Exp 1: RBF m=2 L=2, ArcCos m=1 L=1
  Published baselines from Acero & Belanche (2025) printed inline.
  Exact RBF SVM skipped for n_train > RBF_N_LIMIT.

Output: logs/exp2_benchmark_<ts>.txt  +  results/exp2_benchmark.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   make_linear_svm, make_rbf_svm, make_flat_rff,
                   import_ml_msvm, PUBLISHED, hms, banner)

ML_MSVM = import_ml_msvm()

# ─── Configuration ────────────────────────────────────────────────────────────
RBF_N_LIMIT = 10_000
P = 1000
EXP_ID = "exp2_benchmark"

DATASETS = [
    # tag             display       regime  published_key
    ("wine",          "Wine",          1, None),
    ("breast_cancer", "Breast Cancer", 1, "Breast Cancer"),
    ("ionosphere",    "Ionosphere",    1, "Ionosphere"),
    ("sonar",         "Sonar",         1, None),
    ("glass",         "Glass",         1, "Glass"),
    ("magic",         "Magic",         2, "Magic"),
    ("spambase",      "Spambase",      2, "Spambase"),
    ("covertype_sub", "Cover Type",    2, "Cover Type"),
    ("mnist",         "MNIST",         3, None),
    ("fashion",       "Fashion-MNIST", 3, None),
]

# Best configs from Exp 1 (update after running Exp 1)
BEST_CONFIGS = [
    ("ML-MSVM RBF  m=2 L=1", "rbf",        1, 2),
    ("ML-MSVM RBF  m=2 L=2", "rbf",        2, 2),
    ("ML-MSVM Arc  m=1 L=1", "arc_cosine", 1, 1),
    ("ML-MSVM Arc  m=1 L=2", "arc_cosine", 2, 1),
]

R1R2_REPEATS = 10
R3_REPEATS   = 5


def run(log_path: str, csv_path: str):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)

    try:
        banner("Experiment 2 — Main Benchmark",
               "Models: Linear SVM, Exact RBF SVM (R1 only), Flat RFF,",
               "        ML-MSVM RBF m=2 L={1,2}, ML-MSVM ArcCos m=1 L={1,2}",
               "Published comparison: Acero & Belanche (2025) inline",
               f"P={P}  RBF SVM skipped if n_train > {RBF_N_LIMIT}")

        t_total = time.perf_counter()
        pub = PUBLISHED["acero2025"]

        for tag, name, regime, pub_key in DATASETS:
            repeats = R3_REPEATS if regime == 3 else R1R2_REPEATS
            X, y = load(tag)
            d, n = X.shape[1], len(y)
            splits = make_splits(X, y, regime, repeats)
            n_train = len(splits[0][0])
            n_test  = len(splits[0][2])
            n_cls   = len(np.unique(y))
            skip_rbf = n_train > RBF_N_LIMIT or regime == 3
            proto = (f"{repeats}×90/10" if regime < 3
                     else f"{repeats}×{n_train}/{n_test}")

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, classes={n_cls}) "
                  f"[Regime {regime} | {proto}]")
            if skip_rbf:
                print(f"  (Exact RBF SVM skipped: n_train={n_train} > {RBF_N_LIMIT})")
            print(f"{'='*72}")

            models: list[tuple[str, object, str, int, int]] = [
                # (label, model, kernel, L, m)
                ("Linear SVM",      make_linear_svm(),           "linear",      0, 0),
                ("Flat RFF RBF",    make_flat_rff(ML_MSVM, P, "rbf"),      "rbf",    0, 1),
                ("Flat RFF ArcCos", make_flat_rff(ML_MSVM, P, "arc_cosine"),"arc_cosine",0,1),
            ]
            if not skip_rbf:
                models.insert(1, ("RBF SVM (exact)", make_rbf_svm(), "rbf_exact", 0, 0))
            for lbl, kern, L, m in BEST_CONFIGS:
                models.append((lbl, make_mlmsvm(ML_MSVM, L, m, P, kern), kern, L, m))

            print(f"  {'Model':<36s}  {'acc':>8s}  {'±':>6s}  {'t/run':>7s}  {'note'}")
            print(f"  {'─'*72}")

            results = {}
            for model_label, model, kernel, L, m in models:
                accs, times = [], []
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    mc = clone(model)
                    t0 = time.perf_counter()
                    mc.fit(Xtr, ytr)
                    accs.append(mc.score(Xte, yte))
                    times.append(time.perf_counter() - t0)
                    csv_w.write(dict(
                        exp_id=EXP_ID, dataset=name,
                        n_total=n, n_train=n_train, n_test=n_test,
                        d=d, n_classes=n_cls,
                        model=model_label, kernel=kernel, L=L, m=m, P=P,
                        split_id=i, acc=accs[-1], time_s=times[-1],
                    ))
                acc_mean = float(np.mean(accs))
                acc_std  = float(np.std(accs))
                t_mean   = float(np.mean(times))
                results[model_label] = acc_mean
                print(f"  {model_label:<36s}  {acc_mean:.4f}  {acc_std:.4f}  "
                      f"{t_mean:>6.2f}s", flush=True)

            # Published baseline inline
            if pub_key and pub_key in pub:
                ref = pub[pub_key]["acc"]
                our_best = max(results.values())
                delta = our_best - ref
                sign = "+" if delta >= 0 else ""
                print(f"\n  [Published] {pub['name']:<30s}  {ref:.4f}"
                      f"   (our best: {sign}{delta:.4f})")

            # Mehrkanoon note for MNIST
            if name == "MNIST":
                mref = PUBLISHED["mehrkanoon2018"]
                print(f"  [Published] {mref['name']}")
                for k, v in mref.items():
                    if isinstance(v, dict):
                        print(f"    {k}: acc={v['acc']:.4f}  ({v['note']})")

        elapsed = hms(time.perf_counter() - t_total)
        print(f"\n{'█'*60}")
        print(f"  Experiment 2 complete. Total: {elapsed}")
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
    run(os.path.join(args.log_dir, f"exp2_benchmark_{ts}.txt"),
        os.path.join(args.csv_dir, "exp2_benchmark.csv"))
