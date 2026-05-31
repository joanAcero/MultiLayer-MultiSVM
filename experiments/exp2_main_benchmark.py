"""
exp2_main_benchmark.py — Main Head-to-Head Benchmark
=====================================================
Hypothesis
----------
ML-MSVM with best configurations (RBF m=2 L=2; ArcCos m=1 L=1) matches or
exceeds the published accuracy of ML-SVM (Acero & Belanche 2025) across all
regimes, while remaining competitive with the exact RBF SVM on Regime 1 and
outperforming all scalable alternatives on Regimes 2 and 3. The flat RFF SVM
(L=0) serves as the direct depth-ablation baseline.

Protocol
--------
  Regime 1/2 : 10× stratified 90/10 splits
  Regime 3   : 5× fixed 10k train / 2k test
  Exact RBF SVM skipped for n_train > 20 000.
  Published baselines from Acero & Belanche (2025) printed inline.
  Output     : logs/exp2_main_benchmark_<ts>.txt  +  results/exp2_main_benchmark.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   make_linear_svm, make_rbf_svm, make_flat_rff,
                   import_ml_msvm, PUBLISHED, RBF_N_LIMIT, hms, banner)

ML_MSVM = import_ml_msvm()
P       = 1000
EXP_ID  = "exp2_main_benchmark"

DATASETS = [
    # tag             display           regime  pub_key
    ("wine",          "Wine",           1,      None),
    ("breast_cancer", "Breast Cancer",  1,      "Breast Cancer"),
    ("ionosphere",    "Ionosphere",     1,      "Ionosphere"),
    ("sonar",         "Sonar",          1,      None),
    ("glass",         "Glass",          1,      "Glass"),
    ("magic",         "Magic",          2,      "Magic"),
    ("spambase",      "Spambase",       2,      "Spambase"),
    ("covertype_sub", "Cover Type",     2,      "Cover Type"),
    ("mnist",         "MNIST",          3,      None),
    ("fashion",       "Fashion-MNIST",  3,      None),
]
R1R2_REPS = 10
R3_REPS   = 5


def models_for(skip_exact: bool) -> list:
    entries = [
        ("Linear SVM",           make_linear_svm(),              "linear",     0, 0),
        ("Flat RFF RBF (L=0)",   make_flat_rff(ML_MSVM, P, "rbf"),    "rbf",  0, 1),
        ("Flat RFF Arc (L=0)",   make_flat_rff(ML_MSVM, P, "arc_cosine"), "arc_cosine", 0, 1),
        ("ML-MSVM RBF m=2 L=1",  make_mlmsvm(ML_MSVM, 1, 2, P, "rbf"), "rbf", 1, 2),
        ("ML-MSVM RBF m=2 L=2",  make_mlmsvm(ML_MSVM, 2, 2, P, "rbf"), "rbf", 2, 2),
        ("ML-MSVM Arc m=1 L=1",  make_mlmsvm(ML_MSVM, 1, 1, P, "arc_cosine"), "arc_cosine", 1, 1),
        ("ML-MSVM Arc m=1 L=2",  make_mlmsvm(ML_MSVM, 2, 1, P, "arc_cosine"), "arc_cosine", 2, 1),
    ]
    if not skip_exact:
        entries.insert(1, ("RBF SVM (exact)", make_rbf_svm(), "rbf_exact", 0, 0))
    return entries


def run(log_path: str, csv_path: str) -> None:
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Experiment 2 — Main Head-to-Head Benchmark",
               "Baselines: Linear SVM, Exact RBF SVM (R1 only), Flat RFF (RBF & Arc)",
               "ML-MSVM: RBF m=2 L={1,2}  |  ArcCos m=1 L={1,2}",
               f"P={P}  |  Exact RBF SVM skipped if n_train > {RBF_N_LIMIT}",
               "Acero & Belanche (2025) published results printed inline")
        t_start = time.perf_counter()
        pub = PUBLISHED["acero2025"]

        for tag, name, regime, pub_key in DATASETS:
            reps = R3_REPS if regime == 3 else R1R2_REPS
            X, y = load(tag)
            d, n, n_cls = X.shape[1], len(y), len(np.unique(y))
            splits = make_splits(X, y, regime, reps)
            n_tr, n_te = len(splits[0][0]), len(splits[0][2])
            skip_exact = n_tr > RBF_N_LIMIT or regime == 3
            proto = f"{reps}×90/10" if regime < 3 else f"{reps}×{n_tr}/{n_te}"

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, K={n_cls})  [Regime {regime} | {proto}]")
            if skip_exact:
                print(f"  [Exact RBF SVM: SKIPPED — n_train={n_tr} > {RBF_N_LIMIT}]")
            print(f"{'='*72}")
            print(f"  {'model':<38s}  {'acc':>8s}  {'std':>6s}  {'t/run':>8s}")
            print(f"  {'─'*65}")

            results = {}
            for lbl, model, kernel, L, m in models_for(skip_exact):
                accs, ts = [], []
                for i, (Xtr, ytr, Xte, yte) in enumerate(splits):
                    mc = clone(model)
                    t = time.perf_counter()
                    mc.fit(Xtr, ytr)
                    accs.append(mc.score(Xte, yte))
                    ts.append(time.perf_counter() - t)
                    csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                        n_total=n, n_train=n_tr, n_test=n_te, d=d,
                        n_classes=n_cls, model=lbl, kernel=kernel, L=L, m=m, P=P,
                        split_id=i, acc=accs[-1], time_s=ts[-1]))
                results[lbl] = np.mean(accs)
                print(f"  {lbl:<38s}  {np.mean(accs):.4f}  {np.std(accs):.4f}  "
                      f"{np.mean(ts):>7.2f}s", flush=True)

            if pub_key and pub_key in pub:
                ref = pub[pub_key]["acc"]
                delta = max(results.values()) - ref
                sign = "+" if delta >= 0 else ""
                print(f"\n  [Ref] {pub['name'][:40]:<40s}  {ref:.4f}"
                      f"   (ours best: {sign}{delta:+.4f})")
            if name == "MNIST":
                mref = PUBLISHED["mehrkanoon2018"]
                r = mref["MNIST"]
                print(f"  [Ref] {mref['name'][:40]:<40s}  {r['acc']:.4f}"
                      f"  ({r['note']})")

        print(f"\n  Experiment 2 complete. {hms(time.perf_counter()-t_start)}")
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
    run(os.path.join(a.log_dir, f"exp2_main_benchmark_{ts}.txt"),
        os.path.join(a.csv_dir, "exp2_main_benchmark.csv"))
