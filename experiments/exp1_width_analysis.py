"""
exp1_width_analysis.py — Width Analysis: Effect of m (SVMs per block)
======================================================================
Hypothesis
----------
The jump from m=1 to m=2 captures most of the accuracy gain for the RBF kernel;
further increases yield diminishing returns due to saturation of representational
diversity. The arc-cosine kernel, by contrast, peaks at m=1 on high-dimensional
data because its feature geometry is concentrated and additional weight vectors
dilute rather than diversify the projection. Both kernels are tested jointly on
each dataset with the same splits, enabling direct comparison.

Protocol
--------
  Datasets : Wine, Breast Cancer, Ionosphere, Sonar, Glass (R1),
             Magic, Spambase, Cover Type subset (R2)
  Splits   : 10x stratified 90/10 (matching Acero & Belanche 2025)
  Varied   : m ∈ {1, d//4, d//2, d, 2d}  ×  L ∈ {1, 2, 3}
             × kernel ∈ {rbf, arc_cosine}
  Fixed    : P = 1000
  Output   : logs/exp1_width_<timestamp>.txt  +  results/exp1_width.csv
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (Tee, CSVWriter, load, make_splits, make_mlmsvm,
                   import_ml_msvm, m_values, hms, banner)

ML_MSVM = import_ml_msvm()

# ─── Configuration ────────────────────────────────────────────────────────────
DATASETS = [
    # tag             display_name     regime
    ("wine",          "Wine",          1),
    ("breast_cancer", "Breast Cancer", 1),
    ("ionosphere",    "Ionosphere",    1),
    ("sonar",         "Sonar",         1),
    ("glass",         "Glass",         1),
    ("magic",         "Magic",         2),
    ("spambase",      "Spambase",      2),
    ("covertype_sub", "Cover Type",    2),
]
DEPTHS   = [1, 2, 3]
KERNELS  = [("rbf", "RBF"), ("arc_cosine", "ArcCos")]
P        = 1000
REPEATS  = 10
EXP_ID   = "exp1_width"


def run(log_path: str, csv_path: str):
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    csv = CSVWriter(csv_path)

    try:
        banner("Experiment 1 — Width Analysis (m sweep, both kernels)",
               f"P={P}  Depths={DEPTHS}  Repeats={REPEATS}x 90/10",
               "Hypothesis: m=1→2 captures main gain (RBF); m=1 optimal (ArcCos, high-d)")

        t_total = time.perf_counter()

        for tag, name, regime in DATASETS:
            X, y = load(tag)
            d, n = X.shape[1], len(y)
            ms = m_values(d)
            splits = make_splits(X, y, regime, REPEATS)
            n_train = len(splits[0][0])
            n_test  = len(splits[0][2])

            print(f"\n{'='*72}")
            print(f"  {name}  (n={n}, d={d}, classes={len(np.unique(y))})")
            print(f"  m values: {ms}   splits: {REPEATS}×{n_train}/{n_test}")
            print(f"{'='*72}")

            # Header
            print(f"  {'Model':<30s}  {'kernel':>8s}  {'L':>2s}  {'m':>5s}  "
                  f"{'acc':>8s}  {'±':>6s}  {'t/run':>7s}")
            print(f"  {'─'*70}")

            best_acc = -1.0
            block_results = []   # collect for per-kernel summary

            for kernel, klabel in KERNELS:
                for L in DEPTHS:
                    for m in ms:
                        from sklearn.base import clone
                        model = make_mlmsvm(ML_MSVM, L, m, P, kernel)
                        accs, times = [], []
                        for Xtr, ytr, Xte, yte in splits:
                            mc = clone(model)
                            t0 = time.perf_counter()
                            mc.fit(Xtr, ytr)
                            accs.append(mc.score(Xte, yte))
                            times.append(time.perf_counter() - t0)
                            # Write one CSV row per split
                            csv.write(dict(
                                exp_id=EXP_ID, dataset=name,
                                n_total=n, n_train=n_train, n_test=n_test,
                                d=d, n_classes=len(np.unique(y)),
                                model=f"{klabel}_L{L}_m{m}",
                                kernel=kernel, L=L, m=m, P=P,
                                split_id=len(accs)-1,
                                acc=accs[-1], time_s=times[-1],
                            ))

                        acc_mean = float(np.mean(accs))
                        acc_std  = float(np.std(accs))
                        t_mean   = float(np.mean(times))
                        if acc_mean > best_acc:
                            best_acc = acc_mean
                        label = f"{klabel} L={L} m={m}"
                        print(f"  {label:<30s}  {klabel:>8s}  {L:>2d}  {m:>5d}  "
                              f"{acc_mean:.4f}  {acc_std:.4f}  {t_mean:>6.2f}s",
                              flush=True)
                        block_results.append((klabel, L, m, acc_mean, acc_std, t_mean))

            # ── Per-kernel summary ────────────────────────────────────────────
            print()
            for kernel, klabel in KERNELS:
                rows = [(L, m, a, s, t) for (kl, L, m, a, s, t) in block_results
                        if kl == klabel]
                best_L = max(DEPTHS, key=lambda L:
                    max(a for (_L, m, a, s, t) in rows if _L == L))
                best_m = max(ms, key=lambda m:
                    max(a for (L, _m, a, s, t) in rows if _m == m))
                best_combo = max(rows, key=lambda r: r[2])
                print(f"  [{klabel}] best overall: L={best_combo[0]} m={best_combo[1]}"
                      f" acc={best_combo[2]:.4f}")
                print(f"  [{klabel}] best m per depth: " +
                      "  ".join(f"L={L}→m={max((m for (l,m,a,s,t) in rows if l==L), key=lambda m: max(a for (l2,m2,a,s,t) in rows if l2==L and m2==m))}"
                                for L in DEPTHS))

        elapsed = hms(time.perf_counter() - t_total)
        print(f"\n{'█'*60}")
        print(f"  Experiment 1 complete. Total: {elapsed}")
        print(f"  Log: {log_path}")
        print(f"  CSV: {csv_path}")
        print(f"{'█'*60}")

    finally:
        sys.stdout = tee._stream
        tee.close()
        csv.close()


def parse_args():
    p = argparse.ArgumentParser(description="Exp1: Width analysis")
    p.add_argument("--log_dir",  default="logs")
    p.add_argument("--csv_dir",  default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = os.path.join(args.log_dir,  f"exp1_width_{ts}.txt")
    csv = os.path.join(args.csv_dir, "exp1_width.csv")
    run(log, csv)
