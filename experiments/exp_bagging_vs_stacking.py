"""
exp_bagging_vs_stacking.py — Does the learned W-combination (stacking) beat
plain weight-averaging (true bagging), for BINARY and MULTICLASS, and how does
the answer depend on n and on the per-SVM partition size?

Design:
  - aggregate_W=False → STACKING: full W (P x m*K), downstream SVM learns combination.
  - aggregate_W=True  → BAGGING : average the m votes → W (P x K), one mean vote/class.
  - Identical disjoint partitions, overfitting SVMs (block_C), arc-cosine kernel, P.
  - SWEEP n_total over {100k..500k} for BOTH datasets:
        HIGGS     (binary,      d=28, ~1.1M pool)
        CoverType (7-class,     d=54,  581k pool)   ← replaces MNIST (only 70k)
  - SWEEP m over {1,10,20,50,100}. Each disjoint SVM sees n_total/m samples; if that
    partition drops below MIN_SAMPLES_PER_SVM we SKIP the cell (a large-m failure on
    too-little data would confound "m too big" with "each SVM data-starved"). The log
    prints n/SVM so the partition size is always visible.

Vote-decorrelation diagnostics (per block, on the pre-collapse W):
    hard_rank, stable_rank (||W||_F^2/||W||_2^2), mean_cos_sim (same-class votes
    compared across the m SVMs — multiclass-correct).
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from mlsvm_extensions import DiverseMLMSVM
try:
    from mlsvm_extensions import __DIAG_VERSION__
except ImportError:
    __DIAG_VERSION__ = None

EXP_ID    = "exp_bagging_vs_stacking"
N_SWEEP   = [100_000, 200_000, 300_000, 400_000, 500_000]
TEST_SIZE = 20_000
P         = 1000
BLOCK_C   = 100.0
FINAL_C   = 1.0
M_VALUES  = [1, 10, 20, 50, 100]
L         = 1
KERNEL    = "arc_cosine"
MODES     = ["disjoint", "disjoint_featsub"]
AGG       = [("stacking", False), ("bagging_avg", True)]
N_SEEDS   = 3
# Each disjoint SVM trains on n_total/m samples. Below this many samples a single
# SVM cannot learn a useful boundary, so a large-m failure would be confounded with
# data starvation rather than telling us anything about m itself. We SKIP any (n, m)
# whose partition is smaller than this, and flag it in the log.
MIN_SAMPLES_PER_SVM = 300


def make_model(m, mode, aggregate, seed):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=BLOCK_C, final_C=FINAL_C,
        aggregate_W=aggregate, feature_frac="sqrt",
        random_state=seed, normalize_inter_layer=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run_one(X, y, name, n_total, test_size, csv_w):
    """Run the full mode x agg x m grid at one (dataset, n_total)."""
    d, K = X.shape[1], len(np.unique(y))
    pool = len(y)
    # Cap requested n_total to what the dataset can supply (keep test_size aside)
    n_eff = min(n_total, pool - test_size)
    if n_eff < 1000:
        print(f"  [{name}] n_total={n_total:,} infeasible (pool={pool:,}); skipping.")
        return
    capped = " (CAPPED to pool)" if n_eff < n_total else ""

    print(f"\n{'='*94}")
    print(f"  {name}  (d={d}, K={K})   n_total={n_eff:,}{capped}   test={test_size:,}")
    print(f"{'='*94}")
    print(f"  {'mode':17s} {'agg':12s} {'m':>3s} {'n/SVM':>7s} "
          f"{'train':>7s} {'test':>7s} {'Wcols':>6s} {'stable_rk':>9s} {'cos_sim':>8s} {'t/run':>7s}")
    print("  " + "─"*100)

    for mode in MODES:
        for agg_name, agg_flag in AGG:
            for m in M_VALUES:
                if m == 1 and agg_flag:           # m=1: stacking==bagging, run once
                    continue
                n_per_svm = n_eff // m
                if m > 1 and n_per_svm < MIN_SAMPLES_PER_SVM:
                    print(f"  {mode:17s} {agg_name:12s} {m:>3d} {n_per_svm:>7d} "
                          f"  SKIP (n/SVM={n_per_svm} < {MIN_SAMPLES_PER_SVM}: "
                          f"partition too small to learn)", flush=True)
                    continue
                tr_l, te_l, wc_l, sr_l, cs_l, t_l = [], [], [], [], [], []
                for seed in range(N_SEEDS):
                    rng = np.random.RandomState(3000 + seed)
                    idx = rng.permutation(pool)[: n_eff + test_size]
                    Xs, ys = X[idx], y[idx]
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xs, ys, test_size=test_size, random_state=seed, stratify=ys)
                    try:
                        model = make_model(m, mode, agg_flag, seed)
                        t = time.perf_counter()
                        model.fit(Xtr, ytr)
                        dt = time.perf_counter() - t
                        tr = model.score(Xtr, ytr); te = model.score(Xte, yte)
                        diag = model.named_steps["clf"].W_diagnostics_[0]
                        tr_l.append(tr); te_l.append(te); t_l.append(dt)
                        wc_l.append(diag["n_cols"]); sr_l.append(diag["stable_rank"])
                        cs_l.append(diag["mean_cos_sim"])
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                            n_total=n_eff, n_train=len(Xtr), n_test=test_size,
                            d=d, n_classes=K, model=f"{mode}_{agg_name}_m{m}",
                            kernel=KERNEL, mode=mode, aggregation=agg_name,
                            m=m, P=P, block_C=BLOCK_C, seed=seed,
                            train_acc=round(tr,4), acc=round(te,4),
                            W_cols=diag["n_cols"], stable_rank=diag["stable_rank"],
                            mean_cos_sim=diag["mean_cos_sim"], time_s=round(dt,2)))
                    except Exception as e:
                        print(f"  {mode} {agg_name} m={m} seed={seed} FAILED: {e}", flush=True)
                if tr_l:
                    print(f"  {mode:17s} {agg_name:12s} {m:>3d} {len(Xtr)//m:>7d} "
                          f"{np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} "
                          f"{int(np.mean(wc_l)):>6d} {np.mean(sr_l):>9.2f} "
                          f"{np.mean(cs_l):>8.4f} {np.mean(t_l):>6.1f}s", flush=True)
            print()


def run(log_path, csv_path, datasets, n_sweep):
    # Fail fast if an outdated mlsvm_extensions.py is on the path: the
    # multiclass cosine-similarity fix must be present, otherwise cos_sim
    # for m=1 comes out negative/≈0 instead of 1.0 and the diagnostics lie.
    if __DIAG_VERSION__ != "2024-cos-per-class-fix":
        raise RuntimeError(
            f"Stale mlsvm_extensions.py loaded (__DIAG_VERSION__={__DIAG_VERSION__!r}). "
            "Copy the corrected mlsvm_extensions.py next to this script and clear "
            "any __pycache__ before running.")
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Bagging (avg votes) vs Stacking (learned W) — n sweep",
               f"n_total sweep = {[f'{n//1000}k' for n in n_sweep]}  P={P}  block_C={BLOCK_C}",
               f"m={M_VALUES}  modes={MODES}  agg={[a[0] for a in AGG]}  seeds={N_SEEDS}",
               "Q: does learned W-combination beat averaging, and how does it scale with n?")
        t0 = time.perf_counter()

        # Load each dataset ONCE, then sweep n_total (avoids repeated disk loads)
        loaded = {}
        for tag, name, test_size in datasets:
            try:
                X, y = load(tag)
                loaded[name] = (X, y, test_size)
            except Exception as e:
                print(f"  [{name}] LOAD FAILED: {e}")

        for n_total in n_sweep:
            print(f"\n\n{'#'*94}")
            print(f"#  N_TOTAL = {n_total:,}")
            print(f"{'#'*94}")
            for name, (X, y, test_size) in loaded.items():
                run_one(X, y, name, n_total, test_size, csv_w)

        print(f"\n  exp_bagging_vs_stacking complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING THE RESULTS:")
        print("  - stacking test > bagging_avg test → learned combination beats averaging.")
        print("  - Track the stacking−bagging gap as n grows: does it widen, shrink, vanish?")
        print("  - cos_sim now compares SAME-class votes across SVMs (multiclass-correct).")
        print("    Low cos_sim = decorrelated 'different opinions'; ~1 = redundant votes.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--n_sweep", type=int, nargs="+", default=N_SWEEP,
                   help="list of n_total values, e.g. --n_sweep 100000 200000 300000")
    p.add_argument("--higgs_only", action="store_true")
    p.add_argument("--covertype_only", action="store_true")
    a = p.parse_args()

    datasets = [("higgs", "HIGGS", TEST_SIZE), ("covertype", "CoverType", TEST_SIZE)]
    if a.higgs_only:     datasets = [("higgs", "HIGGS", TEST_SIZE)]
    if a.covertype_only: datasets = [("covertype", "CoverType", TEST_SIZE)]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_bagging_vs_stacking_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_bagging_vs_stacking.csv"),
        datasets, a.n_sweep)
