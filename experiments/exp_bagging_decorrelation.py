"""
exp_bagging_decorrelation.py — Does TRUE bagging produce decorrelated SVM votes?

Design (agreed with supervisor):
  - Hard dataset: HIGGS (binary, d=28). N_TOTAL samples (default 100k), variable.
  - m in {1, 10, 20, 50}: data split into m DISJOINT partitions of n/m each.
  - Each SVM overfits its own slice: block_C = 100.  Arc-cosine kernel.  P = 1000.
  - Final head classifier: C = 1.0.
  - Two strategies:
        disjoint           : data partition only.
        disjoint_featsub   : data partition + sqrt(P) random features per SVM.
  - Bagging theory wants DECORRELATED, high-variance base learners. We measure
    whether the per-SVM weight rows of W actually "differ in opinion":
        hard_rank      rank(W) at tolerance
        stable_rank    ||W||_F^2 / ||W||_2^2  (effective #independent directions)
        mean_cos_sim   mean pairwise cosine similarity of weight rows (lower = more
                       decorrelated; ~0 means near-orthogonal "different opinions")
  - Also report TRAIN and TEST accuracy. If votes don't differ (high cos_sim, low
    stable_rank), m>1 is useless. We compare which strategy decorrelates most.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from mlsvm_extensions import DiverseMLMSVM

EXP_ID   = "exp_bagging_decorrelation"
# ── Tunable knobs ─────────────────────────────────────────────────────────
N_TOTAL  = 100_000          # total samples to play with (set smaller/larger freely)
TEST_SIZE = 20_000          # held-out test rows (drawn separately from the pool)
P        = 1000             # random features per block
BLOCK_C  = 100.0            # high C so each partition SVM overfits its slice
FINAL_C  = 1.0              # head regularisation
M_VALUES = [1, 10, 20, 50]  # number of disjoint partitions / SVMs per block
L        = 1                # single block: isolates the bagging effect cleanly
KERNEL   = "arc_cosine"
MODES    = ["disjoint", "disjoint_featsub"]
N_SEEDS  = 3                # repeat with different partitionings


def make_model(m, mode, seed):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=BLOCK_C, final_C=FINAL_C,
        feature_frac="sqrt", random_state=seed,
        normalize_inter_layer=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Bagging Decorrelation (TRUE disjoint partitions)",
               f"HIGGS, N_TOTAL={N_TOTAL:,}, P={P}, block_C={BLOCK_C}, final_C={FINAL_C}",
               f"m={M_VALUES}  modes={MODES}  kernel={KERNEL}  L={L}  seeds={N_SEEDS}",
               "Q: do disjoint+overfit SVMs produce decorrelated votes (low cos-sim)?")
        t0 = time.perf_counter()

        try:
            X, y = load("higgs")
        except Exception as e:
            print(f"  HIGGS LOAD FAILED: {e}"); return
        d, K = X.shape[1], len(np.unique(y))
        sqrtP = int(round(np.sqrt(P)))
        print(f"  HIGGS loaded: {len(y):,} rows, d={d}, K={K}")
        print(f"  Pool → use {N_TOTAL:,} train + {TEST_SIZE:,} test")
        print(f"  disjoint_featsub uses sqrt(P)={sqrtP} features per SVM\n")

        print(f"  {'mode':17s} {'m':>3s} {'n/SVM':>7s} {'seed':>4s} "
              f"{'train':>7s} {'test':>7s} {'hard_rk':>7s} {'stable_rk':>9s} {'cos_sim':>7s} {'t/run':>7s}")
        print("  " + "─" * 92)

        for mode in MODES:
            for m in M_VALUES:
                tr_l, te_l, hr_l, sr_l, cs_l, t_l = [], [], [], [], [], []
                for seed in range(N_SEEDS):
                    rng = np.random.RandomState(1000 + seed)
                    # Sample a fresh train+test split from the pool each seed
                    idx = rng.permutation(len(y))[: N_TOTAL + TEST_SIZE]
                    Xs, ys = X[idx], y[idx]
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xs, ys, test_size=TEST_SIZE, random_state=seed, stratify=ys)
                    n_per = len(Xtr) // m
                    try:
                        model = make_model(m, mode, seed)
                        t = time.perf_counter()
                        model.fit(Xtr, ytr)
                        dt = time.perf_counter() - t
                        tr = model.score(Xtr, ytr)
                        te = model.score(Xte, yte)
                        diag = model.named_steps["clf"].W_diagnostics_[0]
                        tr_l.append(tr); te_l.append(te); t_l.append(dt)
                        hr_l.append(diag["hard_rank"]); sr_l.append(diag["stable_rank"])
                        cs_l.append(diag["mean_cos_sim"])
                        csv_w.write(dict(exp_id=EXP_ID, dataset="HIGGS",
                            n_total=N_TOTAL, n_train=len(Xtr), n_test=TEST_SIZE,
                            d=d, n_classes=K, model=f"{mode}_m{m}", kernel=KERNEL,
                            mode=mode, m=m, P=P, block_C=BLOCK_C, n_per_svm=n_per,
                            seed=seed, train_acc=round(tr,4), acc=round(te,4),
                            hard_rank=diag["hard_rank"], stable_rank=diag["stable_rank"],
                            mean_cos_sim=diag["mean_cos_sim"], time_s=round(dt,2)))
                    except Exception as e:
                        print(f"  {mode:17s} m={m} seed={seed} FAILED: {e}", flush=True)
                if tr_l:
                    print(f"  {mode:17s} {m:>3d} {len(Xtr)//m:>7d} {'avg':>4s} "
                          f"{np.mean(tr_l):>7.4f} {np.mean(te_l):>7.4f} "
                          f"{np.mean(hr_l):>7.1f} {np.mean(sr_l):>9.2f} "
                          f"{np.mean(cs_l):>7.3f} {np.mean(t_l):>6.1f}s", flush=True)
            print()

        print(f"  exp_bagging_decorrelation complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING THE RESULTS:")
        print("  - mean_cos_sim near 0  → SVMs have 'different opinions' (good for bagging)")
        print("  - mean_cos_sim near 1  → redundant votes; m>1 is useless")
        print("  - stable_rank ≈ m      → votes span m independent directions (ideal)")
        print("  - stable_rank ≈ 1      → votes collapse to one direction (bad)")
        print("  - whichever mode gives LOWER cos_sim decorrelates votes most.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    p.add_argument("--n_total", type=int, default=N_TOTAL)
    a = p.parse_args()
    N_TOTAL = a.n_total
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_bagging_decorrelation_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_bagging_decorrelation.csv"))
