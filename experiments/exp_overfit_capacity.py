"""
exp_overfit_capacity.py — Can the architecture overfit (reach ~100% train accuracy)?

Rationale (agreed with supervisor): a viable architecture must be able to overfit a
small training set when given sufficient capacity. If it cannot drive TRAIN accuracy
to ~1.0, it has a representational bottleneck (most likely the P→(m·K) W-projection)
and is a weak starting point regardless of generalisation.

Design:
  - SMALL fixed training set (N_TRAIN, default 3000) so 100% train acc is achievable.
  - Sweep every capacity axis:
        P     in {250, 1000, 4000}        (features per block)
        m     in {1, 10, 50}              (SVMs per block; disjoint bagging)
        L     in {1, 2, 3}               (depth)
        C     in {1, 100, 10000}         (block_C: large C = less regularisation)
  - Track TRAIN accuracy primarily (test reported for context).
  - Diversity mode: 'disjoint' (the new bagging scheme) and 'same_c' (baseline,
    all SVMs see the full data) so we can see if bagging changes overfitting capacity.
  - Run on HIGGS (hard) and MNIST (structured) to contrast.

If train accuracy saturates < 1.0 as P and C grow, the bottleneck is structural,
not optimisation — and the W-projection dimension (m·K) is the prime suspect.
"""
from __future__ import annotations
import argparse, datetime, os, sys, time, itertools
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from utils import Tee, CSVWriter, load, hms, banner
from mlsvm_extensions import DiverseMLMSVM

EXP_ID    = "exp_overfit_capacity"
N_TRAIN   = 3000
N_TEST    = 3000
P_VALUES  = [250, 1000, 4000]
M_VALUES  = [1, 10, 50]
L_VALUES  = [1, 2, 3]
C_VALUES  = [1.0, 100.0, 10000.0]
MODES     = ["disjoint", "same_c"]
KERNEL    = "arc_cosine"
DATASETS  = [("higgs", "HIGGS"), ("mnist", "MNIST")]


def make_model(P, m, L, C, mode, seed):
    clf = DiverseMLMSVM(
        num_layers=L, svms_per_block=m, rff_features=P,
        kernel=KERNEL, arc_cosine_degree=1,
        diversity_mode=mode, block_C=C, final_C=C,   # large final_C too: let head overfit
        feature_frac="sqrt", random_state=seed,
        normalize_inter_layer=True,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def run(log_path, csv_path):
    tee = Tee(sys.stdout, log_path); sys.stdout = tee
    csv_w = CSVWriter(csv_path)
    try:
        banner("Exp — Overfitting Capacity (can the architecture reach ~100% train?)",
               f"N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, kernel={KERNEL}",
               f"P={P_VALUES}  m={M_VALUES}  L={L_VALUES}  C={C_VALUES}  modes={MODES}",
               "Q: does TRAIN accuracy reach 1.0 as capacity grows? If not → bottleneck.")
        t0 = time.perf_counter()

        for tag, name in DATASETS:
            try:
                X, y = load(tag)
            except Exception as e:
                print(f"\n  [{name}] LOAD FAILED: {e}"); continue
            d, K = X.shape[1], len(np.unique(y))
            rng = np.random.RandomState(0)
            idx = rng.permutation(len(y))[: N_TRAIN + N_TEST]
            Xs, ys = X[idx], y[idx]
            Xtr, Xte, ytr, yte = train_test_split(
                Xs, ys, test_size=N_TEST, random_state=0, stratify=ys)

            print(f"\n{'='*78}")
            print(f"  {name}  (d={d}, K={K})  train={len(Xtr)}  test={len(Xte)}")
            print(f"{'='*78}")
            print(f"  {'mode':9s} {'P':>5s} {'m':>3s} {'L':>2s} {'C':>7s} "
                  f"{'TRAIN':>7s} {'test':>7s} {'Wcols':>6s} {'stable_rk':>9s} {'t':>6s}")
            print("  " + "─"*74)

            best_train = {}
            for mode in MODES:
                for P, m, L, C in itertools.product(P_VALUES, M_VALUES, L_VALUES, C_VALUES):
                    # same_c with m>1 on small data is just repeated SVMs; still informative
                    label = f"{mode}_P{P}_m{m}_L{L}_C{int(C)}"
                    try:
                        model = make_model(P, m, L, C, mode, 0)
                        t = time.perf_counter()
                        model.fit(Xtr, ytr)
                        dt = time.perf_counter() - t
                        tr = model.score(Xtr, ytr)
                        te = model.score(Xte, yte)
                        clf = model.named_steps["clf"]
                        last = clf.W_diagnostics_[-1]
                        csv_w.write(dict(exp_id=EXP_ID, dataset=name,
                            n_train=len(Xtr), n_test=len(Xte), d=d, n_classes=K,
                            model=label, kernel=KERNEL, mode=mode, P=P, m=m, L=L,
                            block_C=C, train_acc=round(tr,4), acc=round(te,4),
                            W_cols=last["n_cols"], stable_rank=last["stable_rank"],
                            time_s=round(dt,2)))
                        best_train[mode] = max(best_train.get(mode, 0), tr)
                        # Only print the high-capacity rows to keep the log readable
                        if P == max(P_VALUES) or tr > 0.99:
                            print(f"  {mode:9s} {P:>5d} {m:>3d} {L:>2d} {C:>7.0f} "
                                  f"{tr:>7.4f} {te:>7.4f} {last['n_cols']:>6d} "
                                  f"{last['stable_rank']:>9.2f} {dt:>5.1f}s", flush=True)
                    except Exception as e:
                        print(f"  {label} FAILED: {e}", flush=True)
            for mode in MODES:
                print(f"\n  [{name}] best TRAIN accuracy ({mode}): {best_train.get(mode,0):.4f}")

        print(f"\n  exp_overfit_capacity complete. {hms(time.perf_counter()-t0)}")
        print("\n  READING THE RESULTS:")
        print("  - If best TRAIN reaches ~1.0 → architecture HAS sufficient capacity.")
        print("  - If TRAIN saturates < 1.0 even at max P and C → structural bottleneck.")
        print("  - Compare W_cols (= m*K) vs where train saturates: if train rises with")
        print("    W_cols, the P→(m*K) projection is the limiting dimension.")
    finally:
        sys.stdout = tee._stream; tee.close(); csv_w.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--csv_dir", default="results")
    a = p.parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run(os.path.join(a.log_dir, f"exp_overfit_capacity_{ts}.txt"),
        os.path.join(a.csv_dir, "exp_overfit_capacity.csv"))
