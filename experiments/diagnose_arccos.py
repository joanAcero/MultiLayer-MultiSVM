"""
diagnose_arccos.py — Run this FIRST on your real machine before the full suite.

It measures, on real MNIST, how many TRON iterations the ArcCos block SVM needs
and whether the bounded solver (block_max_iter=2000) helps. Takes ~3 minutes.
This tells us whether the n=10k slowness is (a) hitting the iteration cap, or
(b) genuinely slow convergence — which determines the right setting.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from utils import load
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning

print("Loading MNIST...", flush=True)
X, y = load("mnist", verbose=False)
X = StandardScaler().fit_transform(X)
rng = np.random.default_rng(0)
P = 1000
Omega = rng.standard_normal((P, X.shape[1]))

def arccos_phi(Xs):
    return np.sqrt(2.0/P) * np.maximum(0.0, Xs @ Omega.T)

print(f"\nReal MNIST ArcCos block SVM — iteration count and timing")
print(f"{'n':>7s} {'tol':>6s} {'max_it':>7s} {'time':>8s} {'n_iter':>8s} {'hit_cap':>8s}")
print("─"*48)
for n in [5000, 10000, 20000]:
    idx = rng.choice(len(y), n, replace=False)
    phi = arccos_phi(X[idx]); ys = y[idx]
    for tol, mi in [(1e-4, 5000), (1e-3, 2000), (1e-2, 1000)]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svm = LinearSVC(C=1.0, dual=False, tol=tol, max_iter=mi, random_state=0)
            t = time.perf_counter(); svm.fit(phi, ys); el = time.perf_counter()-t
            hit = any(issubclass(x.category, ConvergenceWarning) for x in w)
        ni = getattr(svm,'n_iter_',-1)
        if hasattr(ni,'__len__'): ni = int(max(ni))
        print(f"{n:>7d} {tol:>6.0e} {mi:>7d} {el:>7.2f}s {ni:>8} {str(hit):>8s}", flush=True)
    print()
print("If hit_cap=True → looser tol/lower max_iter will speed it up a lot.")
print("If hit_cap=False but slow → genuine slow convergence; the bounded")
print("  solver caps the worst case but MNIST ArcCos stays moderately slow.")
