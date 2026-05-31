# ML-MSVM Experiment Plan
**Thesis:** Deep kernel networks: neural architectures with SVM foundations  
**Author:** Joan Acero · **Supervisor:** Lluís Belanche

---

## Repository structure

```
TFM/
├── mlsvm/
│   └── ml_msvm.py              ← ML_MSVMClassifier
└── experiments/
    ├── utils.py                ← shared utilities (import, loaders, factories)
    ├── exp1_width_analysis.py
    ├── exp2_main_benchmark.py
    ├── exp3_scalability.py
    ├── exp4_learning_curves.py
    ├── exp5_depth_p_analysis.py
    ├── exp6_c_spread.py
    ├── exp7_nystroem.py
    ├── exp8_arccosine_degree.py
    ├── exp9_scalability_timing.py
    ├── exp10_final_c.py
    ├── run_all.sh
    ├── data/                   ← auto-created, caches SUSY/HIGGS .npz
    ├── logs/                   ← auto-created, one .txt per run
    └── results/                ← auto-created, one .csv per experiment
```

---

## Running

```bash
cd ~/Master-Data-Science-FIB-UPC/TFM/experiments

# Run all 10 experiments in recommended order (overnight):
bash run_all.sh

# Run specific experiments:
bash run_all.sh 9 1 3

# Use a specific Python:
PYTHON=python3.11 bash run_all.sh
```

---

## Evaluation protocol

| Regime | Datasets | n | Splits | Protocol |
|--------|----------|---|--------|----------|
| 1 — SVM strong | Wine, Breast Cancer, Ionosphere, Sonar, Glass | ≤ 600 | 10× stratified 90/10 | Matches Acero & Belanche (2025) |
| 2 — High-n tabular | Magic, Spambase, Cover Type subset | 5k–19k | 10× stratified 90/10 | Primary target regime |
| 3 — High-n image | MNIST, Fashion-MNIST | 70k | 5× fixed 10k/2k | Standard in literature |
| 4 — Large-scale | SUSY, Cover Type Full, HIGGS | 400k–580k | 5× stratified 80/20 | Scalability demonstration |

### CSV schema (all experiments)
One row per (model, split evaluation), written immediately to disk.

```
exp_id, dataset, n_total, n_train, n_test, d, n_classes,
model, kernel, L, m, P, split_id, acc, time_s, timestamp
```

### Exact RBF SVM limit
Skipped when `n_train > 20 000`. At that scale the kernel matrix
exceeds 3.2 GB and training time is prohibitive (O(n³)).

---

## Published baselines for direct comparison

**Acero & Belanche (2025)** — ML-SVM, Table 3 (test accuracy = 1 − error):

| Dataset | SVM (RBF) | MLP | ML-SVM |
|---------|-----------|-----|--------|
| Glass | 0.710 | 0.820 | 0.820 |
| Breast Cancer | 0.990 | 0.970 | 0.992 |
| Ionosphere | 0.890 | 0.940 | 0.950 |
| Magic | 0.840 | 0.830 | 0.850 |
| Spambase | 0.690 | 0.920 | 0.850 |
| Cover Type | 0.690 | 0.720 | 0.790 |

**Mehrkanoon & Suykens (2018)** — DHNKN, Neurocomputing:
MNIST accuracy ≈ 0.9756 (60k train / 10k test). Not directly comparable
to our 10k/2k protocol; noted for reference only.

---

## Experiments

### Experiment 1 — Width Analysis
**File:** `exp1_width_analysis.py`  
**Hypothesis:** For RBF, m=2 is universally near-optimal; further increases yield
diminishing returns. For ArcCos, m=1 is optimal on high-d data. Both kernels show
the same saturation at small m but differ in the threshold. The inter-layer
standardisation fix (ArcCos) provides measurable accuracy gain for L≥2.  
**Varied:** m ∈ {1,2,3,d/4,d/2,d} × L ∈ {1,2,3,4} × kernel ∈ {RBF, ArcCos}  
**Fixed:** P=1000  
**Datasets:** All 10 (R1+R2+R3)  
**Extra:** ArcCos L=2 m=1 ablation with `normalize_inter_layer=False`

---

### Experiment 2 — Main Benchmark
**File:** `exp2_main_benchmark.py`  
**Hypothesis:** ML-MSVM (best config) matches or exceeds Acero & Belanche ML-SVM
on all regimes and outperforms all scalable alternatives on R2 and R3.  
**Models:** Linear SVM · Exact RBF SVM (R1 only) · Flat RFF (RBF & ArcCos) ·
ML-MSVM RBF m=2 L={1,2} · ML-MSVM ArcCos m=1 L={1,2}  
**Datasets:** All 10  
**Published baseline:** Acero & Belanche (2025) numbers printed inline

---

### Experiment 3 — Scalability on Large Datasets
**File:** `exp3_scalability.py`  
**Hypothesis:** ML-MSVM remains feasible and competitive at n > 400k where
the exact SVM is completely intractable. The architecture outperforms the
Linear SVM baseline by recovering nonlinear structure.  
**Datasets:** SUSY (500k pool / 5M total, d=18) · Cover Type Full (581k, d=54)
· HIGGS (500k pool / 11M total, d=28)  
**Note:** SUSY and HIGGS are cached as .npz in `data/` after first download.

---

### Experiment 4 — Learning Curves
**File:** `exp4_learning_curves.py`  
**Hypothesis:** ML-MSVM accuracy improves monotonically with n and retains its
advantage over the flat RFF baseline at all scales. The exact SVM is competitive
only until n ≈ 20k, after which it becomes infeasible. ML-MSVM fills this gap.  
**Datasets:** MNIST (up to 60k train) · SUSY (up to 400k train)  
**Output:** acc(n) and time(n) for all models → two thesis figures

---

### Experiment 5 — Depth × P Interaction
**File:** `exp5_depth_p_analysis.py`  
**Hypothesis A (depth):** Depth benefit is conditional on data complexity.
On Magic (low-d, tabular), L=1 suffices. On MNIST (high-d, image), L=3,4
provides additional gain when P is large.  
**Hypothesis B (P):** P saturation scales with boundary complexity, not d.
Magic saturates at P≈500; Spambase and MNIST require P≈2000.  
**Varied:** L ∈ {1,2,3,4} × P ∈ {250,500,1000,2000,3000,5000}  
**Datasets:** Magic · Spambase · MNIST

---

### Experiment 6 — C-Spread Ablation
**File:** `exp6_c_spread.py`  
**Hypothesis:** The logspace spread of C values across the m block SVMs is
essential. Without spread, m>1 degenerates to m=1 repeated. The default
logspace(−2,+1,4) balances coverage and stability.  
**Varied:** Five spread strategies with m=4 SVMs  
**Datasets:** Magic · Spambase · MNIST

---

### Experiment 7 — Nystroem Comparison
**File:** `exp7_nystroem.py`  
**Hypothesis:** ML-MSVM's architectural depth compensates for the approximation
advantage of the Nystroem method. At matched P, Nystroem is more accurate
per feature but is not composable into layers.  
**Varied:** P ∈ {500, 1000, 2000}  
**Datasets:** All Regime 2 + MNIST

---

### Experiment 8 — Arc-Cosine Degree Ablation
**File:** `exp8_arccosine_degree.py`  
**Hypothesis:** Degree=1 (ReLU) is optimal on tabular data. Degree=0 (step)
may be more robust in low-n high-d settings. Degree=2 (quadratic) over-smooths.  
**Varied:** degree ∈ {0, 1, 2} × L ∈ {1, 2}  
**Datasets:** Wine · Sonar · Magic · Spambase · MNIST

---

### Experiment 9 — Scalability Timing (Hero Experiment)
**File:** `exp9_scalability_timing.py`  
**Hypothesis:** ML-MSVM training time scales linearly with n while the exact
SVM grows quadratically. The accuracy gap between ML-MSVM and Flat RFF
widens at intermediate n, showing that depth adds value beyond approximation.  
**Datasets:** MNIST (up to 60k train) · SUSY (up to 400k train)  
**Output:** The primary scalability figure for the thesis.

---

### Experiment 10 — Head final_C Sensitivity
**File:** `exp10_final_c.py`  
**Hypothesis:** The architecture is robust to the choice of final_C over at
least two orders of magnitude, justifying the default of 1.0. If sensitive,
final_C should be added to the hyperparameter search.  
**Varied:** final_C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}  
**Datasets:** Magic · Spambase · MNIST
