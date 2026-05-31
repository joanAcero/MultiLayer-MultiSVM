# ML-MSVM Experiment Plan
**Thesis:** Deep kernel networks: neural architectures with SVM foundations  
**Author:** Joan Acero  
**Supervisor:** Lluís Belanche

---

## 1. Motivation and Gap Analysis

The following gaps in the experimental evidence were identified after the initial
benchmarking runs:

| Gap | Status before this plan |
|-----|------------------------|
| Width analysis (m): only one kernel at a time | Partial |
| Baseline: Mehrkanoon DHNKN | **Missing** |
| Baseline: Acero & Belanche ML-SVM (2025) | **Missing** |
| Regime 1 clean benchmark (fixed code) | **Missing** |
| Large-scale datasets (n > 100k) | **Missing** |
| Training time vs n (scalability) | **Missing** |
| Learning curves (accuracy vs n) | **Missing** |
| P sweep on Regime 3 (d=784) | **Missing** |
| Depth sweep L > 2 on complex data | **Missing** |

---

## 2. Published Baselines for Direct Comparison

The following numbers are taken from the literature and will be printed
alongside our results on the same datasets. Our code does NOT reimplement
these methods; it reproduces the experimental conditions.

### Acero & Belanche (2025) — ML-SVM, Table 3 (test error → accuracy)

| Dataset       | SVM (RBF) | MLP    | ML-SVM | 
|---------------|-----------|--------|--------|
| Glass         | 0.710     | 0.820  | 0.820  |
| Breast Cancer | 0.990     | 0.970  | 0.992  |
| Ionosphere    | 0.890     | 0.940  | 0.950  |
| Magic         | 0.840     | 0.830  | 0.850  |
| Spambase      | 0.690     | 0.920  | 0.850  |
| Cover Type    | 0.690     | 0.720  | 0.790  |

(Note: these come from Table 3 test errors, converted to accuracy.)

### Mehrkanoon & Suykens (2018) — DHNKN, Neurocomputing

Datasets used in their paper that overlap with ours:
MNIST, and several tabular benchmarks. Their reported test accuracy on
MNIST (RFF-based, 2-layer) is approximately 0.9756 (60k train / 10k test).
Our Regime 3 protocol uses 10k train / 2k test, so direct comparison
is not straightforward; we will note this explicitly in the results.

---

## 3. Datasets

### Regime 1 — Low n, any d (SVMs are strong)
| Dataset       | n     | d  | Classes | Source  |
|---------------|-------|----|---------|---------|
| Wine          | 178   | 13 | 3       | sklearn |
| Breast Cancer | 569   | 30 | 2       | sklearn |
| Ionosphere    | 351   | 34 | 2       | OpenML  |
| Sonar         | 208   | 60 | 2       | OpenML  |
| Glass         | 214   | 9  | 6       | OpenML  |

### Regime 2 — High n, moderate d (primary target)
| Dataset         | n       | d  | Classes | Source  |
|-----------------|---------|----|---------|---------|
| Magic           | 19,020  | 10 | 2       | OpenML  |
| Spambase        | 4,601   | 57 | 2       | OpenML  |
| Cover Type (sub)| 10,000  | 54 | 7       | OpenML  |
| SUSY            | 500,000 | 18 | 2       | OpenML  |
| Cover Type Full | 581,012 | 54 | 7       | OpenML  |
| HIGGS (sub)     | 500,000 | 28 | 2       | OpenML  |

### Regime 3 — High n, high d (secondary target)
| Dataset         | n      | d   | Classes | Source  |
|-----------------|--------|-----|---------|---------|
| MNIST           | 70,000 | 784 | 10      | OpenML  |
| Fashion-MNIST   | 70,000 | 784 | 10      | OpenML  |

---

## 4. Experiment List

### Experiment 1 — Width Analysis (exp1_width_analysis.py)
**Hypothesis:** The jump from m=1 to m=2 captures most available accuracy gain;
further increases yield diminishing returns. The optimal m depends on the kernel:
RBF benefits from m≥2 on all datasets, while arc-cosine peaks at m=1 on high-d data.  
**Datasets:** All Regime 1 and 2 (excluding large-scale).  
**Protocol:** 10× stratified 90/10 splits.  
**Varied:** m ∈ {1, d//4, d//2, d, 2d} × L ∈ {1,2,3} × kernel ∈ {RBF, ArcCos}.  
**Output:** accuracy, std, time per run, CSV.

### Experiment 2 — Main Benchmark (exp2_main_benchmark.py)
**Hypothesis:** ML-MSVM with best configuration (RBF m=2 L=2 or ArcCos m=1 L=1)
matches or exceeds Acero & Belanche ML-SVM and approaches Mehrkanoon DHNKN accuracy,
while being backpropagation-free and convexly trained.  
**Datasets:** All (R1, R2 standard, R3).  
**Protocol:** R1/R2: 10× 90/10. R3: 5× 10k/2k.  
**Models:** Linear SVM, Exact RBF SVM (R1 only), Flat RFF, ML-MSVM best configs.  
**Published comparison:** Acero & Belanche numbers printed inline.  
**Output:** accuracy, std, time, CSV + comparison table.

### Experiment 3 — Scalability on Large Datasets (exp3_scalability.py)
**Hypothesis:** ML-MSVM training time scales linearly with n (primal solver),
while exact RBF SVM becomes infeasible above n≈10k. ML-MSVM maintains competitive
accuracy at n=500k where no alternative kernel method can train in reasonable time.  
**Datasets:** SUSY (500k), Cover Type Full (581k), HIGGS (500k).  
**Protocol:** 5× stratified 80/20 splits.  
**Models:** Linear SVM, Flat RFF, ML-MSVM RBF m=2 L=2, ML-MSVM ArcCos m=1 L=1.  
**Output:** accuracy, std, time per run, CSV. Exact RBF SVM not run (infeasible).

### Experiment 4 — Learning Curves / n-Scaling (exp4_learning_curves.py)
**Hypothesis:** ML-MSVM accuracy improves monotonically with n and maintains
advantage over Linear SVM and Flat RFF at all scales. Exact RBF SVM
(competitive at small n) becomes infeasible before ML-MSVM loses its advantage.  
**Datasets:** MNIST and SUSY (most contrast between methods).  
**Protocol:** Fixed test set (5k MNIST, 50k SUSY). Train on
n ∈ {1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k (SUSY only)}.  
**Models:** Linear SVM, Exact RBF SVM (n ≤ 10k), Flat RFF, ML-MSVM best.  
**Output:** acc and time per (model, n_train) pair, CSV.

### Experiment 5 — Depth and P Analysis (exp5_depth_p_analysis.py)
**Hypothesis A (depth):** On simple/low-d data, L=1 is optimal. On complex/high-d
data (MNIST), deeper architectures (L=3,4) with sufficient P provide meaningful gains.  
**Hypothesis B (P):** P saturation threshold scales with data complexity, not
merely with d. P=1000 suffices for d≤10; P=2000 is needed for d≥57.  
**Datasets:** Magic (simple, low-d), Spambase (complex, high-d), MNIST (complex, high-d).  
**Protocol:** 5× splits per config.  
**Varied:** L ∈ {1,2,3,4}, P ∈ {250,500,1000,2000,3000,5000}.  
**Output:** accuracy, std, time, CSV.

---

## 5. Evaluation Protocol

### Split strategy
| Regime | n       | Protocol                          | Rationale                                    |
|--------|---------|-----------------------------------|----------------------------------------------|
| 1      | ≤2000   | 10× stratified 90/10              | Matches Acero & Belanche 2025                |
| 2      | 2k–100k | 10× stratified 90/10              | Consistent with R1; enough variance estimate |
| 3      | >100k   | 5× fixed 10k train / 2k test      | Controls compute; standard in literature     |
| Large  | >500k   | 5× stratified 80/20               | Full power; only feasible for fast models    |

### Metrics reported per experiment
- `acc_mean`: mean test accuracy across splits
- `acc_std`: standard deviation of test accuracy
- `time_mean`: mean wall-clock training time (seconds) per split
- `time_std`: std of training time
- `n_train`, `n_test`, `d`, `n_classes`: dataset metadata
- `kernel`, `L`, `m`, `P`: model hyperparameters

### CSV schema
All experiments write to a CSV file with one row per (model, split):
```
exp_id, dataset, n_total, n_train, n_test, d, n_classes,
model, kernel, L, m, P, split_id, acc, time_s, timestamp
```

---

## 6. Execution Order
```bash
bash run_all_experiments.sh
```
Runs experiments sequentially, each writing to its own log and CSV.
Individual experiments can be run standalone; all are resumable.
