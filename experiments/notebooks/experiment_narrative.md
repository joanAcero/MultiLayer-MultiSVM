# ML-MSVM — Experimental Narrative

**Thesis:** Deep Kernel Networks: Neural Architectures with SVM Foundations  
**Author:** Joan Acero Pousa · **Supervisor:** Lluís Belanche · **FIB-UPC**

---

## Prologue: Global Settings and Why They Matter

Before describing any individual experiment, the following settings are fixed throughout the
entire suite. Understanding them is essential to interpreting the results.

### P = 1000 Random Features

Each block generates a random projection Φ ∈ ℝ^(n×P) where P is the number of random
features. P=1000 is a practical compromise: large enough that the kernel approximation is
reasonable, small enough that Φ fits in RAM for n up to ~200k (Φ memory = n·P·8 bytes;
at n=200k this is 1.6 GB). Exp 5 studies what happens as P varies from 250 to 5000.

### tol = 1e-2 for every LinearSVC in the suite

This is the convergence tolerance of the TRON primal solver inside sklearn's LinearSVC.
**All LinearSVC instances use the same tolerance**: block SVMs, the head SVM, and all
external baselines (Linear SVM, Nyström+LinearSVC). This is a deliberate design choice
for fairness: with identical solver settings, timing differences between models reflect
only the feature space and the number of solves, not the precision of the optimisation.

Why 1e-2 instead of sklearn's default 1e-4? During development we discovered that the
arc-cosine feature map (ReLU) creates a mildly ill-conditioned Hessian for TRON at large n,
causing the solver to run hundreds of iterations at tol=1e-4. At tol=1e-2 the solver
converges in 5–30 iterations with identical accuracy (verified empirically: accuracy
difference < 0.001 on all tested datasets) but ~30× faster.

### Adaptive dual selection (dual = "auto")

LinearSVC can solve either the dual (cost O(n·P) per iteration, good for n < P) or the
primal (cost O(P·n) per iteration, good for n ≥ P). Setting `dual="auto"` lets sklearn
choose based on n vs P. Inside ML-MSVM blocks, we replicate this logic manually. The
threshold is n < P=1000: for small training sets the dual is preferred, for large sets
the primal.

### Canonical configurations

After Exp 1 (width analysis), two configurations were fixed as defaults for both kernels
in all subsequent experiments unless explicitly varied:

- **ML-MSVM Arc:** m=1 SVM per block, L=2 layers (2 blocks + head)
- **ML-MSVM RBF:** m=2 SVMs per block, L=2 layers (5 total SVMs: 2+2+1)

The rationale and limitations of this choice are examined critically in Exp 13.

### Block C values

Each SVM in a block has a regularisation parameter C. The original scheme used a logspace
spread `C_values = logspace(−2, 2, m)`. Exp 6 shows this hurts; from Exp 11 onward,
**uniform C=1.0 for all block SVMs** is used.

---

## Experiment 1 — Width Analysis

### Purpose

Study how block width (m, number of SVMs per block) and depth (L, number of blocks)
affect accuracy, for both the RBF and arc-cosine kernels across all dataset regimes.

### Settings

- P = 1000, C_values = logspace(−2, 2, m)
- L ∈ {1, 2, 3, 4}, m ∈ {1, 2, 3, 4, 6, 10} (capped at 10 to prevent combinatorial blow-up)
- Datasets: Wine, Breast Cancer, Ionosphere, Sonar, Glass (R1), Magic, Spambase,
  Cover Type-sub (R2), MNIST, Fashion-MNIST (R3)
- Ablation: arc-cosine with `normalize_inter_layer=False` at L=2, m=1

### Results

**Regime 1 (small n):** Both kernels reach near-ceiling accuracy at m=1 on Wine (0.983)
and Breast Cancer (0.972–0.981). Variance is high (~0.02–0.04 std) due to small test sets.
No width or depth benefit is visible above noise for binary Regime-1 datasets. Glass
(K=6, n=214) is the exception: some m>1 configs reach 0.746 vs 0.714 for arc-cosine m=1.

**Spambase (d=57, K=2):** Arc-cosine m=1 is the best arc-cosine configuration (0.9399).
Width beyond m=1 hurts arc-cosine with c_spread: m=2 drops to 0.9338 (−0.006). RBF is
flatter: m=1 L=1=0.9356, m=1 L=4=0.9362; width adds almost nothing for RBF either.

**Magic (d=10, K=2):** Both kernels plateau around 0.867 at any (m, L). Width and depth
do nothing measurable on this low-dimensional binary dataset.

**Cover Type-sub (d=54, K=7):** The most sensitive dataset. RBF benefits visibly from
width: m=1 L=1=0.7292, m=10 L=2=0.7492 (+0.020). Arc-cosine is already strong at m=1
L=1 (0.7512) and gains less from width (m=3 L=2=0.7597, marginal beyond).

**MNIST (d=784, K=10):** Arc-cosine m=1 L=2=0.9234. Adding width with c_spread hurts
(m=2 L=1=0.9199). RBF gains slightly from depth: m=1 L=2=0.9231 vs m=1 L=1=0.9161.

**Normalisation ablation:** Setting `normalize_inter_layer=False` changes accuracy by at
most 0.004 on any tested (dataset, config) pair. Inter-layer normalisation is a safe
default but not a critical component.

### Conclusions

1. **Arc-cosine peaks at m=1** for binary tasks; the logspace-C scheme at m>1 actively
   hurts. This motivates Exp 6 and 11.
2. **RBF benefits modestly from m=2** (Cover Type, Magic). Beyond m=2, gains are negligible.
3. **Depth (L) contributes most on multiclass structured data** (Cover Type: +0.02 from L=1 to L=2).
4. **Low-d binary problems** (Magic, Spambase) are insensitive to both m and L.
5. These findings establish the canonical configurations: Arc m=1, RBF m=2, both at L=2.

---

## Experiment 2 — Main Head-to-Head Benchmark

### Purpose

Comprehensive accuracy comparison on the full dataset suite (Regimes 1–3) against five
baselines and two published references.

### Settings

- Canonical configs: ML-MSVM Arc m=1 L=1 and L=2; ML-MSVM RBF m=2 L=1 and L=2
- Baselines: Linear SVM, Exact RBF SVM (R1/R2 only), Flat RFF RBF (L=0), Flat RFF Arc (L=0)
- Exact RBF skipped where n_train > 50k
- Published references: ML-SVM (Acero & Belanche 2025, inline), DHNKN (Mehrkanoon &
  Suykens 2018, MNIST only, different 60k/10k protocol)

### Results

| Dataset | Linear | RBF exact | Flat RBF | Flat Arc | MSVM Arc L2 | MSVM RBF L2 | ML-SVM pub. |
|---------|--------|-----------|---------|---------|-------------|-------------|------------|
| Wine (R1) | 0.994 | 0.983 | 0.989 | 0.983 | 0.983 | 0.983 | — |
| Breast C. (R1) | 0.968 | 0.975 | 0.981 | 0.970 | 0.970 | 0.968 | 0.992 |
| Ionosphere (R1) | 0.867 | 0.939 | 0.931 | 0.928 | 0.936 | 0.939 | 0.950 |
| Sonar (R1) | 0.767 | 0.848 | 0.810 | 0.862 | 0.843 | 0.829 | — |
| Glass (R1) | 0.636 | 0.736 | 0.723 | 0.696 | 0.746 | 0.727 | 0.820 |
| Magic (R2) | 0.785 | 0.866 | 0.863 | 0.867 | 0.867 | 0.866 | 0.850 |
| Spambase (R2) | 0.925 | 0.937 | 0.935 | 0.940 | 0.940 | 0.933 | 0.850 |
| CoverType-sub (R2) | 0.711 | 0.729 | 0.730 | 0.750 | 0.757 | 0.747 | 0.790 |
| MNIST (R3) | 0.875 | — | 0.917 | 0.923 | 0.923 | 0.922 | 0.976† |
| Fashion (R3) | 0.800 | — | 0.847 | 0.837 | 0.835 | 0.848 | — |

†DHNKN uses full 60k/10k protocol; our MNIST uses 10k/2k (Regime 3).

### Conclusions

1. **Against exact RBF SVM:** ML-MSVM matches the exact RBF SVM within ~0.01 on Regime-1
   datasets where both can run, at substantially lower cost for large n.
2. **Against published ML-SVM:** ML-MSVM beats it substantially on Spambase (+0.090) and
   Magic (+0.017). It falls short on Glass (−0.074, high variance with n=214) and
   Cover Type-sub (−0.033, the published protocol used more training data).
3. **Kernel split first appears here:** Arc-cosine wins on Spambase and Cover Type-sub;
   RBF wins on Fashion. This pattern becomes the main theme of Exp 13.
4. **Linear SVM is strongly beaten on R2/R3 datasets** (+0.06 to +0.08pt by all RFF methods),
   confirming that the feature map adds value beyond raw linear classification.

---

## Experiment 3 — Large-scale Feasibility

### Purpose

Test ML-MSVM in the regime where exact SVM is computationally infeasible. This is the
primary design target: n up to 400k where no kernel matrix can be formed.

### Settings

- n_train ∈ {50k, 100k, 200k, 400k}, fixed 50k test set, 3 seeds
- No exact RBF SVM (all n exceed the 50k limit)
- Models: Linear SVM, Flat RFF RBF, Flat RFF Arc, ML-MSVM RBF m=2 L=2, ML-MSVM Arc m=1 L=1
- SUSY: 5M rows, 1.1M sub-pool; Cover Type Full: 581k; HIGGS: 11M rows, 1.1M sub-pool

### Results

**SUSY (d=18, K=2):**

| n_train | Linear | Flat RBF | Flat Arc | MSVM RBF | MSVM Arc |
|---------|--------|---------|---------|---------|---------|
| 50k | 0.787 | 0.802 | 0.799 | 0.802 | 0.799 |
| 200k | 0.787 | 0.803 | 0.802 | 0.803 | 0.802 |
| 400k | 0.787 | 0.803 | 0.802 | **0.804** | 0.802 |

All models saturate near 0.80 by n=25k. RBF holds a consistent ~0.002 edge over
arc-cosine on this low-d (d=18) physics dataset. ML-MSVM adds marginal value over
the flat RFF baseline.

**Cover Type Full (d=54, K=7):**

| n_train | Linear | Flat RBF | Flat Arc | MSVM RBF | MSVM Arc |
|---------|--------|---------|---------|---------|---------|
| 50k | 0.710 | 0.749 | 0.773 | 0.774 | **0.775** |
| 200k | 0.711 | 0.760 | 0.781 | 0.782 | **0.782** |
| 400k | 0.710 | 0.764 | 0.781 | **0.783** | **0.783** |

Arc-cosine provides a sustained +1.7pt advantage over RBF at the flat level (0.781 vs
0.764 at n=400k), and both ML-MSVM variants match or exceed the flat arc-cosine baseline.
The linear SVM is stuck at 0.71 — the feature map is essential for this multiclass problem.

**HIGGS (d=28, K=2, full 1.1M pool):**

| n_train | Linear | Flat RBF | Flat Arc | MSVM RBF | MSVM Arc |
|---------|--------|---------|---------|---------|---------|
| 50k | 0.644 | 0.682 | 0.677 | 0.683 | 0.677 |
| 200k | 0.644 | 0.682 | 0.677 | 0.683 | 0.677 |

HIGGS is known to plateau near 0.64–0.68 for all shallow kernel methods; deep neural
networks reach ~0.88 by exploiting high-order feature interactions. RBF holds a small
advantage (~0.005) over arc-cosine on this physics dataset.

### Conclusions

1. **ML-MSVM is feasible at n=400k** for all three datasets with modest hardware (peak Φ
   memory 1.6 GB at n=200k, training times 80–1640s per fit).
2. **The kernel choice is the dominant accuracy factor on Cover Type.** Arc-cosine is +1.7pt
   over RBF at the flat level — a gap that persists at all training sizes.
3. **On SUSY and HIGGS (low/moderate d), RBF marginally beats arc-cosine.** The dimensional
   split observed in Exp 2 is confirmed at scale.
4. **HIGGS accuracy is bounded by the depth of the method, not by n.** Shallow kernel methods
   have a ceiling around 0.68; this is a genuine limitation to report honestly.

---

## Experiment 4 — Learning Curves

### Purpose

Dense n-grid study on MNIST and SUSY to characterise the accuracy-vs-n and time-vs-n
curves for all models, including Flat RFF Arc (which was missing from earlier scalability
studies).

### Settings

- n_train from 500 to 200k (SUSY) or 60k (MNIST), 3 seeds
- All five models including Flat RFF Arc
- Fixed 50k test (SUSY) or 5k test (MNIST)

### Results

MNIST and SUSY results replicate Exp 9 exactly at overlapping n values (deterministic
seeding). The dense grid adds intermediate points confirming the monotone growth pattern.

Key additional observation: **Flat RFF Arc and ML-MSVM Arc m=1 L=1 track each other
very closely on SUSY at all n**, with ML-MSVM adding at most 0.001 over the flat baseline.
On MNIST, ML-MSVM Arc surpasses Flat Arc more clearly, especially above n=10k.

### Conclusions

Consistent with Exp 9. Serves primarily as a robustness confirmation and supplies the
full learning-curve plot data for the thesis. No conclusions that contradict Exp 9.

---

## Experiment 5 — Depth × P Interaction

### Purpose

Test whether additional layers substitute for additional random features. Hypothesis:
at small P, each extra layer extracts more information from the same feature budget;
at large P, the single layer already captures most of the signal and depth adds little.

### Settings

- Arc-cosine m=1, C=1.0; L ∈ {1, 2, 3, 4}; P ∈ {250, 500, 1000, 2000, 3000, 5000}
- Datasets: Magic, Spambase, MNIST (most sensitive to approximation quality)

### Results (MNIST, arc-cosine)

| P | L=1 | L=2 | L=3 | L=4 | Depth gain (L2−L1) |
|---|-----|-----|-----|-----|-------------------|
| 250 | 0.879 | 0.885 | 0.885 | 0.885 | **+0.006** |
| 500 | 0.904 | 0.911 | 0.909 | 0.909 | +0.007 |
| 1000 | 0.920 | 0.921 | 0.921 | 0.921 | +0.001 |
| 2000 | 0.934 | 0.935 | 0.936 | 0.936 | +0.001 |
| 5000 | 0.943 | 0.945 | 0.945 | 0.945 | +0.002 |

The depth benefit is **largest at small P and diminishes as P grows.** At P=250 adding
a second layer recovers +0.006 accuracy; at P=1000 the gain is +0.001 (within noise of
the 5-seed evaluation). At P=5000 both L=1 and L=2 converge to 0.943–0.945.

On Magic and Spambase (simpler decision boundaries), depth adds nothing at any P.

### Conclusions

1. **Depth is a width-efficiency mechanism.** The primary value of stacking blocks is in
   getting higher accuracy *at low P*, not in breaking through any accuracy ceiling.
2. **At the standard P=1000, depth contributes minimally on MNIST.** This means the
   canonical L=2 is slightly better than L=1 but the gap is small.
3. **For resource-constrained deployment**, L=2 at P=500 approaches the accuracy of L=1 at
   P=1000 at half the feature-map computation cost.

---

## Experiment 6 — C-spread Ablation

### Purpose

Test whether diversity through logspace-C values across the m block SVMs is beneficial.
Compares five C-spread schemes on RBF m=4 L=2.

### Settings

- RBF kernel, m=4, L=2, P=1000
- Schemes: same-1.0 (all C=1), same-0.1 (all C=0.1), narrow (1e-1 to 10), default (1e-2 to 100), wide (1e-3 to 1000)
- Datasets: Magic, Spambase, MNIST (10 seeds each)

### Results

| Scheme | Magic | Spambase | MNIST |
|--------|-------|---------|-------|
| same-1.0 | 0.865 | **0.935** | **0.924** |
| narrow | 0.868 | 0.932 | 0.917 |
| default (logspace) | 0.867 | 0.931 | 0.917 |
| wide | **0.868** | 0.930 | 0.913 |
| same-0.1 | 0.858 | 0.928 | 0.902 |

On Magic, differences across all schemes are within one standard deviation (~0.007) and
no reliable conclusion can be drawn. On Spambase and MNIST, **same-1.0 is clearly best**:
it outperforms the default logspace by +0.004 on Spambase and +0.007 on MNIST, both
larger than the per-split standard deviations. The wide spread (very small minimum C)
is worst on MNIST (0.913 vs 0.924 for same-1.0).

### Conclusions

1. **C-spread diversity does not help and often hurts.** Uniform C=1.0 is the best-performing
   scheme on the two datasets where differences are statistically distinguishable.
2. **Very small C values (0.001 or lower) are harmful** — the severely under-regularised SVMs
   produce weak, noisy weight vectors that degrade the downstream projection.
3. From this experiment onward, **C=1.0 is used for all block SVMs**.
4. This result motivates Exp 11: if C-spread is not the right diversity mechanism, what is?

---

## Experiment 7 — Nyström Comparison

### Purpose

Nyström approximation is a strong, data-adaptive baseline for kernel approximation that
selects P landmark points from the training data to form an explicit feature map.
This experiment tests whether the random, data-independent approach of ML-MSVM can
compete with this adaptive method.

### Settings

- P ∈ {500, 1000, 2000}; canonical ML-MSVM configs and Flat RFF both kernels
- Nyström baseline: sklearn Nystroem with RBF kernel, γ=scale, P landmarks
- 20 seeds on Regime-2 datasets, 10 on MNIST

### Results (P=2000)

| Dataset | Flat RBF | Nyström | MSVM RBF | MSVM Arc | Arc − Nyström |
|---------|---------|--------|---------|---------|--------------|
| Magic | 0.864 | **0.871** | 0.869 | 0.869 | −0.002 |
| Spambase | 0.937 | 0.937 | 0.940 | **0.945** | +0.008 |
| Cover Type | 0.732 | 0.740 | 0.745 | **0.763** | +0.022 |
| MNIST | 0.922 | 0.933 | 0.931 | **0.937** | +0.003 |

On Magic (d=10, simple boundary), the Nyström baseline wins: data-adaptive landmarks
exploit the local structure that random features miss. On all higher-d datasets,
ML-MSVM Arc exceeds Nyström. The Cover Type advantage (+0.022) is clearly significant
given the 20-seed evaluation (std ≈ 0.01).

### Conclusions

1. **ML-MSVM Arc outperforms Nyström on all tested datasets with d ≥ 54**, despite Nyström's
   data-adaptive advantage. The depth + arc-cosine kernel combination provides a richer
   representation than a single data-adapted RBF feature map.
2. **On low-d problems (Magic), data-adaptive methods are preferred.** Nyström wins by a
   small margin that may not be significant given the 20-seed evaluation std.
3. This positions ML-MSVM as competitive with the strongest single-layer kernel approximation
   method for structured, higher-dimensional data.

---

## Experiment 8 — Arc-cosine Degree

### Purpose

The arc-cosine kernel family K_n(x, y) is indexed by degree n (Cho & Saul 2010).
Degree 0 uses a step function, degree 1 uses ReLU, degree 2 uses squared ReLU.
This experiment determines which degree is best.

### Settings

- Degrees 0, 1, 2; m=1, L ∈ {1, 2, 3, 4}; P=1000
- Full dataset suite (Regimes 1–3)
- Best accuracy at any L reported

### Results

| Dataset | Degree 0 | Degree 1 | Degree 2 | Winner |
|---------|---------|---------|---------|--------|
| Wine (d=13) | **0.994** | 0.983 | 0.972 | deg=0 |
| Sonar (d=60) | 0.781 | **0.862** | 0.824 | deg=1 |
| Magic (d=10) | 0.834 | 0.868 | **0.869** | deg=1/2 tie |
| Spambase (d=57) | 0.937 | **0.940** | 0.937 | deg=1 |
| MNIST (d=784) | 0.907 | **0.923** | 0.921 | deg=1 |
| MNIST (time) | ~5s | ~6s | **~84s** | deg=0/1 |

Degree 2 at L=2 on MNIST takes approximately 84 seconds per split vs ~6 seconds for
degree 1, for a lower accuracy (0.921 vs 0.923). Degree 2 is dominated on every dataset.

Degree 0 wins only on Wine — a small (n=178), low-d (d=13) problem that is nearly
linearly separable, where the step-function boundary is sufficient and noise-robust.

### Conclusions

1. **Degree 1 (ReLU) is the optimal arc-cosine degree** across the tested datasets. It wins
   or ties on 4 of 5 datasets and is never the worst choice.
2. **Degree 2 is dominated**: similar accuracy to degree 1 but ~14× slower on MNIST due to
   poorer conditioning of the quadratic feature map for the primal LinearSVC solver.
3. **Degree 0 has a niche on small/low-d data**, but degree 1 is the safe default.
4. This validates the theoretical motivation: degree 1 corresponds to one layer of an
   infinite-width ReLU network (Cho & Saul 2010), supporting the paper's framing.

---

## Experiment 9 — Scalability Timing (Hero Experiment)

### Purpose

The primary scalability figure for the thesis. Accuracy and training time as functions
of n, comparing all models including the exact RBF SVM (up to its feasibility limit).

### Settings

- MNIST (d=784, K=10): n_train ∈ {500, 1k, 2k, 5k, 10k, 20k, 40k, 60k}; fixed 5k test; 3 seeds
- SUSY (d=18, K=2): n_train ∈ {1k, 5k, 10k, 25k, 50k, 100k, 200k}; fixed 50k test; 3 seeds
- Exact RBF SVM: skipped above n_train = 50k
- Models: Linear SVM, Exact RBF SVM, Flat RFF RBF, Flat RFF Arc, ML-MSVM RBF m=2 L=2, ML-MSVM Arc m=1 L=1

### Results — MNIST

| n_train | Linear | RBF exact | Flat RBF | Flat Arc | MSVM RBF | MSVM Arc |
|---------|--------|-----------|---------|---------|---------|---------|
| 500 | 0.752 / 0.3s | 0.810 / 0.8s | 0.810 / 0.4s | 0.824 / 0.2s | 0.815 / 1.0s | 0.823 / 0.2s |
| 2,000 | 0.799 | 0.894 / 2.9s | 0.878 | 0.885 | 0.877 | 0.885 |
| 10,000 | 0.873 | 0.937 / 17.8s | 0.916 | 0.925 | 0.919 | 0.923 |
| 20,000 | 0.901 | 0.951 / 48.8s | 0.925 | 0.935 | 0.932 | 0.934 |
| 40,000 | 0.914 | 0.963 / **135s** | 0.933 | 0.942 | 0.940 | 0.942 |
| 60,000 | 0.917 / 75s | **SKIP** | 0.935 / 33s | 0.946 / 22s | 0.942 / 107s | **0.947 / 23s** |

At n=60k (where exact RBF SVM cannot be run):
- ML-MSVM Arc achieves the highest accuracy among runnable models: **0.947 at 23s**
- ML-MSVM RBF reaches 0.942 at **107s** (4.6× slower for −0.005 accuracy)
- Flat RFF Arc reaches 0.946 at 22s — almost as good as ML-MSVM Arc

The exact RBF SVM already took 135s per fit at n=40k (0.963 accuracy) and was then
capped. The timing crossover (exact SVM slower than ML-MSVM) occurs around n=35–40k.

### Results — SUSY

All models converge to 0.787–0.804 by n=25k and add nothing further. RBF holds a
consistent ~0.002 edge over arc-cosine throughout.

| n_train | Linear | Flat RBF | Flat Arc | MSVM RBF | MSVM Arc |
|---------|--------|---------|---------|---------|---------|
| 10,000 | 0.787 | 0.798 | 0.791 | 0.797 | 0.791 |
| 50,000 | 0.787 | 0.802 | 0.799 | 0.802 | 0.799 |
| 200,000 | 0.787 | 0.803 | 0.802 | 0.803 | 0.802 |

Timing on SUSY: ML-MSVM Arc is consistently 2–3× faster than ML-MSVM RBF at each n.
At n=200k: Arc 28s, RBF 80s. The timing difference is disproportionate to the accuracy
difference (0.001).

### Conclusions

1. **On MNIST at large n (≥40k), ML-MSVM Arc delivers the best accuracy among all
   feasible methods (0.947) and does so 4.6× faster than ML-MSVM RBF.**
2. **Flat RFF Arc is almost as good as ML-MSVM Arc on MNIST** (0.946 vs 0.947), suggesting
   that depth adds only marginal value at n=60k, P=1000. The flat baseline is competitive.
3. **The exact RBF SVM is dominated in timing before its accuracy advantage disappears.**
   At n=40k it is already the slowest model (135s vs 15s for Arc) while being the only
   model with clearly higher accuracy (0.963 vs 0.942). This is the central scalability
   trade-off.
4. **On SUSY, all RFF methods are nearly equivalent in accuracy.** The kernel barely matters
   for this low-d dataset.
5. **ML-MSVM RBF is 4.6× slower than ML-MSVM Arc** partly because it has more SVMs (5 vs 2)
   and partly because RBF cosine features are harder to optimise for the linear solver than
   ReLU features. This motivates Exp 13.

---

## Experiment 10 — Head Final_C Sensitivity

### Purpose

Test whether the regularisation parameter of the final head SVM (final_C) is worth tuning.

### Settings

- RBF m=2 L=2; final_C ∈ {0.001, 0.01, 0.1, 1.0, 10, 100, 1000}
- Magic, Spambase, MNIST

### Results

| Dataset | Min acc | Max acc | Range (Δ) |
|---------|---------|---------|----------|
| Magic | 0.8714 | 0.8716 | 0.0002 |
| Spambase | 0.9297 | 0.9323 | 0.0026 |
| MNIST | 0.9193 | 0.9203 | 0.0010 |

The accuracy range across four orders of magnitude of C is less than 0.003 on all datasets.
This is smaller than the per-split standard deviation on all datasets.

### Conclusions

1. **The head SVM's regularisation parameter is irrelevant.** The block representation
   is already well-separated enough that the head linear classifier is operating in a
   regime where C does not meaningfully affect the decision boundary.
2. **Fix final_C = 1.0.** One fewer hyperparameter with no accuracy cost.

---

## Experiment 11 — Diversity Mechanism Ablation

### Purpose

Exp 6 showed that c_spread (logspace C) hurts arc-cosine and does not help RBF. But
m=1 is a degenerate case: with a single SVM, the concept of "diversity" is undefined.
This experiment tests whether *alternative* diversity mechanisms can make m>1 genuinely
beneficial.

**Four mechanisms:**
- `c_spread`: original logspace-C scheme
- `same_c`: all SVMs at C=1.0, same data — theoretically predicts rank-1 W for K=2
  (each SVM solves the same convex problem, converging to the same solution)
- `bootstrap`: each SVM trains on a fresh 80% random subsample of the training set (C=1.0)
- `feature_subset`: each SVM uses a random P//m-dimensional subset of Φ's columns (C=1.0)

### Settings

- m ∈ {1, 2, 3, 4, 6}, L ∈ {1, 2}, P=1000
- Both kernels; 4 datasets: Magic (d=10), Spambase (d=57), Cover Type-sub (d=54, K=7),
  MNIST (d=784, K=10)

### Results — Magic (d=10, K=2)

All modes at any m produce accuracy in the range 0.862–0.868 for arc-cosine and
0.856–0.867 for RBF. Given per-split standard deviations of 0.007–0.009, no diversity
mechanism produces a reliably distinguishable change on this low-d binary dataset.
**No diversity mechanism helps Magic.**

Feature_subset is however notably faster: at m=4 it runs in 2.4–3.8s vs 11–17s for
other modes (problem size reduced by m-fold).

### Results — Spambase (d=57, K=2)

| Kernel | Mode | Best acc (m) | Δ vs m=1 |
|--------|------|-------------|---------|
| Arc | m=1 baseline | 0.9399 | — |
| Arc | c_spread | 0.9349 (m=2) | **−0.005** |
| Arc | same_c | 0.9399 (m=2) | ~0.000 |
| Arc | bootstrap | 0.9410 (m=4) | +0.001 |
| Arc | feature_subset | 0.9408 (m=4) | +0.001 |
| RBF | m=1 baseline | 0.9356 | — |
| RBF | c_spread | 0.9325 (m=2) | −0.003 |
| RBF | same_c | 0.9364 (m=2) | +0.001 |
| RBF | bootstrap | 0.9349 (m=6) | −0.001 |
| RBF | feature_subset | 0.9299 (m=2) | **−0.006** |

c_spread continues to hurt arc-cosine (Δ=−0.005). same_c is flat — confirming rank-1
collapse for K=2: with identical C and data, all SVMs converge to the same solution.
bootstrap and feature_subset give Δ=+0.001 for arc-cosine, within the standard deviation
(~0.010): a consistent directional trend but not statistically distinguishable.

### Results — Cover Type-sub (d=54, K=7) — Multiclass reveals new behaviour

| Kernel/L | Mode | Best acc (m) | Δ vs m=1 |
|---------|------|-------------|---------|
| Arc L=1 | m=1 baseline | 0.7512 | — |
| Arc L=1 | c_spread | 0.7520 (m=4) | +0.001 |
| Arc L=1 | same_c | 0.7512 | ~0.000 |
| Arc L=1 | bootstrap | 0.7464 (m=6) | −0.005 |
| Arc L=2 | m=1 baseline | 0.7573 | — |
| Arc L=2 | **same_c** | **0.7617 (m=4)** | **+0.004** |
| Arc L=2 | bootstrap | 0.7578 (m=4) | +0.001 |
| RBF L=1 | m=1 baseline | 0.7292 | — |
| **RBF L=1** | **c_spread** | **0.7465 (m=6)** | **+0.017** |
| **RBF L=2** | **c_spread** | **0.7522 (m=3)** | **+0.018** |
| RBF L=2 | same_c | 0.7357 | +0.002 |
| RBF L=2 | bootstrap | 0.7347 | +0.001 |

**Critical finding — multiclass changes everything:**

For **arc-cosine**, same_c at m=4 L=2 reaches 0.7617 vs 0.7573 at m=1 (Δ=+0.004).
The reason: with K=7 classes and OvR, each "identical" SVM (same C, same data) trains
7 separate binary classifiers producing 7 distinct weight vectors. So W has m×K=28 columns
even with same_c — not rank-1. The head SVM can exploit this richer (though partly
redundant) representation.

For **RBF**, c_spread at m=2–3 gives +0.017–0.018 over m=1. This is clearly significant
(Δ >> std ≈ 0.013). C-spread diversity, which hurts binary problems, works for RBF on
multiclass: different C values produce differently-regularised per-class boundaries that
complement each other.

The timing at m=4–6 with K=7 is very high: arc-cosine same_c m=4 L=2 takes 65–100s per
split, because each block SVM internally trains K=7 binary SVMs (so the actual number
of LinearSVC solves is m×K per block). Bootstrap at m=6 reaches 104s.

### Results — MNIST (d=784, K=10)

| Kernel/L | Mode | Best acc (m) | Δ vs m=1 | Note |
|---------|------|-------------|---------|------|
| Arc L=1 | baseline | 0.9223 | — | |
| Arc L=1 | c_spread | 0.9257 (m=2) | +0.003 | marginal |
| Arc L=1 | same_c | 0.9219 | ~0.000 | |
| Arc L=1 | **bootstrap** | **0.9244 (m=4)** | **+0.002** | |
| Arc L=2 | baseline | 0.9234 | — | |
| Arc L=2 | **bootstrap** | **0.9290 (m=4)** | **+0.006** | monotone: m=2→3→4 |
| Arc L=2 | same_c | 0.9245 | +0.001 | |
| RBF L=1 | baseline | 0.9161 | — | |
| RBF L=1 | feature_subset | 0.9018 | **−0.014** | severe hurt |
| RBF L=2 | baseline | 0.9231 | — | |
| RBF L=2 | same_c | 0.9244 | +0.001 | |
| RBF L=2 | feature_subset | 0.9111 | **−0.012** | severe hurt |

Arc-cosine L=2 bootstrap shows a **monotonically increasing accuracy pattern** with m:
m=1: 0.9234, m=2: 0.9256, m=3: 0.9284, m=4: 0.9290. The gain of +0.006 is roughly
equal to one standard deviation (std ≈ 0.005–0.006 for these configs), so it is
consistent but should not be over-interpreted as definitively significant.

RBF feature_subset loses 1.2–1.4pt on MNIST: at m=6, each SVM uses only P//6 ≈ 167
random features, far too few for 784-dimensional digit images.

### Conclusions from Exp 11

1. **The diversity mechanism matters more than the number of SVMs.** c_spread (original)
   hurts arc-cosine on binary data; same_c is flat; bootstrap gives the most consistent gain.
2. **For binary tasks (K=2), m=1 remains the best default** for arc-cosine. Bootstrap at
   m=3–4 gives a directionally positive but noise-level benefit on Spambase.
3. **For multiclass tasks (K>2), the picture is fundamentally different:** even same_c
   benefits arc-cosine because OvR provides K genuinely different weight vectors per SVM;
   c_spread benefits RBF because different regularisations capture different class boundaries.
4. **Bootstrap diversity is the most promising mechanism for arc-cosine on high-d data**
   (MNIST: monotone +0.006 at m=4 L=2), vindicating the bagging-SVM intuition.
5. **Feature_subset is fastest** (3–6× speed-up at m=4 vs other modes) but hurts accuracy
   on high-d datasets where each individual SVM needs sufficient features.
6. **RBF is generally better left at m=1 with same_c (no real diversity mechanism helps
   binary RBF)**, while c_spread remains useful for RBF multiclass.

---

## Experiment 12 — RFF Quality: Monte Carlo vs Orthogonal RF vs Quasi-Monte Carlo

### Purpose

Test whether structurally better random feature sampling (Orthogonal Random Features —
ORF, or Quasi-Monte Carlo Sobol — QMC) closes the accuracy gap to the exact RBF SVM,
and whether it affects arc-cosine features similarly.

**Sampling schemes:**
- `standard`: i.i.d. Gaussian rows of Ω (current baseline)
- `orf`: Orthogonal RF (Yu et al. 2016) — rows orthogonalised via QR decomposition; same
  marginal distribution as Gaussian but reduced inter-feature correlation
- `qmc`: Quasi-Monte Carlo — Sobol low-discrepancy sequence transformed to approximate
  N(0,I) via inverse normal CDF; better spectral coverage

### Settings

- L=1, m=1, C=1.0; P ∈ {250, 500, 1000, 2000, 5000}
- Both kernels; MNIST (5 seeds, 10k/2k) and Spambase (7 seeds, 4140/461)
- Exact RBF SVM included as oracle

### Results — MNIST

| P | RBF std | RBF orf | RBF qmc | Arc std | Arc orf | Arc qmc |
|---|---------|--------|--------|---------|--------|--------|
| 250 | 0.875 | 0.879 | 0.875 | 0.878 | 0.880 | 0.887 |
| 500 | 0.900 | 0.902 | 0.903 | 0.906 | 0.910 | 0.907 |
| 1000 | 0.916 | 0.917 | 0.916 | 0.922 | 0.922 | 0.919 |
| 2000 | 0.922 | 0.925 | 0.925 | **0.937** | 0.937 | 0.937 |
| 5000 | 0.930 | 0.930 | 0.928 | **0.946** | 0.945 | 0.946 |
| Exact RBF | **0.938** | | | | | |

**Arc-cosine standard at P=5000 (0.946) exceeds the exact RBF SVM (0.938).** This is not an
approximation-quality result — it shows the arc-cosine kernel has a strictly higher
accuracy ceiling than the RBF kernel for MNIST. The arc-cosine kernel is genuinely a
better model for this task, not merely a faster approximation.

The gap between arc-cosine and RBF standard grows with P: at P=250 the gap is +0.003;
at P=5000 it is +0.016. Arc-cosine converges faster in P and to a higher limit.

ORF and QMC effects are small and inconsistent: differences are at most +0.004 over
standard MC, often within the 5-seed standard deviation (~0.004–0.007).

### Results — Spambase

| P | RBF std | RBF orf | Arc std | Arc orf | Arc qmc |
|---|---------|--------|---------|--------|--------|
| 250 | 0.924 | 0.928 | 0.926 | **0.936** | 0.930 |
| 500 | 0.928 | 0.935 | 0.935 | **0.941** | 0.936 |
| 1000 | 0.935 | 0.934 | 0.941 | **0.945** | 0.940 |
| 2000 | 0.937 | 0.935 | 0.946 | 0.946 | 0.943 |
| 5000 | 0.936 | 0.937 | 0.946 | 0.941 | **0.948** |
| Exact RBF | 0.934 | | | | |

On Spambase, **ORF is more consistently helpful for arc-cosine at low P**: +0.010 at P=250,
+0.006 at P=500, +0.004 at P=1000. The benefit diminishes as P grows and the standard
MC approximation improves. By P=2000 standard and ORF are equivalent.

### Conclusions

1. **Neither ORF nor QMC closes the gap to the exact RBF SVM meaningfully.** At P=5000, the
   RBF standard gap to the oracle is 0.008; ORF reduces it to 0.007. The gap is primarily
   a finite-P kernel approximation effect, not a sampling variance effect.
2. **ORF provides small, consistent benefit at low P for both kernels** (especially arc-cosine
   on Spambase: +0.004–0.010 for P ≤ 1000). This is useful in resource-constrained settings.
3. **QMC (Sobol) is inconsistent** — sometimes better, sometimes worse than standard MC.
   It is not recommended as a reliable improvement.
4. **The arc-cosine kernel converges faster in P and to a strictly higher limit than RBF on
   MNIST.** This is the key finding: the kernel choice (arc vs RBF) dominates the sampling
   scheme choice (std vs ORF vs QMC) at all P values tested.

---

## Experiment 13 — Fair Kernel Comparison

### Purpose

Previous experiments used different canonical configurations for the two kernels (Arc m=1 L=1
vs RBF m=2 L=2), which confounds kernel effects with configuration effects. This experiment
tests both kernels under *identical* (m, L, C) settings to isolate the pure kernel effect.

### Settings

- All block SVMs: C=1.0 (post-Exp 6 fix)
- Configs: Flat (L=0), m1L1, m1L2, m1L3, m2L1, m2L2 — **same for both kernels**
- Baselines: Linear SVM, Exact RBF SVM (where feasible)
- Datasets: Magic, Spambase, Cover Type-sub, MNIST, Fashion-MNIST

### Results — Magic (d=10, K=2)

| Config | Arc | RBF | Arc − RBF |
|--------|-----|-----|-----------|
| Flat | 0.867 | 0.863 | +0.004 |
| m1L1 | 0.868 | 0.864 | +0.004 |
| m1L2 | 0.867 | 0.865 | +0.002 |
| m2L2 | 0.868 | 0.865 | +0.003 |
| Exact RBF | — | 0.866 | — |

Arc holds a +0.003 advantage at every config (within std ≈ 0.007–0.009; directional but
not clearly significant). Depth and width add nothing for either kernel. RBF exact (0.866)
essentially ties arc-cosine flat (0.867): both saturate at the dataset's accuracy ceiling.

### Results — Spambase (d=57, K=2)

| Config | Arc | RBF | Arc − RBF |
|--------|-----|-----|-----------|
| Flat | 0.940 | 0.935 | +0.005 |
| m1L1 | 0.940 | 0.936 | +0.004 |
| m1L2 | 0.940 | 0.936 | +0.004 |
| m2L2 | 0.940 | 0.936 | +0.004 |
| Exact RBF | — | 0.937 | — |

Arc leads by a consistent +0.004–0.005 across all configs (std ≈ 0.011; consistent
directional pattern over 10 splits). Arc flat (0.940) exceeds exact RBF (0.937).

### Results — Cover Type-sub (d=54, K=7)

| Config | Arc | RBF | Arc − RBF | Time Arc | Time RBF |
|--------|-----|-----|-----------|---------|---------|
| Flat | 0.750 | 0.730 | **+0.020** | 9.7s | 5.1s |
| m1L1 | 0.751 | 0.729 | **+0.022** | 9.8s | 5.1s |
| m1L2 | 0.757 | 0.734 | **+0.023** | 12.9s | 9.1s |
| m2L2 | 0.759 | 0.736 | **+0.023** | 26.1s | 17.5s |
| Exact RBF | — | 0.729 | — | 2.0s | — |

Arc dominates by +2.0–2.3pt across every configuration. This gap is clearly significant
(std ≈ 0.009–0.013). Arc is approximately 1.9× slower than RBF at the same config: with
K=7 and OvR, each block SVM internally trains K binary SVMs, and arc-cosine features are
harder to optimise than cosine-RBF features for this structured tabular multiclass problem.

Depth helps both: m=1 goes from 0.751 (L=1) to 0.757 (L=2) for arc-cosine.

### Results — MNIST (d=784, K=10)

| Config | Arc | RBF | Arc − RBF | Time Arc | Time RBF |
|--------|-----|-----|-----------|---------|---------|
| Flat | 0.923 | 0.917 | +0.006 | 3.5s | 5.2s |
| m1L1 | 0.922 | 0.916 | +0.006 | 3.6s | 5.3s |
| m1L2 | 0.923 | 0.923 | **+0.000** | 5.6s | 9.0s |
| m1L3 | 0.923 | 0.923 | **+0.000** | 7.4s | 12.2s |
| m2L2 | 0.923 | 0.923 | **+0.000** | 11.1s | 17.1s |
| Exact RBF | — | 0.938 | — | 11.7s | — |

**At L=1 arc-cosine leads by +0.006; at L≥2 both kernels are tied at 0.923.** The arc-cosine
advantage from Exp 9 was partly a configuration artifact: the canonical Arc m=1 L=1 vs
RBF m=2 L=2 comparison was not apples-to-apples. With equal configs, the kernels converge
in accuracy on MNIST at L=2. **Arc remains ~1.5× faster** at every config.

### Results — Fashion-MNIST (d=784, K=10)

| Config | Arc | RBF | Arc − RBF |
|--------|-----|-----|-----------|
| Flat | 0.837 | 0.847 | **−0.010** |
| m1L1 | 0.835 | 0.847 | **−0.012** |
| m1L2 | 0.835 | 0.850 | **−0.015** |
| m1L3 | 0.835 | 0.851 | **−0.016** |
| m2L2 | 0.835 | 0.850 | **−0.015** |
| Exact RBF | — | 0.862 | — |

**RBF consistently outperforms arc-cosine by 1.0–1.6pt on Fashion, regardless of config.**
Arc is stuck at 0.835 under all settings; RBF benefits from depth (0.847 at L=1 to 0.851
at L=3). The kernel choice drives this difference, not the configuration.

### Timing analysis

| Dataset | Arc m1L1 | RBF m1L1 | Arc faster by |
|---------|---------|---------|--------------|
| Magic | 2.9s | 2.6s | −0.1× (slight RBF edge) |
| Spambase | 0.4s | 0.4s | ≈equal |
| Cover Type | 9.8s | 5.1s | RBF 1.9× faster |
| MNIST | 3.6s | 5.3s | **Arc 1.5× faster** |
| Fashion | 5.1s | 6.3s | **Arc 1.2× faster** |

Arc is faster on high-d datasets (MNIST, Fashion) because ReLU features produce
better-conditioned optimization problems for the TRON solver than cosine-RBF features
at d=784. On Cover Type (K=7), the multiclass overhead reverses this: K=7 binary SVMs
per block make the arc-cosine kernel slower despite better per-feature conditioning.

### Conclusions from Exp 13

1. **Arc-cosine vs RBF: the gap depends on the dataset, not on the configuration.**
   On Cover Type the gap is +2.2pt (strongly significant); on Fashion it is −1.3pt
   (strongly significant in favour of RBF); on MNIST the gap vanishes at L≥2.
2. **On MNIST, the main Arc advantage is a speed advantage (1.5×), not an accuracy advantage.**
   At equal configs and L=2 both kernels reach 0.923. The difference in our canonical
   Exp 9 comparison (0.947 vs 0.942) was partly a configuration artifact.
3. **Fashion-MNIST requires RBF. There is no arc-cosine configuration that recovers the gap.**
   This is robust evidence that the kernel choice is data-structure-dependent, not merely
   dimensionality-dependent (both MNIST and Fashion have d=784).
4. **Depth (L) helps RBF more than arc-cosine on MNIST.** RBF gains ~0.007 from L=1 to L=2;
   arc-cosine gains ~0.001. The layerwise mechanism better complements the RBF kernel on
   high-d digit data.

---

## Overall Conclusions

### What the full experiment suite establishes

**1. ML-MSVM is a viable scalable kernel machine.** It matches the exact RBF SVM on small
datasets (within 0.01 accuracy) and extends to n=400k where exact SVMs are infeasible,
maintaining competitive accuracy. On MNIST at n=60k it achieves 0.947 in 23 seconds.

**2. The kernel choice is the dominant design decision.** The arc-cosine kernel advantage on
Cover Type (+2.2pt, Exp 13) dwarfs any gain from width, depth, P, or sampling scheme.
Conversely, Fashion-MNIST requires RBF regardless of configuration (+1.3pt).

**3. Depth is a width-efficiency mechanism.** Extra layers add most value at small P (Exp 5:
+0.006 at P=250) and are largely neutral at P=1000. Depth is not a qualitative capability
improvement; it is a way to get large-P accuracy from a small-P model.

**4. Three architectural simplifications are supported by the data:**
   - C-spread should be replaced by uniform C=1.0 for all block SVMs (Exp 6)
   - final_C of the head SVM is irrelevant; fix it at 1.0 (Exp 10)
   - Arc-cosine degree 1 (ReLU) dominates degrees 0 and 2 (Exp 8)

**5. The multi-SVM width (m>1) is only beneficial with the right diversity mechanism in the
right setting:**
   - Binary tasks: m=1 is almost always best; bootstrap at m=3–4 gives marginal improvement
   - Multiclass tasks: m=2–4 helps, but through OvR structure (same_c for Arc) or regularisation
     diversity (c_spread for RBF), not through the originally intended C-spread diversity

**6. The arc-cosine kernel has a strictly higher accuracy ceiling than RBF on MNIST.** At
P=5000 it exceeds the exact RBF SVM (0.946 vs 0.938), demonstrating it is a fundamentally
different and for this task better kernel function (Exp 12).

**7. Honest negative results worth reporting:**
   - On SUSY and HIGGS (low/moderate d, physics tabular), all RFF methods plateau near 0.80/0.68.
     ML-MSVM adds less than 0.002 over the flat RFF baseline. HIGGS accuracy is bounded by
     the shallow kernel method ceiling (~0.68); deep networks reach ~0.88.
   - Fashion-MNIST: arc-cosine consistently underperforms RBF regardless of any hyperparameter.
   - Depth does not rescue poor kernel choice: Arc L=4 on Fashion is still 0.835.

---

## Future Directions

**F1. Bootstrap at large n (highest priority).** Bootstrap diversity at m=4 gives +0.006 on
MNIST at n=10k (Exp 11). Does this gain scale to n=60k? Running Exp 9 with ML-MSVM Arc
m=4 L=2 bootstrap would directly extend the hero scalability result and potentially push
the n=60k accuracy beyond 0.947.

**F2. Exact arc-cosine SVM baseline.** The arc-cosine kernel at P=5000 L=1 already exceeds
exact RBF on MNIST. Implementing the Cho & Saul closed-form arc-cosine kernel as a custom
sklearn.SVC kernel would give the true accuracy ceiling and quantify how close P=5000 is
to convergence. This is feasible for n ≤ 20k.

**F3. RBF multiclass with c_spread at large scale.** RBF m=3 L=2 c_spread on Cover Type
sub-10k reaches 0.752 (only 0.007 below Arc m=1 L=2 = 0.759). At n=200k in Exp 3, does
the gap between Arc and RBF (currently +2pt) remain with properly tuned RBF? This directly
tests whether Arc's large-scale advantage is kernel-intrinsic or partly a config artifact.

**F4. ORF at small P as a practical deployment option.** ORF at P=250 gives Arc+0.008 on
Spambase (0.936 vs 0.926). In edge-device deployment where P must be small, ORF is a
free improvement worth including.

**F5. Kernel choice prediction from dataset geometry.** MNIST vs Fashion-MNIST: both d=784,
K=10, similar n — but opposite kernel winner. Cover Type vs SUSY: both tabular, but
Cover Type prefers Arc while SUSY prefers RBF. A geometric characterisation (intrinsic
dimensionality, class boundary structure) that predicts the better kernel would be both
practically useful and theoretically grounded.

**F6. Depth + bootstrap interaction.** Arc bootstrap at m=4 L=2 reaches 0.929 on MNIST.
What about L=3 or L=4 with bootstrap? Depth and bootstrap diversity may interact
(each layer refines the bootstrap-diverse representation further).
