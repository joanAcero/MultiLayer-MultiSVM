# ML-MSVM — Experimental Plan
**Thesis:** Deep Kernel Networks: Neural Architectures with SVM Foundations  
**Author:** Joan Acero Pousa · **Supervisor:** Lluís Belanche · **FIB-UPC**

---

## 1. Architecture Summary

**ML-MSVM** (Multi-Layer Multi-SVM) is a deep kernel machine built as a stack of *blocks*. Each block:
1. Draws a random feature matrix Ω and computes a kernel feature map Φ = φ(XΩᵀ) of shape n×P.
2. Trains m parallel LinearSVMs on Φ, each producing a weight vector.
3. Forms the weight projection W ∈ ℝ^(P×effective_m) and passes X_next = Φ·W to the next block.
4. A final head SVM classifies on the last block's output.

Training is fully **convex and greedy** (no backpropagation). The architecture has two kernel variants:

| Variant | Feature map | Theoretical basis |
|---------|------------|------------------|
| **RBF** | φ(z) = cos(z + b), Ω ~ N(0, γI) | Rahimi & Recht random Fourier features; approximates the shift-invariant RBF kernel (Bochner) |
| **Arc-cosine** | φ(z) = max(0, z), Ω ~ N(0, I) | Cho & Saul (2010); equivalent to one layer of an infinite-width ReLU network; positive-definite non-shift-invariant kernel |

---

## 2. Global Fixed Parameters

These parameters are **fixed throughout all experiments** unless an experiment explicitly varies them.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| P (random features) | 1000 | Balance between approximation quality and memory/speed |
| `tol` (LinearSVC solver) | 1e-2 | Identical for all LinearSVC instances (baselines, block SVMs, head); ensures timing comparisons reflect feature space, not solver settings |
| `block_tol` | 1e-2 | Same as above; with the arc-cosine feature map this prevents O(n²·⁷) solver blow-up |
| `dual` | auto (n < P → dual, n ≥ P → primal) | Adaptive: avoids O(n²) dual for large n |
| `max_iter` | 2000 | Sufficient at tol=1e-2; never binding |
| Arc-cosine degree | 1 (ReLU) | Validated in Exp 8; degree 0 and 2 are dominated |
| `normalize_inter_layer` | True | Prevents inter-layer scale drift for arc-cosine |
| `final_C` | 1.0 | Validated in Exp 10; head is insensitive to C |
| RBF_N_LIMIT | 50,000 | Exact RBF SVM skipped above this training size |

**Evaluation protocol by regime:**

| Regime | Datasets | Splits | Train/Test |
|--------|---------|--------|-----------|
| 1 (small n) | Wine, Breast Cancer, Ionosphere, Sonar, Glass | 10× | 90/10 stratified |
| 2 (medium n) | Magic, Spambase, Cover Type-sub (10k) | 10× | 90/10 stratified |
| 3 (large structured) | MNIST, Fashion-MNIST | 5× | 10k/2k fixed |
| 4 (very large n) | SUSY, HIGGS, Cover Type Full | 3× seeds | fixed pool |

---

## 3. Experiment Overview

| # | Name | Phase | Key variable | Datasets | Purpose |
|---|------|-------|-------------|---------|---------|
| 1 | Width Analysis | Core | m ∈ {1,2,3,4,6,10} × L ∈ {1,2,3,4} | R1+R2+R3 | Does width (more SVMs per block) help, and if so at what m? |
| 2 | Main Benchmark | Core | Fixed canonical configs | R1+R2+R3 | Head-to-head vs baselines and published ML-SVM results |
| 3 | Large-scale Feasibility | Core | n ∈ {50k,100k,200k,400k} | SUSY, CoverType Full, HIGGS | Does ML-MSVM work where exact SVM is infeasible? |
| 4 | Learning Curves | Core | n scaling (dense grid) | MNIST, SUSY | Accuracy and time as functions of n |
| 5 | Depth × P Interaction | Core | L ∈ {1,2,3,4} × P ∈ {250,500,1000,2000,3000,5000} | Magic, Spambase, MNIST | Does depth substitute for width in P? |
| 6 | C-spread Ablation | Core | C-spread schemes | Magic, Spambase, MNIST | Does diversity via logspace-C in block SVMs help? |
| 7 | Nyström Comparison | Core | P ∈ {500,1000,2000} | Magic, Spambase, CoverType, MNIST | How does ML-MSVM compare to the data-adaptive Nyström baseline? |
| 8 | Arc-cosine Degree | Core | degree ∈ {0, 1, 2} | R1+R2+R3 | Which arc-cosine degree is optimal? |
| 9 | Scalability Timing | Core (hero) | n ∈ {500…60k} (MNIST), n ∈ {1k…200k} (SUSY) | MNIST, SUSY | Main scalability figure: accuracy + time vs n |
| 10 | Head final_C | Core | final_C ∈ {1e-3…1e3} | Magic, Spambase, MNIST | Is the head's regularisation parameter important? |
| 11 | Diversity Mechanisms | Refinement | {c_spread, same_c, bootstrap, feature_subset} × m × L | Magic, Spambase, CoverType, MNIST | What diversity strategy actually makes m>1 beneficial? |
| 12 | RFF Quality: MC vs ORF vs QMC | Refinement | {standard, ORF, QMC Sobol} × P × kernel | MNIST, Spambase | Does structured sampling close the gap to the exact RBF SVM? |
| 13 | Fair Kernel Comparison | Refinement | kernel × {m1L1, m1L2, m1L3, m2L1, m2L2} at C=1.0 | Magic, Spambase, CoverType, MNIST, Fashion | Unbiased Arc vs RBF comparison: same configs, same C |

---

## 4. Phase 1 — Core Experiments (1–10)

**Goal:** Establish whether ML-MSVM is a viable, competitive architecture, and understand how its main hyperparameters (m, L, P, kernel) affect accuracy and speed.

**Exp 1** characterises the effect of block width (m) and depth (L) across 10 datasets in regimes 1–3. It establishes the **canonical configurations**: arc-cosine peaks at m=1 for most binary datasets; RBF benefits modestly from m=2; the multiclass Cover Type dataset is the one case where both depth and width clearly help.

**Exp 2** is the main academic benchmark. It compares ML-MSVM against five baselines (Linear SVM, exact RBF SVM, Flat RFF RBF, Flat RFF Arc, Nyström) and against the published results of ML-SVM (Acero & Belanche 2025) and DHNKN (Mehrkanoon & Suykens 2018). ML-MSVM beats the published ML-SVM on Magic and Spambase; it falls short on Glass (small n, high variance) and Cover Type-sub (limited 10k subsample).

**Exp 3** addresses the primary design objective: very large n where exact SVM is infeasible. Trains all models (no exact RBF) on n up to 400k for SUSY (d=18, 5M total), Cover Type Full (d=54, 581k), and HIGGS (d=28, 11M; sub-sampled to 1.1M pool). Confirms that ML-MSVM matches or exceeds all flat-RFF baselines.

**Exp 4** is a dense-n learning-curve study on MNIST and SUSY. Provides the accuracy-vs-n and time-vs-n data for plotting; results are consistent with Exp 9.

**Exp 5** tests the hypothesis that *depth substitutes for width in P*: at small P, extra layers recover accuracy; at large P, all layers converge. This is confirmed on MNIST for the arc-cosine variant.

**Exp 6** tests the original C-spread diversity scheme. Uniform C=1.0 is shown to equal or outperform logspace-C across all tested datasets. This is a negative result for the original design motivation and leads directly to Exp 11.

**Exp 7** adds the Nyström+LinearSVC baseline (data-adaptive landmarks, stronger approximation in principle). ML-MSVM Arc beats Nyström on all higher-d datasets; Nyström wins on Magic (low-d), where data-adaptive landmarks give an advantage.

**Exp 8** determines the optimal arc-cosine degree. Degree 1 (ReLU) is optimal or tied in 4 of 5 datasets. Degree 2 gives comparable accuracy on MNIST but is ~15× slower and is never the best choice.

**Exp 9** is the *hero experiment*: accuracy and training time vs n for both MNIST and SUSY, covering all models including the exact RBF SVM (up to n=50k). This produces the main scalability figures for the thesis.

**Exp 10** confirms that the head SVM's regularisation parameter final_C is irrelevant: accuracy varies by less than 0.003 across four orders of magnitude of C. It is fixed at 1.0.

---

## 5. Phase 2 — Architecture Refinement (11–13)

**Motivation:** Phase 1 raised three open questions: (a) what diversity mechanism actually helps when m>1? (b) does better random-feature sampling close the accuracy gap to the exact SVM? (c) are the observed Arc vs RBF differences due to kernel choice or to the different canonical configurations used?

**Exp 11** tests four block-diversity strategies for m>1 SVMs across all four regimes. The key finding is that the mechanism matters: c_spread (the original design) hurts arc-cosine at m>1; bootstrap subsampling is the only mechanism that consistently and monotonically improves arc-cosine on high-d data (MNIST); for RBF on multiclass problems, c_spread unexpectedly helps (+1.8pt on Cover Type); and feature-subset sampling is the fastest option (3–6× speed-up) but often hurts accuracy.

**Exp 12** tests whether Orthogonal Random Features (ORF) or Quasi-Monte Carlo Sobol sampling reduces the need for large P. ORF provides small, consistent improvement for RBF at low P (P≤500). Neither method substantially closes the gap to the exact RBF SVM, which turns out to be governed by the kernel approximation quality rather than sampling variance. Notably, the arc-cosine kernel at P=5000 exceeds the exact RBF SVM accuracy on MNIST, evidence that it is a genuinely different (and for this task, better) kernel.

**Exp 13** directly addresses the fairness question: Arc and RBF are tested under *identical* configurations (same m, same L, same C=1.0) across five datasets. The arc-cosine advantage on Cover Type (+2.0–2.3pt) is fully configuration-stable. On MNIST at L≥2 the two kernels are statistically tied in accuracy; Arc is consistently ~1.5× faster. On Fashion-MNIST, RBF consistently outperforms Arc by 1.2–1.5pt regardless of configuration.

---

## 6. Baselines and Reference Points

| Baseline | Where used | Role |
|----------|-----------|------|
| Linear SVM | All experiments | Lower bound; shows feature-map benefit |
| Exact RBF SVM (C=1, γ=scale) | Exp 2, 9, 12, 13 (when n ≤ 50k) | Gold-standard accuracy; becomes infeasible at large n |
| Flat RFF RBF (ML-MSVM L=0) | Exp 2, 3, 4, 9, 13 | Isolates kernel-approximation quality |
| Flat RFF Arc (ML-MSVM L=0) | Exp 2, 3, 4, 9, 13 | Isolates whether *depth* or *kernel choice* drives any gain |
| Nyström + LinearSVC | Exp 7 | Data-adaptive landmark approximation; stronger approximation in theory |
| ML-SVM (Acero & Belanche 2025) | Exp 2 (inline) | Published predecessor; the direct baseline for improvement |
| DHNKN (Mehrkanoon & Suykens 2018) | Exp 2 (MNIST inline) | Reference deep hybrid network; different protocol (60k/10k) |

---

## 7. Key Open Questions for Future Work

1. Does bootstrap diversity (m=4, Exp 11) scale its +0.6pt MNIST gain to the large-n regime (n=60k, Exp 9)?
2. What is the *exact* arc-cosine SVM accuracy on MNIST? (implementing the Cho & Saul closed-form as a custom sklearn kernel would set the ceiling for Exp 12's convergence analysis)
3. Does RBF + c_spread (which works for multiclass Cover Type in Exp 11) maintain its +1.8pt gain at n=200k in the large-scale regime of Exp 3?
4. What geometric property of a dataset (intrinsic dimensionality, class structure) predicts whether arc-cosine or RBF is the better kernel? The MNIST vs Fashion-MNIST split in Exp 13 suggests it is not input dimensionality alone.
