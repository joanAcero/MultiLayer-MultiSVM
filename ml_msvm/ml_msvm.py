from __future__ import annotations

"""
Author: Joan Acero Pousa
ML-MSVM: MultiLayer Multi-SVM

Each layer applies two successive transformations:
  1. Nonlinear: feature map  X^(l) -> Phi^(l)   in R^{n x P}
  2. Linear:    SVM projection  Phi^(l) W^(l) -> X^(l+1)  in R^{n x m}

Two feature maps are supported, controlled by the `kernel` parameter:

  kernel='rbf'  (default)
    Standard Random Fourier Features (Rahimi & Recht, 2007), approximating the
    Gaussian RBF kernel via Bochner's theorem:
      Phi_j(x) = sqrt(2/P) * cos(omega_j^T x + b_j),  omega_j ~ N(0, 2*gamma*I)

  kernel='arc_cosine'
    Arc-cosine random features (proposed from Cho & Saul, 2010, Eq. 1),
    approximating the arc-cosine kernel of degree n via Monte Carlo:
      Phi_j(x) = sqrt(2/P) * max(0, w_j^T x)^n,  w_j ~ N(0, I)
    Each block is equivalent to one layer of an infinite-width neural network
    with activation sigma_n(u) = max(0,u)^n (ReLU for n=1).
    Note: no phase variable b_j is needed.

In both cases W^(l) = [w_1^(l), ..., w_m^(l)] collects the weight vectors of m
parallel SVMs. For binary problems W has shape (P, m); for K-class OvR it has
shape (P, m*K), preserving all class-specific directions.

No backpropagation. Every SVM solve is convex.

Example
-------
    from ml_msvm import ML_MSVMClassifier

    # Standard RBF variant
    clf = ML_MSVMClassifier(
        num_layers=2, svms_per_block=4,
        C_values=[0.01, 0.1, 1.0, 10.0],
        rff_features=1000, random_state=0,
    )

    # Arc-cosine (ReLU, n=1) variant
    clf = ML_MSVMClassifier(
        num_layers=2, svms_per_block=4,
        C_values=[0.01, 0.1, 1.0, 10.0],
        rff_features=1000, kernel="arc_cosine", arc_cosine_degree=1,
        random_state=0,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    """Frozen state of a single feature-extraction block (hidden layer)."""
    Omega: np.ndarray          # feature frequencies/weights, shape (P, d_in)
    b: np.ndarray              # RFF phases, shape (P,); zeros for arc-cosine
    W: np.ndarray              # SVM weight matrix: (P, m) binary, (P, m*K) K-class OvR
    kernel: str                # 'rbf' or 'arc_cosine'
    arc_cosine_degree: int     # degree n; only used when kernel='arc_cosine' 


@dataclass
class _Head:
    """Final linear SVM classifier built on the last block's representation."""
    coef: np.ndarray        # shape (1, m) for binary, (K, m) for K-class OvR
    intercept: np.ndarray   # shape (1,) for binary, (K,) for K-class OvR


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ML_MSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    MultiLayer Multi-SVM classifier (Proposal 2, no AGOP).

    Parameters
    ----------
    num_layers : int
        Number of feature-extraction blocks (hidden layers).
        If 0, falls back to a plain single-SVM on raw features (baseline).
    svms_per_block : int
        Number of parallel SVMs m trained at each block.
        The inter-layer representation has dimension m (binary) or m*K (K-class).
    C_values : sequence of float, optional
        Regularisation constants for the m SVMs.  Must have length svms_per_block.
        If None, defaults to log-spaced values over [1e-2, 1e2].
    rff_features : int
        Number of random features P per block, for both RBF and arc-cosine kernels.
    final_C : float
        Regularisation constant for the final classification SVM.
    kernel : {'rbf', 'arc_cosine'}, default='rbf'
        Feature map used at each block.
        'rbf'        -- cosine random features (Rahimi & Recht, 2007), approximating
                        the Gaussian RBF kernel via Bochner's theorem.
                        Frequencies sampled as omega_j ~ N(0, 2*gamma*I).
        'arc_cosine' -- arc-cosine random features (Cho & Saul, 2010, Eq. 1),
                        approximating the arc-cosine kernel of degree arc_cosine_degree
                        via Monte Carlo. Weights sampled as w_j ~ N(0, I), no bandwidth.
    arc_cosine_degree : {0, 1, 2}, default=1
        Degree of the arc-cosine kernel. Only used when kernel='arc_cosine'.
        0 -> step/Heaviside activation,  1 -> ReLU,  2 -> quadratic ReLU.
    median_heuristic_subsample : int or None
        Number of points used to estimate the RBF bandwidth gamma via the median
        heuristic. None uses all training points. Ignored for kernel='arc_cosine'.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        num_layers: int = 2,
        svms_per_block: int = 4,
        C_values: Optional[Sequence[float]] = None,
        rff_features: int = 2000,
        final_C: float = 1.0,
        kernel: str = "rbf",
        arc_cosine_degree: int = 1,
        median_heuristic_subsample: Optional[int] = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        self.num_layers = num_layers
        self.svms_per_block = svms_per_block
        self.C_values = C_values
        self.rff_features = rff_features
        self.final_C = final_C
        self.kernel = kernel
        self.arc_cosine_degree = arc_cosine_degree
        self.median_heuristic_subsample = median_heuristic_subsample
        self.random_state = random_state

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ML_MSVMClassifier":
        X, y = check_X_y(X, y, dtype=np.float64)
        self._validate_params()
        rng = np.random.default_rng(self.random_state)

        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        C_list = self._resolve_C_values()

        self.blocks_: List[_Block] = []
        X_curr = X.copy()

        # ------------------------------------------------------------------
        # Hidden layers: each block transforms X^(l) -> X^(l+1) via feature map + SVM
        # ------------------------------------------------------------------
        for _ in range(self.num_layers):
            block, X_curr = self._fit_block(X_curr, y_enc, C_list, rng)
            self.blocks_.append(block)

        # ------------------------------------------------------------------
        # Output layer: single SVM on the final representation
        # ------------------------------------------------------------------
        self.head_ = self._fit_head(X_curr, y_enc, rng)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["blocks_", "head_"])
        X = check_array(X, dtype=np.float64)
        self._check_n_features(X)

        X_curr = self._forward_pass(X)
        scores = X_curr @ self.head_.coef.T + self.head_.intercept
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if scores.ndim == 1:
            return self.classes_[(scores > 0).astype(int)]
        return self.classes_[np.argmax(scores, axis=1)]

    # -----------------------------------------------------------------------
    # Core building blocks
    # -----------------------------------------------------------------------

    def _fit_block(
        self,
        X: np.ndarray,
        y_enc: np.ndarray,
        C_list: List[float],
        rng: np.random.Generator,
    ) -> tuple[_Block, np.ndarray]:
        """
        Train one hidden block and return (block, next_representation).

        Steps
        -----
        1. Map X -> Phi via the chosen feature map (RBF or arc-cosine).
        2. Train m SVMs on (Phi, y) with different C values.
        3. Project Phi through the weight matrix: X_next = Phi @ W.
        """
        d = X.shape[1]

        # Step 1: nonlinear feature transformation
        # gamma is estimated via the median heuristic for RBF; ignored for arc_cosine.
        gamma = self._median_heuristic(X, rng) if self.kernel == "rbf" else 1.0
        Omega, b = self._sample_frequencies(d, gamma, rng)
        Phi = self._feature_map(X, Omega, b)                   # (n, P)

        # Step 2: train m parallel SVMs
        W = self._train_svm_block(Phi, y_enc, C_list, rng)     # (P, m) binary / (P, m*K) multiclass

        # Step 3: linear projection -- pass weights forward, not decision values
        X_next = Phi @ W                                        # (n, m) binary / (n, m*K) multiclass

        block = _Block(Omega=Omega, b=b, W=W,
                       kernel=self.kernel,
                       arc_cosine_degree=self.arc_cosine_degree)
        return block, X_next

    def _fit_head(
        self,
        X: np.ndarray,
        y_enc: np.ndarray,
        rng: np.random.Generator,
    ) -> _Head:
        """Train the final classification SVM on the last representation."""
        svm = self._make_svm(self.final_C, rng).fit(X, y_enc)
        return _Head(
            coef=np.atleast_2d(svm.coef_),
            intercept=np.atleast_1d(svm.intercept_),
        )

    def _train_svm_block(
        self,
        Phi: np.ndarray,
        y_enc: np.ndarray,
        C_list: List[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Train m SVMs on (Phi, y) and stack their weight vectors column-wise.

        For binary classification, coef_ has shape (1, P) -> one column per SVM.
        For K-class OvR, coef_ has shape (K, P) -> K columns per SVM, preserving
        all class-specific directions. This avoids collapsing multiclass structure.

        Returns W of shape (P, m * n_class_rows), where n_class_rows = 1 or K.
        """
        weight_cols = []
        for C_k in C_list:
            svm = self._make_svm(C_k, rng).fit(Phi, y_enc)
            # coef_ shape: (1, P) for binary, (K, P) for K-class OvR.
            # Transpose to (P, n_class_rows) and append each column separately
            # so W grows as (P, m) for binary and (P, m*K) for multiclass.
            coef = np.atleast_2d(svm.coef_)                    # (n_class_rows, P)
            for row in coef:
                weight_cols.append(row)                         # each is (P,)
        return np.column_stack(weight_cols)                     # (P, m * n_class_rows)

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Apply all trained blocks in sequence to produce the final representation."""
        X_curr = X
        for block in self.blocks_:
            Phi = self._feature_map(X_curr, block.Omega, block.b,
                                    kernel=block.kernel,
                                    arc_cosine_degree=block.arc_cosine_degree)
            X_curr = Phi @ block.W
        return X_curr

    # -----------------------------------------------------------------------
    # Feature map (RBF or arc-cosine)
    # -----------------------------------------------------------------------

    def _feature_map(
        self,
        X: np.ndarray,
        Omega: np.ndarray,
        b: np.ndarray,
        kernel: Optional[str] = None,
        arc_cosine_degree: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the P-dimensional feature map for the chosen kernel.

        RBF (default):
            Phi_j(x) = sqrt(2/P) * cos(omega_j^T x + b_j)
            Approximates exp(-gamma ||x-z||^2) via Bochner's theorem.

        Arc-cosine (Cho & Saul, 2010, Eq. 1):
            Phi_j(x) = sqrt(2/P) * max(0, w_j^T x)^n
            Approximates the arc-cosine kernel of degree n via Monte Carlo.
            Each feature is equivalent to one unit of an infinite-width network
            with activation sigma_n(u) = max(0,u)^n (ReLU for n=1).
            No bandwidth parameter: w_j ~ N(0, I) canonically (Cho & Saul, Eq. 1).
            Note: no phase b is used; b is ignored for arc-cosine.
        """
        k = kernel if kernel is not None else self.kernel
        n = arc_cosine_degree if arc_cosine_degree is not None else self.arc_cosine_degree
        scale = np.sqrt(2.0 / self.rff_features)

        if k == "rbf":
            Z = X @ Omega.T + b                                 # (n_samples, P)
            return scale * np.cos(Z)

        elif k == "arc_cosine":
            Z = X @ Omega.T                                     # (n_samples, P); no phase
            # n=0: Heaviside step Theta(z) = 1 if z>0 else 0.
            # Cannot use max(0,z)^0: numpy evaluates 0.0**0=1.0, giving 1
            # for ALL z including z<0, producing a constant feature map.
            if n == 0:
                return scale * (Z > 0).astype(float)
            return scale * np.maximum(0.0, Z) ** n

        else:
            raise ValueError(f"Unknown kernel '{k}'. Choose 'rbf' or 'arc_cosine'.")

    def _sample_frequencies(
        self, d: int, gamma: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample feature map weights and phases.

        RBF:
            omega_j ~ N(0, 2*gamma*I),  b_j ~ Unif[0, 2*pi]
            Scale follows from Bochner's theorem for the Gaussian kernel.
            gamma is estimated via the median heuristic.

        Arc-cosine:
            w_j ~ N(0, I)  -- isotropic, no bandwidth parameter.
            This is the canonical distribution from Cho & Saul's integral
            representation (their Eq. 1), which has covariance I by construction.
            The gamma argument is ignored: arc-cosine features have no bandwidth.
            The scale of the kernel is determined by the data norms ||x||, ||y||,
            not by a free parameter. Since data is StandardScaled before the model,
            N(0, I) is the correct and sufficient choice.
            b = 0 (no phase needed; arc-cosine features are real by construction).
        """
        if self.kernel == "rbf":
            scale = float(np.sqrt(2.0 * gamma))
            Omega = rng.normal(0.0, scale, size=(self.rff_features, d))  # (P, d)
            b = rng.uniform(0.0, 2.0 * np.pi, size=self.rff_features)    # (P,)
        elif self.kernel == "arc_cosine":
            Omega = rng.normal(0.0, 1.0, size=(self.rff_features, d))    # (P, d)
            b = np.zeros(self.rff_features)                               # unused
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'.")
        return Omega, b

    def _median_heuristic(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> float:
        """Estimate gamma = 1 / (2 * median(||xi - xj||^2)) via subsampling.

        Only called for kernel='rbf'. Arc-cosine features use N(0,I) weights
        with no bandwidth parameter, so this method is skipped for that kernel.
        """
        n = X.shape[0]
        sub = self.median_heuristic_subsample
        if sub is not None and sub < n:
            idx = rng.choice(n, size=sub, replace=False)
            X = X[idx]
        sq_dists = pdist(X, metric="sqeuclidean")
        med = float(np.median(sq_dists))
        return 1.0 / (2.0 * med) if med > 0.0 else 1.0

    # -----------------------------------------------------------------------
    # SVM factory
    # -----------------------------------------------------------------------

    def _make_svm(self, C: float, rng: np.random.Generator) -> LinearSVC:
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        return LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed)

    # -----------------------------------------------------------------------
    # Parameter helpers
    # -----------------------------------------------------------------------

    def _resolve_C_values(self) -> List[float]:
        """Return the list of C values, defaulting to log-spaced if not given."""
        if self.C_values is not None:
            return list(self.C_values)
        return list(np.logspace(-2, 2, num=self.svms_per_block))

    def _validate_params(self) -> None:
        if self.num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if self.svms_per_block < 1:
            raise ValueError("svms_per_block must be >= 1")
        if self.rff_features < 1:
            raise ValueError("rff_features must be >= 1")
        if self.final_C <= 0.0:
            raise ValueError("final_C must be > 0")
        if self.C_values is not None and len(self.C_values) != self.svms_per_block:
            raise ValueError(
                f"C_values has {len(self.C_values)} entries "
                f"but svms_per_block={self.svms_per_block}"
            )
        if self.kernel not in ("rbf", "arc_cosine"):
            raise ValueError(f"kernel must be 'rbf' or 'arc_cosine', got '{self.kernel}'")
        if self.arc_cosine_degree not in (0, 1, 2):
            raise ValueError("arc_cosine_degree must be 0, 1, or 2")
        if (
            self.median_heuristic_subsample is not None
            and self.median_heuristic_subsample < 2
        ):
            raise ValueError("median_heuristic_subsample must be None or >= 2")

    def _check_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )