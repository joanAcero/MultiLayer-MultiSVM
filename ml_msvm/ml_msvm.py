"""
ml_msvm.py  –  FIXED VERSION
=============================
Bugs fixed vs. original:
  1. L=0 bug: when num_layers=0, _fit_head() was called on raw X without any
     RFF feature map. Fixed by applying one RFF block (no SVM, no W) so the
     final head trains on the P-dimensional feature space as intended.
  2. Arc-cosine inter-layer standardization: X_next = Phi @ W was passed
     to the next block raw. For arc-cosine features this causes scale drift
     across layers (empirically max_abs grows ~5x per layer). Fixed by
     standardizing X_next to zero mean / unit std after each block when
     kernel='arc_cosine', using a per-block StandardScaler stored for inference.

These are the ONLY changes to _fit_block, fit, _forward_pass, and the _Block
dataclass. All other logic is identical to the original.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class _Block:
    """Frozen state of a single feature-extraction block (hidden layer)."""
    Omega: np.ndarray           # feature frequencies/weights, shape (P, d_in)
    b: np.ndarray               # RFF phases, shape (P,); zeros for arc-cosine
    W: Optional[np.ndarray]     # SVM weight matrix, None only for the L=0 RFF block
    kernel: str
    arc_cosine_degree: int
    # FIX 2: per-block scaler, fitted on X_next during training, applied at inference
    scaler: Optional[StandardScaler] = None


@dataclass
class _Head:
    coef: np.ndarray
    intercept: np.ndarray


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ML_MSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    MultiLayer Multi-SVM classifier (Proposal 2, no AGOP).

    Parameters
    ----------
    num_layers : int
        Number of feature-extraction blocks.
        0 -> single RFF map + linear SVM (flat RFF baseline).
    svms_per_block : int  (m)
    C_values : sequence of float, optional
    rff_features : int  (P)
    final_C : float
    kernel : {'rbf', 'arc_cosine'}
    arc_cosine_degree : {0, 1, 2}
    median_heuristic_subsample : int or None
    random_state : int or None
    normalize_inter_layer : bool
        If True (default), standardize X_next between arc-cosine blocks.
        Has no effect for kernel='rbf' (RBF features are already bounded [-1,1]*sqrt(2/P)).
        Set to False only for ablation studies.
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
        normalize_inter_layer: bool = True,
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
        self.normalize_inter_layer = normalize_inter_layer

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ML_MSVMClassifier":
        X, y = check_X_y(X, y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        C_list = self._resolve_C_values()

        self.blocks_: List[_Block] = []
        X_curr = X.copy()

        if self.num_layers == 0:
            # FIX 1: L=0 path — apply one RFF map, then train the head on Phi.
            # The block has W=None (no SVM projection) and no scaler.
            block, X_curr = self._fit_rff_only_block(X_curr, rng)
            self.blocks_.append(block)
        else:
            for _ in range(self.num_layers):
                block, X_curr = self._fit_block(X_curr, y_enc, C_list, rng)
                self.blocks_.append(block)

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

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # -----------------------------------------------------------------------
    # Core building blocks
    # -----------------------------------------------------------------------

    def _fit_rff_only_block(self, X: np.ndarray, rng: np.random.Generator):
        """
        FIX 1: L=0 baseline block.
        Applies the RFF (or arc-cosine) feature map but does NOT train SVMs
        and does NOT project via W.  The head then trains on the P-dimensional Phi.
        """
        d = X.shape[1]
        gamma = self._median_heuristic(X, rng) if self.kernel == "rbf" else 1.0
        Omega, b = self._sample_frequencies(d, gamma, rng)
        Phi = self._feature_map(X, Omega, b)
        block = _Block(Omega=Omega, b=b, W=None, kernel=self.kernel,
                       arc_cosine_degree=self.arc_cosine_degree, scaler=None)
        return block, Phi

    def _fit_block(self, X, y_enc, C_list, rng):
        """
        Train one hidden block and return (block, next_representation).
        FIX 2: for arc-cosine kernel, standardize X_next before returning.
        """
        d = X.shape[1]
        gamma = self._median_heuristic(X, rng) if self.kernel == "rbf" else 1.0
        Omega, b = self._sample_frequencies(d, gamma, rng)
        Phi = self._feature_map(X, Omega, b)
        W = self._train_svm_block(Phi, y_enc, C_list, rng)
        X_next = Phi @ W

        # FIX 2: standardize inter-layer representation for arc-cosine
        scaler = None
        if self.kernel == "arc_cosine" and self.normalize_inter_layer:
            scaler = StandardScaler()
            X_next = scaler.fit_transform(X_next)

        block = _Block(Omega=Omega, b=b, W=W, kernel=self.kernel,
                       arc_cosine_degree=self.arc_cosine_degree, scaler=scaler)
        return block, X_next

    def _fit_head(self, X, y_enc, rng):
        svm = self._make_svm(self.final_C, rng).fit(X, y_enc)
        return _Head(coef=np.atleast_2d(svm.coef_),
                     intercept=np.atleast_1d(svm.intercept_))

    def _train_svm_block(self, Phi, y_enc, C_list, rng):
        weight_cols = []
        for C_k in C_list:
            svm = self._make_svm(C_k, rng).fit(Phi, y_enc)
            coef = np.atleast_2d(svm.coef_)
            for row in coef:
                weight_cols.append(row)
        return np.column_stack(weight_cols)

    def _forward_pass(self, X: np.ndarray) -> np.ndarray:
        """Apply all trained blocks in sequence. Handles both L=0 and L>0."""
        X_curr = X
        for block in self.blocks_:
            Phi = self._feature_map(X_curr, block.Omega, block.b,
                                    kernel=block.kernel,
                                    arc_cosine_degree=block.arc_cosine_degree)
            if block.W is None:
                # L=0 RFF-only block: output is Phi directly
                X_curr = Phi
            else:
                X_curr = Phi @ block.W
                # FIX 2: apply the stored scaler if present
                if block.scaler is not None:
                    X_curr = block.scaler.transform(X_curr)
        return X_curr

    # -----------------------------------------------------------------------
    # Feature map
    # -----------------------------------------------------------------------

    def _feature_map(self, X, Omega, b, kernel=None, arc_cosine_degree=None):
        kernel = kernel or self.kernel
        n = arc_cosine_degree if arc_cosine_degree is not None else self.arc_cosine_degree
        P = Omega.shape[0]
        Z = X @ Omega.T
        if kernel == "rbf":
            return np.sqrt(2.0 / P) * np.cos(Z + b)
        else:
            act = np.maximum(0.0, Z)
            if n == 0:
                act = (act > 0).astype(float)
            elif n == 2:
                act = act ** 2
            return np.sqrt(2.0 / P) * act

    def _sample_frequencies(self, d, gamma, rng):
        P = self.rff_features
        if self.kernel == "rbf":
            Omega = rng.standard_normal((P, d)) * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            Omega = rng.standard_normal((P, d))
            b = np.zeros(P)
        return Omega, b

    def _median_heuristic(self, X, rng):
        n = X.shape[0]
        sub = self.median_heuristic_subsample
        if sub is not None and n > sub:
            idx = rng.choice(n, sub, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X
        dists = pdist(X_sub, metric="sqeuclidean")
        med = np.median(dists)
        return 1.0 / (2.0 * med) if med > 0 else 1.0

    def _make_svm(self, C, rng):
        seed = int(rng.integers(0, 2**31))
        return LinearSVC(C=C, dual=False, max_iter=5000, random_state=seed)

    def _resolve_C_values(self):
        if self.C_values is not None:
            return list(self.C_values)
        return list(np.logspace(-2, 2, num=self.svms_per_block))

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")