"""
ml_msvm.py — PATCHED VERSION
Fixes applied:
  1. (existing) L=0 bug: _fit_rff_only_block
  2. (existing) Arc-cosine inter-layer standardization (normalize X_next)
  3. (NEW) Arc-cosine Phi normalization before block SVM: prevents TRON
     from hitting max_iter on ill-conditioned non-negative features.
     Root cause of ArcCos O(n^2.7) timing on MNIST at n > 5k.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


@dataclass
class _Block:
    Omega: np.ndarray
    b: np.ndarray
    W: Optional[np.ndarray]
    kernel: str
    arc_cosine_degree: int
    scaler: Optional[StandardScaler] = None       # FIX 2: inter-layer scaler
    phi_scaler: Optional[StandardScaler] = None   # FIX 3: pre-SVM Phi scaler


@dataclass
class _Head:
    coef: np.ndarray
    intercept: np.ndarray


class ML_MSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    MultiLayer Multi-SVM classifier (Proposal 2, no AGOP).

    Parameters
    ----------
    num_layers : int
        0 -> flat RFF SVM baseline.
    svms_per_block : int  (m)
    C_values : sequence of float, optional
    rff_features : int  (P)
    final_C : float
    kernel : {'rbf', 'arc_cosine'}
    arc_cosine_degree : {0, 1, 2}
    median_heuristic_subsample : int or None
    random_state : int or None
    normalize_inter_layer : bool
        Standardize X_next between arc-cosine blocks (FIX 2).
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ML_MSVMClassifier":
        X, y = check_X_y(X, y, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        C_list = self._resolve_C_values()

        self.blocks_: List[_Block] = []
        X_curr = X.copy()

        if self.num_layers == 0:
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

    def _fit_rff_only_block(self, X: np.ndarray, rng: np.random.Generator):
        """FIX 1 + FIX 3: L=0 block. Applies feature map, normalises for arc-cosine."""
        d = X.shape[1]
        gamma = self._median_heuristic(X, rng) if self.kernel == "rbf" else 1.0
        Omega, b = self._sample_frequencies(d, gamma, rng)
        Phi = self._feature_map(X, Omega, b)
        # FIX 3
        phi_scaler = None
        if self.kernel == "arc_cosine":
            phi_scaler = StandardScaler()
            Phi = phi_scaler.fit_transform(Phi)
        block = _Block(Omega=Omega, b=b, W=None, kernel=self.kernel,
                       arc_cosine_degree=self.arc_cosine_degree,
                       scaler=None, phi_scaler=phi_scaler)
        return block, Phi

    def _fit_block(self, X, y_enc, C_list, rng):
        """FIX 2 + FIX 3: Train one hidden block."""
        d = X.shape[1]
        gamma = self._median_heuristic(X, rng) if self.kernel == "rbf" else 1.0
        Omega, b = self._sample_frequencies(d, gamma, rng)
        Phi = self._feature_map(X, Omega, b)

        # FIX 3: normalise ArcCos Phi before the SVM solve.
        # max(0,z) features are non-negative with heterogeneous per-feature
        # variance. Phi^T Phi is ill-conditioned, causing TRON primal to fail
        # to converge (hits max_iter=5000) from n ~ 5k on MNIST (d=784).
        # StandardScaler makes features zero-mean unit-variance → good conditioning.
        phi_scaler = None
        if self.kernel == "arc_cosine":
            phi_scaler = StandardScaler()
            Phi = phi_scaler.fit_transform(Phi)

        W = self._train_svm_block(Phi, y_enc, C_list, rng)
        X_next = Phi @ W

        # FIX 2: standardize inter-layer representation
        scaler = None
        if self.kernel == "arc_cosine" and self.normalize_inter_layer:
            scaler = StandardScaler()
            X_next = scaler.fit_transform(X_next)

        block = _Block(Omega=Omega, b=b, W=W, kernel=self.kernel,
                       arc_cosine_degree=self.arc_cosine_degree,
                       scaler=scaler, phi_scaler=phi_scaler)
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
        """Apply all trained blocks. Handles L=0 and L>0, and phi_scaler."""
        X_curr = X
        for block in self.blocks_:
            Phi = self._feature_map(X_curr, block.Omega, block.b,
                                    kernel=block.kernel,
                                    arc_cosine_degree=block.arc_cosine_degree)
            # FIX 3: apply Phi normalisation if present
            if block.phi_scaler is not None:
                Phi = block.phi_scaler.transform(Phi)
            if block.W is None:
                X_curr = Phi
            else:
                X_curr = Phi @ block.W
                # FIX 2: apply inter-layer scaler if present
                if block.scaler is not None:
                    X_curr = block.scaler.transform(X_curr)
        return X_curr

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