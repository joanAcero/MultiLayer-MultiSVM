"""
mlsvm_extensions.py
===================
Research extensions to ML_MSVMClassifier — subclasses that add new
diversity and RFF-quality modes WITHOUT touching the original ml_msvm.py.

Classes
-------
DiverseMLMSVM      Adds diversity_mode={'c_spread','same_c','bootstrap','feature_subset'}
                   to study what actually makes m>1 beneficial.
QMC_MLMSVMClassifier  Adds rff_mode={'standard','orf','qmc'} to study whether
                   better random feature sampling closes the gap to the exact SVM.
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

# ── auto-locate ml_msvm.py (same pattern as utils.import_ml_msvm) ───────────
def _import_base():
    """Return (ML_MSVMClassifier, module) using the same discovery logic as utils.py."""
    experiments_dir = Path(__file__).parent.resolve()
    for base in [experiments_dir.parent, experiments_dir.parent.parent, experiments_dir]:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        for mod_path in ("mlsvm.ml_msvm", "ml_msvm.ml_msvm", "ml_msvm"):
            try:
                mod = importlib.import_module(mod_path)   # registers in sys.modules automatically
                cls = getattr(mod, "ML_MSVMClassifier", None)
                if cls is None or not (isinstance(cls, type) and hasattr(cls, "fit")):
                    continue
                return cls, mod
            except (ImportError, ModuleNotFoundError, AttributeError):
                pass
    # Filesystem fallback (same as utils.py)
    for f in sorted(experiments_dir.parent.rglob("ml_msvm.py")):
        spec = importlib.util.spec_from_file_location("ml_msvm_found", f)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules["ml_msvm_found"] = mod   # register BEFORE exec so @dataclass works
        spec.loader.exec_module(mod)
        cls = getattr(mod, "ML_MSVMClassifier", None)
        if cls:
            print(f"[mlsvm_extensions] Loaded ML_MSVMClassifier from {f}")
            return cls, mod
    raise ImportError("Cannot find ML_MSVMClassifier. Run: find ~ -name ml_msvm.py")

ML_MSVMClassifier, _base_mod = _import_base()
_Block                        = _base_mod._Block


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DiverseMLMSVM
# ══════════════════════════════════════════════════════════════════════════════

class DiverseMLMSVM(ML_MSVMClassifier):
    """
    ML-MSVM with pluggable block-diversity strategies.

    diversity_mode : str
        'c_spread'      — logspace C values (original design, shown by exp6 to hurt).
        'same_c'        — all block SVMs at C=1.0, same training data. Expected to
                          produce near-identical weight vectors (rank-1 W).
        'bootstrap'     — all C=1.0; each SVM trains on a random 80 % subsample
                          (with replacement). True "bagging-SVM" diversity.
        'feature_subset'— all C=1.0; each SVM trains on a disjoint random subset
                          of floor(P / m) features. RF-style column diversity.

    bootstrap_ratio : float   Fraction of training rows used per bootstrap SVM.
    """

    def __init__(
        self,
        num_layers: int = 2,
        svms_per_block: int = 4,
        C_values: Optional[Sequence[float]] = None,
        rff_features: int = 1000,
        final_C: float = 1.0,
        kernel: str = "arc_cosine",
        arc_cosine_degree: int = 1,
        median_heuristic_subsample: Optional[int] = 1000,
        random_state: Optional[int] = None,
        normalize_inter_layer: bool = True,
        block_tol: float = 1e-2,
        block_max_iter: int = 1000,
        diversity_mode: str = "c_spread",
        bootstrap_ratio: float = 0.80,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            svms_per_block=svms_per_block,
            C_values=C_values,
            rff_features=rff_features,
            final_C=final_C,
            kernel=kernel,
            arc_cosine_degree=arc_cosine_degree,
            median_heuristic_subsample=median_heuristic_subsample,
            random_state=random_state,
            normalize_inter_layer=normalize_inter_layer,
            block_tol=block_tol,
            block_max_iter=block_max_iter,
        )
        self.diversity_mode  = diversity_mode
        self.bootstrap_ratio = bootstrap_ratio

    # ── resolve C values depending on mode ───────────────────────────────────
    def _resolve_C_values(self):
        if self.diversity_mode == "c_spread":
            # Original logspace scheme
            if self.C_values is not None:
                return list(self.C_values)
            if self.svms_per_block == 1:
                return [1.0]
            return list(np.logspace(-2, 2, num=self.svms_per_block))
        else:
            # Bootstrap, feature_subset, or same_c: uniform C=1.0
            return [1.0] * self.svms_per_block

    # ── diversity-aware block training ───────────────────────────────────────
    def _train_svm_block(self, Phi, y_enc, C_list, rng):
        """Override: dispatch to the chosen diversity strategy."""
        n, P = Phi.shape
        m = len(C_list)

        if self.diversity_mode == "feature_subset":
            return self._train_feature_subset(Phi, y_enc, m, P, rng)
        elif self.diversity_mode == "bootstrap":
            return self._train_bootstrap(Phi, y_enc, m, P, n, rng)
        else:
            # 'c_spread' or 'same_c': standard loop (same data, vary C)
            return super()._train_svm_block(Phi, y_enc, C_list, rng)

    def _train_bootstrap(self, Phi, y_enc, m, P, n, rng):
        """Each SVM trains on an independent bootstrap subsample (C=1.0 for all)."""
        weight_cols = []
        k = max(2, int(self.bootstrap_ratio * n))
        for _ in range(m):
            idx = rng.choice(n, k, replace=True)
            svm = self._make_svm(1.0, rng, (k, P), "block").fit(Phi[idx], y_enc[idx])
            coef = np.atleast_2d(svm.coef_)
            for row in coef:
                weight_cols.append(row)
        return np.column_stack(weight_cols)

    def _train_feature_subset(self, Phi, y_enc, m, P, rng):
        """Each SVM uses floor(P/m) randomly chosen features (C=1.0 for all).
        Weight vectors are zero-padded back to P dimensions so W has shape (P, m*K_eff).
        """
        k = max(1, P // m)
        weight_cols = []
        for _ in range(m):
            feat_idx = rng.choice(P, k, replace=False)
            Phi_sub   = Phi[:, feat_idx]
            svm = self._make_svm(1.0, rng, Phi_sub.shape, "block").fit(Phi_sub, y_enc)
            coef = np.atleast_2d(svm.coef_)               # (K_eff, k)
            full = np.zeros((coef.shape[0], P))
            full[:, feat_idx] = coef
            for row in full:
                weight_cols.append(row)
        return np.column_stack(weight_cols)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  QMC_MLMSVMClassifier
# ══════════════════════════════════════════════════════════════════════════════

class QMC_MLMSVMClassifier(ML_MSVMClassifier):
    """
    ML-MSVM with pluggable random-feature sampling quality.

    rff_mode : str
        'standard'  — i.i.d. Gaussian rows of Ω (current baseline).
        'orf'       — Orthogonal Random Features (Yu et al., 2016).
                      Rows of Ω are structured-orthogonal; same marginal distribution
                      as Gaussian but lower inter-feature correlation → lower
                      approximation variance.
        'qmc'       — Quasi-Monte Carlo features via Sobol low-discrepancy sequence
                      (scipy.stats.qmc). Better spectral coverage per P features.
                      Falls back to 'orf' if scipy.qmc is unavailable.

    Both ORF and QMC are drop-in replacements: they generate the same Ω shape and
    the rest of the architecture is unchanged.
    """

    def __init__(
        self,
        num_layers: int = 1,
        svms_per_block: int = 1,
        C_values: Optional[Sequence[float]] = None,
        rff_features: int = 1000,
        final_C: float = 1.0,
        kernel: str = "rbf",
        arc_cosine_degree: int = 1,
        median_heuristic_subsample: Optional[int] = 1000,
        random_state: Optional[int] = None,
        normalize_inter_layer: bool = True,
        block_tol: float = 1e-2,
        block_max_iter: int = 1000,
        rff_mode: str = "standard",
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            svms_per_block=svms_per_block,
            C_values=C_values,
            rff_features=rff_features,
            final_C=final_C,
            kernel=kernel,
            arc_cosine_degree=arc_cosine_degree,
            median_heuristic_subsample=median_heuristic_subsample,
            random_state=random_state,
            normalize_inter_layer=normalize_inter_layer,
            block_tol=block_tol,
            block_max_iter=block_max_iter,
        )
        self.rff_mode = rff_mode

    def _sample_frequencies(self, d, gamma, rng):
        """Dispatch to the chosen sampling strategy."""
        P    = self.rff_features
        mode = self.rff_mode
        if mode == "orf":
            return self._sample_orf(d, gamma, P, rng)
        elif mode == "qmc":
            try:
                return self._sample_qmc(d, gamma, P, rng)
            except (ImportError, Exception):
                return self._sample_orf(d, gamma, P, rng)   # graceful fallback
        else:
            return super()._sample_frequencies(d, gamma, rng)

    # ── ORF ─────────────────────────────────────────────────────────────────
    def _sample_orf(self, d, gamma, P, rng):
        """
        Orthogonal Random Features (Yu et al. 2016).
        Rows of Ω are built from orthogonal d×d blocks, each scaled by
        chi(d) norms to preserve the N(0,I) marginal distribution.
        """
        n_blocks = int(np.ceil(P / d))
        rows = []
        for _ in range(n_blocks):
            G = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(G)                         # orthonormal rows
            # Chi(d) scaling: norms of independent d-dim Gaussian vectors
            S = np.sqrt(np.sum(rng.standard_normal((d, d)) ** 2, axis=1))
            rows.append(Q * S[:, np.newaxis])
        Omega = np.vstack(rows)[:P]                        # (P, d)
        if self.kernel == "rbf":
            Omega = Omega * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            b = np.zeros(P)
        return Omega, b

    # ── QMC (Sobol) ─────────────────────────────────────────────────────────
    def _sample_qmc(self, d, gamma, P, rng):
        """
        Quasi-Monte Carlo features via Sobol low-discrepancy sequences.
        Applies the inverse-normal CDF to convert uniform [0,1]^d Sobol
        points into approximately N(0,I) samples.
        """
        from scipy.stats import qmc, norm as sp_norm
        seed = int(rng.integers(0, 2**31))
        # Sobol requires power-of-2 sample count; round up
        P2 = int(2 ** np.ceil(np.log2(P)))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        sampler.fast_forward(1)                            # skip all-zeros point
        u = sampler.random(P2)[:P]                        # (P, d) in [0,1]
        # Clip to avoid ±inf at the tails
        u = np.clip(u, 1e-6, 1 - 1e-6)
        Omega = sp_norm.ppf(u)                            # (P, d) ~ N(0,I)
        if self.kernel == "rbf":
            Omega = Omega * np.sqrt(2 * gamma)
            b = rng.uniform(0, 2 * np.pi, P)
        else:
            b = np.zeros(P)
        return Omega, b