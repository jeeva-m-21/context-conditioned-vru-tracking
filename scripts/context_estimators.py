"""Context Estimators and Feature Extraction Module.

Implements:
1. Feature extractors:
   - Core Pruned Radio Context (phi_radio in R^5): [dist_norm, occ_0, occ_1, occ_2, cbr]
   - Extrapolated Sender-Link Geometry (phi_link in R^5): [dist_link_norm, occ_link_0, occ_link_1, occ_link_2, cbr]
   - Full 10-D Context (phi_full in R^10): radio context + time-of-day (sin, cos) + VRU class (ped, cyc, other)
   - Feature ablation representations (a)-(l)
   - Strictly Causal Pre-Outcome Gating Features I_{k^-} in R^4:
     [Delta d_prox, s_id, Delta t_sched, Delta tau_age]

2. Estimators:
   - Fixed q_bar (from training data only)
   - Rolling Delivery-History EMA baseline
   - L-BFGS Logistic Regression
   - Multilayer Perceptron (MLP) 5 -> 64 -> 32 -> 1
   - Eligibility Gating Model gamma(I_{k^-}) (Shared-gamma and Hypothesis-Conditioned gamma_1/gamma_0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class Normalizer:
    dist_mean: float = 35.0
    dist_std: float = 18.0
    cbr_mean: float = 0.45
    cbr_std: float = 0.15

    def normalize_dist(self, d: float) -> float:
        return (d - self.dist_mean) / max(1.0, self.dist_std)

    def normalize_cbr(self, c: float) -> float:
        return (c - self.cbr_mean) / max(0.01, self.cbr_std)


def extract_phi_radio(
    dist: float,
    occ: int,
    cbr: float,
    normalizer: Normalizer,
) -> np.ndarray:
    """Core Pruned Radio Context phi_radio in R^5."""
    d_norm = normalizer.normalize_dist(dist)
    occ_oh = [1.0 if occ == i else 0.0 for i in range(3)]
    c_norm = normalizer.normalize_cbr(cbr)
    return np.array([d_norm, occ_oh[0], occ_oh[1], occ_oh[2], c_norm], dtype=np.float32)


def extract_phi_full(
    dist: float,
    occ: int,
    cbr: float,
    tod_hours: float,
    vru_class: str,
    normalizer: Normalizer,
) -> np.ndarray:
    """Full 10-D Context phi_full in R^10."""
    radio_5d = extract_phi_radio(dist, occ, cbr, normalizer)
    # Time of day encoding: sin, cos of 24h cycle
    tod_rad = 2.0 * np.pi * (tod_hours / 24.0)
    tod_sin = math.sin(tod_rad)
    tod_cos = math.cos(tod_rad)
    # VRU class one-hot: [Pedestrian, Cyclist, Other]
    vru_oh = [
        1.0 if vru_class == "Pedestrian" else 0.0,
        1.0 if vru_class == "Cyclist" else 0.0,
        0.0,
    ]
    return np.concatenate([radio_5d, [tod_sin, tod_cos], vru_oh]).astype(np.float32)


def extract_phi_feature_ablation(
    variant: str,
    dist: float,
    occ: int,
    cbr: float,
    tod_hours: float,
    vru_class: str,
    ema_val: float,
    dist_link: float,
    occ_link: int,
    normalizer: Normalizer,
) -> np.ndarray:
    """Systematic feature ablation vectors (a) through (l)."""
    d_norm = normalizer.normalize_dist(dist)
    occ_oh = [1.0 if occ == i else 0.0 for i in range(3)]
    c_norm = normalizer.normalize_cbr(cbr)

    if variant == "cbr_only":  # (b)
        return np.array([c_norm], dtype=np.float32)
    elif variant == "dist_only":  # (c)
        return np.array([d_norm], dtype=np.float32)
    elif variant == "occ_only":  # (d)
        return np.array(occ_oh, dtype=np.float32)
    elif variant == "dist_occ":  # (e)
        return np.array([d_norm, *occ_oh], dtype=np.float32)
    elif variant == "dist_cbr":  # (f)
        return np.array([d_norm, c_norm], dtype=np.float32)
    elif variant == "core_radio":  # (g)
        return np.array([d_norm, *occ_oh, c_norm], dtype=np.float32)
    elif variant == "link_geometry":  # (h)
        d_link_norm = normalizer.normalize_dist(dist_link)
        occ_link_oh = [1.0 if occ_link == i else 0.0 for i in range(3)]
        return np.array([d_link_norm, *occ_link_oh, c_norm], dtype=np.float32)
    elif variant == "core_tod":  # (i)
        tod_rad = 2.0 * np.pi * (tod_hours / 24.0)
        return np.array([d_norm, *occ_oh, c_norm, math.sin(tod_rad), math.cos(tod_rad)], dtype=np.float32)
    elif variant == "core_class":  # (j)
        v_oh = [1.0 if vru_class == "Pedestrian" else 0.0, 1.0 if vru_class == "Cyclist" else 0.0]
        return np.array([d_norm, *occ_oh, c_norm, *v_oh], dtype=np.float32)
    elif variant == "full_10d":  # (k)
        return extract_phi_full(dist, occ, cbr, tod_hours, vru_class, normalizer)
    elif variant == "core_ema":  # (l)
        return np.array([d_norm, *occ_oh, c_norm, ema_val], dtype=np.float32)
    else:
        return np.array([d_norm, *occ_oh, c_norm], dtype=np.float32)


def extract_pre_outcome_gating_features(
    d_prox: float,
    s_id: float,
    dt_sched: float,
    tau_age: float,
) -> np.ndarray:
    """Strictly Causal Pre-Outcome Feature Vector I_{k^-} in R^4."""
    # Normalized Euclidean distance, ID pairing consistency, schedule residual, staleness
    d_prox_norm = min(5.0, d_prox / 3.0)  # normalized by gating gate ~ 3m
    s_id_clip = float(np.clip(s_id, 0.0, 1.0))
    dt_sched_norm = min(5.0, dt_sched)
    tau_age_norm = min(10.0, tau_age / 5.0)  # normalized by 5 frames
    return np.array([d_prox_norm, s_id_clip, dt_sched_norm, tau_age_norm], dtype=np.float32)


class ReceptionEstimator:
    """Wrapper for Logistic Regression and MLP Availability Estimators."""

    def __init__(
        self,
        model_type: str = "mlp",  # 'logistic', 'mlp', 'fixed', 'ema'
        q_fixed: float = 0.50,
        ema_alpha: float = 0.75,
        seed: int = 42,
    ):
        self.model_type = model_type
        self.q_fixed = q_fixed
        self.ema_alpha = ema_alpha
        self.seed = seed
        self.model: Optional[Union[LogisticRegression, MLPClassifier]] = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model_type == "fixed" or self.model_type == "ema":
            self.q_fixed = float(np.mean(y)) if len(y) > 0 else 0.50
            self.is_fitted = True
            return

        if self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=500,
                random_state=self.seed,
            )
            self.model.fit(X, y)
            self.is_fitted = True
        elif self.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=100,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=self.seed,
            )
            self.model.fit(X, y)
            self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_type == "fixed":
            return np.full(X.shape[0], self.q_fixed)
        if not self.is_fitted or self.model is None:
            return np.full(X.shape[0], self.q_fixed)
        probs = self.model.predict_proba(X)[:, 1]
        return np.clip(probs, 0.05, 0.95)


class SoftEligibilityGatingModel:
    """Soft Eligibility Gating Model gamma(I_{k^-}) under Sustained Silence."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.shared_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed)
        self.gamma1_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed)
        self.gamma0_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed)
        self.is_fitted = False

    def fit(
        self,
        X_shared: np.ndarray,
        y_shared: np.ndarray,
        X_e1: np.ndarray,
        y_e1: np.ndarray,
        X_e0: np.ndarray,
        y_e0: np.ndarray,
    ):
        self.shared_model.fit(X_shared, y_shared)
        if len(X_e1) > 0 and len(np.unique(y_e1)) > 1:
            self.gamma1_model.fit(X_e1, y_e1)
        if len(X_e0) > 0 and len(np.unique(y_e0)) > 1:
            self.gamma0_model.fit(X_e0, y_e0)
        self.is_fitted = True

    def predict_shared_gamma(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Default prior: high if close and fresh, decays with age
            # Feature: [d_prox, s_id, dt_sched, tau_age]
            logit = 2.0 - 0.8 * X[:, 0] + 1.2 * X[:, 1] - 0.4 * X[:, 2] - 0.5 * X[:, 3]
            return 1.0 / (1.0 + np.exp(-logit))
        return np.clip(self.shared_model.predict_proba(X)[:, 1], 0.01, 0.99)

    def predict_conditioned_gamma(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (gamma_1, gamma_0) for true track vs false/clutter track."""
        if not self.is_fitted:
            g_shared = self.predict_shared_gamma(X)
            return g_shared, np.clip(g_shared * 0.4, 0.01, 0.8)
        g1 = np.clip(self.gamma1_model.predict_proba(X)[:, 1], 0.01, 0.99)
        g0 = np.clip(self.gamma0_model.predict_proba(X)[:, 1], 0.01, 0.99)
        return g1, g0
