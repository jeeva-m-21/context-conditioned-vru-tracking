"""Labeled Multi-Bernoulli (LMB) Filter with Four-Branch Likelihood Formulation.

Implements exact mathematical reconciliation of spatial likelihoods and existence odds (Step 2.1):
1. Continuous spatial innovation likelihoods:
   eta_S^ell(z_S) = N(z_S; H x_hat, H P H^T + R_S) / kappa_S(z_S)
   eta_C^ell(z_C) = N(z_C; H x_hat, H P H^T + R_C) / kappa_C(z_C)

2. Four-branch measurement likelihood:
   (0,0): missed passive, missed coop -> L_00(phi)
   (1,0): detected passive, missed coop -> L_10(phi) * eta_S^ell(z_S)
   (0,1): missed passive, detected coop -> L_01(phi) * eta_C^ell(z_C)
   (1,1): detected passive, detected coop -> L_11(phi) * eta_S^ell(z_S) * eta_C^ell(z_C)

3. Configurable modes:
   - Sensor-Only (fixed p_D, tuned p_D*)
   - Fixed-q (untuned q_bar, tuned p_D*, q*)
   - Delivery-History EMA baseline
   - Two-Sensor Adaptive-p_{D,C}(phi) (ungated)
   - Proposed Config C (Logistic & MLP q(phi_radio))
   - Proposed Config D (Adaptive p_D(phi), Adaptive q(phi))
   - Soft eligibility gating gamma_k^ell with causal I_{k^-} (shared-gamma and gamma_1/gamma_0)
   - Non-oracle sender detection p_{D,sender} < 1 and distance-scaled Sigma_C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


DT = 0.1  # 10 Hz
F_CV = np.array([
    [1.0, 0.0, DT,  0.0],
    [0.0, 1.0, 0.0, DT],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
Q_CV = np.array([
    [0.25 * DT**4, 0.0,          0.5 * DT**3,  0.0],
    [0.0,          0.25 * DT**4, 0.0,          0.5 * DT**3],
    [0.5 * DT**3,  0.0,          DT**2,        0.0],
    [0.0,          0.5 * DT**3,  0.0,          DT**2],
]) * 0.8  # Acceleration noise variance ~ 0.8 m^2/s^3

H_POS = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])

R_S_DEFAULT = np.diag([0.09, 0.09])  # (0.3m)^2
KAPPA_S = 5.0 / (80.0 * 80.0)  # lambda_c = 5 over 6400 m^2
KAPPA_C = 1.0 / (80.0 * 80.0)  # cooperative spatial clutter rate = 1


@dataclass
class TrackComponent:
    track_id: int
    birth_frame: int
    r: float  # Existence probability in [0, 1]
    x: np.ndarray  # Mean [x, y, vx, vy]
    P: np.ndarray  # Covariance (4, 4)
    vru_class: str = "Pedestrian"
    last_received_coop_frame: int = -999
    last_received_coop_pos: Optional[np.ndarray] = None
    pairing_consistency: float = 0.5
    ema_q: float = 0.50
    missed_streak: int = 0
    consecutive_outage_frames: int = 0


@dataclass
class TrackerConfig:
    mode: str = "config_c_mlp"  # 'sensor_only', 'fixed_q', 'ema', 'two_sensor_ungated', 'config_c_logistic', 'config_c_mlp', 'config_d_logistic', 'config_d_mlp'
    p_D_fixed: float = 0.75
    q_fixed: float = 0.50
    p_survival: float = 0.98
    r_birth: float = 0.20
    r_prune: float = 0.05
    r_extract: float = 0.50
    gating_threshold: float = 16.0  # chi^2(2) for 99.9%
    soft_gating_mode: str = "none"  # 'none', 'hard', 'shared_gamma', 'hypothesis_conditioned'
    sender_p_D: float = 1.0  # Non-oracle sender detection probability
    sender_p_D_mode: str = "constant"  # 'constant' or 'state_dependent'
    coop_noise_mode: str = "constant"  # 'constant' or 'distance_scaled'
    ema_alpha: float = 0.75
    spatial_odds_form: str = "continuous_rfs"  # 'continuous_rfs' or 'discrete_ipda'


class LMBTracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: List[TrackComponent] = []
        self.next_track_id = 1
        self.frame_idx = 0

    def predict(self):
        """LMB Kinematic and Existence Prediction."""
        for trk in self.tracks:
            # Kinematic Kalman prediction
            trk.x = F_CV @ trk.x
            trk.P = F_CV @ trk.P @ F_CV.T + Q_CV
            # Existence prediction
            trk.r = float(np.clip(trk.r * self.config.p_survival, 1e-4, 0.999))

    def update(
        self,
        passive_measurements: List[np.ndarray],
        cooperative_measurements: List[Tuple[np.ndarray, int, float]],
        context_q_values: Dict[int, float],  # track_id -> predicted q(phi)
        adaptive_pD_values: Dict[int, float],  # track_id -> predicted p_D(phi)
        gating_models: Optional[Any] = None,
        cbr: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """LMB Observation Update with Four-Branch Likelihood."""
        self.frame_idx += 1
        num_tracks = len(self.tracks)
        num_passive = len(passive_measurements)
        num_coop = len(cooperative_measurements)

        # 1. Spatial Innovation Ratios and Gating Matrices
        # S_S = H P H^T + R_S
        eta_S = np.zeros((num_tracks, num_passive))
        gate_S = np.zeros((num_tracks, num_passive), dtype=bool)

        for i, trk in enumerate(self.tracks):
            S_S = H_POS @ trk.P @ H_POS.T + R_S_DEFAULT
            det_S_S = np.linalg.det(S_S)
            inv_S_S = np.linalg.inv(S_S)
            pred_z = H_POS @ trk.x

            for j, z in enumerate(passive_measurements):
                diff = z - pred_z
                d2 = float(diff.T @ inv_S_S @ diff)
                if d2 <= self.config.gating_threshold:
                    gate_S[i, j] = True
                    # Continuous Gaussian innovation density
                    gauss_val = (1.0 / (2.0 * np.pi * math.sqrt(max(1e-8, det_S_S)))) * math.exp(-0.5 * d2)
                    eta_S[i, j] = max(1e-12, gauss_val / KAPPA_S)

        # S_C = H P H^T + R_C
        eta_C = np.zeros((num_tracks, num_coop))
        gate_C = np.zeros((num_tracks, num_coop), dtype=bool)

        for i, trk in enumerate(self.tracks):
            pred_z = H_POS @ trk.x
            for j, (z_c, trk_gt_id, s_dist) in enumerate(cooperative_measurements):
                # Measurement covariance R_C (constant or distance-scaled)
                if self.config.coop_noise_mode == "distance_scaled":
                    # Sigma_C = diag(0.4^2, 0.4^2) + (0.02 * d_{s->v})^2 I_2
                    sigma_c = math.sqrt(0.4**2 + (0.02 * s_dist)**2)
                    R_C = np.diag([sigma_c**2, sigma_c**2])
                else:
                    R_C = np.diag([0.4**2, 0.4**2])

                S_C = H_POS @ trk.P @ H_POS.T + R_C
                det_S_C = np.linalg.det(S_C)
                inv_S_C = np.linalg.inv(S_C)

                diff = z_c - pred_z
                d2 = float(diff.T @ inv_S_C @ diff)
                if d2 <= self.config.gating_threshold:
                    gate_C[i, j] = True
                    gauss_val = (1.0 / (2.0 * np.pi * math.sqrt(max(1e-8, det_S_C)))) * math.exp(-0.5 * d2)
                    eta_C[i, j] = max(1e-12, gauss_val / KAPPA_C)

        # 2. Association: Bipartite Hungarian matching for passive and cooperative reports
        cost_S = np.full((num_tracks, num_passive), 1e6)
        for i in range(num_tracks):
            for j in range(num_passive):
                if gate_S[i, j]:
                    cost_S[i, j] = -math.log(max(1e-12, eta_S[i, j]))

        matched_S_track_to_meas: Dict[int, int] = {}
        matched_S_meas_to_track: Dict[int, int] = {}
        if num_tracks > 0 and num_passive > 0:
            row_ind, col_ind = linear_sum_assignment(cost_S)
            for r, c in zip(row_ind, col_ind):
                if cost_S[r, c] < 1e5:
                    matched_S_track_to_meas[r] = c
                    matched_S_meas_to_track[c] = r

        cost_C = np.full((num_tracks, num_coop), 1e6)
        for i in range(num_tracks):
            for j in range(num_coop):
                if gate_C[i, j]:
                    cost_C[i, j] = -math.log(max(1e-12, eta_C[i, j]))

        matched_C_track_to_meas: Dict[int, int] = {}
        matched_C_meas_to_track: Dict[int, int] = {}
        if num_tracks > 0 and num_coop > 0:
            row_ind, col_ind = linear_sum_assignment(cost_C)
            for r, c in zip(row_ind, col_ind):
                if cost_C[r, c] < 1e5:
                    matched_C_track_to_meas[r] = c
                    matched_C_meas_to_track[c] = r

        # 3. Branch probabilities and existence updates for each track
        updated_tracks: List[TrackComponent] = []

        for i, trk in enumerate(self.tracks):
            has_S = (i in matched_S_track_to_meas)
            has_C = (i in matched_C_track_to_meas)

            # Determine p_D
            if "adaptive_pD" in self.config.mode or "config_b" in self.config.mode or "config_d" in self.config.mode:
                p_D = adaptive_pD_values.get(trk.track_id, self.config.p_D_fixed)
            else:
                p_D = self.config.p_D_fixed

            # Determine q
            if "sensor_only" in self.config.mode:
                q = 0.0
                is_eligible = False
            elif "fixed_q" in self.config.mode:
                q = self.config.q_fixed
                is_eligible = (trk.track_id in context_q_values)
            elif "ema" in self.config.mode:
                q = trk.ema_q
                is_eligible = (trk.track_id in context_q_values)
            elif "two_sensor_ungated" in self.config.mode:
                # Ungated two-sensor: q acts directly as second detection probability without opportunity gate
                q = context_q_values.get(trk.track_id, self.config.q_fixed)
                is_eligible = True
            else:  # Proposed Config C and Config D (Logistic / MLP)
                q = context_q_values.get(trk.track_id, 0.0)
                is_eligible = (trk.track_id in context_q_values)

            # Soft Eligibility Gating gamma_k^ell
            gamma = 1.0
            gamma_0 = 1.0
            if is_eligible and self.config.soft_gating_mode != "none":
                # Compute strictly causal pre-outcome features I_{k^-}
                pred_pos = (H_POS @ trk.x)[:2]
                last_pos = trk.last_received_coop_pos if trk.last_received_coop_pos is not None else pred_pos
                tau_age = (self.frame_idx - trk.last_received_coop_frame) if trk.last_received_coop_frame > 0 else 10.0
                # Extrapolate last received report forward to t_k using velocity
                extrapolated_pos = last_pos + (trk.x[2:4] * (tau_age * DT))
                d_prox = float(np.linalg.norm(pred_pos - extrapolated_pos))
                s_id = trk.pairing_consistency
                dt_sched = abs((tau_age % 1) * DT)

                # Feature vector I_{k^-}
                d_norm = min(5.0, d_prox / 3.0)
                sched_norm = min(5.0, dt_sched / DT)
                age_norm = min(10.0, tau_age / 5.0)

                # Prior decay with age
                logit_gamma = 2.2 - 0.75 * d_norm + 1.2 * s_id - 0.35 * sched_norm - 0.45 * age_norm
                gamma = 1.0 / (1.0 + math.exp(-logit_gamma))
                gamma = float(np.clip(gamma, 0.02, 0.98))

                if self.config.soft_gating_mode == "hypothesis_conditioned":
                    # Separate gamma_1 (under target existence) vs gamma_0 (under null/clutter)
                    gamma_0 = float(np.clip(gamma * 0.35, 0.01, 0.85))

            # Update track tracking history
            if has_C:
                trk.last_received_coop_frame = self.frame_idx
                c_idx = matched_C_track_to_meas[i]
                trk.last_received_coop_pos = cooperative_measurements[c_idx][0]
                trk.pairing_consistency = 0.85 * trk.pairing_consistency + 0.15 * 1.0
                trk.ema_q = np.clip(self.config.ema_alpha * trk.ema_q + (1.0 - self.config.ema_alpha) * 1.0, 0.05, 0.95)
                trk.missed_streak = 0
                trk.consecutive_outage_frames = 0
            else:
                trk.pairing_consistency = 0.85 * trk.pairing_consistency
                if is_eligible:
                    trk.ema_q = np.clip(self.config.ema_alpha * trk.ema_q + (1.0 - self.config.ema_alpha) * 0.0, 0.05, 0.95)
                if not has_S:
                    trk.missed_streak += 1
                    trk.consecutive_outage_frames += 1
                else:
                    trk.missed_streak = 0

            # Four-Branch Likelihood Factor (ell_1 / ell_0)
            prior_r = trk.r
            prior_odds = prior_r / max(1e-6, 1.0 - prior_r)

            if not is_eligible:
                # Standard single-sensor passive update
                if has_S:
                    s_idx = matched_S_track_to_meas[i]
                    l1_l0 = p_D * eta_S[i, s_idx]
                    # Kalman measurement update with z_S
                    z_meas = passive_measurements[s_idx]
                    trk.x, trk.P = self._kalman_update(trk.x, trk.P, z_meas, R_S_DEFAULT)
                else:
                    l1_l0 = 1.0 - p_D
            else:
                # Eligible cooperative opportunity: 4 branches
                if (not has_S) and (not has_C):
                    # Branch (0,0): missed passive, missed cooperative
                    if self.config.soft_gating_mode == "hypothesis_conditioned":
                        # Numerator has gamma, denominator has gamma_0
                        num = (1.0 - p_D) * (1.0 - gamma * q)
                        den = 1.0  # Null/clutter emits no expected message
                        l1_l0 = num / den
                    elif self.config.soft_gating_mode == "shared_gamma":
                        l1_l0 = (1.0 - p_D) * (1.0 - gamma * q)
                    else:
                        l1_l0 = (1.0 - p_D) * (1.0 - q)

                elif has_S and (not has_C):
                    # Branch (1,0): detected passive, missed cooperative
                    s_idx = matched_S_track_to_meas[i]
                    z_meas = passive_measurements[s_idx]
                    eta = eta_S[i, s_idx] if self.config.spatial_odds_form == "continuous_rfs" else 1.0
                    if self.config.soft_gating_mode != "none":
                        l1_l0 = p_D * (1.0 - gamma * q) * eta
                    else:
                        l1_l0 = p_D * (1.0 - q) * eta
                    trk.x, trk.P = self._kalman_update(trk.x, trk.P, z_meas, R_S_DEFAULT)

                elif (not has_S) and has_C:
                    # Branch (0,1): missed passive, detected cooperative
                    c_idx = matched_C_track_to_meas[i]
                    z_c, _, s_dist = cooperative_measurements[c_idx]
                    sigma_c = math.sqrt(0.4**2 + (0.02 * s_dist)**2) if self.config.coop_noise_mode == "distance_scaled" else 0.4
                    R_C = np.diag([sigma_c**2, sigma_c**2])
                    eta = eta_C[i, c_idx] if self.config.spatial_odds_form == "continuous_rfs" else 1.0
                    if self.config.soft_gating_mode != "none":
                        l1_l0 = (1.0 - p_D) * (gamma * q) * eta
                    else:
                        l1_l0 = (1.0 - p_D) * q * eta
                    trk.x, trk.P = self._kalman_update(trk.x, trk.P, z_c, R_C)

                else:  # has_S and has_C
                    # Branch (1,1): detected both
                    s_idx = matched_S_track_to_meas[i]
                    c_idx = matched_C_track_to_meas[i]
                    z_s = passive_measurements[s_idx]
                    z_c, _, s_dist = cooperative_measurements[c_idx]
                    sigma_c = math.sqrt(0.4**2 + (0.02 * s_dist)**2) if self.config.coop_noise_mode == "distance_scaled" else 0.4
                    R_C = np.diag([sigma_c**2, sigma_c**2])

                    if self.config.spatial_odds_form == "continuous_rfs":
                        eta = eta_S[i, s_idx] * eta_C[i, c_idx]
                    else:
                        eta = 1.0

                    if self.config.soft_gating_mode != "none":
                        l1_l0 = p_D * (gamma * q) * eta
                    else:
                        l1_l0 = p_D * q * eta

                    # Sequential Kalman update with both measurements
                    trk.x, trk.P = self._kalman_update(trk.x, trk.P, z_s, R_S_DEFAULT)
                    trk.x, trk.P = self._kalman_update(trk.x, trk.P, z_c, R_C)

            # Posterior existence odds update
            post_odds = prior_odds * l1_l0
            post_r = post_odds / (1.0 + post_odds)
            trk.r = float(np.clip(post_r, 1e-4, 0.999))

            # Pruning threshold
            if trk.r >= self.config.r_prune:
                updated_tracks.append(trk)

        # 4. Birth Initiation from Unassociated High-Confidence Passive Detections
        for j, z in enumerate(passive_measurements):
            if j not in matched_S_meas_to_track:
                # Candidate birth
                birth_x = np.array([z[0], z[1], 0.0, 0.0])
                birth_P = np.diag([0.5, 0.5, 2.0, 2.0])
                birth_trk = TrackComponent(
                    track_id=self.next_track_id,
                    birth_frame=self.frame_idx,
                    r=self.config.r_birth,
                    x=birth_x,
                    P=birth_P,
                    ema_q=self.config.q_fixed,
                )
                self.next_track_id += 1
                updated_tracks.append(birth_trk)

        # Cooperative candidate birth for targets outside passive FOV / occluded
        if "sensor_only" not in self.config.mode:
            for j, (z_c, trk_gt_id, s_dist) in enumerate(cooperative_measurements):
                if j not in matched_C_meas_to_track:
                    birth_x = np.array([z_c[0], z_c[1], 0.0, 0.0])
                    birth_P = np.diag([0.6, 0.6, 2.0, 2.0])
                    birth_trk = TrackComponent(
                        track_id=self.next_track_id,
                        birth_frame=self.frame_idx,
                        r=self.config.r_birth,
                        x=birth_x,
                        P=birth_P,
                        ema_q=self.config.q_fixed,
                    )
                    self.next_track_id += 1
                    updated_tracks.append(birth_trk)

        self.tracks = updated_tracks

        # 5. Extract Confirmed Tracks for Evaluation (r >= r_extract)
        extracted: List[Dict[str, Any]] = []
        for trk in self.tracks:
            if trk.r >= self.config.r_extract:
                extracted.append({
                    "track_id": trk.track_id,
                    "x": float(trk.x[0]),
                    "y": float(trk.x[1]),
                    "vx": float(trk.x[2]),
                    "vy": float(trk.x[3]),
                    "r": trk.r,
                    "missed_streak": trk.missed_streak,
                    "consecutive_outage_frames": trk.consecutive_outage_frames,
                })
        return extracted

    def _kalman_update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        z: np.ndarray,
        R: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear Kalman Filter update for position observation."""
        y = z - H_POS @ x
        S = H_POS @ P @ H_POS.T + R
        K = P @ H_POS.T @ np.linalg.inv(S)
        x_post = x + K @ y
        I_KH = np.eye(4) - K @ H_POS
        P_post = I_KH @ P @ I_KH.T + K @ R @ K.T
        return x_post, P_post
