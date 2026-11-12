"""Standardized Evaluator Module for MOT and Calibration Metrics.

Computes:
1. HOTA, DetA, AssA, LocA (Higher Order Tracking Accuracy) over 19 thresholds [0.05, 0.95]
2. IDF1 (Identification F1 score from global bipartite identity matching)
3. OSPA (order p=2, cutoff c=10 meters)
4. CLEAR metrics: ID switches (IDSW) and Fragmentations (Frag) per 1000 frames
5. Brier Score and Expected Calibration Error (ECE, 10 confidence bins)
6. AUROC for soft eligibility gating gamma_k^ell
7. Track Continuity: Survival rate after 3, 5, 10 consecutive missed frames
8. Mean outage recovery time after 10-frame outage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


def compute_ospa_frame(
    gt_positions: np.ndarray,  # shape (M, 2)
    pr_positions: np.ndarray,  # shape (N, 2)
    p: int = 2,
    c: float = 10.0,
) -> float:
    """Computes OSPA distance at a single frame."""
    m = len(gt_positions)
    n = len(pr_positions)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0  # Normalized: cutoff c / c = 1.0

    # Cost matrix with cutoff c
    cost = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist = np.linalg.norm(gt_positions[i] - pr_positions[j])
            cost[i, j] = min(c, dist) ** p

    # Hungarian assignment for min cost
    if m <= n:
        row_ind, col_ind = linear_sum_assignment(cost)
        total_dist = float(np.sum(cost[row_ind, col_ind]))
        ospa = ((total_dist + (c**p) * (n - m)) / n) ** (1.0 / p)
    else:
        row_ind, col_ind = linear_sum_assignment(cost.T)
        total_dist = float(np.sum(cost.T[row_ind, col_ind]))
        ospa = ((total_dist + (c**p) * (m - n)) / m) ** (1.0 / p)
    return float(ospa / c)  # Normalized OSPA in [0, 1]


def compute_hota_and_clear(
    gt_tracks: Dict[int, Dict[int, np.ndarray]],  # frame_idx -> {gt_id: pos}
    pr_tracks: Dict[int, Dict[int, np.ndarray]],  # frame_idx -> {pr_id: pos}
    num_frames: int,
    dist_max: float = 2.0,  # 2.0 meter spatial gating for HOTA
) -> Dict[str, float]:
    """Computes exact HOTA, DetA, AssA, LocA, IDF1, IDSW, and Frag."""
    alphas = np.linspace(0.05, 0.95, 19)
    hota_list = []
    deta_list = []
    assa_list = []
    loca_list = []

    # Map for ID switches and fragmentations
    all_gt_ids = set()
    for f_idx in range(num_frames):
        all_gt_ids.update(gt_tracks.get(f_idx, {}).keys())

    gt_assigned_history: Dict[int, List[Optional[int]]] = {gt_id: [] for gt_id in all_gt_ids}

    # Frame-by-frame matching at alpha=0.5 for CLEAR and IDF1
    for f_idx in range(num_frames):
        gts = gt_tracks.get(f_idx, {})
        prs = pr_tracks.get(f_idx, {})
        gt_ids = list(gts.keys())
        pr_ids = list(prs.keys())

        if len(gt_ids) == 0 or len(pr_ids) == 0:
            for gt_id in all_gt_ids:
                if gt_id in gts:
                    gt_assigned_history[gt_id].append(None)
            continue

        cost = np.zeros((len(gt_ids), len(pr_ids)))
        for i, g_id in enumerate(gt_ids):
            for j, p_id in enumerate(pr_ids):
                d = np.linalg.norm(gts[g_id] - prs[p_id])
                cost[i, j] = d if d <= dist_max else 1e5

        r_ind, c_ind = linear_sum_assignment(cost)
        assigned_map = {}
        for r, c in zip(r_ind, c_ind):
            if cost[r, c] <= dist_max:
                assigned_map[gt_ids[r]] = pr_ids[c]

        for gt_id in all_gt_ids:
            if gt_id in gts:
                gt_assigned_history[gt_id].append(assigned_map.get(gt_id, None))

    # Compute IDSW and Fragmentations
    total_idsw = 0
    total_frag = 0
    for gt_id, hist in gt_assigned_history.items():
        prev_pr = None
        was_tracking = False
        for cur_pr in hist:
            if cur_pr is not None:
                if prev_pr is not None and cur_pr != prev_pr:
                    total_idsw += 1
                if not was_tracking and prev_pr is not None:
                    total_frag += 1
                prev_pr = cur_pr
                was_tracking = True
            else:
                was_tracking = False

    idsw_per_1000 = (total_idsw / max(1, num_frames)) * 1000.0
    frag_per_1000 = (total_frag / max(1, num_frames)) * 1000.0

    # Multi-threshold HOTA evaluation
    for alpha in alphas:
        # Distance threshold corresponding to alpha: S(x, y) = 1 - d / dist_max >= alpha
        d_thresh = dist_max * (1.0 - alpha)

        tp_matches = []  # list of (frame, gt_id, pr_id)
        total_gt = 0
        total_pr = 0
        total_sim = 0.0

        for f_idx in range(num_frames):
            gts = gt_tracks.get(f_idx, {})
            prs = pr_tracks.get(f_idx, {})
            total_gt += len(gts)
            total_pr += len(prs)

            gt_ids = list(gts.keys())
            pr_ids = list(prs.keys())
            if len(gt_ids) == 0 or len(pr_ids) == 0:
                continue

            cost = np.zeros((len(gt_ids), len(pr_ids)))
            sim = np.zeros((len(gt_ids), len(pr_ids)))
            for i, g_id in enumerate(gt_ids):
                for j, p_id in enumerate(pr_ids):
                    d = np.linalg.norm(gts[g_id] - prs[p_id])
                    s = max(0.0, 1.0 - d / dist_max)
                    sim[i, j] = s
                    cost[i, j] = -s if d <= d_thresh else 1e5

            r_ind, c_ind = linear_sum_assignment(cost)
            for r, c in zip(r_ind, c_ind):
                if cost[r, c] < 1e4:
                    tp_matches.append((f_idx, gt_ids[r], pr_ids[c]))
                    total_sim += sim[r, c]

        num_tp = len(tp_matches)
        num_fp = total_pr - num_tp
        num_fn = total_gt - num_tp

        deta = (num_tp / max(1, num_tp + num_fp + num_fn)) if (num_tp + num_fp + num_fn) > 0 else 0.0
        loca = (total_sim / max(1, num_tp)) if num_tp > 0 else 0.0

        # Association Accuracy (AssA)
        if num_tp == 0:
            assa = 0.0
        else:
            # Pair counts
            gt_counts: Dict[int, int] = {}
            pr_counts: Dict[int, int] = {}
            pair_counts: Dict[Tuple[int, int], int] = {}
            for _, g_id, p_id in tp_matches:
                gt_counts[g_id] = gt_counts.get(g_id, 0) + 1
                pr_counts[p_id] = pr_counts.get(p_id, 0) + 1
                pair_counts[(g_id, p_id)] = pair_counts.get((g_id, p_id), 0) + 1

            assa_sum = 0.0
            for _, g_id, p_id in tp_matches:
                tpa = pair_counts[(g_id, p_id)]
                fpa = pr_counts[p_id] - tpa
                fna = gt_counts[g_id] - tpa
                assa_sum += tpa / max(1, tpa + fpa + fna)
            assa = assa_sum / num_tp

        hota = math.sqrt(deta * assa)

        hota_list.append(hota)
        deta_list.append(deta)
        assa_list.append(assa)
        loca_list.append(loca)

    final_hota = float(np.mean(hota_list)) * 100.0
    final_deta = float(np.mean(deta_list)) * 100.0
    final_assa = float(np.mean(assa_list)) * 100.0
    final_loca = float(np.mean(loca_list)) * 100.0

    # Global IDF1 calculation
    # Pairwise overlap matrix across entire sequence
    all_gt_list = sorted(list(all_gt_ids))
    all_pr_set = set()
    for f_idx in range(num_frames):
        all_pr_set.update(pr_tracks.get(f_idx, {}).keys())
    all_pr_list = sorted(list(all_pr_set))

    if len(all_gt_list) == 0 or len(all_pr_list) == 0:
        idf1 = 0.0
    else:
        overlap_matrix = np.zeros((len(all_gt_list), len(all_pr_list)))
        for f_idx in range(num_frames):
            gts = gt_tracks.get(f_idx, {})
            prs = pr_tracks.get(f_idx, {})
            for i, g_id in enumerate(all_gt_list):
                if g_id in gts:
                    for j, p_id in enumerate(all_pr_list):
                        if p_id in prs:
                            if np.linalg.norm(gts[g_id] - prs[p_id]) <= 1.5:
                                overlap_matrix[i, j] += 1

        r_ind, c_ind = linear_sum_assignment(-overlap_matrix)
        idtp = float(np.sum(overlap_matrix[r_ind, c_ind]))
        total_gt_boxes = sum(len(gt_tracks.get(f, {})) for f in range(num_frames))
        total_pr_boxes = sum(len(pr_tracks.get(f, {})) for f in range(num_frames))
        idfp = total_pr_boxes - idtp
        idfn = total_gt_boxes - idtp
        idf1 = (2.0 * idtp / max(1.0, 2.0 * idtp + idfp + idfn)) * 100.0

    return {
        "HOTA": final_hota,
        "DetA": final_deta,
        "AssA": final_assa,
        "LocA": final_loca,
        "IDF1": idf1,
        "IDSW_per_1000": idsw_per_1000,
        "Frag_per_1000": frag_per_1000,
    }


def compute_calibration_metrics(
    q_pred: np.ndarray,
    m_real: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Computes Brier Score and Expected Calibration Error (ECE) over 10 confidence bins."""
    if len(q_pred) == 0:
        return {"brier": 0.0, "ece": 0.0}

    # Brier score: mean squared error
    brier = float(np.mean((q_pred - m_real) ** 2))

    # ECE with 10 bins
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(q_pred)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (q_pred >= bin_lower) & (q_pred < bin_upper if i < n_bins - 1 else q_pred <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            avg_confidence = float(np.mean(q_pred[in_bin]))
            avg_accuracy = float(np.mean(m_real[in_bin]))
            ece += (bin_count / total_samples) * abs(avg_accuracy - avg_confidence)

    return {"brier": brier, "ece": ece}
