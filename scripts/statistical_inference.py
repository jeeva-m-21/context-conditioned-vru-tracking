"""Unified Statistical Inference Module for Hierarchical Block Bootstrapping.

Implements:
1. Hierarchical / Block Bootstrap (10,000 resamples, seed=42):
   - Resamples at the scenario / site level first, then sequences within scenarios.
   - Carries across-seed variance (seeds 1-5) directly into bootstrap distribution.
2. Holm-Bonferroni step-down multiplicity adjustment.
3. Sequence-level block-bootstrapping for Brier Score and ECE.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def hierarchical_block_bootstrap(
    df: pd.DataFrame,
    metric_col: str,
    baseline_variant: str,
    proposed_variant: str,
    n_boot: int = 10000,
    seed: int = 42,
    lower_is_better: bool = False,
) -> Dict[str, float]:
    """Runs 10,000 hierarchical block-bootstrap resamples.

    Preserves across-seed variance (seeds 1-5) without pre-averaging.
    Returns:
    - mean_diff: proposed - baseline
    - ci_lower: 2.5 percentile
    - ci_upper: 97.5 percentile
    - p_value: empirical two-sided bootstrap p-value
    """
    rng = np.random.default_rng(seed)

    scenarios = df["scenario_id"].unique()
    num_scenarios = len(scenarios)

    # Pre-filter
    df_base = df[df["variant"] == baseline_variant]
    df_prop = df[df["variant"] == proposed_variant]

    # Map (scenario_id, sequence_id, seed) -> value
    base_map = df_base.set_index(["scenario_id", "sequence_id", "seed"])[metric_col].to_dict()
    prop_map = df_prop.set_index(["scenario_id", "sequence_id", "seed"])[metric_col].to_dict()

    # Pre-group sequence IDs by scenario
    seqs_by_scenario = df.groupby("scenario_id")["sequence_id"].unique().to_dict()
    seeds = df["seed"].unique()

    diff_samples = np.zeros(n_boot)

    for b in range(n_boot):
        # 1. Resample scenarios with replacement
        sampled_scenarios = rng.choice(scenarios, size=num_scenarios, replace=True)

        batch_base = []
        batch_prop = []

        for sc in sampled_scenarios:
            avail_seqs = seqs_by_scenario[sc]
            # 2. Resample sequences within this scenario with replacement
            sampled_seqs = rng.choice(avail_seqs, size=len(avail_seqs), replace=True)

            for sq in sampled_seqs:
                # 3. Resample across-seed replicates
                for sd in seeds:
                    val_b = base_map.get((sc, sq, sd), None)
                    val_p = prop_map.get((sc, sq, sd), None)
                    if val_b is not None and val_p is not None:
                        batch_base.append(val_b)
                        batch_prop.append(val_p)

        if len(batch_base) > 0:
            diff_samples[b] = np.mean(batch_prop) - np.mean(batch_base)

    mean_diff = float(np.mean(diff_samples))
    ci_lower = float(np.percentile(diff_samples, 2.5))
    ci_upper = float(np.percentile(diff_samples, 97.5))

    # Empirical two-sided p-value against null difference = 0
    # Proportion of resampled diffs that cross 0
    if lower_is_better:
        # Negative diff is improvement
        p_val = 2.0 * min(np.mean(diff_samples >= 0), np.mean(diff_samples <= 0))
    else:
        # Positive diff is improvement
        p_val = 2.0 * min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))
    p_val = float(np.clip(p_val, 1e-4, 1.0))

    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }


def holm_bonferroni_adjustment(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down adjustment to control family-wise error rate."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by ascending p-value
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(m)

    running_max = 0.0
    for k, idx in enumerate(sorted_indices):
        # Multiplier: (m - k)
        adjusted = (m - k) * p_values[idx]
        running_max = max(running_max, adjusted)
        adj_p[idx] = min(1.0, running_max)

    return list(adj_p)


def sequence_block_bootstrap_calibration(
    seq_records: List[Dict[str, Any]],  # list of dicts with 'seq_id', 'q_pred', 'm_real'
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Sequence-level block bootstrap for Brier score and ECE confidence intervals."""
    rng = np.random.default_rng(seed)
    n_seq = len(seq_records)
    if n_seq == 0:
        return {"brier": (0.0, 0.0, 0.0), "ece": (0.0, 0.0, 0.0)}

    brier_samples = np.zeros(n_boot)
    ece_samples = np.zeros(n_boot)

    for b in range(n_boot):
        sample_indices = rng.choice(n_seq, size=n_seq, replace=True)
        sampled_q = []
        sampled_m = []
        for idx in sample_indices:
            sampled_q.extend(seq_records[idx]["q_pred"])
            sampled_m.extend(seq_records[idx]["m_real"])

        sampled_q = np.array(sampled_q)
        sampled_m = np.array(sampled_m)
        if len(sampled_q) > 0:
            brier_samples[b] = np.mean((sampled_q - sampled_m) ** 2)
            # 10-bin ECE
            bin_boundaries = np.linspace(0.0, 1.0, 11)
            ece_val = 0.0
            tot = len(sampled_q)
            for i in range(10):
                in_bin = (sampled_q >= bin_boundaries[i]) & (
                    sampled_q < bin_boundaries[i + 1] if i < 9 else sampled_q <= bin_boundaries[i + 1]
                )
                cnt = np.sum(in_bin)
                if cnt > 0:
                    ece_val += (cnt / tot) * abs(np.mean(sampled_m[in_bin]) - np.mean(sampled_q[in_bin]))
            ece_samples[b] = ece_val

    brier_mean = float(np.mean(brier_samples))
    brier_ci = (float(np.percentile(brier_samples, 2.5)), float(np.percentile(brier_samples, 97.5)))
    ece_mean = float(np.mean(ece_samples))
    ece_ci = (float(np.percentile(ece_samples, 2.5)), float(np.percentile(ece_samples, 97.5)))

    return {
        "brier": (brier_mean, brier_ci[0], brier_ci[1]),
        "ece": (ece_mean, ece_ci[0], ece_ci[1]),
    }
