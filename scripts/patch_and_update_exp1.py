#!/usr/bin/env python3
"""
scripts/patch_and_update_exp1.py

Targeted update for Experiment 1:
1. Re-runs 'two_sensor_ungated' across all 3 datasets, 2 channel modes (stochastic, external_physics),
   and 5 seeds using the corrected ungated two-sensor filter and unconditioned fixed prediction logging.
2. Runs 'config_d_logistic' and 'config_d_mlp' on external_physics channel across all 3 datasets and 5 seeds.
   (For stochastic channel, retrieves exact matched evaluations from exp2_attribution_ad_raw.csv).
3. Merges updated records into results/revision_package/raw_logs/exp1_main_benchmark_raw.csv.
4. Correctly computes Brier and ECE for all methods over held-out transmission opportunities.
"""

from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_generator import build_dataset_manifests, SequenceData
from context_estimators import ReceptionEstimator
from lmb_tracker import TrackerConfig
from evaluator import compute_calibration_metrics
from run_experiments import train_estimators_on_split, run_tracker_on_sequence

NUM_WORKERS = 10
SEEDS = [1, 2, 3, 4, 5]
RAW_LOGS_DIR = Path("results/revision_package/raw_logs")


def _worker_run(args) -> Dict[str, Any]:
    (seq, cfg, ch_mode, rec_est, norm, seed, meta) = args
    metrics, _, q_pred_list, m_real_list = run_tracker_on_sequence(
        seq, cfg, ch_mode, rec_est, norm, seed=seed
    )
    row = {**meta, **metrics}
    row["_q_pred"] = q_pred_list
    row["_m_real"] = m_real_list
    return row


def main():
    print("Loading datasets and existing raw logs...", flush=True)
    manifest_dir = Path("data/manifests")
    datasets = build_dataset_manifests(manifest_dir)

    exp1_csv = RAW_LOGS_DIR / "exp1_main_benchmark_raw.csv"
    exp2_csv = RAW_LOGS_DIR / "exp2_attribution_ad_raw.csv"

    df1_orig = pd.read_csv(exp1_csv)
    df2_orig = pd.read_csv(exp2_csv)

    # Filter out old two_sensor_ungated and any old config_d rows from df1
    df1_kept = df1_orig[~df1_orig["variant"].isin(["two_sensor_ungated", "config_d_logistic", "config_d_mlp"])].copy()

    # Prepare tasks to run
    tasks = []

    # 1. two_sensor_ungated on stochastic and external_physics
    # 2. config_d_logistic and config_d_mlp on external_physics
    variants_to_run = [
        ("two_sensor_ungated", ["stochastic", "external_physics"], TrackerConfig(mode="two_sensor_ungated", p_D_fixed=0.68, q_fixed=0.55)),
        ("config_d_logistic", ["external_physics"], TrackerConfig(mode="config_d_logistic", p_D_fixed=0.68)),
        ("config_d_mlp", ["external_physics"], TrackerConfig(mode="config_d_mlp", p_D_fixed=0.68)),
    ]

    for d_name, all_seqs in datasets.items():
        train_seqs = [s for s in all_seqs if s.split == "train"]
        test_seqs = [s for s in all_seqs if s.split == "test"]

        for seed in SEEDS:
            # Fit estimators
            log_stoch, mlp_stoch, _, norm_stoch, q_bar_stoch = train_estimators_on_split(
                train_seqs, channel_mode="stochastic", seed=seed
            )
            log_ext, mlp_ext, _, norm_ext, q_bar_ext = train_estimators_on_split(
                train_seqs, channel_mode="external_physics", seed=seed
            )
            ch_models = {
                "stochastic": (log_stoch, mlp_stoch, norm_stoch, q_bar_stoch),
                "external_physics": (log_ext, mlp_ext, norm_ext, q_bar_ext),
            }

            for v_name, ch_modes, cfg_template in variants_to_run:
                for ch_mode in ch_modes:
                    log_est, mlp_est, norm, q_bar_train = ch_models[ch_mode]
                    cfg = TrackerConfig(
                        mode=cfg_template.mode,
                        p_D_fixed=cfg_template.p_D_fixed,
                        q_fixed=cfg_template.q_fixed,
                    )
                    rec_est = mlp_est if "mlp" in v_name else (
                        log_est if "logistic" in v_name else ReceptionEstimator(model_type="fixed", q_fixed=cfg.q_fixed)
                    )

                    for seq in test_seqs:
                        meta = {
                            "dataset": d_name,
                            "channel": ch_mode,
                            "variant": v_name,
                            "seed": seed,
                            "scenario_id": seq.scenario_id,
                            "sequence_id": seq.sequence_id,
                            "ablation_variant": "core_radio",
                        }
                        tasks.append((seq, cfg, ch_mode, rec_est, norm, seed, meta))

    print(f"Executing {len(tasks)} targeted tasks across {NUM_WORKERS} workers...", flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(_worker_run, tasks, chunksize=16))

    # Add config_d_logistic and config_d_mlp from exp2 for stochastic channel
    exp2_config_d = df2_orig[df2_orig["variant"].isin(["config_d_logistic", "config_d_mlp"])].copy()
    exp2_config_d["channel"] = "stochastic"
    exp2_config_d["ablation_variant"] = "core_radio"
    # Ensure all columns present
    for col in df1_kept.columns:
        if col not in exp2_config_d.columns and not col.startswith("_"):
            exp2_config_d[col] = np.nan

    # Compute calibration metrics across newly run tasks
    group_map = {}
    for r in results:
        key = (r["dataset"], r["channel"], r["variant"], r["seed"])
        if key not in group_map:
            group_map[key] = {"q": [], "m": []}
        group_map[key]["q"].extend(r.pop("_q_pred"))
        group_map[key]["m"].extend(r.pop("_m_real"))

    for r in results:
        key = (r["dataset"], r["channel"], r["variant"], r["seed"])
        cal = compute_calibration_metrics(np.array(group_map[key]["q"]), np.array(group_map[key]["m"]), n_bins=10)
        r["brier"] = cal["brier"]
        r["ece"] = cal["ece"]

    df_new = pd.DataFrame(results)

    # Combine: df1_kept + df_new + exp2_config_d
    cols = [c for c in df1_kept.columns if not c.startswith("_")]
    combined_df = pd.concat([df1_kept[cols], df_new[cols], exp2_config_d[cols]], ignore_index=True)

    combined_df.to_csv(exp1_csv, index=False)
    print(f"Successfully updated {exp1_csv} ({len(combined_df)} rows).", flush=True)


if __name__ == "__main__":
    main()
