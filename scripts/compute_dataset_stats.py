"""Compute Dataset and Opportunity Statistics Table (Table I-Data).

Generates:
- results/dataset_statistics.csv
- results/table_dataset_stats.tex
- results/revision_package/tables/table1_dataset_stats.tex

Outputs for each dataset (R-LiViT, DAIR-V2X-Seq/SPD, V2X-Sim):
- Scenarios/sites count and total sequences
- Exact split counts: N_train (70%), N_val (15%), N_test (15%)
- Total evaluation frames and unique VRU tracks (Pedestrian vs Cyclist) per split
- Eligible opportunities (G_k^ell = 1), empirical mean q_bar, Bernoulli variance Var(M),
  context-explained variance V_context = Var(E[M | phi]), and residual variance V_residual.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from dataset_generator import build_dataset_manifests, SequenceData
from channel_emulation import ChannelEmulator


def compute_statistics():
    data_dir = Path("data/manifests")
    datasets = build_dataset_manifests(data_dir)
    channel = ChannelEmulator(seed=42)

    rows = []

    for d_name, seqs in datasets.items():
        sites = set(s.site_id for s in seqs)
        scenarios = set(s.scenario_id for s in seqs)
        total_seqs = len(seqs)

        # Group by split
        splits_data = {"train": [], "val": [], "test": []}
        for s in seqs:
            splits_data[s.split].append(s)

        n_train = len(splits_data["train"])
        n_val = len(splits_data["val"])
        n_test = len(splits_data["test"])

        for split_name in ["train", "val", "test", "total"]:
            cur_seqs = seqs if split_name == "total" else splits_data[split_name]

            num_frames = sum(s.num_frames for s in cur_seqs)

            # Unique tracks by class
            num_pedestrians = 0
            num_cyclists = 0
            total_opps = 0
            received_opps = 0
            predicted_qs = []
            realized_ms = []

            for s in cur_seqs:
                channel.reset(seed=hash(s.sequence_id) % 100000)
                for vru in s.vrus:
                    if vru.vru_class == "Pedestrian":
                        num_pedestrians += 1
                    else:
                        num_cyclists += 1

                    # Count opportunities
                    for k in range(vru.start_frame, vru.end_frame):
                        idx = k - vru.start_frame
                        pos = vru.trajectory[idx, :2]
                        occ = vru.occlusions[idx]
                        dist = float(np.linalg.norm(pos))
                        cbr = s.channel_busy_ratio[k]

                        # Check if within sender range (eligible opportunity G=1)
                        is_eligible = False
                        for sender in s.senders:
                            s_pos = sender.trajectory[k, :2]
                            if np.linalg.norm(pos - s_pos) <= 40.0:
                                is_eligible = True
                                break

                        if is_eligible:
                            total_opps += 1
                            # Stochastic channel outcome
                            q_pred = channel.compute_smooth_q(dist, occ, cbr)
                            m_real = 1.0 if channel.rng.random() < q_pred else 0.0
                            received_opps += int(m_real)
                            predicted_qs.append(q_pred)
                            realized_ms.append(m_real)

            realized_ms = np.array(realized_ms)
            predicted_qs = np.array(predicted_qs)

            q_bar = float(np.mean(realized_ms)) if len(realized_ms) > 0 else 0.0
            var_m = float(np.var(realized_ms)) if len(realized_ms) > 0 else 0.0
            v_context = float(np.var(predicted_qs)) if len(predicted_qs) > 0 else 0.0
            v_residual = max(0.0, var_m - v_context)

            rows.append({
                "dataset": d_name,
                "split": split_name,
                "num_sites": len(sites),
                "num_scenarios": len(scenarios),
                "num_seqs": len(cur_seqs),
                "eval_frames": num_frames,
                "pedestrians": num_pedestrians,
                "cyclists": num_cyclists,
                "total_vrus": num_pedestrians + num_cyclists,
                "eligible_opps": total_opps,
                "q_bar": q_bar,
                "var_m": var_m,
                "v_context": v_context,
                "v_residual": v_residual,
            })

    df = pd.DataFrame(rows)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/revision_package/tables").mkdir(parents=True, exist_ok=True)

    csv_path = Path("results/dataset_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Generate LaTeX Table
    tex_content = generate_latex_table(df)
    tex_path1 = Path("results/table_dataset_stats.tex")
    tex_path2 = Path("results/revision_package/tables/table1_dataset_stats.tex")
    tex_path1.write_text(tex_content)
    tex_path2.write_text(tex_content)
    print(f"Saved {tex_path1} and {tex_path2}")


def generate_latex_table(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Dataset, Partitioning, and Cooperative-Opportunity Statistics (Table I).}",
        r"\label{tab:dataset_stats}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"Dataset & Split & Seqs & Frames & Pedestrians & Cyclists & Total VRUs & Opps ($G=1$) & $\bar{q}$ & $V_{\text{context}}$ & $V_{\text{residual}}$ \\",
        r"\midrule",
    ]

    for d_name in ["R-LiViT", "DAIR-V2X-Seq", "V2X-Sim"]:
        sub_df = df[df["dataset"] == d_name]
        for idx, row in sub_df.iterrows():
            sp = row["split"].capitalize()
            bold = (row["split"] == "total")
            prefix = r"\textbf{" if bold else ""
            suffix = r"}" if bold else ""

            d_str = d_name if row["split"] == "train" else ""
            line = (
                f"{d_str} & {prefix}{sp}{suffix} & "
                f"{prefix}{int(row['num_seqs'])}{suffix} & "
                f"{prefix}{int(row['eval_frames'])}{suffix} & "
                f"{prefix}{int(row['pedestrians'])}{suffix} & "
                f"{prefix}{int(row['cyclists'])}{suffix} & "
                f"{prefix}{int(row['total_vrus'])}{suffix} & "
                f"{prefix}{int(row['eligible_opps'])}{suffix} & "
                f"{prefix}{row['q_bar']:.4f}{suffix} & "
                f"{prefix}{row['v_context']:.4f}{suffix} & "
                f"{prefix}{row['v_residual']:.4f}{suffix} \\\\"
            )
            lines.append(line)
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    compute_statistics()
