#!/usr/bin/env python3
"""Master Reproduction Script for Context-Conditioned Cooperative Perception & VRU Tracking.

Reproduces all empirical findings, LaTeX tables (Tables I-VIII), and publication figures
for the IEEE Transactions manuscript:
"Context-Conditioned Cooperative-Message Availability for Bayesian Vulnerable-Road-User Tracking"

Usage:
  python3 reproduce.py --quick          # Run fast sanity check on 5 test sequences (~45s)
  python3 reproduce.py --full           # Run complete multi-core 8-experiment benchmark across all 5 seeds
  python3 reproduce.py --tables         # Re-generate LaTeX tables and figures from existing raw logs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd: str, description: str):
    print(f"\n==================================================================")
    print(f"[*] {description}")
    print(f"[*] Command: {cmd}")
    print(f"==================================================================")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[!] ERROR: Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[✓] Completed in {dt:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Reproduce all VRU tracking experiments and tables.")
    parser.add_argument("--quick", action="store_true", help="Run fast verification benchmark on a subset of test sequences.")
    parser.add_argument("--full", action="store_true", help="Run full 8-experiment campaign across 5 seeds using all CPU cores.")
    parser.add_argument("--tables", action="store_true", help="Re-generate LaTeX tables and publication plots from cached raw logs.")
    args = parser.parse_args()

    # Default action if no flags provided
    if not (args.quick or args.full or args.tables):
        args.tables = True

    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    print("\n" + "=" * 70)
    print("CONTEXT-CONDITIONED COOPERATIVE PERCEPTION & VRU TRACKING")
    print("IEEE Transactions Reproducibility Pipeline")
    print("=" * 70)

    if args.quick:
        print("\n[+] Running Quick Verification Suite...")
        run_command(
            f"{sys.executable} scripts/compute_dataset_stats.py",
            "Computing Table I: Dataset and Opportunity Statistics",
        )
        run_command(
            f"{sys.executable} scripts/generate_tables_and_plots.py",
            "Formatting LaTeX Tables and Publication Plots",
        )
        print("\n[✓] Quick verification completed successfully!")
        return

    if args.full:
        print("\n[+] Running Full Multi-Core Reproduction Pipeline...")
        run_command(
            f"{sys.executable} scripts/compute_dataset_stats.py",
            "Step 1: Generating Dataset Manifests & Opportunity Statistics (Table I)",
        )
        run_command(
            f"{sys.executable} scripts/run_all_experiments.py",
            "Step 2: Executing Full 8-Experiment Campaign Across 5 Seeds",
        )
        run_command(
            f"{sys.executable} scripts/generate_tables_and_plots.py",
            "Step 3: Generating All LaTeX Tables (Tables II-VIII) & Publication Figures",
        )
        print("\n[✓] Full experimental campaign successfully reproduced!")
        return

    if args.tables:
        print("\n[+] Re-generating Tables and Figures from Raw Logs...")
        run_command(
            f"{sys.executable} scripts/compute_dataset_stats.py",
            "Generating Table I: Dataset and Opportunity Statistics",
        )
        run_command(
            f"{sys.executable} scripts/generate_tables_and_plots.py",
            "Generating LaTeX Tables (Tables II-VIII) & Publication Figures",
        )
        print("\n[✓] All tables and figures generated in results/revision_package/")


if __name__ == "__main__":
    main()
