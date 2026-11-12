# VRU Cooperative Tracking Benchmark: Reproducibility Package

This package contains the code, evaluation harness, and scripts to reproduce the experiments, tables, and figures for context-conditioned cooperative perception in multi-object VRU tracking.

## Repository Contents

* `scripts/`: Modular Python scripts implementing channel emulation, dataset generation, context estimators, LMB tracking filter, evaluation metrics, and bootstrap statistics.
* `reproduce.py`: Main CLI entrypoint to run benchmarks and regenerate artifacts.
* `requirements.txt`: Python package dependencies.
* `results/`: Output directories for generated logs, LaTeX tables, and plots.

## Quick Start

### 1. Installation

Ensure Python 3.10+ is installed, then install requirements:

```bash
pip install -r requirements.txt
```

### 2. Execution

* **Fast verification** (generates Table I and formats available tables):
  ```bash
  python3 reproduce.py --quick
  ```

* **Full multi-core benchmark** (runs all 8 experiments across 5 seeds in parallel):
  ```bash
  python3 reproduce.py --full
  ```

* **Real 3-D LiDAR detector experiment** (evaluates PointPillars and CenterPoint on R-LiViT and DAIR-V2X-Seq):
  ```bash
  python3 scripts/run_real_detector_experiment.py
  ```

* **Regenerate tables and plots** (from existing raw logs):
  ```bash
  python3 reproduce.py --tables
  ```

## Generated Outputs

* LaTeX tables: `results/revision_package/tables/`
* Publication figures: `results/revision_package/figures/`
* Raw execution logs: `results/revision_package/raw_logs/`
