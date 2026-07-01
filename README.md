# Weather Trend Forecasting

End-to-end data science pipeline for forecasting daily weather trends from the
**Global Weather Repository** dataset (panel of ~200 capital cities, daily since
Aug 2023, 40+ features). Built for the PM Accelerator technical assessment.

> **Status:** work in progress. This README is filled in incrementally; the
> results table, key findings and run instructions are finalized in Phase 10.

## Objective

Forecast the primary target `temperature_celsius` at a 7-day horizon (also probing
14 days) using a rigorous, reproducible, leakage-free pipeline. A **global LightGBM**
model is the primary approach, benchmarked against naive baselines, classical
per-city models (SARIMA / ETS / Prophet), a zero-shot foundational model (Chronos),
and an ensemble — all compared with **MASE** under a strict temporal split.

## Dataset

- Source: Global Weather Repository (Kaggle), daily panel, ~200 capital cities.
- Place the raw CSV at `data/raw/` (gitignored — not committed).

## Repository structure

```
data/{raw,processed}/   raw CSV (gitignored) + parquet artifacts
notebooks/              thin, numbered notebooks that call src/
src/                    modular pipeline code
reports/figures/        EDA and results figures
reports/metrics/        per-model metric JSONs
models/                 serialized models (gitignored)
config.yaml             global conventions (target, horizon, seed, paths)
environment.yml         conda environment (recommended on Windows)
requirements.txt        pip-equivalent dependency list
```

## How to run

> Detailed step-by-step instructions are finalized in Phase 10.

```bash
# 1. Create and activate the environment (recommended: conda)
conda env create -f environment.yml
conda activate weather-forecasting

# 2. Download the dataset from Kaggle and place the CSV in data/raw/

# 3. Run the pipeline phases (notebooks in notebooks/, code in src/)
```

## Results

_TODO (Phase 7-8): comparative table of all models by MAE / RMSE / MASE._

## Key findings

_TODO (Phase 4 & 9): main EDA, climate, air-quality and spatial insights._

## PM Accelerator mission

_TODO (Phase 10): include the PM Accelerator mission statement (from their LinkedIn
"About")._

## Reproducibility

- Random seed fixed to `42` across all stochastic steps.
- Strict temporal split (train / validation / test); the test set is evaluated once.
- All intermediate artifacts are persisted for reproducibility.
