# CLAUDE.md — Project operating rules (persist across sessions)

End-to-end weather **trend forecasting** pipeline (panel of ~200 capital cities,
daily since Aug 2023, 40+ features) for the PM Accelerator technical assessment.
Primary target: `temperature_celsius`. Primary model: **global LightGBM**.

The full step-by-step spec lives in `guia_pipeline_weather.md` (10 phases).
Always follow it in order and produce each phase's output artifact before moving on.

---

## Golden Rules (invariants — never violate)

1. **Temporal split in THREE blocks, never random.** train (oldest ~70%) →
   validation (middle ~15%) → test (newest ~15%), chronologically. The **test set
   is touched exactly once, at the very end**, only for reported metrics — never
   for tuning, feature selection, early stopping, or ensemble weights.
2. **Lags & rolling ALWAYS per city.** `sort_values(['location_name','date'])` then
   `groupby('location_name')` before any lag/rolling. Never mix data across cities.
3. **No lookahead.** A feature at date `t` may only use info from date `<= t`.
   Rolling features carry `shift(1)` so today never enters its own window.
4. **Fit on train only.** Scalers, encoders, imputers and feature selection are
   *fit* on train and *applied* to validation/test. Never the other way around.
5. **Always compare against a baseline.** No result stands without the naive next
   to it. The primary cross-model metric is **MASE**.
6. **Persist every artifact** (clean data, features, models, metrics, figures) for
   reproducibility.

---

## Environment (this machine)

- **OS:** Windows 11 Home (10.0.26200), 64-bit.
- **CPU:** Intel Core Ultra 9 285H — 16 cores / 16 logical. **No GPU / no CUDA.**
- **RAM:** ~31 GB.
- **Python:** Anaconda3 base has Python 3.13; project uses a **dedicated conda env**
  (`weather-forecasting`, Python 3.11) — see `environment.yml`. Always work inside it.
- **Model strategy given CPU-only:**
  - PRIMARY: global **LightGBM** (CPU-native, fast, handles NaN, no scaling).
  - Deep learning (N-BEATS/NHITS/TFT) is OPTIONAL — small configs, never block the
    pipeline on it.
  - Foundational models (Chronos-Bolt) are zero-shot inference only, on a few
    representative cities.
  - Install **PyTorch CPU build**: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Check system resources with real commands and adapt batch sizes / epochs if needed.

---

## Global conventions

| Concept | Value |
|---|---|
| Primary target | `temperature_celsius` |
| Secondary targets (optional) | `air_quality_PM2.5`, `precip_mm` |
| Time index | `last_updated` → `date` (day granularity) |
| Series key | `location_name` (+ `country` to disambiguate) |
| Forecast horizon | `H = 7` days (also probe 14) |
| Random seed | `SEED = 42` (everywhere) |
| Intermediate format | `.parquet` |
| Split quantiles | 0.70 / 0.85 (train / val / test) |

All conventions are also machine-readable in `config.yaml` — read from there, don't
hard-code.

---

## Working conventions

- **Language:** ALL code, comments, docstrings and the README in **English**
  (deliverable goes to an English-speaking review team). Converse with the user
  (Juan) in **Spanish**.
- **Reproducibility:** fix `SEED = 42` in everything stochastic.
- **Structure:** modular code in `src/`, thin notebooks in `notebooks/` that call it.
  Prefer this over giant notebooks.
- **Per phase:** produce the output artifact, `git commit` with a clear message,
  give Juan a 3-4 line summary, then continue.
- **Before heavy/optional installs or long training runs (> a few minutes): ask Juan first.**
- Keep `requirements.txt` (and `environment.yml`) updated as packages are installed.
- After Phase 1 (data audit): STOP and show the audit report + plan before proceeding.

---

## Repo structure

```
data/{raw,processed}/   raw CSV (gitignored) + parquet artifacts
notebooks/              thin, numbered notebooks
src/                    modular pipeline code
reports/figures/        figures (tracked); figures/advanced/ for Phase 9
reports/metrics/        per-model metric JSONs (tracked)
models/                 serialized models (gitignored)
config.yaml             global conventions
```

## Progress

- [x] Phase 0 — Project setup (structure, config, deps) — *env creation pending Juan's OK*
- [ ] Phase 1 — Data audit  ← **needs the raw CSV in data/raw/**
- [ ] Phases 2-10 — see `guia_pipeline_weather.md`
