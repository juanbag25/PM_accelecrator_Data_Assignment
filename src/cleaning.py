"""Phase 2 - Cleaning & preprocessing.

Turns the raw CSV into a clean, ordered, de-duplicated daily panel and writes
``data/processed/clean.parquet`` plus a human-readable ``reports/cleaning_report.md``.

Pipeline (in order):
  1. Parse timestamp -> daily ``date``; sort by (location, country, time).
  2. Consolidate same-city name variants within a country (typos / renamed
     stations) using coordinate proximity + complementary date ranges.
  3. Drop redundant imperial columns and the epoch timestamp.
  4. Resolve intra-day duplicates -> keep the last record per (series, date).
  5. Clip physically impossible values (never delete real extremes) and flag
     statistical outliers on the target with a per-series modified z-score (MAD).
  6. Reindex each series to a continuous daily frequency and impute gaps
     (time interpolation for numerics, ffill/bfill for categoricals); mark
     imputed rows with ``is_imputed``.

Run with:
    conda run -n weather-forecasting python -m src.cleaning
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import load_config, resolve

# Imperial / duplicate columns to drop (metric system kept) + epoch timestamp.
REDUNDANT_COLS = [
    "temperature_fahrenheit", "wind_mph", "pressure_in", "precip_in",
    "visibility_miles", "gust_mph", "feels_like_fahrenheit", "last_updated_epoch",
]

# Physical plausibility limits for clipping. None = unbounded on that side.
PHYSICAL_LIMITS = {
    "temperature_celsius": (-90, 60),
    "feels_like_celsius": (-90, 65),
    "humidity": (0, 100),
    "cloud": (0, 100),
    "moon_illumination": (0, 100),
    "pressure_mb": (870, 1085),
    "wind_kph": (0, 410),      # world record sustained/gust ~408 kph
    "gust_kph": (0, 500),
    "wind_degree": (0, 360),
    "precip_mm": (0, None),
    "visibility_km": (0, None),
    "uv_index": (0, 20),
    "air_quality_Carbon_Monoxide": (0, None),
    "air_quality_Ozone": (0, None),
    "air_quality_Nitrogen_dioxide": (0, None),
    "air_quality_Sulphur_dioxide": (0, None),
    "air_quality_PM2.5": (0, None),
    "air_quality_PM10": (0, None),
}

# Categorical columns (ffill/bfill on reindex); everything else numeric.
CATEGORICAL_COLS = [
    "timezone", "condition_text", "wind_direction", "moon_phase",
    "sunrise", "sunset", "moonrise", "moonset",
]

MAD_THRESHOLD = 3.5  # modified z-score cutoff for the outlier flag


# --------------------------------------------------------------------------- #
def parse_and_sort(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ts, loc, ctry = cfg["time"]["raw_timestamp"], cfg["time"]["series_key"], cfg["time"]["disambiguation_key"]
    df = df.copy()
    df[ts] = pd.to_datetime(df[ts])
    df["date"] = df[ts].dt.normalize()
    return df.sort_values([loc, ctry, ts]).reset_index(drop=True)


def consolidate_name_variants(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Merge same-city name variants within a country.

    Two (country, name) series are merged when their coordinates are within
    ``variant_coord_tol_deg`` AND their date ranges overlap by at most
    ``variant_overlap_tol_days`` (i.e. they are complementary in time, which is
    the signature of a station renamed over time rather than two parallel places).
    The canonical name of a cluster is the member with the most distinct days.
    """
    loc, ctry = cfg["time"]["series_key"], cfg["time"]["disambiguation_key"]
    same_point_tol = cfg["cleaning"]["variant_same_point_deg"]
    coord_tol = cfg["cleaning"]["variant_coord_tol_deg"]
    overlap_tol = cfg["cleaning"]["variant_overlap_tol_days"]

    stats = (df.groupby([ctry, loc])
               .agg(lat=("latitude", "median"), lon=("longitude", "median"),
                    n=("date", "nunique"), first=("date", "min"), last=("date", "max"))
               .reset_index())
    n_days = {(r[ctry], r[loc]): r["n"] for _, r in stats.iterrows()}

    # Union-find over (country, name) keys.
    parent = {k: k for k in n_days}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for country, sub in stats.groupby(ctry):
        recs = sub.to_dict("records")
        for a, b in itertools.combinations(recs, 2):
            dist = float(np.hypot(a["lat"] - b["lat"], a["lon"] - b["lon"]))
            if dist < same_point_tol:
                union((country, a[loc]), (country, b[loc]))  # spelling variant, same point
            elif dist < coord_tol:
                overlap = (min(a["last"], b["last"]) - max(a["first"], b["first"])).days
                if overlap <= overlap_tol:  # nearby + complementary in time -> moved station
                    union((country, a[loc]), (country, b[loc]))

    clusters = defaultdict(list)
    for k in n_days:
        clusters[find(k)].append(k)

    name_map, merges = {}, []
    for members in clusters.values():
        canon = max(members, key=lambda k: n_days[k])[1]
        for m in members:
            name_map[m] = canon
        if len(members) > 1:
            country = members[0][0]
            variants = sorted((m[1] for m in members), key=lambda nm: -n_days[(country, nm)])
            merges.append({"country": country, "canonical": canon,
                           "variants": variants,
                           "days": {v: int(n_days[(country, v)]) for v in variants}})

    keys = list(zip(df[ctry], df[loc]))
    df = df.copy()
    df[loc] = [name_map[k] for k in keys]
    return df, merges


def clip_physical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    clipped = {}
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        n = int(mask.sum())
        if n:
            clipped[col] = n
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df, clipped


def flag_outliers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-series modified z-score (MAD) flag on the target. Does not drop rows."""
    loc, ctry = cfg["time"]["series_key"], cfg["time"]["disambiguation_key"]
    target = cfg["target"]["primary"]
    df = df.copy()
    grp = df.groupby([loc, ctry])[target]
    med = grp.transform("median")
    abs_dev = (df[target] - med).abs()
    mad = abs_dev.groupby([df[loc], df[ctry]]).transform("median")
    mod_z = 0.6745 * (df[target] - med) / mad.replace(0, np.nan)
    df["is_outlier"] = (mod_z.abs() > MAD_THRESHOLD).fillna(False)
    return df


def reindex_and_impute(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Reindex each series to a continuous daily frequency and impute gaps."""
    loc, ctry = cfg["time"]["series_key"], cfg["time"]["disambiguation_key"]
    df = df.copy()
    df["is_imputed"] = False

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in df.columns
                if c not in cat_cols + [loc, ctry, "date", "is_imputed", "is_outlier"]
                and pd.api.types.is_numeric_dtype(df[c])]

    parts = []
    for (l, c), g in df.groupby([loc, ctry], sort=False):
        g = g.drop_duplicates("date").set_index("date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full)
        newly = g["is_imputed"].isna()
        g[loc], g[ctry] = l, c
        g["is_imputed"] = newly.values
        g["is_outlier"] = g["is_outlier"].fillna(False).astype(bool)
        g[num_cols] = g[num_cols].interpolate(method="time", limit_direction="both")
        g[cat_cols] = g[cat_cols].ffill().bfill()
        parts.append(g.rename_axis("date").reset_index())

    out = pd.concat(parts, ignore_index=True)
    stats = {"rows_after_reindex": len(out),
             "imputed_rows": int(out["is_imputed"].sum())}
    return out, stats


# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    loc, ctry, ts = cfg["time"]["series_key"], cfg["time"]["disambiguation_key"], cfg["time"]["raw_timestamp"]
    rep: dict = {"rows_raw": len(df)}

    df = parse_and_sort(df, cfg)
    df, merges = consolidate_name_variants(df, cfg)
    rep["merges"] = merges
    rep["series_before"] = None  # filled below
    rep["n_merged_clusters"] = len(merges)

    df = df.drop(columns=[c for c in REDUNDANT_COLS if c in df.columns])
    df = df.drop(columns=[ts])  # keep daily 'date' as the canonical time index

    before = len(df)
    df = df.drop_duplicates(subset=[loc, ctry, "date"], keep="last").reset_index(drop=True)
    rep["dups_removed"] = before - len(df)

    df, clipped = clip_physical(df)
    rep["clipped"] = clipped

    df = flag_outliers(df, cfg)
    rep["n_outliers"] = int(df["is_outlier"].sum())

    df, ristats = reindex_and_impute(df, cfg)
    rep.update(ristats)

    # Series-length summary (for the min_series_days modeling threshold).
    per = df.groupby([loc, ctry])["date"].nunique()
    thr = cfg["cleaning"]["min_series_days"]
    rep["n_series"] = int(per.size)
    rep["n_series_ge_threshold"] = int((per >= thr).sum())
    rep["min_series_days"] = thr
    return df, rep


def build_report(rep: dict, cfg: dict) -> str:
    L = ["# Cleaning Report — Phase 2\n",
         "Transforms the raw CSV into `data/processed/clean.parquet`.\n",
         "## Row accounting\n",
         f"- Raw rows: **{rep['rows_raw']:,}**",
         f"- Intra-day duplicates removed (keep last): **{rep['dups_removed']:,}**",
         f"- Rows after reindex to daily frequency: **{rep['rows_after_reindex']:,}**",
         f"- Imputed (gap-filled) rows: **{rep['imputed_rows']:,}** "
         f"({100*rep['imputed_rows']/rep['rows_after_reindex']:.1f}%)",
         f"- Outliers flagged (kept, `is_outlier`): **{rep['n_outliers']:,}**\n",
         "## Series\n",
         f"- Distinct series after consolidation: **{rep['n_series']}**",
         f"- Name-variant clusters merged: **{rep['n_merged_clusters']}**",
         f"- Series with >= {rep['min_series_days']} days (used in global modeling): "
         f"**{rep['n_series_ge_threshold']}**\n",
         "## Physical clipping (values clamped, not deleted)\n"]
    if rep["clipped"]:
        L.append("| Column | Values clipped |")
        L.append("| --- | --- |")
        for c, n in sorted(rep["clipped"].items(), key=lambda x: -x[1]):
            L.append(f"| `{c}` | {n:,} |")
    else:
        L.append("_None._")
    L.append("\n## Consolidated name variants (same city, renamed over time)\n")
    if rep["merges"]:
        L.append("| Country | Canonical | Merged variants (days) |")
        L.append("| --- | --- | --- |")
        for m in sorted(rep["merges"], key=lambda x: x["country"]):
            vs = ", ".join(f"{v} ({m['days'][v]}d)" for v in m["variants"])
            L.append(f"| {m['country']} | **{m['canonical']}** | {vs} |")
    else:
        L.append("_None._")
    L.append("")
    return "\n".join(L)


def main() -> None:
    cfg = load_config()
    csv_path = resolve(cfg["data"]["raw_csv"])
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Raw shape={df.shape}")

    clean_df, rep = clean(df, cfg)
    print(f"Clean shape={clean_df.shape}")

    out_path = resolve(cfg["data"]["clean_parquet"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(out_path, index=False)

    report = build_report(rep, cfg)
    rep_path = resolve("reports/cleaning_report.md")
    rep_path.write_text(report, encoding="utf-8")

    print("\n===== CLEANING SUMMARY =====")
    for k in ["rows_raw", "dups_removed", "rows_after_reindex", "imputed_rows",
              "n_outliers", "n_series", "n_merged_clusters", "n_series_ge_threshold"]:
        print(f"  {k}: {rep[k]}")
    print(f"  clipped columns: {rep['clipped']}")
    print(f"\nSaved: {out_path}")
    print(f"Report: {rep_path}")


if __name__ == "__main__":
    main()
