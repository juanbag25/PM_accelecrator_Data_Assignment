"""Phase 1 - Data audit & validation.

Profiles the raw Global Weather Repository panel *before* any cleaning, and
writes a human-readable report to ``reports/data_audit.md``. Nothing here mutates
or persists the dataset; it only measures.

Run with:
    conda run -n weather-forecasting python -m src.data_audit
"""
from __future__ import annotations

import pandas as pd

from .config import load_config, resolve

# Unit-redundancy pairs (metric kept, imperial/duplicate dropped in Phase 2).
UNIT_PAIRS = [
    ("temperature_celsius", "temperature_fahrenheit"),
    ("wind_kph", "wind_mph"),
    ("pressure_mb", "pressure_in"),
    ("precip_mm", "precip_in"),
    ("visibility_km", "visibility_miles"),
    ("gust_kph", "gust_mph"),
]

# Physical plausibility limits (value outside => suspicious / impossible).
PHYSICAL_CHECKS = {
    "temperature_celsius < -90 or > 60": lambda d: (d["temperature_celsius"] < -90) | (d["temperature_celsius"] > 60),
    "humidity outside [0, 100]": lambda d: (d["humidity"] < 0) | (d["humidity"] > 100),
    "cloud outside [0, 100]": lambda d: (d["cloud"] < 0) | (d["cloud"] > 100),
    "moon_illumination outside [0, 100]": lambda d: (d["moon_illumination"] < 0) | (d["moon_illumination"] > 100),
    "pressure_mb <= 0": lambda d: d["pressure_mb"] <= 0,
    "pressure_mb outside [800, 1100]": lambda d: (d["pressure_mb"] < 800) | (d["pressure_mb"] > 1100),
    "precip_mm < 0": lambda d: d["precip_mm"] < 0,
    "wind_kph < 0": lambda d: d["wind_kph"] < 0,
    "gust_kph < 0": lambda d: d["gust_kph"] < 0,
    "visibility_km < 0": lambda d: d["visibility_km"] < 0,
    "uv_index < 0": lambda d: d["uv_index"] < 0,
}

# Key numeric variables to describe (physical-range sanity).
DESCRIBE_COLS = [
    "temperature_celsius", "feels_like_celsius", "humidity", "cloud",
    "pressure_mb", "wind_kph", "gust_kph", "precip_mm", "visibility_km",
    "uv_index", "air_quality_PM2.5", "air_quality_PM10",
]


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if x is None else str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(x, nd: int = 2) -> str:
    try:
        return f"{x:,.{nd}f}"
    except (ValueError, TypeError):
        return str(x)


# --------------------------------------------------------------------------- #
# Audit sections
# --------------------------------------------------------------------------- #
def build_report(df: pd.DataFrame, cfg: dict) -> tuple[str, dict]:
    """Return (markdown_report, key_metrics_dict)."""
    loc = cfg["time"]["series_key"]          # location_name
    ctry = cfg["time"]["disambiguation_key"]  # country
    ts = cfg["time"]["raw_timestamp"]         # last_updated
    target = cfg["target"]["primary"]

    L: list[str] = []
    m: dict = {}

    # Derive a day-level date from the local timestamp (audit only, not persisted).
    date = pd.to_datetime(df[ts]).dt.normalize()
    df = df.assign(date=date)

    # ----- 0. Header ----------------------------------------------------- #
    L.append("# Data Audit — Global Weather Repository\n")
    L.append("Phase 1 output. Profiles the **raw** panel before any cleaning "
             "(no rows/columns are modified here).\n")

    # ----- 1. Overview / initial snapshot (Phase 0.4) -------------------- #
    n_rows, n_cols = df.shape[0], df.shape[1] - 1  # minus the derived 'date'
    mem_mb = df.drop(columns="date").memory_usage(deep=True).sum() / 1e6
    dmin, dmax = df["date"].min(), df["date"].max()
    span_days = (dmax - dmin).days + 1
    m.update(n_rows=n_rows, n_cols=n_cols, date_min=str(dmin.date()),
             date_max=str(dmax.date()), span_days=span_days)

    L.append("## 1. Overview (initial snapshot)\n")
    L.append(md_table(
        ["Metric", "Value"],
        [["Rows", f"{n_rows:,}"],
         ["Columns", n_cols],
         ["Memory (deep)", f"{mem_mb:,.1f} MB"],
         ["Date range (local)", f"{dmin.date()} → {dmax.date()}"],
         ["Calendar span", f"{span_days:,} days (~{span_days/365.25:.1f} years)"],
         ["Dtypes", ", ".join(f"{k}:{v}" for k, v in
                              df.drop(columns='date').dtypes.astype(str).value_counts().items())]],
    ))
    L.append("")

    # ----- 2. Panel structure & series key ------------------------------ #
    n_loc = df[loc].nunique()
    n_ctry = df[ctry].nunique()
    pairs = df[[loc, ctry]].drop_duplicates()
    n_pairs = len(pairs)
    # Homonyms: a location_name appearing under >1 country.
    countries_per_loc = df.groupby(loc)[ctry].nunique()
    homonyms = countries_per_loc[countries_per_loc > 1]
    # Multiple coordinates for the same (location_name, country).
    coords = df.groupby([loc, ctry])[["latitude", "longitude"]].nunique()
    multi_coord = coords[(coords["latitude"] > 1) | (coords["longitude"] > 1)]

    # Days per city (distinct calendar days) and coverage vs each city's own span.
    g = df.groupby([loc, ctry])
    per_city = pd.DataFrame({
        "rows": g.size(),
        "days": g["date"].nunique(),
        "first": g["date"].min(),
        "last": g["date"].max(),
    })
    per_city["own_span"] = (per_city["last"] - per_city["first"]).dt.days + 1
    per_city["coverage"] = per_city["days"] / per_city["own_span"]
    per_city["rows_per_day"] = per_city["rows"] / per_city["days"]

    m.update(n_locations=int(n_loc), n_countries=int(n_ctry), n_series=int(n_pairs),
             n_homonyms=int(len(homonyms)), n_multicoord=int(len(multi_coord)))

    L.append("## 2. Panel structure & series key\n")
    L.append(md_table(
        ["Metric", "Value"],
        [[f"Distinct `{loc}`", n_loc],
         [f"Distinct `{ctry}`", n_ctry],
         [f"Distinct (`{loc}`, `{ctry}`) series", n_pairs],
         [f"`{loc}` used in >1 country (homonyms)", len(homonyms)],
         ["(loc, country) with >1 coordinate", len(multi_coord)]],
    ))
    L.append("")
    L.append("**Days per series** (distinct calendar days per (location, country)):\n")
    desc = per_city["days"].describe()
    L.append(md_table(
        ["stat", "days"],
        [[k, fmt(desc[k], 0 if k == "count" else 1)] for k in
         ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]],
    ))
    L.append("")
    cov = per_city["coverage"]
    L.append("**Temporal coverage** (distinct days / own calendar span):\n")
    L.append(md_table(
        ["Metric", "Value"],
        [["mean coverage", f"{cov.mean():.3f}"],
         ["median coverage", f"{cov.median():.3f}"],
         ["min coverage", f"{cov.min():.3f}"],
         ["series with coverage < 0.90", int((cov < 0.90).sum())],
         ["series with coverage < 0.50", int((cov < 0.50).sum())],
         ["mean rows per present day", f"{per_city['rows_per_day'].mean():.3f}"],
         ["series with >1 row/day (has intra-day dups)",
          int((per_city["rows_per_day"] > 1.0001).sum())]],
    ))
    L.append("")
    if len(homonyms):
        ex = ", ".join(f"{name} ({int(c)})" for name, c in homonyms.head(12).items())
        L.append(f"> **Homonyms** (name → #countries): {ex}"
                 + (" …" if len(homonyms) > 12 else "") + "\n")
    if len(multi_coord):
        ex = ", ".join(f"{i[0]}/{i[1]}" for i in multi_coord.head(8).index)
        L.append(f"> **Multi-coordinate series** (coords change over time): {len(multi_coord)} "
                 f"e.g. {ex}" + (" …" if len(multi_coord) > 8 else "") + "\n")

    # ----- 3. Duplicates ------------------------------------------------ #
    exact = int(df.drop(columns="date").duplicated().sum())
    dup_loc_date = int(df.duplicated(subset=[loc, "date"]).sum())
    dup_key_date = int(df.duplicated(subset=[loc, ctry, "date"]).sum())
    m.update(dup_exact=exact, dup_loc_date=dup_loc_date, dup_key_date=dup_key_date)

    L.append("## 3. Duplicates\n")
    L.append(md_table(
        ["Check", "Count"],
        [["Fully identical rows", exact],
         [f"Duplicate (`{loc}`, date)", dup_loc_date],
         [f"Duplicate (`{loc}`, `{ctry}`, date)", dup_key_date]],
    ))
    L.append(f"\n> Policy (Phase 2): keep the **last** record per "
             f"(`{loc}`, `{ctry}`, date). {dup_key_date:,} rows would be dropped "
             f"({100*dup_key_date/n_rows:.1f}% of the panel).\n")

    # ----- 4. Missingness ---------------------------------------------- #
    na = df.drop(columns="date").isna().mean().sort_values(ascending=False)
    n_missing_cols = int((na > 0).sum())
    m["explicit_missing_cols"] = n_missing_cols

    L.append("## 4. Missingness\n")
    L.append(f"**Explicit NaNs:** {n_missing_cols} of {n_cols} columns contain any NaN.\n")
    if n_missing_cols:
        L.append(md_table(["Column", "% NaN"],
                          [[c, f"{100*v:.2f}%"] for c, v in na.head(15).items() if v > 0]))
    else:
        L.append("_No explicit missing values in any column._")
    L.append("")
    # Implicit missingness = gaps in the daily series (missing calendar days).
    total_expected = int(per_city["own_span"].sum())
    total_present = int(per_city["days"].sum())
    implicit_gap = total_expected - total_present
    m["implicit_gap_days"] = implicit_gap
    L.append("**Implicit missingness (calendar gaps):** even with 0 explicit NaNs, "
             "days can be absent from a city's series.\n")
    L.append(md_table(
        ["Metric", "Value"],
        [["Expected day-slots (sum of own spans)", f"{total_expected:,}"],
         ["Present day-slots (distinct days)", f"{total_present:,}"],
         ["Missing day-slots (gaps)", f"{implicit_gap:,} ({100*implicit_gap/total_expected:.1f}%)"]],
    ))
    L.append("\n> These gaps matter for lags/rolling (Phase 3). Decision to make: "
             "reindex each series to a continuous daily frequency, or accept gaps.\n")

    # ----- 5. Redundant columns (unit pairs) --------------------------- #
    L.append("## 5. Redundant columns (unit conversions)\n")
    rows = []
    for a, b in UNIT_PAIRS:
        if a in df.columns and b in df.columns:
            corr = df[a].corr(df[b])
            ratio = (df[b] / df[a]).replace([float("inf"), float("-inf")], pd.NA).median()
            rows.append([f"`{a}` vs `{b}`", f"{corr:.6f}", f"{ratio:.4f}" if pd.notna(ratio) else "n/a"])
    L.append(md_table(["Pair (keep / drop)", "Pearson r", "median ratio drop/keep"], rows))
    L.append(f"\n> Confirmed redundant → drop imperial columns + `last_updated_epoch` "
             f"in Phase 2 (keep metric system).\n")

    # ----- 6. Physical-range sanity ------------------------------------ #
    L.append("## 6. Physical-range sanity\n")
    desc = df[DESCRIBE_COLS].describe().T
    rows = []
    for c in DESCRIBE_COLS:
        r = desc.loc[c]
        rows.append([f"`{c}`", fmt(r["min"]), fmt(r["25%"]), fmt(r["50%"]),
                     fmt(r["mean"]), fmt(r["75%"]), fmt(r["max"])])
    L.append(md_table(["Variable", "min", "25%", "median", "mean", "75%", "max"], rows))
    L.append("")
    L.append("**Impossible / suspicious values:**\n")
    checks = []
    for label, fn in PHYSICAL_CHECKS.items():
        try:
            cnt = int(fn(df).sum())
        except KeyError:
            continue
        checks.append([label, cnt, f"{100*cnt/n_rows:.2f}%"])
    m["physical_violations"] = {c[0]: c[1] for c in checks}
    L.append(md_table(["Check", "Rows", "% of panel"], checks))
    L.append("\n> Policy (Phase 2): **clip** to physical limits, and flag statistical "
             "outliers on the target with a MAD/IQR rule — never delete real extremes.\n")

    # ----- 7. Target quick look ---------------------------------------- #
    t = df[target]
    L.append("## 7. Target quick look — `temperature_celsius`\n")
    L.append(md_table(
        ["Metric", "Value"],
        [["min / max", f"{t.min():.1f} / {t.max():.1f} °C"],
         ["mean / std", f"{t.mean():.2f} / {t.std():.2f} °C"],
         ["skew", f"{t.skew():.3f}"]],
    ))
    L.append("")

    return "\n".join(L), m


def main() -> None:
    cfg = load_config()
    csv_path = resolve(cfg["data"]["raw_csv"])
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Loaded shape={df.shape}")

    report, metrics = build_report(df, cfg)

    out_path = resolve(cfg["paths"]["audit_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print("\n===== KEY METRICS =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
