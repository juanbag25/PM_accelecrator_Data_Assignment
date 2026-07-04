"""Phase 3 - Feature engineering.

Builds the model-ready feature table from ``data/processed/clean.parquet`` and
writes ``data/processed/features.parquet``.

Golden rules honored here:
  * All lags & rolling windows are computed PER SERIES (location_name, country),
    on a date-sorted frame.
  * Rolling features carry ``shift(1)`` so the current day never enters its own
    window (no lookahead).
  * Nothing is fit on data here (encodings are deterministic), so there is no
    train/val/test leakage to worry about at this stage.

Run with:
    conda run -n weather-forecasting python -m src.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pycountry_convert as pcc

from .config import load_config, resolve

# Meteorological seasons for the Northern Hemisphere (month -> season).
_NORTH_SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
                 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
                 10: "autumn", 11: "autumn"}
_SWAP = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}

_MOON_ORDER = {"New Moon": 0, "Waxing Crescent": 1, "First Quarter": 2,
               "Waxing Gibbous": 3, "Full Moon": 4, "Waning Gibbous": 5,
               "Last Quarter": 6, "Waning Crescent": 7}

_CONTINENT = {"AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
              "SA": "South America", "OC": "Oceania", "AN": "Antarctica"}

# Country names pycountry_convert cannot resolve: legitimate aliases + a handful
# of source-side glitches where the country leaked in another language
# (e.g. '火鸡' = "turkey" the bird, an autotranslation of Turkey).
_CONTINENT_OVERRIDE = {
    "Cote d'Ivoire": "Africa", "Democratic Republic of Congo": "Africa",
    "Fiji Islands": "Oceania", "Kosovo": "Europe", "Kyrghyzstan": "Asia",
    "Seychelles Islands": "Africa", "Timor-Leste": "Asia", "Vatican City": "Europe",
    "Belgica": "Europe", "Belgica ": "Europe", "Belgique": "Europe",
    "Belgica": "Europe", "Estonie": "Europe", "Inde": "Asia", "Jemen": "Asia",
    "Komoren": "Africa", "Letonia": "Europe", "Malasia": "Asia", "Marrocos": "Africa",
    "Mexique": "North America", "Polonia": "Europe", "Polonia ": "Europe",
    "Saint-Vincent-et-les-Grenadines": "North America", "Saudi Arabien": "Asia",
    "Sudkorea": "Asia", "Turkmenistan": "Asia", "USA United States of America": "North America",
    # Non-ASCII source glitches
    "Bélgica": "Europe", "Malásia": "Asia", "Polônia": "Europe", "Südkorea": "Asia",
    "Гватемала": "North America", "Польша": "Europe", "Турция": "Asia",
    "كولومبيا": "South America", "火鸡": "Asia",
}

# Categorical feature columns kept as pandas 'category' dtype (LightGBM-native).
_CATEGORICAL = ["hemisphere", "climate_zone", "season", "continent"]

# Columns consumed by feature construction and dropped from the final table.
_DROP_AFTER = ["condition_text", "wind_direction", "sunrise", "sunset",
               "moonrise", "moonset", "moon_phase", "timezone"]


# --------------------------------------------------------------------------- #
def _country_to_continent(name: str) -> str:
    try:
        code = pcc.country_name_to_country_alpha2(name)
        cont = _CONTINENT.get(pcc.country_alpha2_to_continent_code(code))
        if cont:
            return cont
    except Exception:  # pycountry raises bare KeyError-like errors for unknown names
        pass
    return _CONTINENT_OVERRIDE.get(name, "Unknown")


def _condition_family(text: str) -> str:
    t = str(text).lower()
    if "thunder" in t or "storm" in t:
        return "storm"
    if any(w in t for w in ("snow", "sleet", "blizzard", "ice")):
        return "snow"
    if any(w in t for w in ("rain", "drizzle", "shower")):
        return "rain"
    if any(w in t for w in ("fog", "mist", "haze")):
        return "fog"
    if "cloud" in t or "overcast" in t:
        return "cloudy"
    if "clear" in t or "sunny" in t:
        return "clear"
    return "other"


def _to_hours(series: pd.Series) -> pd.Series:
    """Parse '04:50 AM' style times into fractional hours (NaN if unparseable)."""
    t = pd.to_datetime(series, format="%I:%M %p", errors="coerce")
    return t.dt.hour + t.dt.minute / 60.0


# --------------------------------------------------------------------------- #
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["dayofyear"] = d.dayofyear
    df["weekofyear"] = d.isocalendar().week.astype("int32")  # pandas 3.0: weekofyear removed
    df["quarter"] = d.quarter
    df["dayofweek"] = d.dayofweek
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
    # Cyclical encodings so Dec and Jan sit next to each other.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_geo(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ctry = cfg["time"]["disambiguation_key"]
    df["hemisphere"] = np.where(df["latitude"] >= 0, "N", "S")
    df["abs_lat"] = df["latitude"].abs()
    df["climate_zone"] = np.select(
        [df["abs_lat"] < 23.5, df["abs_lat"] < 66.5],
        ["tropical", "temperate"], default="polar")
    # continent: map the ~200 unique country names once, then broadcast.
    cmap = {c: _country_to_continent(c) for c in df[ctry].unique()}
    df["continent"] = df[ctry].map(cmap)
    # Season depends on hemisphere (do NOT map month->season blindly).
    base = df["month"].map(_NORTH_SEASON)
    df["season"] = np.where(df["hemisphere"].to_numpy() == "S",
                            base.map(_SWAP).to_numpy(), base.to_numpy())
    return df


def add_lags_rolling(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    keys = [cfg["time"]["series_key"], cfg["time"]["disambiguation_key"]]
    target = cfg["target"]["primary"]
    lags = cfg["features"]["lags"]
    windows = cfg["features"]["rolling_windows"]

    df = df.sort_values(keys + ["date"]).reset_index(drop=True)  # re-confirm order
    gt = df.groupby(keys, sort=False)[target]
    for k in lags:
        df[f"temp_lag_{k}"] = gt.shift(k)
    for w in windows:
        # shift(1) BEFORE rolling => today never enters its own window.
        df[f"temp_rollmean_{w}"] = gt.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
        df[f"temp_rollstd_{w}"] = gt.transform(lambda s, w=w: s.shift(1).rolling(w).std())
    for col in ["humidity", "pressure_mb"]:
        gc = df.groupby(keys, sort=False)[col]
        for w in windows:
            df[f"{col}_rollmean_{w}"] = gc.transform(lambda s, w=w: s.shift(1).rolling(w).mean())
    # Differences.
    df["temp_diff_1"] = df[target] - df["temp_lag_1"]
    df["temp_diff_7"] = df[target] - df["temp_lag_7"]
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["temp_minus_feels"] = df["temperature_celsius"] - df["feels_like_celsius"]
    day_len = _to_hours(df["sunset"]) - _to_hours(df["sunrise"])
    df["day_length_h"] = day_len.where(day_len >= 0, day_len + 24)  # handle wrap
    df["pm_ratio"] = df["air_quality_PM2.5"] / (df["air_quality_PM10"] + 1e-6)
    # Wind direction as cyclical (better than one-hot for a circular quantity).
    df["wind_sin"] = np.sin(np.deg2rad(df["wind_degree"]))
    df["wind_cos"] = np.cos(np.deg2rad(df["wind_degree"]))
    # Moon phase as an ordinal 0..7 over the lunar cycle.
    df["moon_phase_ord"] = df["moon_phase"].map(_MOON_ORDER).fillna(-1).astype("int8")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # condition_text -> weather family -> one-hot (per the spec).
    fam = df["condition_text"].map(_condition_family)
    dummies = pd.get_dummies(fam, prefix="cond").astype("int8")
    df = pd.concat([df, dummies], axis=1)
    # Low-cardinality descriptors kept as category dtype (LightGBM-native).
    for c in _CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df = add_calendar(df)
    df = add_geo(df, cfg)
    df = add_lags_rolling(df, cfg)
    df = add_derived(df)
    df = encode_categoricals(df)
    df = df.drop(columns=[c for c in _DROP_AFTER if c in df.columns])
    return df


def main() -> None:
    cfg = load_config()
    clean_path = resolve(cfg["data"]["clean_parquet"])
    print(f"Loading {clean_path} ...")
    df = pd.read_parquet(clean_path)
    print(f"Clean shape={df.shape}")

    feats = build_features(df, cfg)
    out_path = resolve(cfg["data"]["features_parquet"])
    feats.to_parquet(out_path, index=False)

    new_cols = [c for c in feats.columns if c not in df.columns]
    lag_roll = [c for c in feats.columns if c.startswith(("temp_lag", "temp_roll",
                "humidity_roll", "pressure_mb_roll", "temp_diff"))]
    print(f"\nFeatures shape={feats.shape}  (+{len(new_cols)} new columns)")
    print(f"Continents mapped: {dict(feats['continent'].value_counts())}")
    print(f"Condition one-hot: {[c for c in feats.columns if c.startswith('cond_')]}")
    print(f"Lag/rolling/diff cols: {len(lag_roll)}")
    print("NaN in lag/rolling cols (expected at each series' start):")
    na = feats[lag_roll].isna().sum()
    print(f"  total NaN={int(na.sum())}, max per col={int(na.max())} "
          f"(<= (max_window)*n_series is normal)")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
