# Data Audit — Global Weather Repository

Phase 1 output. Profiles the **raw** panel before any cleaning (no rows/columns are modified here).

## 1. Overview (initial snapshot)

| Metric | Value |
| --- | --- |
| Rows | 151,047 |
| Columns | 41 |
| Memory (deep) | 65.4 MB |
| Date range (local) | 2024-05-16 → 2026-07-03 |
| Calendar span | 779 days (~2.1 years) |
| Dtypes | float64:23, str:11, int64:7 |

## 2. Panel structure & series key

| Metric | Value |
| --- | --- |
| Distinct `location_name` | 268 |
| Distinct `country` | 211 |
| Distinct (`location_name`, `country`) series | 286 |
| `location_name` used in >1 country (homonyms) | 15 |
| (loc, country) with >1 coordinate | 203 |

**Days per series** (distinct calendar days per (location, country)):

| stat | days |
| --- | --- |
| count | 286 |
| mean | 527.1 |
| std | 342.1 |
| min | 1.0 |
| 25% | 29.0 |
| 50% | 772.0 |
| 75% | 774.0 |
| max | 776.0 |

**Temporal coverage** (distinct days / own calendar span):

| Metric | Value |
| --- | --- |
| mean coverage | 0.878 |
| median coverage | 0.992 |
| min coverage | 0.003 |
| series with coverage < 0.90 | 46 |
| series with coverage < 0.50 | 37 |
| mean rows per present day | 1.005 |
| series with >1 row/day (has intra-day dups) | 198 |

> **Homonyms** (name → #countries): Beirut (2), Bern (2), Bogot (2), Grenada (2), Kingstown (2), Lom (3), Mbabane (2), Moroni (4), New Delhi (2), Palau (2), Riga (2), Sanaa (2) …

> **Multi-coordinate series** (coords change over time): 203 e.g. 'S Gravenjansdijk/Belgium, Abu Dhabi/United Arab Emirates, Abuja/Nigeria, Accra/Ghana, Addis Ababa/Ethiopia, Addis Abeba/Ethiopia, Airai/Palau, Algiers/Algeria …

## 3. Duplicates

| Check | Count |
| --- | --- |
| Fully identical rows | 0 |
| Duplicate (`location_name`, date) | 283 |
| Duplicate (`location_name`, `country`, date) | 283 |

> Policy (Phase 2): keep the **last** record per (`location_name`, `country`, date). 283 rows would be dropped (0.2% of the panel).

## 4. Missingness

**Explicit NaNs:** 0 of 41 columns contain any NaN.

_No explicit missing values in any column._

**Implicit missingness (calendar gaps):** even with 0 explicit NaNs, days can be absent from a city's series.

| Metric | Value |
| --- | --- |
| Expected day-slots (sum of own spans) | 168,587 |
| Present day-slots (distinct days) | 150,764 |
| Missing day-slots (gaps) | 17,823 (10.6%) |

> These gaps matter for lags/rolling (Phase 3). Decision to make: reindex each series to a continuous daily frequency, or accept gaps.

## 5. Redundant columns (unit conversions)

| Pair (keep / drop) | Pearson r | median ratio drop/keep |
| --- | --- | --- |
| `temperature_celsius` vs `temperature_fahrenheit` | 0.999997 | 3.1240 |
| `wind_kph` vs `wind_mph` | 0.999988 | 0.6211 |
| `pressure_mb` vs `pressure_in` | 0.999795 | 0.0295 |
| `precip_mm` vs `precip_in` | 0.997925 | 0.0000 |
| `visibility_km` vs `visibility_miles` | 0.992500 | 0.6000 |
| `gust_kph` vs `gust_mph` | 0.999992 | 0.6210 |

> Confirmed redundant → drop imperial columns + `last_updated_epoch` in Phase 2 (keep metric system).

## 6. Physical-range sanity

| Variable | min | 25% | median | mean | 75% | max |
| --- | --- | --- | --- | --- | --- | --- |
| `temperature_celsius` | -29.80 | 16.00 | 23.70 | 21.33 | 27.90 | 79.30 |
| `feels_like_celsius` | -36.70 | 15.90 | 25.00 | 22.13 | 29.80 | 81.30 |
| `humidity` | 2.00 | 51.00 | 72.00 | 66.91 | 86.00 | 100.00 |
| `cloud` | 0.00 | 0.00 | 27.00 | 39.56 | 75.00 | 100.00 |
| `pressure_mb` | 947.00 | 1,010.00 | 1,014.00 | 1,014.06 | 1,018.00 | 3,006.00 |
| `wind_kph` | 3.60 | 6.10 | 10.80 | 12.79 | 17.60 | 2,963.20 |
| `gust_kph` | 3.60 | 10.00 | 15.10 | 18.10 | 24.00 | 2,970.40 |
| `precip_mm` | 0.00 | 0.00 | 0.00 | 0.13 | 0.02 | 42.24 |
| `visibility_km` | 0.00 | 10.00 | 10.00 | 9.52 | 10.00 | 32.00 |
| `uv_index` | 0.00 | 0.10 | 1.70 | 3.21 | 6.00 | 16.30 |
| `air_quality_PM2.5` | 0.17 | 6.90 | 13.65 | 23.43 | 26.82 | 1,614.10 |
| `air_quality_PM10` | -1,848.15 | 9.70 | 19.24 | 47.03 | 40.15 | 6,037.29 |

**Impossible / suspicious values:**

| Check | Rows | % of panel |
| --- | --- | --- |
| temperature_celsius < -90 or > 60 | 1 | 0.00% |
| humidity outside [0, 100] | 0 | 0.00% |
| cloud outside [0, 100] | 0 | 0.00% |
| moon_illumination outside [0, 100] | 0 | 0.00% |
| pressure_mb <= 0 | 0 | 0.00% |
| pressure_mb outside [800, 1100] | 2 | 0.00% |
| precip_mm < 0 | 0 | 0.00% |
| wind_kph < 0 | 0 | 0.00% |
| gust_kph < 0 | 0 | 0.00% |
| visibility_km < 0 | 0 | 0.00% |
| uv_index < 0 | 0 | 0.00% |

> Policy (Phase 2): **clip** to physical limits, and flag statistical outliers on the target with a MAD/IQR rule — never delete real extremes.

## 7. Target quick look — `temperature_celsius`

| Metric | Value |
| --- | --- |
| min / max | -29.8 / 79.3 °C |
| mean / std | 21.33 / 9.52 °C |
| skew | -0.829 |
