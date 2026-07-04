# Cleaning Report — Phase 2

Transforms the raw CSV into `data/processed/clean.parquet`.

## Row accounting

- Raw rows: **151,047**
- Intra-day duplicates removed (keep last): **283**
- Rows after reindex to daily frequency: **165,140**
- Imputed (gap-filled) rows: **14,376** (8.7%)
- Outliers flagged (kept, `is_outlier`): **531**

## Series

- Distinct series after consolidation: **265**
- Name-variant clusters merged: **12**
- Series with >= 90 days (used in global modeling): **222**

## Physical clipping (values clamped, not deleted)

| Column | Values clipped |
| --- | --- |
| `pressure_mb` | 2 |
| `air_quality_PM10` | 2 |
| `temperature_celsius` | 1 |
| `feels_like_celsius` | 1 |
| `wind_kph` | 1 |
| `gust_kph` | 1 |
| `air_quality_Carbon_Monoxide` | 1 |
| `air_quality_Sulphur_dioxide` | 1 |

## Consolidated name variants (same city, renamed over time)

| Country | Canonical | Merged variants (days) |
| --- | --- | --- |
| Belgium | **'S Gravenjansdijk** | 'S Gravenjansdijk (431d), 'S Gravenjansdyk (12d) |
| Cambodia | **Phnom Penh** | Phnom Penh (772d), Phnum Penh (2d) |
| China | **Beijing** | Beijing (771d), Beijing Shi (1d) |
| Costa Rica | **San Ignacio** | San Ignacio (439d), San Jose (284d), San Andres (26d), San Juan (15d) |
| Ethiopia | **Addis Ababa** | Addis Ababa (772d), Addis Abeba (2d) |
| Jamaica | **Port Royal** | Port Royal (420d), Kingston (300d), Norman Gardens (35d), Bournemouth Gardens (11d), Newport East (6d) |
| Kuwait | **Kuwait City** | Kuwait City (773d), Kuwait (1d) |
| Myanmar | **Yangon** | Yangon (424d), Rangoon (307d) |
| Palau | **Airai** | Airai (371d), Koror (338d), Aakip (22d), Adkip (20d), Meyungs (14d), Achelap (7d) |
| Philippines | **Manila** | Manila (773d), Kiyabo (1d) |
| San Marino | **San Marino** | San Marino (771d), City Of San Marino (1d) |
| Tonga | **Nuku`Aloia** | Nuku`Aloia (761d), Nuku'alofa (2d) |
