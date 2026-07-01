# Runbook del Pipeline — Weather Trend Forecasting

> **Qué es esto.** Una guía paso a paso del pipeline completo, diseñada para dos lectores a la vez:
> - **Para un humano:** cada fase arranca con un objetivo en lenguaje claro y termina con un "qué tenés que haber logrado".
> - **Para una IA que la ejecute:** cada paso tiene una *acción concreta* (con las funciones y librerías exactas), una *regla de decisión* cuando hay que elegir, el *archivo de salida* que produce y un *check* para verificar que funcionó.
>
> **Cómo leerla.** Las fases van en orden y cada una depende de la anterior. Si sos una IA ejecutando: respetá el orden, producí el artefacto de salida de cada fase antes de pasar a la siguiente, y nunca rompas las **Reglas de Oro** de abajo.

---

## TL;DR — Camino recomendado (lo que sí o sí hay que hacer)

1. **Limpiar** el dataset entendiendo que es un **panel** (~200 ciudades, una serie diaria por ciudad desde ago-2023).
2. **Crear features** temporales + lags + rolling + geográficas (la parte que más mueve la aguja).
3. **Modelo primario: LightGBM global** (un solo modelo para todas las ciudades, con lags y variables exógenas). Es el que mejor encaja y el que probablemente gane.
4. **Comparar** contra: baselines (obligatorio), clásicos por ciudad (SARIMA/Prophet) y un **modelo fundacional zero-shot (Chronos)** como diferenciador.
5. **Ensamblar** los mejores (stacking) y evaluar **siempre con split temporal y MASE**.
6. **Análisis avanzados** (anomalías, clima con Mann-Kendall, aire vs clima, espacial) + empaquetar el repo.

Todo lo demás de esta guía es el detalle de cómo hacer cada uno de esos puntos bien.

---

## Convenciones globales (la IA las usa en todo el pipeline)

| Concepto | Valor |
|---|---|
| Target principal | `temperature_celsius` |
| Targets secundarios (opcional) | `air_quality_PM2.5`, `precip_mm` |
| Índice temporal | `last_updated` (parseado a `datetime`) → columna `date` (a nivel día) |
| Clave de serie | `location_name` (+ `country` para desambiguar ciudades homónimas) |
| Horizonte de pronóstico | `H = 7` días (probar también `H = 14`) |
| Semilla aleatoria | `SEED = 42` |
| Formato intermedio | `.parquet` (rápido y conserva tipos) |
| Carpetas de salida | `data/processed/`, `reports/figures/`, `reports/metrics/`, `models/` |

## Reglas de Oro (invariantes — nunca violarlas)

1. **Nunca split aleatorio; siempre por fecha, en tres bloques.** La división es **train (pasado) → validation (medio) → test (futuro reciente)**, en orden cronológico. El **test se toca una sola vez, al final, y jamás se usa para decidir** (ni tuning, ni features, ni early stopping, ni pesos del ensemble): todas esas decisiones van contra validation.
2. **Lags y rolling SIEMPRE por ciudad.** Hacé `sort` por `[location_name, date]` y usá `groupby('location_name')` antes de cualquier lag/rolling. Jamás mezclar datos entre ciudades.
3. **No mirar el futuro.** Cualquier feature en la fila de fecha `t` solo puede usar información de fecha `≤ t`. (Por eso los rolling llevan un `shift(1)`.)
4. **Fit solo con train.** Scalers, encoders, imputers y selección de features se *ajustan* con train y se *aplican* a validation/test. Nunca al revés.
5. **Siempre comparar contra baseline.** Ningún resultado vale sin el naive al lado. La métrica de comparación entre modelos es **MASE**.
6. **Guardar cada artefacto** (datos limpios, features, modelos, métricas, figuras). Reproducibilidad ante todo.

---

# FASE 0 — Setup del proyecto

**Objetivo (humano):** dejar el proyecto listo para trabajar: estructura de carpetas, dependencias y datos cargados.

**Paso 0.1 — Estructura de carpetas**
- *Acción:* crear el árbol `data/{raw,processed}/`, `notebooks/`, `src/`, `reports/{figures,metrics}/`, `models/`, y los archivos `requirements.txt`, `config.yaml`, `README.md`, `.gitignore` (con `data/raw/` adentro para no subir el CSV pesado).
- *Output:* esqueleto del repo.
- *Check:* las carpetas existen y `.gitignore` ignora `data/raw/`.

**Paso 0.2 — requirements.txt**
- *Acción:* listar dependencias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `statsmodels`, `pmdarima`, `prophet`, `lightgbm`, `xgboost`, `shap`, `pymannkendall`, `pycountry-convert`, `geopandas`, `folium`, y opcionales `neuralforecast`/`darts`, `autogluon.timeseries`, `optuna`, `streamlit`.
- *Output:* `requirements.txt`.
- *Check:* `pip install -r requirements.txt` corre sin errores en un entorno limpio.

**Paso 0.3 — config.yaml**
- *Acción:* volcar las Convenciones Globales (target, horizonte, semilla, rutas) en `config.yaml` para no hardcodear nada.
- *Output:* `config.yaml`.

**Paso 0.4 — Carga inicial**
- *Acción:* `df = pd.read_csv('data/raw/GlobalWeatherRepository.csv')`. Registrar `df.shape`, `df.dtypes`, `df.head()`, uso de memoria.
- *Output:* notebook `01_setup.ipynb` con el snapshot inicial.
- *Check:* el CSV carga y tiene ~40+ columnas.

---

# FASE 1 — Auditoría y validación de datos

**Objetivo (humano):** entender qué tenés entre manos *antes* de tocar nada: cuántas ciudades, qué rango de fechas, qué está roto.

**Paso 1.1 — Perfilado de la estructura de panel**
- *Acción:* calcular `df['location_name'].nunique()` (n ciudades), `df['country'].nunique()`, rango de fechas (`last_updated.min()/max()`), y días por ciudad (`df.groupby('location_name').size().describe()`).
- *Output:* sección en `reports/data_audit.md`.
- *Check:* confirmás que hay ~195-200 ciudades, cada una con cientos de días → es un panel, no una serie sola.

**Paso 1.2 — Duplicados (ciudad, fecha)**
- *Acción:* `dups = df.duplicated(subset=['location_name','last_updated']).sum()`.
- *Regla de decisión:* si hay duplicados → política = **quedarse con el último registro** por `(location_name, date)`.
- *Output:* número de duplicados registrado en el audit.

**Paso 1.3 — Reporte de faltantes**
- *Acción:* `df.isna().mean().sort_values(ascending=False)`.
- *Output:* tabla de % de NaN por columna en el audit.
- *Check:* identificás qué columnas tienen muchos faltantes para tratarlas distinto.

**Paso 1.4 — Confirmar columnas redundantes**
- *Acción:* verificar correlación ≈ 1.0 entre pares de unidades (Celsius/Fahrenheit, mph/kph, mb/in, mm/in, km/miles).
- *Output:* lista confirmada de columnas a dropear (se usa en Fase 2).

**Paso 1.5 — Sanity de rangos físicos**
- *Acción:* `df.describe()` sobre numéricas; chequear mínimos/máximos imposibles (humedad fuera de 0-100, temperaturas absurdas, presiones de 0).
- *Output:* lista de límites físicos por variable (se usa en Fase 2).
- *Check:* tenés mapeado qué valores son físicamente imposibles vs extremos reales.

> ✅ **Fin de Fase 1:** existe `reports/data_audit.md` con n ciudades, rango de fechas, % de faltantes, columnas redundantes y rangos sospechosos.

---

# FASE 2 — Limpieza y preprocesamiento

**Objetivo (humano):** dejar un dataset limpio, ordenado y sin información duplicada, conservando los extremos reales (no borrar olas de calor).

**Paso 2.1 — Parsear fechas y ordenar**
- *Acción:* `df['last_updated'] = pd.to_datetime(df['last_updated'])`; crear `df['date'] = df['last_updated'].dt.normalize()`; `df = df.sort_values(['location_name','last_updated'])`.
- *Check:* `df['last_updated'].dtype` es datetime y el df queda ordenado por ciudad y fecha.

**Paso 2.2 — Dropear columnas redundantes**
- *Acción:* eliminar las imperiales/duplicadas, conservando el sistema métrico:
  `['temperature_fahrenheit','wind_mph','pressure_in','precip_in','visibility_miles','gust_mph','feels_like_fahrenheit']`. También `last_updated_epoch` (usás `last_updated`).
- *Output:* df con ~33 columnas.

**Paso 2.3 — Resolver duplicados**
- *Acción:* `df = df.drop_duplicates(subset=['location_name','date'], keep='last')`.
- *Check:* `df.duplicated(subset=['location_name','date']).sum() == 0`.

**Paso 2.4 — Imputar faltantes**
- *Regla de decisión por tipo:*
  - **Numéricas:** interpolación temporal *por ciudad* →
    `df[col] = df.groupby('location_name')[col].transform(lambda s: s.interpolate(method='linear', limit_direction='both'))`. Fallback `ffill`/`bfill` si queda algún hueco en los bordes.
  - **Categóricas** (`condition_text`, `wind_direction`, `moon_phase`): rellenar con la moda de la ciudad o `'Unknown'`.
- *Check:* `df.isna().sum().sum() == 0` en las columnas que se van a usar (salvo lags/rolling, que se crean después).

**Paso 2.5 — Outliers (recortar + flaggear, NO borrar)**
- *Acción:*
  1. Aplicar **límites físicos** con `clip` (ej.: `humidity` a [0,100]).
  2. Crear flag estadístico por ciudad: `is_outlier` usando z-score modificado (MAD) o IQR sobre el target, *sin eliminar* las filas.
- *Output:* `data/processed/clean.parquet` con la columna `is_outlier`.
- *Check:* la cantidad de filas se mantiene (no se borraron extremos); existe la columna flag.

> ✅ **Fin de Fase 2:** `data/processed/clean.parquet` limpio, ordenado, sin duplicados, sin NaN en las columnas base, con extremos flaggeados pero conservados.

---

# FASE 3 — Feature engineering

**Objetivo (humano):** crear las "pistas" que los modelos van a usar. Esta fase es la que más diferencia los resultados. Todo lag/rolling va **por ciudad**.

**Paso 3.1 — Garantizar el orden**
- *Acción:* `df = df.sort_values(['location_name','date'])` (re-confirmar antes de crear features temporales).

**Paso 3.2 — Features de calendario**
- *Acción:* derivar de `date`: `year, month, day, dayofyear, weekofyear, quarter, dayofweek, is_weekend, days_since_start`.

**Paso 3.3 — Encoding cíclico**
- *Acción:*
  ```python
  df['month_sin'] = np.sin(2*np.pi*df['month']/12);  df['month_cos'] = np.cos(2*np.pi*df['month']/12)
  df['doy_sin']   = np.sin(2*np.pi*df['dayofyear']/365.25); df['doy_cos'] = np.cos(2*np.pi*df['dayofyear']/365.25)
  ```
- *Por qué:* así diciembre y enero quedan "pegados" para el modelo.

**Paso 3.4 — Features geográficas**
- *Acción:*
  - `hemisphere = np.where(df['latitude']>=0, 'N', 'S')`
  - `abs_lat = df['latitude'].abs()`
  - `climate_zone`: `abs_lat<23.5 →'tropical'`, `<66.5 →'templada'`, else `'polar'`
  - `continent`: con `pycountry_convert` a partir de `country` (envolver en try/except para países que no mapean).

**Paso 3.5 — Estación según hemisferio**
- *Acción:* función `season(month, hemisphere)` que invierte las estaciones para el hemisferio sur. NO mapear mes→estación a secas.

**Paso 3.6 — Lags del target (por ciudad)**
- *Acción:*
  ```python
  for k in [1, 2, 3, 7, 14, 30]:
      df[f'temp_lag_{k}'] = df.groupby('location_name')['temperature_celsius'].shift(k)
  ```

**Paso 3.7 — Rolling (por ciudad, con shift para evitar leakage)**
- *Acción:*
  ```python
  for w in [7, 14, 30]:
      g = df.groupby('location_name')['temperature_celsius']
      df[f'temp_rollmean_{w}'] = g.transform(lambda s: s.shift(1).rolling(w).mean())
      df[f'temp_rollstd_{w}']  = g.transform(lambda s: s.shift(1).rolling(w).std())
  ```
  Repetir media móvil para `humidity` y `pressure_mb`.
- *Regla crítica:* el `shift(1)` antes del `rolling` es lo que impide que el día actual entre en su propio promedio (leakage).

**Paso 3.8 — Diferencias**
- *Acción:* `df['temp_diff_1'] = df['temperature_celsius'] - df['temp_lag_1']`; ídem `temp_diff_7`.

**Paso 3.9 — Features físicas derivadas**
- *Acción:* `temp_minus_feels = temperature_celsius - feels_like_celsius`; `day_length_h` = (hora de `sunset` − hora de `sunrise`); `pm_ratio = air_quality_PM2.5 / (air_quality_PM10 + 1e-6)`.

**Paso 3.10 — Encoding de categóricas**
- *Regla de decisión:*
  - `condition_text`: agrupar en familias (despejado / nublado / lluvia / nieve / tormenta / niebla) y luego **one-hot**.
  - `wind_direction`: mejor usar `sin/cos` de `wind_degree` que one-hot (es cíclica).
  - `moon_phase`: **ordinal** (0→7 según el ciclo).
- *Nota anti-leakage:* si usás target encoding, ajustalo **solo con train**.

**Paso 3.11 — Manejar NaN de lags/rolling**
- *Regla de decisión:* las primeras `N` filas de cada ciudad tendrán NaN en lags/rolling.
  - Si el modelo es **LightGBM/XGBoost** → dejarlos, manejan NaN nativamente (ventaja, no imputes).
  - Si el modelo es **lineal/red neuronal** → dropear esas filas o imputar.
- *Output:* `data/processed/features.parquet`.
- *Check:* las columnas de lag/rolling existen y solo tienen NaN al inicio de cada serie por ciudad.

> ✅ **Fin de Fase 3:** `data/processed/features.parquet` con calendario, cíclicas, geográficas, lags, rolling, diffs, derivadas y categóricas codificadas.

---

# FASE 4 — EDA (Análisis Exploratorio)

**Objetivo (humano):** entender los datos y generar las visualizaciones que pide el enunciado (temperatura y precipitación), más el análisis de serie temporal.

**Paso 4.1 — Univariado**
- *Acción:* `df.describe()`; histogramas del target y variables clave; boxplots por `continent` y por `climate_zone`.

**Paso 4.2 — Precipitación (tratar zero-inflation)**
- *Acción:* histograma de `precip_mm` (notar el exceso de ceros) y de `np.log1p(precip_mm)`.

**Paso 4.3 — Correlaciones**
- *Acción:* heatmap de correlación **Pearson** y **Spearman** sobre las numéricas limpias.
- *Output:* `reports/figures/corr_pearson.png`, `corr_spearman.png`.

**Paso 4.4 — Visualizaciones de temperatura (REQUERIDO)**
- *Acción:* serie temporal de 4-5 ciudades representativas (distintos climas/hemisferios); promedio global mensual; heatmap ciudad×mes; **choropleth** de temperatura media por país (`plotly`/`geopandas`).

**Paso 4.5 — Visualizaciones de precipitación (REQUERIDO)**
- *Acción:* patrón estacional de lluvia; lluvia por región/continente.

**Paso 4.6 — Análisis de serie temporal**
- *Acción:* sobre 1-2 ciudades representativas: descomposición **STL** (`statsmodels.tsa.seasonal.STL`); gráficos **ACF/PACF** (`plot_acf`, `plot_pacf`); tests **ADF** (`adfuller`) y **KPSS** (`kpss`).
- *Output:* figuras + interpretación breve en `reports/eda_notes.md`.

**Paso 4.7 — Espacial**
- *Acción:* scatter/choropleth de temperatura y de AQI sobre el mapa.

> ✅ **Fin de Fase 4:** figuras en `reports/figures/` + `reports/eda_notes.md` con los hallazgos (incluye descomposición y resultado de estacionariedad).

---

# FASE 5 — Selección e importancia de features

**Objetivo (humano):** quedarte con las pistas que de verdad sirven, usando varias técnicas y eligiendo por consenso. El enunciado pide explícitamente "different techniques".

**Paso 5.1 — Quitar multicolinealidad**
- *Acción:* sobre la matriz de correlación, dropear uno de cada par con `|corr| > 0.95`. Opcional: VIF iterativo eliminando hasta que todos los VIF < 10.

**Paso 5.2 — Filter**
- *Acción:* rankear con `mutual_info_regression` y `f_regression` (sklearn).

**Paso 5.3 — Embedded**
- *Acción:* coeficientes de `LassoCV` (sobre features escaladas) + `LGBMRegressor().feature_importances_`.

**Paso 5.4 — Model-agnostic**
- *Acción:* `permutation_importance` sobre un LightGBM de validación + **SHAP** (`shap.TreeExplainer` → `summary_plot`).
- *Output:* `reports/figures/shap_summary.png`.

**Paso 5.5 — Consenso**
- *Acción:* armar una tabla con el ranking de cada método y quedarte con las features que aparecen en el top de **≥2 métodos** (o las top-K por rank promedio).
- *Output:* `reports/selected_features.json` + figuras de importancia.
- *Check:* tenés una lista final de features justificada por varios métodos.

> ✅ **Fin de Fase 5:** `selected_features.json` con la lista definitiva y los gráficos de importancia (SHAP incluido).

---

# FASE 6 — Setup de evaluación

**Objetivo (humano):** dejar definido *cómo* se va a medir todo, antes de entrenar el primer modelo. Esto evita comparaciones tramposas.

**Paso 6.1 — Definir el problema**
- *Acción:* fijar target (`temperature_celsius`), horizonte `H`, y los dos framings: **global** (todas las ciudades juntas) y **per-city** (clásicos sobre ciudades representativas).

**Paso 6.2 — Split temporal en TRES bloques (train / validation / test)**
- *Acción:*
  ```python
  q_train, q_val = df['date'].quantile([0.70, 0.85])
  train = df[df['date'] <  q_train]                           # ~70% más viejo → ENTRENAR
  val   = df[(df['date'] >= q_train) & (df['date'] < q_val)]  # ~15% medio    → DECIDIR
  test  = df[df['date'] >= q_val]                             # ~15% más nuevo → REPORTAR
  ```
- *Regla de decisión (qué se hace con cada bloque):*
  - **train** → entrenar los modelos.
  - **validation** → tuning de hiperparámetros, *early stopping* (LightGBM/redes), decisiones de feature selection y pesos del ensemble/stacking.
  - **test** → se toca UNA sola vez, al final, solo para las métricas reportadas. Nunca para elegir nada.
- *Orden obligatorio:* los tres bloques van cronológicos (train → val → test), nunca mezclados. El test es siempre el futuro más reciente.
- *Check:* `train.date.max() < val.date.min()` y `val.date.max() < test.date.min()`; `val` y `test` cubren cada uno al menos `H` días.

**Paso 6.3 — CV y refit final (opcional, recomendado)**
- *Acción:*
  - **CV (alternativa al bloque fijo de val):** para un tuning más robusto podés reemplazar el bloque fijo de validation por `TimeSeriesSplit(n_splits=5)` sobre `train+val`. El bloque fijo es más simple; la CV usa mejor los datos pero cuesta más cómputo. Elegí una de las dos, no mezcles roles.
  - **Refit final:** una vez fijados hiperparámetros y features con train+val, reentrená el modelo definitivo sobre `train+val` juntos y recién ahí predecí `test` (así aprovechás los datos más recientes, que en clima son los más relevantes).
  - **Casos aparte:** los clásicos per-city (SARIMA/Prophet) se evalúan con backtesting de origen móvil (`darts`/`sktime`), no necesitan un bloque fijo de val. Los modelos fundacionales zero-shot no entrenan → van directo a test.

**Paso 6.4 — Métricas**
- *Acción:* implementar `MAE`, `RMSE`, `MAPE`, `sMAPE`, `R²` y **`MASE`** (= MAE del modelo / MAE del seasonal naive). MASE es la métrica de comparación principal.

**Paso 6.5 — Harness de evaluación**
- *Acción:* función `evaluate(y_true, y_pred) -> dict` que devuelve todas las métricas y guarda a `reports/metrics/{modelo}.json`.
- *Output:* `src/evaluation.py`.
- *Check:* corrés `evaluate` sobre un naive y obtenés un JSON con las 6 métricas.

> ✅ **Fin de Fase 6:** existe `src/evaluation.py`, los tres bloques temporales (train / validation / test) están definidos y MASE funciona.

---

# FASE 7 — Modelos (qué probar y en qué orden)

**Objetivo (humano):** entrenar de lo simple a lo sofisticado, comparando todo contra el baseline. **El modelo primario recomendado es LightGBM global**; el resto se implementa para comparar y cubrir requisitos.

> **Por qué LightGBM global es el primario:** este dataset es un panel tabular con muchas variables exógenas (humedad, presión, etc.) y series no demasiado largas. Ahí los modelos de gradient boosting suelen ganarle a los clásicos univariados y a las redes, son rápidos, toleran NaN y no necesitan escalado. Es la apuesta más segura para el mejor número.

### Tier 0 — Baselines (OBLIGATORIO)
- *Acción:* `naive` (= `temp_lag_1`), `seasonal_naive` (= `temp_lag_7`, o lag de 365 si hay un año completo), `moving_average` (= `temp_rollmean_7`). Evaluar cada uno.
- *Output:* métricas baseline → son el **piso a superar**.

### Tier 1 — Clásicos por ciudad (para 3-5 ciudades representativas)
- *Acción:*
  - **SARIMA:** `pmdarima.auto_arima(serie, seasonal=True, m=7)` (m=7 para estacionalidad semanal; probar también modelado anual).
  - **ETS / Holt-Winters:** `statsmodels` `ExponentialSmoothing`.
  - **Prophet:** con estacionalidad anual + semanal activadas.
- *Para qué:* cubre la consigna "usar `last_updated` para time series" y da las descomposiciones lindas para el informe.

### Tier 2 — ML global (PRIMARIO)
- *Acción:* `LGBMRegressor` (y `XGBRegressor` como alternativa) entrenado sobre `selected_features`, prediciendo el target. Tuning con `optuna` (opcional) o hiperparámetros razonables. Aprovechar que LightGBM maneja NaN nativo.
- *Output:* `models/lgbm_global.pkl` + métricas. **Suele ser el ganador.**

### Tier 3 — Deep Learning global (OPCIONAL, para mostrar amplitud)
- *Acción:* `N-BEATS` y `NHITS` vía `neuralforecast`/`darts`; opcional **TFT** (maneja covariables + multi-serie y es interpretable).
- *Nota:* requiere más cómputo; solo si hay tiempo.

### Tier 4 — Modelos fundacionales (DIFERENCIADOR)
- *Acción:* **Chronos** (Bolt) en modo **zero-shot** vía `autogluon.timeseries` o HuggingFace, sobre las ciudades representativas; opcional **TimesFM**.
- *Para qué:* mostrar un modelo de 2026 prediciendo sin entrenamiento y compararlo contra los clásicos. Casi nadie lo va a hacer → es tu factor de diferenciación.

- *Output de la fase:* tabla comparativa `reports/model_comparison.md` con todas las métricas lado a lado.
- *Check:* todos los modelos están medidos con el **mismo split y las mismas métricas**, y todos se comparan contra el baseline vía MASE.

> ✅ **Fin de Fase 7:** `reports/model_comparison.md` con baselines, clásicos, LightGBM global, (DL opcional) y Chronos, todos comparables.

---

# FASE 8 — Ensemble

**Objetivo (humano):** combinar los mejores modelos para sacar un número mejor que el de cualquiera solo. El enunciado lo pide explícitamente.

**Paso 8.1 — Elegir candidatos**
- *Acción:* tomar los 2-4 mejores por MASE en validación.

**Paso 8.2 — Combinaciones simples**
- *Acción:* promedio simple y **promedio ponderado** (peso ∝ 1/error_validación).

**Paso 8.3 — Stacking**
- *Acción:* entrenar un meta-modelo (`Ridge` o `LGBMRegressor`) sobre las predicciones de los modelos base. Generá esas predicciones **sin leakage**: o con predicciones *out-of-fold* de una CV sobre train, o prediciendo sobre el bloque de **validation** (para eso lo tenés). El meta-modelo nunca ve el test hasta la evaluación final.

**Paso 8.4 — Evaluar**
- *Acción:* medir el ensemble en test y compararlo con el mejor individual.
- *Output:* `models/ensemble.pkl` + fila final en `model_comparison.md`.
- *Check:* el ensemble iguala o supera al mejor modelo individual (si no, documentar por qué y quedarse con el mejor individual).

> ✅ **Fin de Fase 8:** ensemble entrenado, evaluado y comparado.

---

# FASE 9 — Análisis avanzados (los "unique analyses")

**Objetivo (humano):** los análisis que demuestran nivel y cubren los puntos avanzados del enunciado.

**Paso 9.1 — Detección de anomalías**
- *Acción:* `IsolationForest` y `LocalOutlierFactor` sobre las features meteorológicas; además, anomalías sobre el **residuo de STL**. Cruzar con eventos de clima extremo reales y mapearlas.
- *Output:* `reports/figures/advanced/anomalies_map.png`.

**Paso 9.2 — Análisis climático**
- *Acción:* **Mann-Kendall** + **pendiente de Sen** (`pymannkendall`) por ciudad/región sobre la temperatura → ¿hay tendencia y de cuánto por año? Calcular anomalías vs baseline climático. Comparar hemisferios.

**Paso 9.3 — Impacto ambiental (aire vs clima)**
- *Acción:* correlaciones de los contaminantes (`PM2.5, PM10, O3, NO2, SO2, CO`) contra `temperature, humidity, wind, pressure`. Regresión para predecir AQI desde el clima. Ranking de ciudades por contaminación. Explorar **inversión térmica** (días sin viento + alta presión + AQI alto).

**Paso 9.4 — Análisis espacial**
- *Acción:* **KMeans** para agrupar ciudades por perfil climático (features agregadas por ciudad); choropleths; opcional **Moran's I** (autocorrelación espacial).

**Paso 9.5 — Patrones geográficos**
- *Acción:* temperatura por continente y por banda de latitud; relación `abs_lat` vs temperatura.
- *Output:* `reports/figures/advanced/*.png` + `reports/advanced_findings.md` con las conclusiones.

> ✅ **Fin de Fase 9:** figuras avanzadas + `advanced_findings.md` con los insights de clima, aire, anomalías y geografía.

---

# FASE 10 — Empaquetado y entrega

**Objetivo (humano):** dejar el repo listo para que el equipo evaluador lo entienda y lo corra.

**Paso 10.1 — README.md**
- *Acción:* incluir objetivo, descripción del dataset, estructura del repo, **cómo correrlo** (paso a paso), resumen de resultados (tabla de modelos), hallazgos clave y la **misión de PM Accelerator** (sacarla del "About" de su LinkedIn — es requisito).

**Paso 10.2 — Reproducibilidad**
- *Acción:* verificar `requirements.txt`, fijar `SEED`, ordenar notebooks/scripts numerados en `src/`.

**Paso 10.3 — Dashboard (opcional, suma puntos)**
- *Acción:* app de `Streamlit` con selección de ciudad, gráficos y predicciones.

**Paso 10.4 — Demo y publicación**
- *Acción:* grabar el video demo de 1-2 min; dejar el repo **público y open-source** (o privado con los colaboradores que pide el enunciado).
- *Check:* clonás el repo en limpio, instalás dependencias y corre de punta a punta.

> ✅ **Fin de Fase 10:** repo público reproducible + README con misión PM Accelerator + (opcional) dashboard y video.

---

## Checklist final (mapeo a los requisitos del enunciado)

| Requisito del enunciado | Dónde se cubre |
|---|---|
| Data cleaning & preprocessing | Fases 1-2 |
| EDA + viz de temperatura y precipitación | Fase 4 |
| Modelo de forecasting + métricas | Fases 6-7 |
| Usar `last_updated` para time series | Fases 3 y 7 (Tier 1) |
| Anomaly detection | Fase 9.1 |
| Múltiples modelos + ensemble | Fases 7-8 |
| Climate analysis | Fase 9.2 |
| Environmental impact (aire) | Fase 9.3 |
| Feature importance | Fase 5 |
| Spatial analysis | Fases 4.7 y 9.4 |
| Geographical patterns | Fase 9.5 |
| Misión PM Accelerator | Fase 10.1 |
| README + requirements + repo público | Fase 10 |

---

## Notas de ejecución para una IA

- **Seguí el orden de fases.** No empieces a modelar (Fase 7) sin el `selected_features.json` (Fase 5) y el split (Fase 6).
- **Producí el artefacto de salida de cada fase** antes de avanzar (parquet, json, figuras, métricas).
- **Ante una decisión con "Regla de decisión":** seguí la regla; si los datos no encajan en ningún caso, registralo y elegí la opción más conservadora.
- **Si una librería pesada falla** (`neuralforecast`, `autogluon`): son OPCIONALES (Tiers 3-4). Documentá el fallo y seguí; el pipeline mínimo viable es Tier 0 + Tier 1 + Tier 2 + ensemble.
- **Validá las Reglas de Oro en cada fase** (sobre todo: split temporal, lags por ciudad, fit solo en train). Son la causa #1 de resultados falsamente buenos.
