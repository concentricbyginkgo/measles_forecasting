# Run metadata configuration schema

This document defines the **CSV metadata format** used to drive batch model execution via [`fitOne.py`](fitOne.py). Each row is one model configuration (geography or pooled selection, predictor set, algorithm, and optionally a random seed).

For the **tabular disease dataset** consumed by the loader, see [`../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md`](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md).

**Canonical example:** [`input/metadata_example.csv`](input/metadata_example.csv)  
**Typical generator:** [`../grid_search/create_final_mod_metadata.R`](../grid_search/create_final_mod_metadata.R) (writes e.g. `run_metadata.csv`).

---

## 1. Execution entry point

[`fitOne(metadata, ROW, run_name)`](fitOne.py) selects the row where `metadata['ROW_ID'] == ROW`, then reads the columns below. **`predictor`** must be a string that **`ast.literal_eval`** parses into a Python **`dict`** (predictor column names → non-negative integer lags). **`environmentalArg`** is optional (see §2); when present it must parse to a **`dict`** or it is treated as empty. **`depVar`** is optional (see §2); when omitted the outcome defaults to **`cases_1M`** for backward compatibility. **Binary classification metrics** (sensitivity, F1, etc.): [`fitOne.py`](fitOne.py) passes **`evaluate_binary_metrics=True`** and an explicit labeller — by default [`MeaslesModelEval.defaultBinaryMetric`](MeaslesModelEval.py) (**≥ 2** on the **depVar** scale), or **`lambda x: x >= T`** when **`binary_outbreak_threshold`** is a numeric **T ≥ 0** in the **same units as `depVar`** (e.g. incidence per million if `depVar` is `cases_1M`, or a raw count threshold if `depVar` is a case-count column) (see §2). Set **`evaluate_binary_metrics`** to false to force regression-only summaries (classification fields as NaN) for that row. Direct use of model wrappers in Python defaults to regression-only unless you pass **`evaluate_binary_metrics=True`** and **`binaryLabelMetric`**.

Merged predictor specification:

```text
indepVars = literal_eval(predictor)  ∪  environmentalArg_dict
```

where `environmentalArg_dict` is **`{}`** if the **`environmentalArg`** column is missing, blank, or not a parseable dict (otherwise `ast.literal_eval` of the cell). Order: base predictors from `predictor`, then keys from `environmentalArg`.

---

## 2. Configuration parameters (structured table)

| Column | Required by `fitOne` | Type | Default / if omitted | Description |
|--------|----------------------|------|----------------------|-------------|
| **`ROW_ID`** | **Yes** | Integer (recommended) or string unique per row | — | Primary key for the run; written into outputs and used in filenames (e.g. `{ROW_ID}_Summary.csv`, `{ROW_ID}_{GEO_ID}_Projection.csv` per country curve in a run). Compilation scripts often coerce summary filenames to numeric IDs. |
| **`geography`** | **Yes** | String | — | Passed as the first argument (`geography`) to the model wrapper and stored internally as `selection`: must map to a value in `GEO_ID` from the model training data or a **pooled filter key** such as `cluster:1` (must exist in `prepData()['filters']`). See [MODEL_INPUT_SCHEMA §4](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md). |
| **`model`** | **Yes** | String (controlled vocabulary) | — | Selects model class; must match one of the **exact strings** in §3 (case and spacing matter for most entries). |
| **`depVar`** | No | String (column name) | **`cases_1M`** if the column is missing or the cell is blank / NaN | Dependent variable passed to the model wrapper (second argument after `geography`). Must exist in the model input CSV and have a preprocessor rule in [`PreprocessorConfig.csv`](input/PreprocessorConfig.csv) (same as [`MODEL_INPUT_SCHEMA`](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md) §3). Does **not** change [`MeaslesDataLoader`](../model/MeaslesDataLoader.py) **`validityColumn`** truncation (still `cases_1M` unless you change the loader). |
| **`predictor`** | **Yes** | String: Python **`dict`** literal | — | Disease / core predictors and their lags, e.g. `"{'MCV2': 0, 'cases_1M_12z': 0}"`. Keys must exist in the preprocessed curve and in [`PreprocessorConfig.csv`](input/PreprocessorConfig.csv). |
| **`environmentalArg`** | No | String: Python **`dict`** literal, or empty | **`{}`** (no extra predictors) if the column is missing, the cell is blank / NaN, or parsing fails | Additional predictors (often climate), e.g. `"{'mean_temp': 3, 'mean_precip_mm_per_day': 3}"`. Same key and typing rules as `predictor`. You may omit the column entirely, or use a blank cell, or the literal `"{}"`. |
| **`Seed`** | No | Integer | **`1337`** if the column is missing or the cell is empty / NaN | Random seed for the estimator (`randomState` in the wrapper). Omit the column when you do not need reproducibility per row. |
| **`evaluate_binary_metrics`** | No | Boolean-like | **On** (same as legacy batch) if the column is missing or blank | If the cell parses as **false** (`0`, `false`, `no`, …), [`fitOne.py`](fitOne.py) turns **off** outbreak classification metrics for that row (even if a threshold column is set). Any other non-empty value is treated as **on** for threshold/default resolution below. |
| **`binary_outbreak_threshold`** | No | Number **≥ 0** | — (if off, see `evaluate_binary_metrics`; if on and no valid threshold, **≥ 2** via explicit `defaultBinaryMetric` from `fitOne`) | Outbreak labelling threshold *T* on the **same scale as `depVar`** (not forced to “per million”). When set to a numeric *T* **≥ 0**, `fitOne` passes **`binaryLabelMetric = lambda x: x >= T`** for observed and predicted **depVar** values. Non-numeric or *T* < 0 is ignored (falls through to default ≥2 when metrics are on). |
| **`MODEL_ID`** | No | String | — | Hash or label for design tracking; not read by `fitOne.py`. |
| **`num_predictors`** | No | Integer | — | Count of predictors in `predictor`; informational for pipelines; not read by `fitOne.py`. |
| **`Rep`** | No | Integer | — | Replicate index for sweeps; not read by `fitOne.py`. |
| **`v1_ROW_ID`** | No | Integer or string | — | Legacy / crosswalk; not read by `fitOne.py`. |


### 2.1 Values fixed in code (not in metadata today)

| Setting | Current behavior | To change |
|---------|------------------|-----------|
| **`testSize`**, preprocessor URL, `missingVarResponse`, etc. | Model class defaults in [`MeaslesModelEval.py`](MeaslesModelEval.py) | Use notebooks / direct API for full control; metadata path is intentionally minimal. |

---

## 3. Accepted `model` strings (`fitOne.py`)

These strings are matched **exactly** (after reading from CSV). Typos or alternate labels will fall through with **`model` undefined** and can cause failures.

| `model` value | Python wrapper / constructor |
|---------------|------------------------------|
| `neural prophet` | [`npLaggedTTS`](MeaslesModelEval.py) |
| `gradient boosting` | [`sklGradientBoostingRegression`](MeaslesModelEval.py) |
| `AdaBoost regressor` | `sklGeneric` with `AdaBoostRegressor` |
| `Bagging regressor` | `sklGeneric` with `BaggingRegressor` |
| `Extra Trees` | `sklGeneric` with `ExtraTreesRegressor` |
| `Random Forest` | `sklGeneric` with `RandomForestRegressor` |
| `ElasticNet` | `sklGeneric` with `ElasticNet` |
| `SGD` | `sklGeneric` with `SGDRegressor` |
| `SVR` | `sklGeneric` with `SVR` |
| `BayesianRidge` | `sklGeneric` with `BayesianRidge` |
| `KernelRidge` | `sklGeneric` with `KernelRidge` |
| `CatBoost` | `sklGeneric` with `CatBoostRegressor` |
| `Linear regression` | `sklGeneric` with `LinearRegression` |
| `XGBRegressor` | `sklGeneric` with `XGBRegressor` |
| `LGBMR` | `sklGeneric` with `LGBMRegressor` |
| `diverse` | Ensemble preset (`ensembleModels`) |
| `diverse low n` | Ensemble preset |
| `boosted heavy` | Ensemble preset |
| `boosted alpha` | Ensemble preset |

**Pipeline note:** [`create_final_mod_metadata.R`](../grid_search/create_final_mod_metadata.R) currently includes **`"XGBoost"`** in its `models` vector; **`fitOne.py` does not map that string**—use **`XGBRegressor`** in exported metadata so the row matches §3.

---

## 4. `predictor` and `environmentalArg` typing rules

**`predictor` (required)**

- **Format:** Single-line string, valid Python literal for a **`dict`**: `{'col_a': 0, 'col_b': 3}`  
  Use **single quotes** around keys if possible to avoid CSV escaping issues; keys must match column names exactly.
- **Keys:** Must match **column names** in [`processed_measles_model_data.csv`](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md) (or your replacement dataset) and have preprocessor entries where applicable.
- **Values:** **Integers ≥ 0** — lag in months for that regressor (`indepVars` in `MeaslesModelEval`).

**`environmentalArg` (optional)**

- Omit the column, leave the cell empty, or use `"{}"` for **no** environmental (or secondary) predictors beyond `predictor`.
- If provided and non-blank, it must **`ast.literal_eval`** to a **`dict`**; non-dict or parse errors become **`{}`** in `fitOne.py`.

---

## 5. Machine-readable schema

JSON Schema (draft-07) for one metadata row: [`schemas/run_metadata.schema.json`](schemas/run_metadata.schema.json).

---

## 6. Minimal valid row (conceptual)

```csv
ROW_ID,geography,predictor,model
999999,NGA,"{'MCV2': 0}",CatBoost
```

or with explicit empty environmental block:

```csv
ROW_ID,geography,predictor,environmentalArg,model
999999,NGA,"{'MCV2': 0}","{}",CatBoost
```

A **blank** `environmentalArg` cell or omitting the column is equivalent to `{}`.

With no **`Seed`** column (or a blank seed), [`fitOne.py`](fitOne.py) uses **`randomState = 1337`**. Add **`Seed`** when you need a distinct seed per configuration.

With no **`depVar`** column (or a blank cell), the outcome defaults to **`cases_1M`**. Example with another regression target and **5/M** binary evaluation to align with [`model_output_processing/2_compile_time_series_tables.R`](../model_output_processing/2_compile_time_series_tables.R) Shiny columns:

```csv
ROW_ID,geography,depVar,binary_outbreak_threshold,predictor,model
999999,NGA,cases_1M,5,"{'MCV2': 0}",CatBoost
```

(Include optional columns such as `MODEL_ID` or `Rep` as needed for your bookkeeping.)

---

## 7. Related files

| File | Role |
|------|------|
| [`fitOne.py`](fitOne.py) | Reads metadata columns and maps `model` → class |
| [`MeaslesModelEval.py`](MeaslesModelEval.py) | Training, defaults, preprocessor |
| [`input/PreprocessorConfig.csv`](input/PreprocessorConfig.csv) | Per-column preprocessing |
| [`../model_output_processing/1_compile_summary_table.R`](../model_output_processing/1_compile_summary_table.R) | Joins `ROW_ID` to summaries |
