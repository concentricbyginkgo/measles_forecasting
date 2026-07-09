# Preprocessor methods reference

This document lists the **supported method names** that may appear in the `Methods` column of [`input/PreprocessorConfig.csv`](input/PreprocessorConfig.csv) (or a Google Sheet / remote CSV loaded the same way). Implementation lives in [`EpiPreprocessor.py`](EpiPreprocessor.py).

Column-wise transforms run during curve preparation (`prepCurve` → `preprocessDf`). Pooled-model encoding uses a separate path (`encodeMergedDf` / `getEncoderMethod`); see [MODEL_INPUT_SCHEMA §4.3](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md).

---

## 1. Config format

[`getGoogleSheetConfig`](EpiPreprocessor.py) loads a CSV with:

| Column | Role |
|--------|------|
| **Attribute** (index) | Column name in the model-input curve, **or** a full model `method` string for encoder rows (e.g. `Scikit-learn generic: XGBRegressor`). |
| **Methods** | Space-separated list of method tokens applied **in order** to that attribute. |

Example row: `cases_1M` → `zero_min linear` runs `zero_min`, then `linear`.

Unknown tokens are **ignored** with a console warning (`getPreprocessorMethods`). Rows with empty `Methods` are dropped at load time.

Default local path: `input/PreprocessorConfig.csv` (`tempConfigURL`).

---

## 2. Fixed methods

Exact token names (no suffix). Each takes `(df, column)`.

| Method | Purpose |
|--------|---------|
| **`zero_fill`** | Fill missing values with `0`. |
| **`back_fill`** | Back-fill leading (outside) missing values from the first valid entry. |
| **`forward_fill`** | Forward-fill trailing (outside) missing values from the last valid entry. |
| **`back_truncate`** | Drop rows before the first non-null in the column. |
| **`forward_truncate`** | Drop rows after the last non-null in the column. |
| **`first_valid_overwrite`** | Replace the entire column with its first valid value. |
| **`average_valid_overwrite`** | Replace the entire column with the mean of valid values. |
| **`linear`** | Linear interpolation inside gaps; if only one unique value, fill equal-value gaps. |
| **`neuralprophet`** | NeuralProphet interpolation with fixed seed `1337` (see also `neuralprophet_seed_`). |
| **`timesfm`** | Placeholder; currently returns the frame unchanged. |
| **`pass_unchanged`** | No-op (copy). Also used as the encoder slot for methods that should not one-hot `ID`. |
| **`january_only`** | Keep January values; set other months to NaN (requires `ds`). |
| **`july_only`** | Keep July values; set other months to NaN (requires `ds`). |
| **`bool_to_int`** | Cast booleans to integers. |
| **`yn_to_int`** | Map yes/no-like strings (`y`/`n` after lowercasing first character) to `1`/`0`. |
| **`flip_bool`** | Invert boolean values. |
| **`zero_min`** | Clamp values below zero to `0`. |

---

## 3. Modifiable methods (prefix + argument)

Tokens must **start with** one of these prefixes. Everything after the prefix is passed as the method argument.

| Prefix | Argument | Purpose |
|--------|----------|---------|
| **`interpolate_via_`** | SciPy `interp1d` kind, e.g. `linear`, `nearest`, `cubic`, `previous`, `next` | 1D interpolation on `ds` (Unix seconds); skips if no NaNs; single-unique-value series uses gap fill. Example: `interpolate_via_cubic`. |
| **`remove_flag_`** | Flag value (cast to float/int to match column dtype when needed) | Replace that flag with NaN. Example: `remove_flag_-999`. |
| **`divide_by_`** | Numeric factor | Divide the column by the factor. Example: `divide_by_1000`. |
| **`multiply_by_`** | Numeric factor | Multiply the column by the factor. Example: `multiply_by_100`. |
| **`drop_tailing_`** | Integer `n` | Drop the last `n` rows of the frame. Example: `drop_tailing_12`. |
| **`neuralprophet_seed_`** | Integer seed | NeuralProphet fill of NaNs with that seed (results cached under `store/`). Example: `neuralprophet_seed_42`. |
| **`discard_last_`** | Integer `n` | Discard the last `n` rows (`n > 0`). Example: `discard_last_3`. |
| **`remap_by_`** | Python dict literal (as a single token; use underscores in keys as needed) | Remap string values by `str.lower().startswith` against keys (keys normalized: lowercased, `_` → space). Example: `remap_by_{'yes':1,'no':0}` (exact quoting must match how the CSV is parsed). |
| **`check_coverage_`** | Dict literal with `proportion` and optional `tail` | If the fraction of NaNs in the (optional tail) window is **≥** `proportion`, set the whole column to NaN. |
| **`check_gaps_`** | Dict literal with `length` and optional `tail` | If the longest contiguous NaN stretch in the (optional tail) window is **≥** `length`, set the whole column to NaN. |

---

## 4. Encoders (pooled / merged curves)

Encoders are **not** applied as ordinary per-column `Methods` on predictors. For pooled sklearn-style fits, [`getEncoderMethod`](EpiPreprocessor.py) reads the config row whose **Attribute** equals the model `method` string and expects a token of the form:

```text
{encoderName}_on_{args}
```

| Encoder base | Config pattern | Purpose |
|--------------|----------------|---------|
| **`encode_onehot`** | `encode_onehot_on_ID` | One-hot dummy columns for `ID` (geography). Default if no encoder row is found for the method. |
| **`encode_ordinal`** | `encode_ordinal_on_{col}_by_{ref}` | Ordinal codes for `{col}` ranked by last value of `{ref}` (see `encodeOrdinal`). |

**`pass_unchanged`** on a method-string Attribute disables encoding (used for NeuralProphet lagged regressors). Details and preprocessor-config requirements for pooled runs: [MODEL_INPUT_SCHEMA §4.3–4.4](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md).

Registered base names in code: `baseEncoders` → `encode_ordinal`, `encode_onehot`.

---

## 5. Related files

| Topic | File |
|-------|------|
| Method implementations | [`EpiPreprocessor.py`](EpiPreprocessor.py) |
| Per-column / per-method config | [`input/PreprocessorConfig.csv`](input/PreprocessorConfig.csv) |
| When columns need config rows | [`MODEL_INPUT_SCHEMA.md`](../data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md) |
| Curve prep calling `preprocessDf` | [`MeaslesModelEval.py`](MeaslesModelEval.py) (`prepCurve`) |
