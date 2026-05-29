# EpiFlowML
A Modular Framework for Standardized and Reproducible Epidemiological Forecasting
Contact: ameadows@ginkgobioworks.com

## Overview

This repository provides a complete pipeline for measles outbreak forecasting using machine learning approaches. The system processes epidemiological, climate, and socioeconomic data to predict measles case incidence at the country level.

### Quick Start
1. **Data Processing**: Run scripts 1-7 in `data_ingestion_pipeline/` (R dependencies: see `requirements-r.txt`)
2. **Grid Search**: Use R scripts in `grid_search/` to identify optimal predictors
3. **Environment Setup**: Create Python environment with `pip install -r requirements.txt` (see Environment Setup section)
4. **Model Training**: Use `model/FinalModelStage1Runs.ipynb` or `model/RunFromFunction.ipynb` for model training and forecasting
5. **Output Compilation**: Run scripts in `model_output_processing/` to compile model outputs for visualization
6. **Model Validation** (Optional): Launch the Shiny app in `shiny_standalone/` for interactive model validation and visualization

## Repository Structure

```
epiflowml/
├── requirements.txt           # Python dependencies (version-pinned)
├── requirements-r.txt         # R dependencies
├── data_ingestion_pipeline/   # R scripts for model input data processing (1-7)
├── grid_search/               # R scripts for predictor selection and metadata generation
├── model/                     # Core Python modules and Jupyter notebooks
│   ├── output/               # Model run outputs (organized by run_name)
│   │   └── {run_name}/
│   │       ├── tables/        # Individual time series projection files
│   │       └── scores/        # Individual model summary files
├── model_output_processing/   # R scripts to compile model outputs for visualization
├── model_comparison_pipeline/ # Model evaluation and visualization tools
└── shiny_standalone/          # Interactive Shiny web application for model validation
    └── data/                  # Compiled data for Shiny app
        └── tables/
            ├── selection/     # Selection period time series tables
            └── validation/    # Validation period time series tables
```

## Data Ingestion Pipeline

The data ingestion pipeline contains **7 R scripts** that process raw case and predictor data into a clean, consistent format for model training:

### Scripts Overview
1. **`1_case_processing.R`** - Processes WHO measles case data, creates outbreak indicators
2. **`2_precip_processing.R`** - Extracts and processes gridded precipitation data (compute-intensive)
3. **`3_temperature_processing.R`** - Processes gridded temperature data (very compute-intensive)
4. **`4_social_data_processing.R`** - Processes socioeconomic indicators (see `README_Social_Series.txt`)
5. **`5_road_data_processing.R`** - Processes road density as development proxy
6. **`6_SIA_processing.R`** - **NEW**: Processes Supplementary Immunization Activities data
7. **`7_combine_all_datasets.R`** - Combines all processed datasets into final model input

### R Dependencies
Install required R packages from the project list:
```r
install.packages(readLines("requirements-r.txt"))
```

### Important Notes
- **Scripts 1-6 must complete successfully before running script 7**
- Scripts 2 & 3 process gridded climate data (.nc files) - **extremely compute and memory intensive**
- Climate processing is parallelized - tune parameters for your machine specifications
- Manual downloading of raw datasets required (links provided in scripts)

### Model input schema (Python contract)

Ingestion must produce a **long-format** CSV that the Python loader can read (default: `model/input/processed_measles_model_data.csv`). Structural requirements, experiment-specific columns (`depVar`, `indepVars`), preprocessor alignment, and measles-specific coupling (for example `cases_1M` truncation in the loader) are documented in:

- **[`data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md`](data_ingestion_pipeline/MODEL_INPUT_SCHEMA.md)** — human-readable contract, **pooled (global-local) grouping and `selection` keys** (see §4), and new-pathogen checklist  
- **[`data_ingestion_pipeline/schemas/model_input.schema.json`](data_ingestion_pipeline/schemas/model_input.schema.json)** — JSON Schema (required: `ISO3`, `date`; additional columns allowed)  
- **[`model/input/example_minimal_model_input.csv`](model/input/example_minimal_model_input.csv)** — header-only example matching the canonical measles column set (no data rows)

**Path alignment:** [`data_ingestion_pipeline/7_combine_all_datasets.R`](data_ingestion_pipeline/7_combine_all_datasets.R) writes `model_training_data.csv` at the repo root by default. Copy or symlink that file to `model/input/processed_measles_model_data.csv` (or change `MeaslesDataLoader.prepData` `defaultLoc`) before training.

## Grid Search Pipeline

The `grid_search/` directory contains R scripts for predictor selection and model metadata generation:

### Scripts Overview
- **`univariate_country_test.R`** - Performs univariate analysis to identify significant predictors by country
- **`create_final_mod_metadata.R`** - Generates metadata for model configurations based on predictor analysis

### Key Outputs
- **`univariate_country_results.csv`** - Results of univariate predictor analysis
- **`correlation_results.csv`** - Correlation analysis between predictors
- **`metadata_example.csv`** - Example metadata format for model configuration

### Run metadata configuration schema

Batch runs driven by CSV metadata (e.g. `fitOne.py`, `RunFromFunction.ipynb`) use a fixed set of columns and **`model`** string literals. For a **parameter table, types, defaults, and accepted `model` values**, see:

- **[`model/METADATA_RUN_SCHEMA.md`](model/METADATA_RUN_SCHEMA.md)** — structured specification aligned with [`fitOne.py`](model/fitOne.py)
- **[`model/schemas/run_metadata.schema.json`](model/schemas/run_metadata.schema.json)** — JSON Schema (draft-07) for one metadata row

This pipeline helps identify the most relevant predictors for each country before running the full machine learning models.

## Complete Workflow

The typical workflow from data to visualization:

1. **Data Ingestion** → Process raw data into `model/input/processed_measles_model_data.csv`
2. **Grid Search** → Generate metadata with optimal predictor combinations
3. **Model Training** → Run models using notebooks, outputs saved to `model/output/{run_name}/`
4. **Output Compilation** → Compile outputs using `model_output_processing/` scripts
5. **Visualization** → Launch Shiny app to explore results

### Model Training Output

When you run model training (via `RunFromFunction.ipynb` or `FinalModelStage1Runs.ipynb`), outputs are saved to:
```
model/output/{run_name}/
├── tables/
│   └── {ROW_ID}_{geography}_Projection.csv  # Time series projections
└── scores/
    └── {ROW_ID}_Summary.csv                # Model performance metrics
```

### Post-Training Processing

After training, compile outputs for visualization:
1. **Compile Summaries**: Run `1_compile_summary_table.R` (set `run_type = "selection"` or `"validation"`)
2. **Compile Time Series**: Run `2_compile_time_series_tables.R` (set `run_type` to match)
3. **Launch Shiny App**: The app will automatically load compiled outputs

## Forecasting Models

### Environment Setup

**Recommended**: Use mamba/conda with Python 3.11, then install from the project's dependency list:

```bash
mamba create -n epiflowml python=3.11
mamba activate epiflowml
pip install -r requirements.txt
```

The `requirements.txt` file lists all Python dependencies with version pins. For a fully locked environment, run `pip freeze > requirements-lock.txt` after installing.

### Core Python Modules (`model/`)

The repository includes several Python modules and Jupyter notebooks:

#### Core Modules
- **`MeaslesDataLoader.py`** - Data loading and preprocessing
- **`MeaslesModelEval.py`** - Model evaluation and cross-validation  
- **`EpiPreprocessor.py`** - Epidemiological data preprocessing
- **`fitOne.py`** - Individual model fitting functions
- **`EpiAnnealer.py`** - Advanced optimization and hyperparameter tuning
- **`ModelSweeps.py`** - Mass model comparison using multiple ML algorithms
- **`LossFunctions.py`** - Custom loss functions for model evaluation
- **`SeasonalityMetrics.py`** - Seasonality analysis and trend detection

#### Jupyter Notebooks
- **`FinalModelStage1Runs.ipynb`** - Primary notebook for model training and forecasting
- **`RunFromFunction.ipynb`** - Alternative model training workflow using metadata-driven approach
- **`TTSEval.ipynb`** - Interactive time series evaluation and testing with model function documentation

### Model Training and Forecasting

The main workflow uses the Jupyter notebooks in the `model/` directory:

#### Primary Notebook (`FinalModelStage1Runs.ipynb`)
The main notebook for model training and forecasting. This notebook integrates all the core modules to:

- Load and preprocess data using `MeaslesDataLoader.py`
- Perform model selection using `ModelSweeps.py` 
- Evaluate models using `MeaslesModelEval.py`
- Generate forecasts and projections

#### Metadata-Driven Training (`RunFromFunction.ipynb`)
Alternative workflow that uses metadata from the grid search pipeline to systematically train models:

- Reads metadata configurations from `model/input/metadata_example.csv` (schema: [`model/METADATA_RUN_SCHEMA.md`](model/METADATA_RUN_SCHEMA.md))
- Uses `fitOne.py` functions for individual model training
- Supports batch processing of multiple model configurations
- Integrates with the grid search pipeline outputs
- Outputs organized by `run_name` in `model/output/{run_name}/`
  - Individual projection files: `tables/{ROW_ID}_{geography}_Projection.csv`
  - Individual summary files: `scores/{ROW_ID}_Summary.csv`

#### Time Series Evaluation (`TTSEval.ipynb`)
Specialized notebook for time series model evaluation and testing.

#### Advanced Features
- **Hyperparameter optimization** via `EpiAnnealer.py` # !NOTE! This is experimental
- **Seasonality analysis** using `SeasonalityMetrics.py`
- **Custom loss functions** defined in `LossFunctions.py`
- **Multi-algorithm comparison** through `ModelSweeps.py`

## Model Features

### Example Supported ML Algorithm Examples
- **XGBoost** - Gradient boosting framework
- **CatBoost** - Categorical boosting
- **LightGBM** - Gradient boosting
- **Random Forest** - Ensemble method
- **Bagging Regressor** - Bootstrap aggregating
- **Gradient Boosting** - Scikit-learn implementation

### Data Sources (Measles Case Study)
- **Epidemiological**: WHO measles surveillance data
- **Climate**: Gridded precipitation and temperature data
- **Socioeconomic**: World Bank indicators, migration data
- **Infrastructure**: Road density data
- **Immunization**: MCV1/MCV2 coverage, SIA campaigns
- **Travel**: Air passenger flows

### Key Features
- Geography-specific and pooled modeling approaches
- Time series cross-validation
- Outbreak probability predictions
- Environmental and social determinants integration
- Comprehensive model evaluation metrics

## Data Requirements

The model expects `input/processed_measles_model_data.csv` generated by the data ingestion pipeline. Key variables include:

- **Target**: `cases_1M` (cases per million population)
- **Outbreak indicators**: Various threshold-based outbreak definitions
- **Climate**: Temperature and precipitation aggregates
- **Socioeconomic**: Birth rates, migration, development indicators
- **Immunization**: Vaccination coverage and campaign data

## Model Output Processing Pipeline

After model training, the `model_output_processing/` directory contains R scripts to compile individual model outputs into formats suitable for visualization and analysis.

### Directory Structure

Model outputs are organized by run name:
```
model/output/{run_name}/
├── tables/          # Individual projection files: {ROW_ID}_{geography}_Projection.csv
└── scores/          # Individual summary files: {ROW_ID}_Summary.csv
```

### Compilation Scripts

1. **`1_compile_summary_table.R`** - Compiles individual summary statistics into a single table
   - Reads all `{ROW_ID}_Summary.csv` files from `model/output/{run_name}/scores/`
   - Combines them with metadata
   - Outputs: `model/output/{run_name}_{run_type}_compiled_summary.csv`
   - **Important**: Set `run_type` to `"selection"` or `"validation"` to indicate the evaluation window

2. **`2_compile_time_series_tables.R`** - Compiles time series projections by geography
   - Reads all `{ROW_ID}_{geography}_Projection.csv` files from `model/output/{run_name}/tables/`
   - Combines projections by geography and merges with observed data
   - Outputs to `shiny_standalone/data/tables/{run_type}/` (where `run_type` is "selection" or "validation")
   - **Important**: Set `run_type` to match your compilation - determines output subdirectory

### Usage

For **selection** runs:
```r
run_name <- "my_selection_run"
run_type <- "selection"
# Run both compilation scripts with these settings
```

For **validation** runs:
```r
run_name <- "my_validation_run"
run_type <- "validation"
# Run both compilation scripts with these settings
```

### Key Features

- **Run Type Flag**: Distinguishes between selection and validation evaluation windows
- **Automatic Directory Organization**: Tables written to appropriate subdirectories
- **Error Handling**: Comprehensive validation and error reporting
- **Metadata Integration**: Combines model outputs with configuration metadata

## Output Format

### Individual Model Outputs

All projection files (`{ROW_ID}_{geography}_Projection.csv`) contain:
- **`ID`**: Unique geography string identifier
- **`ds`**: Date/timestamp  
- **`y`**: Observed measles incidence (if available)
- **`yhat1`**: Model-projected incidence
- **`ROW_ID`**: Metadata identifier
- Additional columns: `MODEL_ID`, outbreak indicators, cumulative values

All summary files (`{ROW_ID}_Summary.csv`) contain:
- Performance metrics: Test/Train MSE, MAE, R²
- Binary classification metrics: Sensitivity, Specificity, F1 Score
- Model configuration: method, geography, predictor variables
- Data quality metrics: coverage, seasonality scores

### Compiled Outputs

- **Compiled Summary**: Single CSV with all model performance metrics and metadata
- **Compiled Time Series**: Geography-level CSVs with all model projections for that geography

## Interactive Model Validation (`shiny_standalone/`)

The `shiny_standalone/` directory contains a comprehensive Shiny web application for interactive model validation and visualization.

### Prerequisites

Before running the Shiny app, you must compile your model outputs using the scripts in `model_output_processing/`:

1. Run `1_compile_summary_table.R` for both selection and validation runs (set `run_type` appropriately)
2. Run `2_compile_time_series_tables.R` for both selection and validation runs (set `run_type` appropriately)

The compiled outputs will be automatically placed in:
- `shiny_standalone/data/tables/selection/` - Selection period time series
- `shiny_standalone/data/tables/validation/` - Validation period time series

### Key Features
- **Interactive Geography Selection** - Choose from geographies with unique string identifiers (e.g. ISO3 codes) for detailed analysis
- **Model Performance Metrics** - View detailed performance statistics in interactive data tables
- **Epidemiological Curve Visualization** - Compare observed vs predicted case counts over time
- **Binary Outcome Analysis** - Visualize outbreak prediction accuracy using heatmaps
- **Model Selection & Validation** - Separate analysis for training and validation periods
- **Automatic Data Loading** - Automatically loads compiled summaries from `model/output/` directory

### Application Structure
- **`ui.R`** - User interface definition with responsive Bootstrap layout
- **`server.R`** - Server logic for data processing and visualization
- **`global.R`** - Global variables, functions, and data loading
  - Automatically loads compiled summaries (selection and/or validation)
  - Falls back to sample data if compiled summaries not found
- **`data/`** - Compiled datasets for visualization
  - **`tables/selection/`** - Selection period time series by geography
  - **`tables/validation/`** - Validation period time series by geography
  - **`cutoff_date_by_country.csv`** - Cutoff dates for evaluation periods
  - **`sample_summaryTable.csv`** - Sample summary table (fallback)
- **`www/`** - Static web assets (CSS, images, favicon)
- **Documentation** - Complete user guide (`Measles_Model_Validation_App_Documentation.html`)

### Running the Application

```r
# Install required R packages (from project root)
install.packages(readLines("requirements-r.txt"))

# Update run_name in global.R to match your compiled outputs (default: "test_run")
# Then run the application
shiny::runApp("shiny_standalone/")
```

### Data Loading

The app automatically:
1. Looks for compiled summaries: `model/output/{run_name}_{run_type}_compiled_summary.csv`
2. Combines selection and validation summaries if both exist
3. Falls back to generic compiled summary if run_type-specific not found
4. Falls back to sample data if no compiled summaries found

Update the `run_name` variable in `global.R` to match your model run name.

The application provides an intuitive interface for exploring model performance across different countries and time periods, making it easy to validate model predictions and compare performance metrics.

## Testing

The repository includes a comprehensive test suite in the `test/` directory to validate code quality, syntax, and workflow integrity.

### Test Suite Overview

The test suite includes four main test categories:

1. **`test_python_imports.py`** - Python module validation
   - Tests that all core Python modules can be imported
   - Validates Python syntax for all `.py` files in `model/`
   - Handles missing dependencies gracefully (warnings only)

2. **`test_r_syntax.R`** - R script validation
   - Validates syntax for all R scripts in `model_output_processing/`
   - Ensures compilation scripts parse correctly
   - Catches syntax errors before runtime

3. **`test_file_structure.py`** - Repository structure validation
   - Verifies required directories exist (`model/`, `model_output_processing/`, etc.)
   - Checks for required files (core modules, scripts, README)
   - Validates metadata file location

4. **`test_compilation_scripts.R`** - Integration tests
   - Creates sample data and tests compilation workflow
   - Validates that summary and table compilation scripts work correctly
   - Tests output structure and required columns
   - Validates error handling

5. **`run_all_tests.sh`** - Master test runner
   - Executes all test suites in sequence
   - Provides comprehensive test summary
   - Returns appropriate exit codes for CI/CD integration

### Running Tests

#### Run All Tests (Recommended)
```bash
cd /path/to/epiflowml
./test/run_all_tests.sh
```

This will run all test suites and provide a summary of results.

#### Run Individual Test Suites

**Python tests:**
```bash
# Test Python imports and syntax
python3 test/test_python_imports.py

# Test file structure
python3 test/test_file_structure.py
```

**R tests:**
```bash
# Test R script syntax
Rscript test/test_r_syntax.R

# Test compilation scripts (integration test)
Rscript test/test_compilation_scripts.R
```

### Test Requirements

- **Python 3.11+** with packages from `requirements.txt` (see Environment Setup section)
- **R 4.0+** with packages from `requirements-r.txt`
- All dependencies from the main workflow must be installed

### Test Data

The `test/data/` directory contains sample data files used for integration testing:
- Sample metadata files
- Sample summary statistics
- Sample time series projections

Test data is automatically generated by `test_compilation_scripts.R` and stored in `test/output/`.

### Continuous Integration

Tests are automatically run via GitHub Actions on:
- Push to `main` or `develop` branches
- Pull requests

The CI pipeline runs:
1. Python tests (imports, syntax, file structure)
2. R tests (syntax validation)
3. Integration tests (compilation scripts)

See `.github/workflows/build-test.yml` for CI configuration.

### Adding New Tests

When adding new functionality to the repository:

1. **New Python modules**: Add import tests to `test_python_imports.py`
2. **New R scripts**: Add syntax tests to `test_r_syntax.R`
3. **New workflows**: Add integration tests to `test_compilation_scripts.R` or create new test files
4. **New required files/directories**: Update `test_file_structure.py`

For detailed test documentation, see `test/README.md`.

## Documentation

- **Social data README**: `data_ingestion_pipeline/README_Social_Series.txt`
- **Shiny app user guide**: `shiny_standalone/Measles_Model_Validation_App_Documentation.html`
- **Test suite documentation**: `test/README.md`

## Notes for Public Use

This repository has been prepared for public use:
- ✅ **Local data processing** - All data sources use local files
- ✅ **Complete environment specification** - Python: `requirements.txt` (version-pinned); R: `requirements-r.txt`
- ✅ **Comprehensive documentation** - Updated README and comments

Users must run the data ingestion pipeline to generate required input files, as processed data files are not included due to size constraints.

## Citation

When using this code, please cite the associated research and acknowledge the data sources as detailed in the data inventory.

## Terms of Use

Creative Commons Attribution 4.0 (CC BY 4.0)

This work is based on research funded by the Gates Foundation. The findings and conclusions contained within are those of the authors and do not necessarily reflect positions or policies of the Gates Foundation.

