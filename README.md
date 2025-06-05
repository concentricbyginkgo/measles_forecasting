# Measles Forecasting
AI-enabled measles forecasting using machine learning models

## Overview

This repository provides a complete pipeline for measles outbreak forecasting using machine learning approaches. The system processes epidemiological, climate, and socioeconomic data to predict measles case incidence at the country level.

### Quick Start
1. **Data Processing**: Run scripts 1-7 in `data_ingestion_pipeline/` 
2. **Environment Setup**: Create Python environment using `environment.yml`
3. **Model Training**: Use `alpha_model/RunFromMetadata.ipynb` for grid search or `alpha_model/RunAlphaModels.ipynb` for forecasting
4. **Beta Models**: Explore advanced methods in `beta_model/BetaMethodsComparison.ipynb`

## Repository Structure

```
measles_forecasting/
├── data_ingestion_pipeline/    # R scripts for data processing (1-7)
├── model/                      # Core Python modules
├── alpha_model/               # Alpha model notebooks and configs
├── beta_model/                # Beta model development
├── model_comparison_pipeline/ # Model evaluation tools
└── other_deliverables/        # Documentation and data inventory
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

### Important Notes
- **Scripts 1-6 must complete successfully before running script 7**
- Scripts 2 & 3 process gridded climate data (.nc files) - **extremely compute and memory intensive**
- Climate processing is parallelized - tune parameters for your machine specifications
- Manual downloading of raw datasets required (links provided in scripts)
- Complete data inventory available in `other_deliverables/data_inventory/`

## Forecasting Models

### Environment Setup

**Recommended**: Use mamba/conda with Python 3.11

#### Option 1: Using the provided environment file
```bash
conda env create -f alpha_model/environment.yml
conda activate measles_forecasting
```

#### Option 2: Manual installation
```bash
mamba create -n measles_forecasting python=3.11
mamba activate measles_forecasting
mamba install neuralprophet scikit-learn statsmodels jupyterlab pandas numpy \
              geopandas multiprocess matplotlib scipy country_converter seaborn \
              xgboost catboost lightgbm
```

### Core Python Modules (`model/`)

The repository includes four main Python modules:

- **`MeaslesDataLoader.py`** - Data loading and preprocessing
- **`MeaslesModelEval.py`** - Model evaluation and cross-validation  
- **`EpiPreprocessor.py`** - Epidemiological data preprocessing
- **`fitOne.py`** - Individual model fitting functions

### Alpha Model (`alpha_model/`)

#### Grid Search Application (`RunFromMetadata.ipynb`)
Performs systematic parameter search to find optimal models for each country.

**Required metadata fields:**
- **`ROW_ID`**: Unique identifier for parameter set
- **`model`**: ML model type (`XGBRegressor`, `CatBoost`, `gradient boosting`, `Random Forest`, `Bagging regressor`)
- **`predictor`**: Primary predictor variables and lags `{'predictor': lag}`
- **`environmentalArg`**: Environmental predictors `{'predictor': lag}`
- **`country`**: ISO3 country code or geographic grouping
- **`Seed`**: Random seed for reproducibility

**Usage:**
```python
for i in range(len(meta_df)):
    fi.fitOne(metadata=meta_df, ROW=i, run_name='sweep_20250106')
```

**Outputs:**
- Summary statistics: `output/<run_name>/scores/<ROW_ID>_Summary.csv`
- Projections: `output/<run_name>/tables/<ROW_ID>_<ISO3>_Projection.csv`

#### Forward Projection (`RunAlphaModels.ipynb`)
Generates future forecasts using pre-selected optimal parameters.

**Requirements:**
- `input/alpha_model_by_country.csv` - Parameter specifications per country
- Trained models from grid search

**Outputs:**
- Future projections: `output/tables/<ROW_ID>_<ISO3>_Projection.csv`

### Beta Model (`beta_model/`)

Advanced model development and comparison:

- **`BetaMethodsComparison.ipynb`** - Comparative analysis of modeling approaches
- **`beta_run_specs.csv`** - Configuration file for beta model parameters

## Model Features

### Supported ML Algorithms
- **XGBoost** - Gradient boosting framework
- **CatBoost** - Categorical boosting
- **LightGBM** - Gradient boosting
- **Random Forest** - Ensemble method
- **Bagging Regressor** - Bootstrap aggregating
- **Gradient Boosting** - Scikit-learn implementation

### Data Sources
- **Epidemiological**: WHO measles surveillance data
- **Climate**: Gridded precipitation and temperature data
- **Socioeconomic**: World Bank indicators, migration data
- **Infrastructure**: Road density data
- **Immunization**: MCV1/MCV2 coverage, SIA campaigns
- **Travel**: Air passenger flows

### Key Features
- Country-specific and clustered modeling approaches
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

## Output Format

All projection files contain:
- **`ID`**: Country ISO3 code
- **`ds`**: Date/timestamp  
- **`y`**: Observed measles incidence
- **`yhat1`**: Model-projected incidence
- **`ROW_ID`**: Metadata identifier

## Model Comparison Pipeline

Additional tools for model evaluation and comparison are available in `model_comparison_pipeline/`.

## Documentation

- **Data inventory**: `other_deliverables/data_inventory/`
- **Social data README**: `data_ingestion_pipeline/README_Social_Series.txt`
- **Project documentation**: `other_deliverables/`

## Notes for Public Use

This repository has been prepared for public use:
- ✅ **Local data processing** - All data sources use local files
- ✅ **Complete environment specification** - Dependencies clearly defined
- ✅ **Comprehensive documentation** - Updated README and comments

Users must run the data ingestion pipeline to generate required input files, as processed data files are not included due to size constraints.

## Citation

When using this code, please cite the associated research and acknowledge the data sources as detailed in the data inventory.

