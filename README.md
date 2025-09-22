# Measles Forecasting
AI-enabled measles forecasting using machine learning models

## Overview

This repository provides a complete pipeline for measles outbreak forecasting using machine learning approaches. The system processes epidemiological, climate, and socioeconomic data to predict measles case incidence at the country level.

### Quick Start
1. **Data Processing**: Run scripts 1-7 in `data_ingestion_pipeline/` 
2. **Grid Search**: Use R scripts in `grid_search/` to identify optimal predictors
3. **Environment Setup**: Create Python environment with required packages (see Environment Setup section)
4. **Model Training**: Use `model/FinalModelStage1Runs.ipynb` or `model/RunFromFunction.ipynb` for model training and forecasting
5. **Model Validation** (Optional): Launch the Shiny app in `shiny_standalone/` for interactive model validation and visualization

## Repository Structure

```
measles_forecasting/
├── data_ingestion_pipeline/   # R scripts for data processing (1-7)
├── grid_search/               # R scripts for predictor selection and metadata generation
├── model/                     # Core Python modules and Jupyter notebooks
├── model_comparison_pipeline/ # Model evaluation and visualization tools
└── shiny_standalone/          # Interactive Shiny web application for model validation
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

## Grid Search Pipeline

The `grid_search/` directory contains R scripts for predictor selection and model metadata generation:

### Scripts Overview
- **`univariate_country_test.R`** - Performs univariate analysis to identify significant predictors by country
- **`create_final_mod_metadata.R`** - Generates metadata for model configurations based on predictor analysis

### Key Outputs
- **`univariate_country_results.csv`** - Results of univariate predictor analysis
- **`correlation_results.csv`** - Correlation analysis between predictors
- **`metadata_example.csv`** - Example metadata format for model configuration

This pipeline helps identify the most relevant predictors for each country before running the full machine learning models.

## Forecasting Models

### Environment Setup

**Recommended**: Use mamba/conda with Python 3.11

```bash
mamba create -n measles_forecasting python=3.11
mamba activate measles_forecasting
mamba install neuralprophet scikit-learn statsmodels jupyterlab pandas numpy \
              geopandas multiprocess matplotlib scipy country_converter seaborn \
              xgboost catboost lightgbm ordpy
```

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

- Reads metadata configurations from `grid_search/metadata_example.csv`
- Uses `fitOne.py` functions for individual model training
- Supports batch processing of multiple model configurations
- Integrates with the grid search pipeline outputs

#### Time Series Evaluation (`TTSEval.ipynb`)
Specialized notebook for time series model evaluation and testing.

#### Advanced Features
- **Hyperparameter optimization** via `EpiAnnealer.py`
- **Seasonality analysis** using `SeasonalityMetrics.py`
- **Custom loss functions** defined in `LossFunctions.py`
- **Multi-algorithm comparison** through `ModelSweeps.py`

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

The `model_comparison_pipeline/` directory contains tools for evaluating and comparing model performance:

- **`model_comparison_viz.Rmd`** - R Markdown document for generating model comparison visualizations
- **`country_output/`** - Individual country-specific model outputs (e.g., NGA.csv, UKR.csv)
- **`summary_output/`** - Aggregated summary tables (summaryTable.csv)
- **Documentation** - Detailed pipeline documentation (PDF)

## Interactive Model Validation (`shiny_standalone/`)

The `shiny_standalone/` directory contains a comprehensive Shiny web application for interactive model validation and visualization:

### Key Features
- **Interactive Country Selection** - Choose from countries with ISO3 codes for detailed analysis
- **Model Performance Metrics** - View detailed performance statistics in interactive data tables
- **Epidemiological Curve Visualization** - Compare observed vs predicted case counts over time
- **Binary Outcome Analysis** - Visualize outbreak prediction accuracy using heatmaps
- **Model Selection & Validation** - Separate analysis for training and validation periods

### Application Structure
- **`ui.R`** - User interface definition with responsive Bootstrap layout
- **`server.R`** - Server logic for data processing and visualization
- **`global.R`** - Global variables, functions, and data loading
- **`data/`** - Sample datasets including cutoff dates and summary tables
- **`www/`** - Static web assets (CSS, images, favicon)
- **Documentation** - Complete user guide (`Measles_Model_Validation_App_Documentation.html`)

### Running the Application
```r
# Install required packages
install.packages(c("shiny", "data.table", "plotly", "ggplot2", "DT", "viridis"))

# Run the application
shiny::runApp("shiny_standalone/")
```

The application provides an intuitive interface for exploring model performance across different countries and time periods, making it easy to validate model predictions and compare performance metrics.

## Documentation

- **Social data README**: `data_ingestion_pipeline/README_Social_Series.txt`
- **Model comparison documentation**: `model_comparison_pipeline/INV-059412_GinkgoBiosecurity_Output 7_Model Comparison Pipeline Documentation_12092024.pdf`
- **Shiny app user guide**: `shiny_standalone/Measles_Model_Validation_App_Documentation.html`

## Notes for Public Use

This repository has been prepared for public use:
- ✅ **Local data processing** - All data sources use local files
- ✅ **Complete environment specification** - Dependencies clearly defined
- ✅ **Comprehensive documentation** - Updated README and comments

Users must run the data ingestion pipeline to generate required input files, as processed data files are not included due to size constraints.

## Citation

When using this code, please cite the associated research and acknowledge the data sources as detailed in the data inventory.

