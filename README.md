# measles_forecasting
AI-enabled measles forecasting

Overview of the steps for running the model:
1. Run scripts in `data_ingestion_pipeline/` (scripts 1-5)
2. Run `data_ingestion_pipeline/` (script 6)
3. Set up python environment
4. Run the model (`alpha_model/RunFromMetadata.ipynb` or `alpha_model/RunAlphaModels.ipynb`)

Details on these steps are found below.

## Data ingestion pipeline
The data ingestion pipeline contains scripts to process raw case and predictor data to a clean, consistent format for model fitting. All processing is done in R. The processed model data are provided, but the pipeline can be used to re-create, or refresh the provided processed dataset. Be advised that the processing of the gridded, temporal climate data is extremely compute and memory heavy. 

The scripts here will recreate `model_training_data.csv`. Scripts 1-5 must be successfully run before running `6_combine_all_datasets.R`. Script `4_social_data_processing.R` has it's own README file `README_Social_Series.txt` because there are several datasets processed within that script. Manual downloading of the raw datasets is necessary, but links to the raw data are provided within the scripts. A data inventory with these links is also present in `other_deliverables/data_inventory/`. You can find additional information on the source of the data, temporal and spatial resolution of the data, license, etc. here as well.

Note that `2_precip_processing.R` and `3_temperature_processing.R` extract and geocode gridded, temporal data (from `.nc` files to data table). This process is compute and memory intensive. Precipitation data is available on a monthly temporal resolution, while temperature data is at a daily resolution. While both datasets take considerable compute time to process, the temperature dataset is especially intensive. The scripts are set up to run parallelized, so be sure to tune these parameters to the specifications of your machine.

## Forecasting model

The forecasting model is run in Python via Jupyter Lab.

### Environment Setup
You'll most likely want to run this in a mamba python 3.11 environment via the miniforge distribution from here: 

[Miniforge Github](https://github.com/conda-forge/miniforge)

*nix users installation:
```
$ curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
$ bash Miniforge3-$(uname)-$(uname -m).sh
```

Once installed, you can open a new terminal tab or run `bash ~/.bashrc` to access mamba. To get started, you'll want to create a mamba environment for this with the required libraries, switch to that environment, and then open jupyter lab. This may be done via the following bash prompts:

```
$ mamba create -n bmgf neuralprophet scikit-learn statsmodels jupyterlab pandas numpy osmnx geopandas multiprocess matplotlib statsmodels scipy country_converter seaborn matplotlib==3.8.3
$ mamba activate bmgf
$ jupyter lab
```
### Basic usage

We provide two applications for running the model: grid search model selection and forward projection. Whichever way you run the analysis, they all use the same data loader, MeaslesDataLoader.py, which loads `model_training_data.csv` unless an alternate file is passed.

### Grid search application
A staged grid search approach was used to find the best-fitting model for each country for the alpha model phase. The [RunFromMetadata.ipynb](alpha_model/RunFromMetadata.ipynb) notebook was used to perform this search. This notebook requires a meta data file that contains all of the specifications for the parameter sets examined in the grid search. The metadata file must include the following fields:

**ROW_ID**: an id corresponding to the metadata row.
model: one of the following names  specifying which ML learning model to use: XGBRegressor, CatBoost, gradient boosting, Random Forest, or Bagging Regressor. Names must match exactly to ensure the dictionary encoded in the model script will recognize and load the correct module.
**predictor**: the predictor variable(s) and predictor lag to evaluate. Must be in the form {‘<predictor>’: <predictor lag}.
**environmentalArg**: environmental predictor variables included in the model. Must also be in the form {‘<predictor>’: <predictor lag}.
**country**: the ISO3 code of the country or the character name of the geography to run the model. Examples of non-country geographies include cluster or unicef region variables in the input data; these will run global-local models of all countries mapping to the geography.
**Seed**: a random seed for reproducibility of the run.

A sample metadata file [run_metadata.csv](alpha_model/run_metadata.csv) is included with an abbreviated list of parameter sets. The grid search is run by calling the fitOne function:  

```
for i in 1:len(meta_df):
    fi.fitOne(metadata = meta_df, ROW = i, run_name = 'sweep_20250106')

```
The function takes the metadata file, a ROW argument (used for subsetting the metadata file, and run_name argument (used for organizing model output in a subdirectory in the output/ folder. 

The run will output summary statistics files to `output/<run_name>/scores/`  in the naming convention `<ROW_ID>_Summary.csv` and model projections to `output/<run_name>/tables/`  in the convention `<ROW_ID>_<country ISO3>_Projection.csv.` Summary statistics files include test and train MSE, MAE, an R2 values, sensitivity, specificity, the ML method, and the ROW_ID from the metadata file. Projection files contain the following fields: ID (the country ISO3 code), ds (the date/timestep), y (the observed measles incidence), yhat1 (the model-projected measles incidence), and ROW_ID (the metadata ROW_ID). All runs will output a summary file, but runs that fail will not write out a projection table. 

### Forward projection application

Once a parameter set is chosen for each country, the forward projection approach will run N number of projection steps past the last observed data present, with the default being 9 months. The [RunAlphaModels.ipynb](alpha_model/RunAlphaModels.ipynb) notebook is used for projection. Projections require `input/alpha_model_by_country.csv`, which specifies the parameter set for each country being projected.

No summary statistics will be output for future projection runs since there will be no observed data to calculate them against. Model projects will be written to `output/tables` in the convention `<ROW_ID>_<country ISO3>_Projection.csv.`

