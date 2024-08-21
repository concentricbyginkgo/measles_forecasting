# measles_forecasting
AI-enabled measles forecasting

## Data ingestion pipeline
The data ingestion pipeline contains scripts to process raw case and predictor data to a clean, consistent format for model fitting. All processing is done in R. The processed model data are provided, but the pipeline can be used to re-create, or refresh the provided processed dataset. Be advised that the processing of the gridded, temporal climate data is extremely compute and memory heavy. 

The scripts here will recreate `model_training_data.csv`. Scripts 1-5 must be successfully run before running `6_combine_all_datasets.R`. Script `4_social_data_processing.R` has it's own README file `README_Social_Series.txt` because there are several datasets processed within that script. Manual downloading of the raw datasets is necessary, but links to the raw data are provided within the scripts. A data inventory with these links is also present in `other_deliverables/data_inventory/`. You can find additional information on the source of the data, temporal and spatial resolution of the data, license, etc. here as well.

Note that `2_precip_processing.R` and `3_temperature_processing.R` extract and geocode gridded, temporal data (from `.nc` files to data table). This process is compute and memory intensive. Precipitation data is available on a monthly temporal resolution, while temperature data is at a daily resolution. While both datasets take considerable compute time to process, the temperature dataset is especially intensive. The scripts are set up to run parallelized, so be sure to tune these parameters to the specifications of your machine.

## Forecasting model

The forecasting model is run in Python.

### Environment Setup
You'll most likely want to run this in a mamba python 3.11 environment via the miniforge distribution from here: 

[Miniforge Github](https://github.com/conda-forge/miniforge)

*nix users:
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

The method has been tested on OSX:x86 and OSX:arm64.

### Basic Usage
There are presently two ways you can run this model, neither of which is necessarily better depending on your goals. Both methods withhold the last twelve months of case and predictor data, project the predictor data forward, estimate the case data from it, then evaluate the results.

### Notebook Usage
If you run analyses within the analysis notebooks, you can access all of the functions of the wrapper class and tweak and tune runs as desired. The only substantive difference between the notebooks at present is how the code is partitioned, so the "WithCode" notebook would be useful if you want to look and understand how things work or even poke around a bit in a temporary copy.

The notebook approach would be specifically useful if there were any additional Scikit-Learn modules you'd want to evaluate. The caveat of this method is that as many of those Scikit-Learn modules are the product of granular research projects, version conflicts and similar edge cases within them may crash the notebook's python kernel. This doesn't cause any significant problems outside of the analysis, but it may break set and forget usage for large sweeps. The result of each analysis is a python dict, analogous to a json. It is trivial to bundle these and save locally like so:

```
import pandas as pd

results = []
for analysis in analyses:
   result = fx(analysis)
   results.append(result)

resultsDf = pd.DataFrame(results)
resultsDf.to_csv("results.csv")
```

### CLI Usage
You can also run these experiments via the CLI, in which case the Scikit-Learn generic class is inaccessible for lack of a clean way to pass generic library references. The CLI has to load some fairly heavy dependencies at each run, so that will add about 5-10 seconds of delay per iteration. On the other hand, if you're running these CLI analyses with a bash script, R, or another method, a crashed analysis shouldn't crash bash. The results of these will be a singular experiment output json saved locally at a file you set for each analysis.

Usage is like so:

```
% python MeaslesModelEval.py nplagged Nigeria cases_1M "{'total_precip_mm_per_day':9}" test.json
82/194 included countries found with noted outbreaks.

Initializing...
Training...
Finding best initial lr: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 209/209 [00:00<00:00, 344.39it/s]
Epoch 220: 100%|██████████████████████████████████████████████| 220/220 [00:00<00:00, 7244.73it/s, loss=0.0579, v_num=5212, MAE=4.250, RMSE=8.400, Loss=0.0548, RegLoss=0.000]
Predicting DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 757.50it/s]
Evaluating...
Task complete! Results written to: test.json

Results:
{'Test MAE': 3.8484136526996853,
 'Test MSE': 20.164550994626428,
 'Test R2': -0.8553428376157688,
 'Train MAE': 4.370380609797452,
 'Train MSE': 119.81935461450068,
 'Train R2': 0.2678163799086263,
 'depVar': 'cases_1M',
 'indepVars': {'total_precip_mm_per_day': 9},
 'method': 'NeuralProphet lagged regressors',
 'model args': '{}',
 'predictor projection': 'NeuralProphet autoregression',
 'random state': 1337,
 'withheld': 12}
```

Whichever way you run an analysis, they all use the same data loader, MeaslesDataLoader.py, which loads `model_training_data.csv`.

### Plumbing Considerations
Some of these ML models are a bit slow to train. Likewise, NeuralProphet is at present used in every model to forecast predictor variables into the future no matter which ML wrapper class you use for the case data. This makes a fair bit of sense for environmental predictors like rain and temperature, less so for anything particularly social, granular, or stochastic. In either case, for any unique experiment you run, the unique setup variables and raw data of that experiment are hashed as part of the memoization framework. The code dictating this process is as follows:


```
self.hash = hashIt((self.curve.to_csv(),
		    testSize,
		    randomState,
		    self.features,
		    self.method,
		    self.projection,
		    self.modelArgs))

```
For each predictor projection or model training run, all of the pertinent class variables will be pickled and stored to a file such as store/{self.hash}.pkl. However, if you change the code after that hash or change library versions, you may induce errors and inconsistencies here. In that case, simply delete the contents of store/ and rerun the experiments. You could even identify the hash from a specific initiated TTS run and delete runs a la carte. Each pickled ML training model is typically 1-4MB, so space could potentially become any issue with larger experiments.

If you are interacting with the code directly in python, you can access and experiment with any of the data within the class object, denoted by self.{attribute}. There is also a method in the notebooks for plotting trained TTS runs. Usage and output is as follows:

```
neuralRun = mm.npLaggedTTS('NGA',
                          'cases_1M',
                          indepVars = {'total_precip_mm_per_day':3},
                          testSize = 12)

neuralRun.train()
mm.plotTTS(neuralRun)

```

