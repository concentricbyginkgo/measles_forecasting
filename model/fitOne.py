###########################################################################
###   fitOne.PY                                                         ###
###      * RUNS FORECAST MODEL FROM METADATA FILE SPECIFICATIONS        ###
###      * OPTIONAL METADATA: depVar, Seed, environmentalArg,           ###
###        binary_outbreak_threshold_per_m (binary classification)     ###
###      * WRITES FORECAST TABLE AND MODEL PERFORMANCE SCORE            ###
###      * WRITES TO RUN_NAME DIRECTORY IN S3                           ###
###                                                                     ###
###########################################################################

import pandas as pd
import numpy as np
import pickle
import warnings
import hashlib
import pickle
import os
import json
import ast

import MeaslesModelEval as mm
import MeaslesDataLoader as md
import EpiPreprocessor as ep

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor


def _optional_binary_label_metric_kw(metadata, row):
    """
    If metadata specifies an incidence threshold (cases per million), return
    {'binaryLabelMetric': f} with f(x) -> 1 if x >= threshold else 0.
    Otherwise return {} so model wrappers use MeaslesModelEval.defaultBinaryMetric.
    Recognized columns (first match wins): binary_outbreak_threshold_per_m,
    binary_outbreak_threshold.
    """
    for col in ('binary_outbreak_threshold_per_m', 'binary_outbreak_threshold'):
        if col not in metadata.columns:
            continue
        raw = metadata.loc[metadata['ROW_ID'] == row, col].values[0]
        if pd.isna(raw) or str(raw).strip() == '':
            return {}
        try:
            th = float(raw)
        except (ValueError, TypeError):
            return {}
        return {'binaryLabelMetric': (lambda t: (lambda x: x >= t))(th)}
    return {}


def fitOne(metadata, ROW, run_name):
    
    #predictorLag = metadata.loc[metadata['ROW_ID'] == ROW,'predictorLag'].values[0]
    country = metadata.loc[metadata['ROW_ID'] == ROW, 'country'].values[0]
    model_name = metadata.loc[metadata['ROW_ID'] == ROW, 'model'].values[0]
    predictor = metadata.loc[metadata['ROW_ID'] == ROW, 'predictor'].values[0]
    indepVars = ast.literal_eval(metadata.loc[metadata['ROW_ID'] == ROW, 'predictor'].values[0])
    if 'environmentalArg' in metadata.columns:
        env_raw = metadata.loc[metadata['ROW_ID'] == ROW, 'environmentalArg'].values[0]
        if pd.notna(env_raw) and str(env_raw).strip() != '':
            try:
                environmentalArg = ast.literal_eval(str(env_raw).strip())
                if not isinstance(environmentalArg, dict):
                    environmentalArg = {}
            except (SyntaxError, ValueError, TypeError):
                environmentalArg = {}
        else:
            environmentalArg = {}
    else:
        environmentalArg = {}
    if 'Seed' in metadata.columns:
        seed_val = metadata.loc[metadata['ROW_ID'] == ROW, 'Seed'].values[0]
        if pd.notna(seed_val) and str(seed_val).strip() != '':
            try:
                randomState = int(seed_val)
            except (ValueError, TypeError):
                randomState = 1337
        else:
            randomState = 1337
    else:
        randomState = 1337
    meta_Row = metadata.loc[metadata['ROW_ID'] == ROW, 'ROW_ID'].values[0]
    if 'depVar' in metadata.columns:
        dep_raw = metadata.loc[metadata['ROW_ID'] == ROW, 'depVar'].values[0]
        if pd.notna(dep_raw) and str(dep_raw).strip() != '':
            depVar = str(dep_raw).strip()
        else:
            depVar = 'cases_1M'
    else:
        depVar = 'cases_1M'

    binary_kw = _optional_binary_label_metric_kw(metadata, ROW)

    prepArgs = dict()
    #indepVars = {predictor:predictorLag}
    indepVars.update(environmentalArg)
    success = False
    initialized = False
    
    tolerableExceptions = ["Insufficient training or testing data following the application of preprocessor rules.",
                      "Insufficient number of unique, valid measurements of the dependent variable.",
                      "Insufficent test data for analysis."]

    nullResult = {'ID':country, 
                  'method':model_name, 
                  'ROW_ID': meta_Row}
    
    if model_name == 'neural prophet':
        model = mm.npLaggedTTS

    if model_name == 'gradient boosting':
        model = mm.sklGradientBoostingRegression

    if model_name == 'AdaBoost regressor':
        model = {'modelName': 'AdaBoost regressor','model':AdaBoostRegressor}

    if model_name == 'Bagging regressor':
        model = {'modelName': 'Bagging regressor','model':BaggingRegressor}

    if model_name == 'Extra Trees':
          model = {'modelName': 'Extra Trees regressor','model':ExtraTreesRegressor}

    if model_name == 'Random Forest':
        model = {'modelName': 'Random Forest regressor','model':RandomForestRegressor}

    if model_name == 'ElasticNet':
        model = {'modelName': 'ElasticNet','model':ElasticNet}

    if model_name == 'SGD':
        model = {'modelName': 'SGDRegressor','model':SGDRegressor}

    if model_name == 'SVR':
        model = {'modelName': 'SVR','model':SVR}

    if model_name == 'BayesianRidge':
        model = {'modelName': 'BayesianRidge','model':BayesianRidge}

    if model_name == 'KernelRidge':
        model = {'modelName': 'KernelRidge','model':KernelRidge}

    if model_name == 'CatBoost':
        model = {'modelName': 'CatBoostRegressor','model':CatBoostRegressor}

    if model_name == 'Linear regression':
        model = {'modelName': 'LinearRegression','model':LinearRegression}

    if model_name == 'XGBRegressor':
        model = {'modelName': 'XGBRegressor','model':XGBRegressor}

    if model_name == 'LGBMR':
        model = {'modelName': 'LGBMRegressor','model':LGBMRegressor}
    
    # ENSEMBLE MODEL SUPPORT - New functionality
    if model_name in ['diverse', 'diverse low n', 'boosted heavy', 'boosted alpha']:
        model = {'ensembleModels': model_name}  # Special flag for ensemble models
    
    
    try:
        if type(model) is not dict:
            mlRun = model(country,
                          depVar,
                          indepVars = indepVars,
                          randomState = randomState,
                          metaRow = meta_Row,
                          prefix = run_name,
                          **binary_kw)
        elif 'ensembleModels' in model:
            # Handle ensemble models using sklGeneric with ensembleModels parameter
            mlRun = mm.sklGeneric(country,
                                  depVar,
                                  indepVars = indepVars,
                                  ensembleModels = model['ensembleModels'],
                                  randomState = randomState,
                                  metaRow = meta_Row,
                                  prefix = run_name,
                                  **binary_kw)
        elif type(model) is dict:
            # Handle single models using sklGeneric with modelArgs parameter
            mlRun = mm.sklGeneric(country,
                                  depVar,
                                  indepVars = indepVars,
                                  modelArgs = model,
                                  randomState = randomState,
                                  metaRow = meta_Row,
                                  prefix = run_name,
                                  **binary_kw)
        initialized = True

       
        mlRun.train()
        #mlRun.finalDf.to_csv(f'output/tables/{mlRun.hash}.csv',index=False)
        mlRun.export()
        success = True
    
    except Exception as e:
        errorStr = str(e)
        print(errorStr)
        if errorStr in tolerableExceptions:
            print("Tolerable error ignored:",errorStr)
            if initialized:
                result = nullResult
            else:
                return
        else:
            #raise e
            scores_dir = f'output/{run_name}/scores'
            if not os.path.exists(scores_dir):
                os.makedirs(scores_dir, exist_ok=True)           
            result = nullResult
            result = pd.DataFrame(data=result, index=[0])
            result.to_csv(f'{scores_dir}/{meta_Row}_Summary.csv', index=False)

    
