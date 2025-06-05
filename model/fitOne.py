###########################################################################
###   fitOne.PY                                                         ###
###      * RUNS FORECAST MODEL FROM METADATA FILE SPECIFICATIONS        ###
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

def fitOne(metadata, ROW, run_name):
    
    #predictorLag = metadata.loc[metadata['ROW_ID'] == ROW,'predictorLag'].values[0]
    country = metadata.loc[metadata['ROW_ID'] == ROW, 'country'].values[0]
    model_name = metadata.loc[metadata['ROW_ID'] == ROW, 'model'].values[0]
    predictor = metadata.loc[metadata['ROW_ID'] == ROW, 'predictor'].values[0]
    indepVars = ast.literal_eval(metadata.loc[metadata['ROW_ID'] == ROW, 'predictor'].values[0])
    environmentalArg = ast.literal_eval(metadata.loc[metadata['ROW_ID'] == ROW, 'environmentalArg'].values[0])
    randomState = metadata.loc[metadata['ROW_ID'] == ROW, 'Seed'].values[0]
    meta_Row = metadata.loc[metadata['ROW_ID'] == ROW, 'ROW_ID'].values[0]
   
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
                          'cases_1M',
                          indepVars = indepVars,
                          randomState = randomState,
                          metaRow = meta_Row,
                          prefix = run_name)
        elif 'ensembleModels' in model:
            # Handle ensemble models using sklGeneric with ensembleModels parameter
            mlRun = mm.sklGeneric(country,
                                  'cases_1M',
                                  indepVars = indepVars,
                                  ensembleModels = model['ensembleModels'],
                                  randomState = randomState,
                                  metaRow = meta_Row,
                                  prefix = run_name)
        elif type(model) is dict:
            # Handle single models using sklGeneric with modelArgs parameter
            mlRun = mm.sklGeneric(country,
                                  'cases_1M',
                                  indepVars = indepVars,
                                  modelArgs = model,
                                  randomState = randomState,
                                  metaRow = meta_Row,
                                  prefix = run_name)
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
            if not os.path.exists(f'output/{run_name}/scores/'):
                os.makedirs(f'output/{run_name}/scores/')           
            result = nullResult
            result = pd.DataFrame(data=result, index=[0])
            result.to_csv(f'output/{run_name}/scores/{meta_Row}_Summary.csv',index=False)

    
