###########################################################################
###   MEASLESMODELEVAL.PY                                               ###
###      * ML WRAPPER MODULE FOR STANDARDIZED EVALUATIONS               ###
###      * INTEGRATES NEURALPROPHET AND SKL METHODS                     ###
###      * CAN TAKE GENERIC SKL CLASSES IF PASSED IN PYTHON             ###
###      * USES AUTOETS TO FORWARD PROJECT SEASONAL PREDICTORS          ###
###      * IMPLEMENTS GLOBAL-LOCAL MODELLING (MULTI GEOGRAPHY)          ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


import pandas as pd
import numpy as np
import pickle

import seaborn as sb

sb.set_context("poster")
sb.set_theme()

import warnings
import hashlib
import pickle
import os
import json

from matplotlib import pyplot as plt
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, recall_score, confusion_matrix

from sklearn import ensemble
from sklearn import linear_model
from sklearn import neural_network
from sklearn import svm

from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsmodels.tsa.seasonal import seasonal_decompose

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from copy import deepcopy
from datetime import datetime

import MeaslesDataLoader as md
import EpiPreprocessor as ep

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")

preppedCountries = md.prepData()

expectedDirectories = ['input',
                       'output/figures',
                       'store']

for directory in expectedDirectories:
    if not os.path.exists(directory):
        os.makedirs(directory)

minAcceptableTotalMonths = 12
minAcceptableTestMonths = 6
defaultMissingVarResponse = 'drop country' #value may equal "drop country", "drop var", or "ignore"
defaultProjectionMethod = 'AutoETS'
defaultBinaryMetric = lambda x: x >= 2
groupingVar = 'ISO3'





##########################################
###   DATA MANAGEMENT FUNCTIONS        ###
##########################################


def setCutoffSize(simObject,
                  df,
                  cutoff,
                  testSize):
    """Given a df and a cutoff date, returns the cutoff size"""
    if simObject.monthsForward != 0:
        return 0

    if cutoff != 'not passed':
        testSize = len(df[df.ds >= cutoff])


    if testSize < minAcceptableTestMonths:
        raise Exception("Insufficent test data for analysis.")

    return testSize



def hashIt(var):
    """Returns an md5 hash from an arbitrary variable"""
    varStr = f'{var}'.encode('utf-8')
    md5_hash_obj = hashlib.md5()
    md5_hash_obj.update(varStr)
    return md5_hash_obj.hexdigest()


def hashObject(obj):
    """Hashes arbitrary memory objects"""
    try:
        data = pickle.dumps(obj)
        return hashlib.sha256(data).hexdigest()
    except Exception as e:
        raise TypeError(f"Object is not picklable: {e}")



def sortDict(dictIn):
    """Returns dict contents in predictable order"""
    return dict(sorted(dictIn.items()))



def getCutoffDate(country):
    """Gets the flex cutoff for a given country, returning "not passed" if one is not available""" 
    try:
        cutoff = preppedCountries['cutoffs'][country]
    except:
        cutoff = 'not passed'

    return cutoff


############################################################
###  CROSS MODEL INIT FUNCTIONS                          ###
############################################################


def prepCurve(simObject,
              dfIn,
              cutoff,
              testSizeIn,
              additionalPrep,
              country):
    """Prepares one curve df, paring out all unnecessary attributes"""
    
    # Loads preprocessor config and merges manual args
    df = dfIn.copy(True)

    # Set & verify cutoff size for initial data load quality checks
    testSize = setCutoffSize(simObject,
                             df,
                             cutoff,
                             testSizeIn)

    # Create local copy of indepVars for per-experiment modification
    indepVars = deepcopy(simObject.indepVars)
    
    if len(df[simObject.depVar].dropna()) <= max(testSize,minAcceptableTotalMonths): #or df[depVar].nunique() < testSize*2:
        raise Exception("Insufficient number of unique, valid measurements of the dependent variable.")
    
    # Pare down df to only needed and present vars
    indepVars = {key:duration for key,duration in indepVars.items() if key in df.columns}
    indepVars = {key:duration for key,duration in indepVars.items() if len(df[key].dropna()) > 0}
    df = df[['ds',simObject.depVar]+list(indepVars.keys())]

    # Apply preprocessor
    for column,methods in additionalPrep.items():
        try:
            config[column] += methods
        except:
            config[column] = methods

    df,preprocessorLog = ep.preprocessDf(df,simObject.preprocessor)
    simObject.unshifted[country] = df.copy(deep=True)
    df.to_csv("shifted_temp.csv")

    
    # Apply lagged regressors
    if simObject.method != 'NeuralProphet lagged regressors':
        for key,duration in indepVars.items():
            df.loc[:,key] = df[key].shift(duration).tolist()

    df = df.dropna()

    # Set & verify cutoff size after preprocessor application
    testSize = setCutoffSize(simObject,
                             df,
                             cutoff,
                             testSize)


    if len(df) <= max(min(simObject.testStatsWindow,testSize)*2,minAcceptableTotalMonths):
        raise Exception("Insufficient training or testing data following the application of preprocessor rules.")

    # Drop indepvars that do not have multiple values for single country fits
    if len(simObject.countries) == 1:
        indepVars = {key:duration for key,duration in indepVars.items() if (df[key][:-testSize].nunique() > 1 or simObject.monthsForward !=0)}
        df = df[['ds',simObject.depVar]+list(indepVars.keys())]


    df.rename({simObject.depVar:'y'},
               axis=1,
               inplace = True)

    return df, preprocessorLog, testSize



def prepCurves(simObject,
               additionalPrep):
    """Loads & preps multiple curve objects"""
    
    curves = dict()
    dropped = []
    for country in simObject.countries:
        # Load individual country
        curve = preppedCountries['curves'][country].copy(deep=True)
        
        # Get cutoff date
        cutoff = getCutoffDate(country)

        if cutoff != cutoff:
            raise Exception("NaT Cutoff date indicates non-viable experiment.")

        try:
            #P reps country curve
            prepped, log, testSize = prepCurve(simObject,
                                               curve,
                                               cutoff,
                                               simObject.testSize,
                                               additionalPrep,
                                               country)
    
            # Tracks runnable variables for the country curve
            validVars = set(prepped.columns)
    
            # Merge data structure
            curves[country] = {'curve':curve,
                               'cutoff':cutoff,
                               'prepped':prepped,
                               'preprocessor log':log,
                               'testSize':testSize}
        except Exception as e:
            dropped.append(('country',country,str(e)))

    if len(curves) == 0:
        raise Exception("All countries pared during curve preparation, experiment has no data.")
        
    simObject.countries = list(curves.keys()) 
    simObject.dropLog += dropped
    simObject.curves = curves


def pareIndepVars(simObject):
    """Removes indepVars that did not exist for every data set"""
    
    keptKeys = set(['ds','y'] + list(simObject.indepVars.keys()))
    initialKeys = deepcopy(keptKeys)
    multipleCurvesLoaded = len(simObject.curves) > 1

    if simObject.missingVarResponse == 'drop country' and multipleCurvesLoaded:
        dropped = []
        for country,data in simObject.curves.items():
            # Identify countries missing required columns, drop them, and log it
            if initialKeys > set(data['prepped'].columns):
                dropped.append(('country',country,"Country dropped due to missing var"))
        for _,country,_ in dropped:
            del simObject.curves[country]
        simObject.countries = list(simObject.curves.keys())

    elif simObject.missingVarResponse == 'drop var' or not multipleCurvesLoaded:
        # Identify the set of variables which exists for all countries
        for country,data in simObject.curves.items():
            keptKeys = keptKeys.intersection(set(data['prepped'].columns))

        # Modify indepVars to reflect only those variables
        simObject.indepVars = {var:delay for var,delay in simObject.indepVars.items() if var in keptKeys}

        # Update prepped curves and hashes
        for country,data in simObject.curves.items():
            data['prepped'] = data['prepped'][list(keptKeys)].copy(deep=True)
            data['indepVars'] = deepcopy(simObject.indepVars)
    
        # Update the dropped log
        dropped = [('var',var,"Var not available in prepped data") for var in initialKeys if var not in keptKeys]


    elif simObject.missingVarResponse == 'ignore':
        dropped = []

    simObject.dropLog += dropped



def prepTTSSets(simObject):
    """Generates TTS objects for multiple data sets"""
    
    ttsSets = dict()
    for country, countryData in simObject.curves.items():
        if simObject.monthsForward == 0:
            endTrim = max(countryData['testSize'] - simObject.testStatsWindow, 0)
            testSize = min(countryData['testSize'], simObject.testStatsWindow)

            if endTrim != 0:            
                trainDf, testDf = train_test_split(countryData['prepped'][:-endTrim],
                                                   shuffle = False,
                                                   test_size = testSize)
            else:
                trainDf, testDf = train_test_split(countryData['prepped'],
                                   shuffle = False,
                                   test_size = testSize)
            
            yTrain = trainDf['y'].values
            yTest = testDf['y'].values
    
            yTestPred = None
            yTrainPred = None
            results = None
    
            ttsSets[country] = {'trainDf':trainDf,
                                'testDf':testDf,
                                'yTrain':yTrain,
                                'yTest':yTest,
                                'yTestPred':None,
                                'yTrainPred':None}
        else:
            trainDf = countryData['prepped']
            yTrain = trainDf['y'].values
            
            ttsSets[country] = {'trainDf':trainDf,
                                'yTrain':yTrain,
                                'yTrainPred':None}
            
    
    simObject.TTSData = ttsSets



def initModel(simObject,
              selection,
              depVar,
              indepVars,
              projectionMethod,
              testSize,
              randomState,
              preprocessor,
              additionalPrep,
              modelArgs,
              missingVarResponse,
              prefix,
              binaryLabelMetric,
              useCache,
              testStatsWindow,
              monthsForward,
              fuzzReplicates,
              fuzzStd):
    """Generalized model initiation across wrappers"""
    
    if monthsForward != 0:
        testSize = 0
        testStatsWindow = 0

    indepVars = sortDict(indepVars)
    validCountries = [country for country in preppedCountries['filters'][selection] if country in preppedCountries['curves'].keys()]
    simObject.initialCountryCount = len(validCountries)
    
    simObject.selection = selection
    simObject.countries = validCountries
    simObject.depVar = depVar
    simObject.indepVars = indepVars
    simObject.testSize = testSize
    simObject.missingVarResponse = missingVarResponse
    simObject.projection = projectionMethod
    simObject.preprocessor = ep.getGoogleSheetConfig(preprocessor)
    simObject.dropLog = []
    simObject.binaryLabeller = binaryLabelMetric
    simObject.useCache = useCache
    simObject.testStatsWindow = testStatsWindow
    simObject.monthsForward = monthsForward
    simObject.fuzzReplicates = fuzzReplicates
    simObject.fuzzStd = fuzzStd
    simObject.unshifted = dict()

    if prefix.endswith('/'):
        if not os.path.exists(f'output/tables/{prefix}'):
            os.makedirs(f'output/tables/{prefix}')
    elif prefix != '' and not prefix.endswith('_'):
        prefix += '_'

    simObject.prefix = prefix

    prepCurves(simObject,
               additionalPrep)

    simObject.multipleCurves = len(simObject.curves) > 1

    pareIndepVars(simObject)
    
    if simObject.dropLog != []:
        print(f'Dropped {simObject.dropLog}')
    
    simObject.modelArgs = modelArgs
    
    if randomState == "stochastic":
        randomState += ' ' + hashIt(os.urandom(32))[:10]
    simObject.randomState = randomState

    prepTTSSets(simObject)
    getMergedFutures(simObject)
    
    simObject.hash = hashIt([simObject.randomState,
                             simObject.indepVars,
                             simObject.method,
                             simObject.projection,
                             simObject.modelArgs,
                             simObject.mergedFutures.to_csv(index=False),
                             simObject.monthsForward,
                             simObject.testStatsWindow,
                             simObject.fuzzReplicates,
                             simObject.fuzzStd])

    
    simObject.varKeys = sorted(list(simObject.indepVars.keys()))
    simObject.mergedTrainDf = simObject.mergedFutures.dropna(subset=['y'])
    simObject.xTrain = simObject.mergedTrainDf[simObject.varKeys].values
    simObject.yTrain = simObject.mergedTrainDf['y'].values
    simObject.trained = None



############################################################
###  CROSS MODEL TRAINING FUNCTIONS                      ###
############################################################



def projectPredictor(country, var, delay, simObject):
    """
    Projects a single predictor n periods into the future.

    Assumptions:
    - simObject.TTSData[country]['trainDf'] contains the training (historical) ds,var.
    - simObject.unshifted[country][var] contains the regressor values *before* lag shift,
      so the last `lag` entries there are used for the forward-aligned stub.
    """

    # Determine lag and horizons
    lag = simObject.indepVars[var] if simObject.projection != 'NeuralProphet autoregression' else 0
    testSize = simObject.curves[country]['testSize']
    monthsForward = simObject.monthsForward
    periods = int(testSize) + int(monthsForward)
    if periods == 0:
        return pd.DataFrame(columns=['ds', var])

    # Training dataframe (historical only)
    df_train = simObject.TTSData[country]['trainDf'][['ds', var]].copy()
    df_train.columns = ['ds', 'y']
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    last_train_date = df_train['ds'].max()

    # How many stub rows go at the head of the projection window?
    n_stub = min(int(lag), periods)
    postLag = max(0, periods - n_stub)

    # uild forward-aligned stub using UNshifted regressor
    forwardAlignedStub = None
    if n_stub > 0:
        unshifted_series = simObject.unshifted[country][var].reset_index(drop=True)

        # Take last n_stub values (pad with NaN if too short)
        if len(unshifted_series) >= n_stub:
            vals = unshifted_series.iloc[-n_stub:].reset_index(drop=True)
        else:
            pad = pd.Series([np.nan] * (n_stub - len(unshifted_series)))
            vals = pd.concat([pad, unshifted_series], ignore_index=True)

        stub_dates = [last_train_date + pd.DateOffset(months=i) for i in range(1, n_stub + 1)]
        forwardAlignedStub = pd.DataFrame({'ds': stub_dates, var: vals.values})

    # Forecast remaining months (postLag)
    if postLag == 0:
        forecast = pd.DataFrame(columns=['ds', var])
    else:
        if simObject.projection == 'AutoETS':
            autoETSInstance = AutoETS(season_length=12)
            df_sf = df_train.copy()
            df_sf['unique_id'] = 'series'
            sf = StatsForecast(df=df_sf,
                               models=[autoETSInstance],
                               freq='MS',
                               n_jobs=-1)
            sf.fit()
            pred = sf.predict(h=postLag).reset_index()
            pred = pred[['ds', 'AutoETS']].rename(columns={'AutoETS': var})
            if n_stub > 0:
                pred['ds'] = pred['ds'] + pd.DateOffset(months=n_stub)
            forecast = pred

        elif simObject.projection == 'NeuralProphet autoregression':
            if not str(simObject.randomState).startswith('stochastic'):
                set_random_seed(simObject.randomState)
            model = NeuralProphet()
            model.fit(df_train, freq='M')
            future = model.make_future_dataframe(df_train, periods=postLag, n_historic_predictions=False)
            fc = model.predict(future)
            fc = fc[pd.to_datetime(fc['ds']) > last_train_date].copy().reset_index(drop=True)
            if 'yhat1' in fc.columns:
                pred = fc[['ds', 'yhat1']].rename(columns={'yhat1': var})
            elif 'y' in fc.columns:
                pred = fc[['ds', 'y']].rename(columns={'y': var})
            else:
                raise RuntimeError("NeuralProphet predict output lacks y/yhat1 column.")
            if n_stub > 0:
                pred['ds'] = pd.to_datetime(pred['ds']) + pd.DateOffset(months=n_stub)
            forecast = pred

        else:
            raise ValueError(f"Unknown projection method: {simObject.projection}")

    # Assemble final result
    parts = []
    if forwardAlignedStub is not None and not forwardAlignedStub.empty:
        parts.append(forwardAlignedStub[['ds', var]])
    if not forecast.empty:
        forecast['ds'] = pd.to_datetime(forecast['ds'])
        parts.append(forecast[['ds', var]])

    if parts:
        result = pd.concat(parts, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['ds', var])

    # Ensure exactly `periods` rows (pad if needed)
    result = result.sort_values('ds').reset_index(drop=True)
    if len(result) < periods:
        needed_dates = [last_train_date + pd.DateOffset(months=i) for i in range(1, periods + 1)]
        existing = set(pd.to_datetime(result['ds']))
        missing = [d for d in needed_dates if d not in existing]
        pad_df = pd.DataFrame({'ds': missing, var: [np.nan] * len(missing)})
        result = pd.concat([result, pad_df], ignore_index=True).sort_values('ds').reset_index(drop=True)
    else:
        result = result.iloc[:periods].reset_index(drop=True)

    return result

    

def getFutureDf(simObject,country):
    """Projects the predictors used in training a sim for a single country"""

    toMerge = pd.DataFrame(columns=['ds'])
    testSize = simObject.curves[country]['testSize']
    trainDf = simObject.TTSData[country]['trainDf'].copy(deep=True)
    
    if simObject.method.startswith('NeuralProphet') and simObject.indepVars == dict():
        future = model.make_future_dataframe(trainDf,
                                             periods = testSize + simObject.monthsForward,
                                             n_historic_predictions=True)
    
    elif simObject.method.startswith('Scikit-learn') and simObject.indepVars == dict() and False:
        raise ValueError("Model cannot be trained without one or more independent variables.")
    
    else:
        future = trainDf
        for indepVar,delay in simObject.indepVars.items():
            predictor = projectPredictor(country,
                                         indepVar,
                                         delay,
                                         simObject)
            
            toMerge = toMerge.merge(predictor,how='outer')


    future = pd.concat([future,toMerge],
                       axis=0,
                       ignore_index=True)

    return future



def mergeCurves(simObject):
    """Creates a merged curve, retaining ID as country"""
    
    toMerge = []
    for country,data in simObject.results.items():
        countryCurve = data['future'].copy(deep=True)
        countryCurve.loc[:,'ID'] = country
        toMerge.append(countryCurve)

    mergedFutures = pd.concat(toMerge,axis=0)
    if not simObject.multipleCurves or simObject.method == 'NeuralProphet lagged regressors': 
        encodingMethod = 'pass_unchanged'
    else:
        # Encode country or other grouping variables to numeric types
        mergedFutures, newColumns, encodingMethod = ep.encodeMergedDf(mergedFutures,
                                                                      simObject.preprocessor,
                                                                      simObject.method)
        # Add encoded grouping variables to features
        for newColumn in newColumns:
            simObject.indepVars[newColumn] = 0
            
    simObject.encodingMethod = encodingMethod
    simObject.mergedFutures = mergedFutures



def getMergedFutures(simObject):
    """Gets and merges all future Dfs"""

    simObject.results = dict()
    
    for country in simObject.countries:
        future = getFutureDf(simObject,country)
        simObject.results[country] = {'future':future}

    mergeCurves(simObject)




def processResults(simObject,
                   forecast,
                   model,
                   metrics = None):
    """Parses data from a forecast to a results object."""

    result = {'model':model,
              'metrics':metrics,
              'forecast':forecast,
              'plotDfs':dict()}
               
    for country,countryForecast in forecast.groupby('ID'):
        result['plotDfs'][country] = standardizeOutput(simObject,
                                                       countryForecast,
                                                       country)
    return result



def updateTTSData(simObject):
    """Updates TTS data with completed run data"""
    
    forecast = simObject.trained['forecast']
    for country,countryForecast in forecast.groupby('ID'):
        testSize = simObject.curves[country]['testSize']
        simObject.TTSData[country]['yTestPred'] = countryForecast['yhat1'][-testSize:].values
        simObject.TTSData[country]['yTrainPred'] = countryForecast['yhat1'][:-testSize].values


def standardizeOutput(simObject,
                      forecast,
                      country):
    """Returns one output table format for all wrappers"""
    
    forecast = forecast.drop(['y'],axis=1)
    forecast = forecast.merge(simObject.curves[country]['prepped'][['ds','y']],
                             how = 'left')
        
    return forecast



def trainSKLearn(simObject, model):
    """Sklearn training handler incooporating fuzzing"""
    if simObject.fuzzReplicates == 0:
        model.fit(simObject.xTrain, simObject.yTrain)
                 
    else:
        rng = np.random.default_rng(abs(simObject.randomState))
    
        # Determine binary columns (only two unique values)
        binaryMask = np.array([len(np.unique(col)) == 2 for col in simObject.xTrain.T])
        floatMask = ~binaryMask
    
        xReplicates = [simObject.xTrain]
        yReplicates = [simObject.yTrain]
    
        for i in range(simObject.fuzzReplicates):
            xFuzzed = simObject.xTrain.copy()
    
            # Fuzz floats
            if floatMask.any():
                floatCols = simObject.xTrain[:, floatMask]
                base = np.maximum(np.abs(floatCols), 1e-8)
                noise = rng.normal(0, simObject.fuzzStd, size=floatCols.shape) * base
                xFuzzed[:, floatMask] += noise

            # Fuzz binaries (flip with some prob)
            if binaryMask.any():
                flips = rng.random(size=simObject.xTrain[:, binaryMask].shape) < simObject.fuzzStd
                xFuzzed[:, binaryMask] = np.logical_xor(simObject.xTrain[:, binaryMask].astype(bool), flips).astype(int)
    
            xReplicates.append(xFuzzed)
            yReplicates.append(simObject.yTrain)
    
        xTrainFuzzed = np.vstack(xReplicates)
        yTrainFuzzed = np.concatenate(yReplicates)

        model.fit(xTrainFuzzed,yTrainFuzzed)


############################################################
###  CROSS MODEL EVALUATION FUNCTIONS                    ###
############################################################


def alignEvalData(original,predicted):
    """Takes two iterables of the same length and drops rows where either is missing a value"""

    merged = pd.DataFrame({'original':original,
                           'predicted':predicted}).dropna()
    
    return merged.original, merged.predicted



def evaluateCountry(simObject,country):
    """Evaluates model performance for a single country from generic TTS parameters"""

    if simObject.monthsForward == 0:    
        countryTTS = simObject.TTSData[country]
        testSize = simObject.curves[country]['testSize']
        testWindow = min(testSize,simObject.testStatsWindow)

        yTest,yTestPred = alignEvalData(countryTTS['yTest'],countryTTS['yTestPred'][:len(countryTTS['yTest'])])
        yTrain,yTrainPred = alignEvalData(countryTTS['yTrain'],countryTTS['yTrainPred'])
    
        yTest = yTest[:testWindow]
        yTestPred = yTestPred[:testWindow]
    
        mseTest = mean_squared_error(yTest, yTestPred)
        maeTest = mean_absolute_error(yTest, yTestPred) 
        r2Test = r2_score(yTest, yTestPred)
        mseTrain = mean_squared_error(yTrain, yTrainPred)
        maeTrain = mean_absolute_error(yTrain, yTrainPred)
        r2Train = r2_score(yTrain, yTrainPred)
    
        yTestBin = yTest.apply(simObject.binaryLabeller).astype(int)
        yTestPredBin = yTestPred.apply(simObject.binaryLabeller).astype(int)
        confMatrix = confusion_matrix(yTestBin, yTestPredBin, labels=[0, 1])
    
        if confMatrix.shape == (1, 1):
            tn, fp, fn, tp = (confMatrix[0, 0], 0, 0, 0) if yTestBin.sum() == 0 else (0, 0, 0, confMatrix[0, 0])
        elif confMatrix.shape == (1, 2):
            tn, fp, fn, tp = (confMatrix[0, 0], confMatrix[0, 1], 0, 0)
        elif confMatrix.shape == (2, 1):
            tn, fp, fn, tp = (confMatrix[0, 0], 0, confMatrix[1, 0], 0)
        else:
            tn, fp, fn, tp = confMatrix.ravel()
    
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = f1_score(yTestBin, yTestPredBin) if (tp > 0) else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
        result = {'Test MSE':mseTest,
                  'Test MAE':maeTest,
                  'Test R2':r2Test,
                  'Train MSE':mseTrain,
                  'Train MAE':maeTrain,
                  'Train R2':r2Train,
                  'Sensitivity':sensitivity,
                  'Specificity':specificity,
                  'F1 Score':f1,
                  'NPV':npv,
                  'True positives':tp,
                  'True negatives':tn,
                  'False positives':fp,
                  'False negatives':fn}
    else:
        result = dict()
        testWindow = 0

    modelParams = {'method':simObject.method,
                   'geography':simObject.selection,
                   'global-local':simObject.initialCountryCount != 1,
                   'predictor projection':simObject.projection,
                   'depVar':simObject.depVar,
                   'indepVars':simObject.indepVars,
                   'withheld':testWindow,
                   'random state':simObject.randomState,
                   'model args':str(simObject.modelArgs),
                   'fuzz replicates':simObject.fuzzReplicates,
                   'fuzz std':simObject.fuzzStd}
    
    result.update(modelParams)
    
    return result



def evaluateModel(simObject):
    """Evaluates model performance for all countries within a given experiment"""

    results = dict()
    for country in simObject.countries:
        result = evaluateCountry(simObject,country)
        results[country] = result

    results = pd.DataFrame(results).T
    indepVars = results['indepVars'].apply(pd.Series)
    results = pd.concat([results.drop(['indepVars'],axis=1),indepVars],
                       axis=1)
    results.loc[:,'initial country count'] = simObject.initialCountryCount
    results.loc[:,'runnable country count'] = len(simObject.countries)
    
    return results



def exportTables(simObject, directory = 'output/tables'):
    """Exports prepped data csvs from a trained run"""

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    hashStr = simObject.hash
    for country, table in simObject.trained['plotDfs'].items():
        fileOut = f'{directory}/{simObject.prefix}{hashStr}_{country}_Projection.csv'
        table.to_csv(fileOut,index=False)
        
    summaryRef = f'{directory}/{simObject.prefix}{hashStr}_Summary.csv'
    evaluateModel(simObject).to_csv(summaryRef)
    

def plotTTS(simObject):
    """Quick plotting function for TTS objects"""

    forecast = simObject.finalDf.copy(deep=True)
    withheld = simObject.curve[-simObject.testSize:].copy(deep=True)
    withheld.rename({'y':'withheld'},
                    axis = 1,
                    inplace = True)
    forecast.rename({'y':'training data','yhat1':'prediction'},
                    axis = 1,
                    inplace = True)
    
    fig, ax = plt.subplots(figsize=(8,6))
    forecast.plot(x = 'ds',
                  y = ['training data','prediction'],
                  alpha = .5,
                  ax = ax)
    withheld.plot(x = 'ds',
                  y = 'withheld',
                  alpha = .5,
                  ax = ax,)
    plt.xlabel("Month/Year")
    plt.ylabel(simObject.depVar)
    plt.title(f"{simObject.country} {simObject.depVar} vs predictors {simObject.indepVars}\nMethod: {simObject.method}")
    plt.savefig(f'output/figures/{simObject.hash}.png')


    
############################################################
###  NEURAL PROPHET LAGGED REGRESSOR TTS CLASS           ###
############################################################



class npLaggedTTS:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 prefix = '',
                 projectionMethod = defaultProjectionMethod,
                 missingVarResponse = defaultMissingVarResponse,
                 testSize = 12,
                 randomState = 1337,
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict(),
                 binaryLabelMetric = defaultBinaryMetric,
                 useCache = False,
                 testStatsWindow = 9,
                 monthsForward = 0,
                 fuzzReplicates = 0,
                 fuzzStd = 0.01):
        """
        Initialize the model parameters
        """

        self.method = 'NeuralProphet lagged regressors'

        initModel(self,
                  country,
                  depVar,
                  indepVars,
                  projectionMethod,
                  testSize,
                  randomState,
                  preprocessor,
                  additionalPrep,
                  dict(),
                  missingVarResponse,
                  prefix,
                  binaryLabelMetric,
                  useCache,
                  testStatsWindow,
                  monthsForward,
                  0,
                  0)


    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        if not str(self.randomState).startswith('stochastic'):
            set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile) and self.useCache:
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            if self.multipleCurves:
                model = NeuralProphet(trend_global_local="global",
                                      season_global_local="local")
            else:
                model = NeuralProphet()
            
            for indepVar,delay in self.indepVars.items():
                if delay != 0:
                    model.add_lagged_regressor(indepVar, n_lags=delay)
                else:
                    model.add_future_regressor(indepVar)

            metrics = model.fit(self.mergedFutures.dropna())
            forecast = model.predict(self.mergedFutures)
            result = processResults(self,
                                    forecast,
                                    model,
                                    metrics=metrics)
            
            if self.useCache:
                with open(cacheFile, 'wb') as fileOut:
                    pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result
        updateTTSData(self)


    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        return results

    
    def export(self):
        """
        Exports country tables
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")
            
        exportTables(self)


        
############################################################
###  SCIKIT LEARN GRADIENT BOOSTING REGRESSION TREES TTS ###
############################################################



class sklGradientBoostingRegression:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 prefix = '',
                 projectionMethod = defaultProjectionMethod,
                 missingVarResponse = defaultMissingVarResponse,
                 testSize = 12,
                 randomState = 1337,
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict(),
                 binaryLabelMetric = defaultBinaryMetric,
                 useCache = False,
                 testStatsWindow = 9,
                 monthsForward = 0,
                 fuzzReplicates = 0,
                 fuzzStd = 0.01):
        """
        Initialize the model parameters
        """
        
        self.method = 'Scikit-learn gradient boosted regression'

        initModel(self,
                  country,
                  depVar,
                  indepVars,
                  projectionMethod,
                  testSize,
                  randomState,
                  preprocessor,
                  additionalPrep,
                  dict(),
                  missingVarResponse,
                  prefix,
                  binaryLabelMetric,
                  useCache,
                  testStatsWindow,
                  monthsForward,
                  fuzzReplicates,
                  fuzzStd)
    
    
    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        if not str(self.randomState).startswith('stochastic'):
            set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile) and self.useCache:
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            if not str(self.randomState).startswith('stochastic'):
                model = ensemble.GradientBoostingRegressor(random_state=self.randomState)
            else:
                model = ensemble.GradientBoostingRegressor()

            trainSKLearn(self,model)
                         
            forecast = self.mergedFutures.copy(deep=True)
            forecast.loc[:,'yhat1'] = model.predict(self.mergedFutures[self.varKeys].values)
            
            result = processResults(self,
                                    forecast,
                                    model)
            
            if self.useCache:
                with open(cacheFile, 'wb') as fileOut:
                    pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result
        updateTTSData(self)


    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        return results

    
    def export(self):
        """
        Exports country tables
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")
            
        exportTables(self)
        


############################################################
###  SCIKIT LEARN ENSEMBLER PREP                         ###
############################################################



def prepEnsemble(models = 'diverse',
                 randomState = 1337,
                 estimators = 100,
                 stackingModelName = 'stacking regressor',
                 finalEstimatorName = 'ridgeCV',
                 passThrough = True,
                 tscv = TimeSeriesSplit(n_splits=5)):
    """Preps ensemble models"""

    modelNames = []
    preppedModels = []
    hashedModels = []    

    if models == 'diverse':
        models = [('rf', ensemble.RandomForestRegressor(n_estimators=100, max_depth=3, n_jobs=-1)),
                  ('svr', svm.LinearSVR(C=1.0, epsilon=0.1, max_iter=10000)),
                  ('lasso', linear_model.Lasso(alpha=0.01, max_iter=1000))]
        passThrough = False

    elif models == 'diverse low n':
        models = [('rf', ensemble.RandomForestRegressor(n_estimators=100, max_depth=3)),
                  ('hgb', ensemble.HistGradientBoostingRegressor()),
                  ('ridge', linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=tscv))]
        passThrough = True
        
    elif models == 'boosted heavy':
        models = [('xgb', XGBRegressor()),
                  ('cat', CatBoostRegressor(verbose=0)),
                  ('lgbm', LGBMRegressor(verbosity=-1))]
        passThrough = True
        
    elif models == 'boosted alpha':
        models = [('xgb', XGBRegressor()),
                  ('cat', CatBoostRegressor(verbose=0)),
                  ('gbr', ensemble.GradientBoostingRegressor)]
        passThrough = True
        
    for model in models:
        if type(model) is tuple:
            (modelName, preppedModel) = model
            try:
                preppedModel = preppedModel()
            except:
                preppedModel = preppedModel
        else:
            modelName = str(model)
            preppedModel = model()
            
        try:
            preppedModel.n_estimators = estimators
        except:
            pass
        try:
            preppedModel.random_state = randomState
        except:
            pass
            
        modelNames.append(modelName)
        preppedModels.append(preppedModel)

    combined = sorted(zip(modelNames, preppedModels), key=lambda x: x[0])
    modelNames, preppedModels = zip(*combined)
    baseModels = [(name,model) for name, model in zip(modelNames,preppedModels)]

    
    ensemblingModel = {'stacking regressor': ensemble.StackingRegressor,
                       'voting regressor': ensemble.VotingRegressor}[stackingModelName]

    finalEstimator = {'ridge': linear_model.Ridge(),
                      'ridgeCV': linear_model.RidgeCV(),
                      'elastic net': linear_model.ElasticNet(),
                      'MLP': neural_network.MLPRegressor(),
                      'SVR': svm.SVR()}[finalEstimatorName]

    model = ensemblingModel(estimators = baseModels,
                            final_estimator = finalEstimator,
                            passthrough = passThrough)

    modelParams = {'model names': modelNames,
                   'stacking model': stackingModelName,
                   'final estimator': finalEstimatorName,
                   'passthrough': passThrough}

    modelStr = json.dumps(modelParams)
    
    stateHash = hashObject(model)

    return model,modelStr,stateHash



############################################################
###  SCIKIT LEARN GENERIC ML WRAPPER TTS CLASS           ###
############################################################



class sklGeneric:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 prefix = '',
                 projectionMethod = defaultProjectionMethod,
                 missingVarResponse = defaultMissingVarResponse,
                 testSize = 12,
                 randomState = 1337,
                 modelArgs = dict(),
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict(),
                 initialized = False,
                 binaryLabelMetric = defaultBinaryMetric,
                 useCache = False,
                 testStatsWindow = 9,
                 monthsForward = 0,
                 fuzzReplicates = 0,
                 fuzzStd = 0.01,
                 ensembleModels = [],
                 ensembleStacker = 'stacking regressor',
                 ensembleEstimator = 'ridgeCV',
                 ensemblePassthrough = True):
        """
        Initialize the model parameters
        """


        if ensembleModels != []:
            preppedModel, modelName, modelState = prepEnsemble(models = ensembleModels,
                                                               randomState = randomState,
                                                               stackingModelName = ensembleStacker,
                                                               finalEstimatorName = ensembleEstimator,
                                                               passThrough = ensemblePassthrough)
            modelArgs = {'model':preppedModel,
                         'modelName':modelName}
            randomState = hash(modelState)
            self.method = f'Scikit-learn generic: {modelName}'
            self.initialized = True

        else:
            self.method = f'Scikit-learn generic: {modelArgs["modelName"]}'
            self.initialized = initialized

        
        
        initModel(self,
                  country,
                  depVar,
                  indepVars,
                  projectionMethod,
                  testSize,
                  randomState,
                  preprocessor,
                  additionalPrep,
                  modelArgs,
                  missingVarResponse,
                  prefix,
                  binaryLabelMetric,
                  useCache,
                  testStatsWindow,
                  monthsForward,
                  fuzzReplicates,
                  fuzzStd)

        

    
    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """
            
        if not self.initialized:
            if not str(self.randomState).startswith('stochastic'):
                try:
                    set_random_seed(self.randomState)
                    model = self.modelArgs['model'](random_state=self.randomState)
                except:
                    model = self.modelArgs['model']()
            else:
                model = self.modelArgs['model']()
        else:
            model = self.modelArgs['model']

        cacheFile = f'store/{self.hash}Trained.pkl'
        if os.path.exists(cacheFile) and self.useCache:
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            trainSKLearn(self,model)
            
            forecast = self.mergedFutures.copy(deep=True)
            forecast.loc[:,'yhat1'] = model.predict(self.mergedFutures[self.varKeys].values)
            
            result = processResults(self,
                                    forecast,
                                    model)
            
            if self.useCache:
                with open(cacheFile, 'wb') as fileOut:
                    pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result
        updateTTSData(self)

    
    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        return results

    
    def export(self):
        """
        Exports country tables
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        exportTables(self)



###########################################################################
###   CLI METHODS                                                       ###
###      * ML WRAPPER MODULE FOR STANDARDIZED EVALUATIONS               ###
###      * INTEGRATES NEURALPROPHET AND SKL METHODS                     ###
###      * CAN TAKE GENERIC SKL CLASSES IF PASS IN PYTHON               ###
###      * USES NEURAL PROPHET TO FORWARD PROJECT SEASONAL PREDICTORS   ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


helpText = """\n\narg 1 - Method
    Must be one of the following:
     - nplagged (NeuralProphet lagged regressors)
     - npfuture (NeuralProphet future regressors)
     - sklgbr (Scikit-learn gradient boosting regression trees)
arg 2 - country
arg 3 - dependent variable
arg 4 - independent variables
    Must be json formatted with not white space characters
     - ex: {'total_precip_mm_per_day':3}
arg 5 - output file
arg 6 - test size (optional, defaults to 12)\n\n"""

if __name__ == "__main__":
    import sys, ast, pprint
    args = sys.argv[1:]

    if len(args) == 0:
        print(helpText)
        quit()
        
    if args[0].lower() in {'help','-h'} or len(args) == 1:
        print(helpText)
        quit()
              
    method = args[0]
    country = args[1]
    depVar = args[2]
    indepVars = ast.literal_eval(args[3])
    fileOut = args[4]
    try:
        testSize = args[5]
    except:
        testSize = 12

    if os.path.exists(fileOut):
        print("Error: file already exists at",fileOut)
        quit()
        

    if method == 'nplagged':
        ttsMethod = npLaggedTTS
    elif method == 'npfuture':
        ttsMethod = npFutureTTS
    elif method == 'sklgbr':
        ttsMethod = sklGradientBoostingRegression

    print("\nInitializing...")
    ttsRun = ttsMethod(country,
                       depVar,
                       indepVars = indepVars,
                       testSize = testSize)
    print("Training...")
    ttsRun.train()
    print("Evaluating...")
    results = ttsRun.evaluate()
    
    with open(fileOut,'w') as fileWriter:
        json.dump(results,fileWriter)

    print("Task complete! Results written to:",fileOut)
    print()
    print("Results:")
    pprint.pprint(results)
    print()
    quit()


