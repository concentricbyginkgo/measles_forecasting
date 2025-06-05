###########################################################################
###   MEASLESMODELEVAL.PY                                               ###
###      * ML WRAPPER MODULE FOR STANDARDIZED EVALUATIONS               ###
###      * INTEGRATES NEURALPROPHET AND SKL METHODS                     ###
###      * CAN TAKE GENERIC SKL CLASSES IF PASS IN PYTHON               ###
###      * USES NEURAL PROPHET TO FORWARD PROJECT SEASONAL PREDICTORS   ###
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

#preppedCountries = md.prepData()

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

    #if testSize < minAcceptableTestMonths:
        #raise Exception("Insufficent test data for analysis.")

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
              additionalPrep):
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
    
    #if len(df[simObject.depVar].dropna()) <= max(testSize,minAcceptableTotalMonths): #or df[depVar].nunique() < testSize*2:
       #raise Exception("Insufficient number of unique, valid measurements of the dependent variable.")
    
    # Pare down df to only needed and present vars
    indepVars = {key:duration for key,duration in indepVars.items() if key in df.columns}
    indepVars = {key:duration for key,duration in indepVars.items() if len(df[key].dropna()) > 0}
    df = df[['ds',simObject.depVar]+list(indepVars.keys())]

    # Apply lagged regressors
    if simObject.method != 'NeuralProphet lagged regressors':
        for key,duration in indepVars.items():
            df.loc[:,key] = df[key].shift(duration).tolist()

    # Apply preprocessor - fix undefined config variable
    config = {}  # Initialize config 
    for column,methods in additionalPrep.items():
        try:
            config[column] += methods
        except:
            config[column] = methods

    df,preprocessorLog = ep.preprocessDf(df,simObject.preprocessor)

    # Set & verify cutoff size after preprocessor application
    testSize = setCutoffSize(simObject,
                             df,
                             cutoff,
                             testSize)

    #if len(df) <= max(min(simObject.testStatsWindow,testSize)*2,minAcceptableTotalMonths):
        #raise Exception("Insufficient training or testing data following the application of preprocessor rules.")

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
        #Load individual country
        curve = preppedCountries['curves'][country].copy(deep=True)
        
        #Get cutoff date
        cutoff = getCutoffDate(country)
        #if cutoff != cutoff:
            #raise Exception("NaT Cutoff date indicates non-viable experiment.")

        try:
            #Preps country curve
            prepped, log, testSize = prepCurve(simObject,
                                               curve,
                                               cutoff,
                                               simObject.testSize,
                                               additionalPrep)
    
            #Tracks runnable variables for the country curve
            validVars = set(prepped.columns)
    
            #Merge data structure
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
            #print("DEBOOO",endTrim,testSize,len(countryData['prepped'][:-endTrim]))
            
            trainDf, testDf = train_test_split(countryData['prepped'][:-endTrim],
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
              indepVars,
              depVar,
              countries,
              prefix,
              method,
              preprocessor,
              useCache,
              testStatsWindow,
              monthsForward,
              metaRow,
              fuzzReplicates,
              fuzzStd,
              randomState = 1337):
    """Generalized model initiation across wrappers"""

    indepVars = sortDict(indepVars)
    simObject.indepVars = indepVars
    simObject.depVar = depVar
    simObject.countries = countries
    simObject.prefix = prefix
    simObject.method = method
    simObject.preprocessor = preprocessor
    simObject.useCache = useCache
    simObject.testStatsWindow = testStatsWindow
    simObject.monthsForward = monthsForward
    simObject.metaRow = metaRow
    simObject.fuzzReplicates = fuzzReplicates
    simObject.fuzzStd = fuzzStd

    # Initialize missing attributes that are needed by prepCurves and other functions
    simObject.dropLog = []
    simObject.testSize = 12  # default test size
    simObject.missingVarResponse = 'drop var'  # default behavior
    simObject.projection = 'AutoETS'  # default projection method
    simObject.binaryLabeller = lambda x: 1 if x > 1 else 0  # default binary labeller
    simObject.initialCountryCount = len(countries)
    simObject.selection = '_'.join(countries) if len(countries) <= 3 else f"{len(countries)}_countries"

    if prefix.endswith('/'):
        if not os.path.exists(f'output/tables/{prefix}'):
            os.makedirs(f'output/tables/{prefix}')
    #elif prefix != '' and not prefix.endswith('_'):
        #prefix += '_'

    simObject.prefix = prefix

    prepCurves(simObject,
               dict())

    simObject.multipleCurves = len(simObject.curves) > 1

    pareIndepVars(simObject)
    
    if simObject.dropLog != []:
        print(f'Dropped {simObject.dropLog}')
    
    simObject.modelArgs = dict()
    
    if randomState == "stochastic":
        randomState += ' ' + hashIt(os.urandom(32))[:10]
    simObject.randomState = randomState

    prepTTSSets(simObject)
    getMergedFutures(simObject)
    
    #curveHashes = [data['hash'] for country,data in simObject.curves.items()]
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
    mergedTrainDf = simObject.mergedFutures.dropna(subset=['y'])
    simObject.xTrain = mergedTrainDf[simObject.varKeys].values
    simObject.yTrain = mergedTrainDf['y'].values
    simObject.trained = None



############################################################
###  CROSS MODEL TRAINING FUNCTIONS                      ###
############################################################


def projectPredictor(country,
                     var,
                     periods,
                     simObject):
    """Projects a single predictor n periods into the future"""

    if simObject.projection != 'NeuralProphet autoregression':
        lag = simObject.indepVars[var]
    else:
        lag = 0
        
    postLag = periods - lag

    df = simObject.TTSData[country]['trainDf'][['ds',var]].copy(deep=True)
    if lag != 0:
        #dateSlice = simObject.curves[country]['curve'][['ds']].iloc[-periods:-postLag].copy(deep=True)
        #varSlice = simObject.curves[country]['curve'][[var]].iloc[-periods-lag:-periods].copy(deep=True)
        dateSlice = simObject.curves[country]['curve'][['ds']].iloc[-periods:-postLag].copy(deep=True)
        varSlice = simObject.curves[country]['prepped'][[var]].iloc[-periods:-periods+lag].copy(deep=True)
        varSlice.index = dateSlice.index
        forwardAlignedStub = dateSlice.merge(varSlice,
                                            left_index = True,
                                            right_index = True)
        df = pd.concat([df,forwardAlignedStub],ignore_index=True)

    df.columns = ['ds','y']

    if simObject.useCache:
        hash = hashIt((df.to_csv(index=False),
                       var,
                       postLag,
                       simObject.randomState,
                       simObject.projection))
        
        cacheFile = f'store/{hash}Predictor.pkl'
        
        if os.path.exists(cacheFile):
            with open(cacheFile, 'rb') as fileIn:
                predictor = pickle.load(fileIn)
            return predictor[-postLag:]


    if df['y'].nunique() == 1:
        lastDate = df['ds'].max()
        yValue = df['y'].iloc[0]        
        newDates = [lastDate + pd.DateOffset(months=i) for i in range(1, postLag + 1)]
        newRows = pd.DataFrame({'ds': newDates, 'y': yValue})
        result = pd.concat([df, newRows], ignore_index=True)
        result.columns = ['ds',var]

    
    elif simObject.projection == 'NeuralProphet autoregression':
        if not str(simObject.randomState).startswith('stochastic'):
            set_random_seed(simObject.randomState)
        model = NeuralProphet()
        metrics = model.fit(df,freq='M')
        future = model.make_future_dataframe(df,
                                             periods=postLag,
                                             n_historic_predictions=True)
        forecast = model.predict(future)
        forecast['y'].fillna(forecast['yhat1'],inplace=True)
    
        result = forecast[['ds','y']]
        result.columns = ['ds',var]

    
    elif simObject.projection == 'AutoETS':
        autoETSInstance = AutoETS(season_length=12)
        df['unique_id'] = 'null'

        statsforecast = StatsForecast(df = df,
                                      models = [autoETSInstance],
                                      freq = 'MS',
                                      n_jobs=-1)
        statsforecast.fit()
        result = statsforecast.predict(postLag)
        result = result.reset_index()[['ds','AutoETS']]
        result.columns = ['ds',var]
        
    
    if simObject.useCache:
        with open(cacheFile, 'wb') as fileOut:
            pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)


    if lag != 0:
        result = pd.concat([forwardAlignedStub,result],ignore_index=True)
        
    return result[-periods:]
    

def getFutureDf(simObject,country):
    """Projects the predictors used in training a sim for a single country"""

    toMerge = pd.DataFrame(columns=['ds'])
    testSize = simObject.curves[country]['testSize']
    trainDf = simObject.TTSData[country]['trainDf'].copy(deep=True)
    
    if simObject.method.startswith('NeuralProphet') and simObject.indepVars == dict():
        future = model.make_future_dataframe(trainDf,
                                             periods=testSize + simObject.monthsForward,
                                             n_historic_predictions=True)
    
    elif simObject.method.startswith('Scikit-learn') and simObject.indepVars == dict() and False:
        raise ValueError("Model cannot be trained without one or more independent variables.")
    
    else:
        future = trainDf
        for indepVar,delay in simObject.indepVars.items():
            predictor = projectPredictor(country,
                                         indepVar,
                                         testSize + simObject.monthsForward,
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
        #mergedFutures = mergedFutures.drop(['ID'],axis=1)
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
    #forecast = forecast.dropna()
        
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

    #print(len(original))
    #print(len(predicted))
    #print(original)
    #print(predicted)
    merged = pd.DataFrame({'original':original,
                           'predicted':predicted}).dropna()
    
    return merged.original, merged.predicted



def evaluateCountry(simObject,country):
    """Evaluates model performance for a single country from generic TTS parameters"""

    if simObject.monthsForward == 0:    
        countryTTS = simObject.TTSData[country]
        testSize = simObject.curves[country]['testSize']
        testWindow = min(testSize,simObject.testStatsWindow)

        ## TODO: some issue is causing an off by one in the length of yTestPred, likely a dropna
        ## suppressing with length matching for now
        
        #yTest,yTestPred = alignEvalData(countryTTS['yTest'],countryTTS['yTestPred'][:testWindow])
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
                  'True positives':tp,
                  'True negatives':tn,
                  'False positives':fp,
                  'False negatives':fn}
    else:
        result = dict()
        testSize = 0

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
    #if simObject.monthsForward != 0:
    #    raise Exception("Model evaluation is currently incompatible with forward projection, please set 'monthsForward' to zero and try again.")
    
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



def exportTables(simObject, directory = 'output/'):
    """Exports prepped data csvs from a trained run"""

    if not os.path.exists(f'{directory}/{simObject.prefix}/'):
        os.makedirs(f'{directory}/{simObject.prefix}/')
    if not os.path.exists(f'{directory}/{simObject.prefix}/scores/'):
        os.makedirs(f'{directory}/{simObject.prefix}/scores/')
    if not os.path.exists(f'{directory}/{simObject.prefix}/tables/'):
        os.makedirs(f'{directory}/{simObject.prefix}/tables/')
    
    hashStr = simObject.hash
    for country, table in simObject.trained['plotDfs'].items():
        table = table[['ID', 'ds', 'y', 'yhat1']]
        table2 = deepcopy(table)
        table2[['ROW_ID']] = simObject.metaRow
        fileOut = f'{directory}{simObject.prefix}/tables/{simObject.metaRow}_{country}_Projection.csv'
        table2.to_csv(fileOut,index=False)
        
    summaryRef = f'{directory}{simObject.prefix}/scores/{simObject.metaRow}_Summary.csv'
    metricSummary = evaluateModel(simObject)
    metricSummary.index.name = 'ID'
    metricSummary.reset_index(inplace=True)
    metricSummary = metricSummary[['ID', 'Test MSE', 'Test MAE', 'Test R2', 
                                   'Train MSE', 'Train MAE', 'Train R2', 
                                   'Sensitivity', 'Specificity', 'F1 Score', 
                                   'method']]
    metricSummary2 = deepcopy(metricSummary)
    metricSummary2['ROW_ID'] = simObject.metaRow
    metricSummary2.to_csv(summaryRef, index=False)    

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
                 indepVars,
                 depVar,
                 countries,
                 prefix,
                 method,
                 preprocessor,
                 useCache = False,
                 testStatsWindow = 9,
                 monthsForward = 0,
                 metaRow = 1,
                 fuzzReplicates = 0,
                 fuzzStd = 0.01,
                 randomState = 1337):
        """
        Initialize the model parameters
        """
        initModel(self,
                 indepVars,
                 depVar,
                 countries,
                 prefix,
                 method,
                 preprocessor,
                 useCache,
                 testStatsWindow,
                 monthsForward,
                 metaRow,
                 fuzzReplicates,
                 fuzzStd,
                 randomState)


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
                print(indepVar,delay)
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
                 indepVars,
                 depVar,
                 countries,
                 prefix,
                 method,
                 preprocessor,
                 useCache = False,
                 testStatsWindow = 9,
                 monthsForward = 0,
                 metaRow = 1,
                 fuzzReplicates = 0,
                 fuzzStd = 0.01,
                 randomState = 1337):
        """
        Initialize the model parameters
        """
        initModel(self,
                 indepVars,
                 depVar,
                 countries,
                 prefix,
                 method,
                 preprocessor,
                 useCache,
                 testStatsWindow,
                 monthsForward,
                 metaRow,
                 fuzzReplicates,
                 fuzzStd,
                 randomState)
    
    
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

            #model.fit(self.xTrain, self.yTrain)
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
                 countries,              # Can be single string or list
                 depVar,
                 indepVars,
                 modelArgs = None,       # For cluster_dev_2 compatibility
                 randomState = 1337,     # For cluster_dev_2 compatibility  
                 preprocessor = None,
                 prefix = 'run',
                 metaRow = 1,
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
        
        # Handle single country string (cluster_dev_2 style) vs country list (main style)
        if isinstance(countries, str):
            countries = [countries]
            
        # Set default preprocessor if none provided
        if preprocessor is None:
            import EpiPreprocessor as ep
            preprocessor = ep.getGoogleSheetConfig(ep.tempConfigURL)
            
        # Initialize the model first
        initModel(self,
                 indepVars,
                 depVar,
                 countries,
                 prefix,
                 'sklGeneric',  # method name
                 preprocessor,
                 useCache,
                 testStatsWindow,
                 monthsForward,
                 metaRow,
                 fuzzReplicates,
                 fuzzStd,
                 randomState)

        # Handle ensemble models (from main branch)
        if ensembleModels != []:
            preppedModel, modelName, modelState = prepEnsemble(models = ensembleModels,
                                                               randomState = randomState,
                                                               stackingModelName = ensembleStacker,
                                                               finalEstimatorName = ensembleEstimator,
                                                               passThrough = ensemblePassthrough)
            self.modelArgs = {'model':preppedModel,
                             'modelName':modelName}
            self.randomState = hash(modelState)
            self.method = f'Scikit-learn generic: {modelName}'
            self.initialized = True
        else:
            # Handle direct model specification (cluster_dev_2 style)
            if modelArgs is not None:
                self.modelArgs = modelArgs
                self.method = f'Scikit-learn generic: {modelArgs["modelName"]}'
                self.initialized = False
            else:
                raise ValueError("Must provide either modelArgs or ensembleModels")
            self.randomState = randomState

    
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
            #model.fit(self.xTrain, self.yTrain)
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


