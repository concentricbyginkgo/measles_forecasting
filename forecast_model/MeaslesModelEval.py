###########################################################################
###   MEASLESMODELEVAL.PY                                               ###
###      * ML WRAPPER MODULE FOR STANDARDIZED EVALUATIONS               ###
###      * INTEGRATES NEURALPROPHET AND SKL METHODS                     ###
###      * CAN TAKE GENERIC SKL CLASSES IF PASS IN PYTHON               ###
###      * USES NEURAL PROPHET TO FORWARD PROJECT SEASONAL PREDICTORS   ###
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

import MeaslesDataLoader as md
import EpiPreprocessor as ep

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")

preppedCountries = md.prepData('../model_training_data.csv')

expectedDirectories = ['input',
                       'ouput',
                       'output/figures',
                       'store']

for directory in expectedDirectories:
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    
##########################################
###   DATA MANAGEMENT FUNCTIONS        ###
##########################################


def prepCurve(dfIn,
              modelName,
              depVar,
              indepVars,
              configURL,
              additionalPrep = {}):
    """Prepares one curve df, paring out all unnecessary attributes"""
    # Loads preprocessor config and merges manual args
    df = dfIn.copy(True)
    config = ep.getGoogleSheetConfig(configURL)
    for column,methods in additionalPrep.items():
        try:
            config[column] += methods
        except:
            config[column] = methods

    # Pare down df to only needed and present vars
    indepVars = {key:duration for key,duration in indepVars.items() if key in df.columns}
    indepVars = {key:duration for key,duration in indepVars.items() if len(df[key].dropna()) > 0}
    df = df[['ds',depVar]+list(indepVars.keys())]

    # Apply lagged regressors
    if modelName != 'NeuralProphet lagged regressors':
        for key,duration in indepVars.items():
            df.loc[:,key] = df[key].shift(duration).tolist()

    # Apply preprocessor
    df,preprocessorLog = ep.preprocessDf(df,config)

    df.rename({depVar:'y'},
               axis=1,
               inplace = True)
    
    return df, indepVars, preprocessorLog



def hashIt(var):
    """Returns an md5 hash from an arbitrary variable"""
    varStr = f'{var}'.encode('utf-8')
    md5_hash_obj = hashlib.md5()
    md5_hash_obj.update(varStr)
    return md5_hash_obj.hexdigest()


def sortDict(dictIn):
    """Returns dict contents in predictable order"""
    return dict(sorted(dictIn.items()))




##########################################
###   GENERIC EVALUATION METHODS       ###
##########################################


def alignEvalData(original,predicted):
    """Takes two iterables of the same length and drops rows where either is missing a value"""
    merged = pd.DataFrame({'original':original,
                           'predicted':predicted}).dropna()
    
    return merged.original, merged.predicted


def evaluateModel(trained):
    """Evaluates any trained model from generic TTS parameters"""

    yTest,yTestPred = alignEvalData(trained.yTest,trained.yTestPred)
    yTrain,yTrainPred = alignEvalData(trained.yTrain,trained.yTrainPred)

    
    mseTest = mean_squared_error(yTest, yTestPred)
    maeTest = mean_absolute_error(yTest, yTestPred)
    r2Test = r2_score(yTest, yTestPred)
    mseTrain = mean_squared_error(yTrain, yTrainPred)
    maeTrain = mean_absolute_error(yTrain, yTrainPred)
    r2Train = r2_score(yTrain, yTrainPred)

    result = {'Test MSE':mseTest,
              'Test MAE':maeTest,
              'Test R2':r2Test,
              'Train MSE':mseTrain,
              'Train MAE':maeTrain,
              'Train R2':r2Train}
    
    return result


def plotTTS(simObject):
    """Quick plotting function for TTS objects"""

    forecast = simObject.finalDf
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
    plt.title(f"{simObject.country} {simObject.depVar} vs predictors {simObject.features}\nMethod: {simObject.method}")
    plt.savefig(f'output/figures/{simObject.hash}.png')


##########################################
###   PREDICTOR PROJECTIONS METHODS    ###
##########################################


def projectPredictor(dfIn,
                     var,
                     periods,
                     method = 'NeuralProphet autoregression',
                     forcePositive = True,
                     randomState = 1337):
    """Projects a single predictor n periods into the future"""
    
    df = dfIn[['ds',var]].copy(deep=True)
    df.columns = ['ds','y']

    hash = hashIt((df.to_csv(index=False),
                   var,
                   periods,
                   forcePositive,
                   method,
                   randomState))
    
    cacheFile = f'store/{hash}Predictor.pkl'
    
    if os.path.exists(cacheFile):
        with open(cacheFile, 'rb') as fileIn:
            predictor = pickle.load(fileIn)
        return predictor[-periods:]

    
    if method == 'NeuralProphet autoregression':
        set_random_seed(randomState)
        model = NeuralProphet()
        metrics = model.fit(df,freq='M')
        future = model.make_future_dataframe(df,
                                             periods=periods,
                                             n_historic_predictions=True)
        forecast = model.predict(future)
        forecast['y'].fillna(forecast['yhat1'],inplace=True)
    
        result = forecast[['ds','y']]
        result.columns = ['ds',var]
        

    with open(cacheFile, 'wb') as fileOut:
        pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

    return result[-periods:]
    

############################################################
###  CROSS MODEL INIT AND SUMMARY FUNCTIONS              ###
############################################################


def initModel(simObject,
              country,
              depVar,
              indepVars,
              projectionMethod,
              testSize,
              randomState,
              preprocessor,
              additionalPrep,
              modelArgs):
    """Generalized model initiation across wrappers"""
    curve = preppedCountries[country].copy(deep=True)
    indepVars = sortDict(indepVars)
    simObject.country = country
    simObject.depVar = depVar
    simObject.curve, simObject.indepVars, simObject.preprocessorLog = prepCurve(curve,
                                                                                simObject.method,
                                                                                depVar,
                                                                                indepVars,
                                                                                preprocessor,
                                                                                additionalPrep)

    simObject.modelArgs = modelArgs
    simObject.features = indepVars
    simObject.varKeys = sorted(list(simObject.indepVars.keys()))
    
    simObject.testSize = testSize
    simObject.randomState = randomState
    simObject.trainDf, simObject.testDf  = train_test_split(simObject.curve,
                                                  shuffle = False,
                                                  test_size = simObject.testSize)
    simObject.projection = projectionMethod
    simObject.hash = hashIt((simObject.curve.to_csv(index=False),
                        testSize,
                        randomState,
                        simObject.features,
                        simObject.method,
                        simObject.projection,
                        simObject.modelArgs,
                        simObject.preprocessorLog))

    simObject.xTrain = simObject.trainDf[simObject.varKeys].values
    simObject.yTrain = simObject.trainDf['y'].values
    simObject.xTest = simObject.testDf[simObject.varKeys].values
    simObject.yTest = simObject.testDf['y'].values

    simObject.trained = None
    simObject.yTestPred = None
    simObject.yTrainPred = None
    simObject.results = None


def standardizeOutput(simObject):
    """Rerturns one output table format for all wrapper"""
    if simObject.method.startswith('NeuralProphet'):
        forecast = simObject.trained['forecast'].copy(deep=True)
        
    if simObject.method.startswith('Scikit-learn'):
        forecast = pd.DataFrame(simObject.trained['forecast'])
        forecast.columns = ['yhat1']
        forecast.loc[:,'ds'] = simObject.curve['ds'].values
        forecast.loc[:,'y'] = simObject.curve['y'].values

    simObject.finalDf = forecast.merge(simObject.curve,how='left')


############################################################
###  NEURAL PROPHET LAGGED REGRESSOR TTS CLASS           ###
############################################################


class npLaggedTTS:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 projectionMethod = 'NeuralProphet autoregression',
                 testSize = 12,
                 randomState = 1337,
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict()):
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
                  dict())


    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile):
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            model = NeuralProphet()
            
            for indepVar,delay in self.indepVars.items():
                print(indepVar,delay)
                if delay != 0:
                    model.add_lagged_regressor(indepVar, n_lags=delay)
                else:
                    model.add_future_regressor(indepVar)
                
            metrics = model.fit(self.trainDf,
                                freq='M')

            toMerge = pd.DataFrame(columns=['ds'])
            if self.features == dict():
                future = model.make_future_dataframe(self.trainDf,
                                                     periods=self.testSize,
                                                     n_historic_predictions=True)
            else:
                future = self.trainDf.copy(deep=True)
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    toMerge = toMerge.merge(predictor,how='outer')

            future = pd.concat([future,toMerge],
                                    axis=0,
                                    ignore_index=True)
            
            forecast = model.predict(future)
            result = {'train':self.trainDf,
                      'metrics':metrics,
                      'forecast':forecast,
                      'model':model,
                      'future':future}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast']['yhat1'][-self.testSize:].values
        self.yTrainPred = self.trained['forecast']['yhat1'][:-self.testSize].values
        standardizeOutput(self)
    

    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        modelParams = {'method':self.method,
                       'predictor projection':self.projection,
                       'depVar':self.depVar,
                       'indepVars':self.features,
                       'withheld':self.testSize,
                       'random state':self.randomState,
                       'model args':str(self.modelArgs)}

        results.update(modelParams)
        return results


############################################################
###  NEURAL PROPHET FUTURE REGRESSOR TTS CLASS           ###
############################################################


class npFutureTTS:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 projectionMethod = 'NeuralProphet autoregression',
                 testSize = 12,
                 randomState = 1337,
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict()):
        """
        Initialize the model parameters
        """
        
        self.method = 'NeuralProphet future regressors'

        initModel(self,
                  country,
                  depVar,
                  indepVars,
                  projectionMethod,
                  testSize,
                  randomState,
                  preprocessor,
                  additionalPrep,
                  dict())

    
    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile):
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            model = NeuralProphet()
            
            for indepVar,delay in self.indepVars.items():
                model.add_future_regressor(indepVar)

            metrics = model.fit(self.trainDf,
                                freq='M')

            toMerge = pd.DataFrame(columns=['ds'])
            if self.features == []:
                future = model.make_future_dataframe(self.trainDf,
                                                     periods=self.testSize,
                                                     n_historic_predictions=True)
            else:
                future = self.trainDf.copy(deep=True)
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    toMerge = toMerge.merge(predictor,how='outer')
                    
            future = pd.concat([future,toMerge],
                               axis=0,
                               ignore_index=True)
            
            forecast = model.predict(future)
            result = {'train':self.trainDf,
                      'metrics':metrics,
                      'future':future,
                      'forecast':forecast,
                      'model':model,}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast']['yhat1'][-self.testSize:].values
        self.yTrainPred = self.trained['forecast']['yhat1'][:-self.testSize].values
        standardizeOutput(self)


    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        modelParams = {'method':self.method,
                       'predictor projection':self.projection,
                       'depVar':self.depVar,
                       'indepVars':self.features,
                       'withheld':self.testSize,
                       'random state':self.randomState,
                       'model args':str(self.modelArgs)}

        results.update(modelParams)
        return results

        
        
############################################################
###  SCIKIT LEARN GRADIENT BOOSTING REGRESSION TREES TTS ###
############################################################


class sklGradientBoostingRegression:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 projectionMethod = 'NeuralProphet autoregression',
                 testSize = 12,
                 randomState = 1337,
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict()):
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
                  dict())

    
    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile) and False:
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            model = GradientBoostingRegressor(random_state=self.randomState)
            model.fit(self.xTrain, self.yTrain)


            if self.features == []:
                raise ValueError("Model cannot be trained without one or more independent variables.")
            else:
                future = self.trainDf.copy(deep=True)
                toMerge = pd.DataFrame(columns=['ds'])
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    toMerge = toMerge.merge(predictor,how='outer')

            future = pd.concat([future,toMerge],
                               axis=0,
                               ignore_index=True)
            
            forecast = model.predict(future[sorted(list(self.features.keys()))].values)
            
            result = {'train':self.trainDf,
                      'future':future,
                      'forecast':forecast,
                      'model':model,}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast'][-self.testSize:]
        self.yTrainPred = self.trained['forecast'][:-self.testSize]
        standardizeOutput(self)
    

    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        modelParams = {'method':self.method,
                       'predictor projection':self.projection,
                       'depVar':self.depVar,
                       'indepVars':self.features,
                       'withheld':self.testSize,
                       'random state':self.randomState,
                       'model args':str(self.modelArgs)}

        results.update(modelParams)
        return results



############################################################
###  SCIKIT LEARN GENERIC ML WRAPPER TTS CLASS           ###
############################################################


class sklGeneric:
    def __init__(self,
                 country,
                 depVar,
                 indepVars,
                 projectionMethod = 'NeuralProphet autoregression',
                 testSize = 12,
                 randomState = 1337,
                 modelArgs = dict(),
                 preprocessor = ep.tempConfigURL,
                 additionalPrep = dict()):
        """
        Initialize the model parameters
        """
        
        self.method = f'Scikit-learn generic: {modelArgs["modelName"]}'
        
        initModel(self,
                  country,
                  depVar,
                  indepVars,
                  projectionMethod,
                  testSize,
                  randomState,
                  preprocessor,
                  additionalPrep,
                  modelArgs)

    
    def train(self):
        """
        Trains the model, loading from cache if previously trained
        """

        cacheFile = f'store/{self.hash}Trained.pkl'
        set_random_seed(self.randomState)
        
        if os.path.exists(cacheFile) and False:
            with open(cacheFile, 'rb') as fileIn:
                self.trained = pickle.load(fileIn)
        
        else:
            model = self.modelArgs['model'](random_state=self.randomState)
            model.fit(self.xTrain, self.yTrain)

            
            if self.features == []:
                raise ValueError("Model cannot be trained without one or more independent variables.")
            else:
                future = self.trainDf.copy(deep=True)
                toMerge = pd.DataFrame(columns=['ds'])
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    toMerge = toMerge.merge(predictor,how='outer')

            future = pd.concat([future,toMerge],
                               axis=0,
                               ignore_index=True)
            
            forecast = model.predict(future[sorted(list(self.features.keys()))].values)
            
            result = {'train':self.trainDf,
                      'future':future,
                      'forecast':forecast,
                      'model':model,}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast'][-self.testSize:]
        self.yTrainPred = self.trained['forecast'][:-self.testSize]
        standardizeOutput(self)

    
    def evaluate(self):
        """
        Returns evaluation data
        """
        if self.trained is None:
            raise ValueError("Model has not been trained yet, call that first.")

        results = evaluateModel(self)
        modelParams = {'method':self.method,
                       'predictor projection':self.projection,
                       'depVar':self.depVar,
                       'indepVars':self.features,
                       'withheld':self.testSize,
                       'random state':self.randomState,
                       'model args':str(self.modelArgs)}

        results.update(modelParams)
        return results



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
