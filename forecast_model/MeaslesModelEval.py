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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

import MeaslesDataLoader as md

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")

preppedCountries = md.prepData()

    
    
##########################################
###   DATA MANAGEMENT FUNCTIONS        ###
##########################################


def prepCurve(df,
              depVar,
              indepVars):
    """Prepares one curve df, paring out all unnecessary attributes"""
    df.rename({depVar:'y'},
               axis=1,
               inplace = True)

    indepVars = {key:duration for key,duration in indepVars.items() if key in df.columns}
    indepVars = {key:duration for key,duration in indepVars.items() if len(df[key].dropna()) > 0}
    df = df[['ds','y']+list(indepVars.keys())]
    df = df.dropna(subset=['y'],axis=0)
    
    return df,indepVars


def hashIt(var):
    """Returns an md5 hash from an arbitrary variable"""
    
    md5_hash_obj = hashlib.md5()
    md5_hash_obj.update(str(var).encode('utf-8'))
    return md5_hash_obj.hexdigest()


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
    forecast = simObject.trained['forecast'].copy(deep=True)
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

    hash = hashIt((df,
                   var,
                   periods,
                   forcePositive,
                   method,
                   randomState))
    
    cacheFile = f'store/{hash}Predictor.pkl'
    
    if os.path.exists(cacheFile):
        with open(cacheFile, 'rb') as fileIn:
            predictor = pickle.load(fileIn)
        return predictor

    
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

    return result





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
                 randomState = 1337):
        """
        Initialize the model parameters
        """
        curve = preppedCountries[country].copy(deep=True)
        self.country = country
        self.depVar = depVar
        self.curve, self.indepVars = prepCurve(curve,
                                               depVar,
                                               indepVars)

        self.modelArgs = dict()
        self.features = indepVars
        varKeys = sorted(list(self.indepVars.keys()))
        
        self.testSize = testSize
        self.randomState = randomState
        self.trainDf, self.testDf  = train_test_split(self.curve,
                                                      shuffle = False,
                                                      test_size = self.testSize)
        self.method = 'NeuralProphet lagged regressors'
        self.projection = projectionMethod
        self.hash = hashIt((self.curve,
                            testSize,
                            randomState,
                            self.features,
                            self.method,
                            self.projection,
                            self.modelArgs))

        self.xTrain = self.trainDf[varKeys].values
        self.yTrain = self.trainDf['y'].values
        self.xTest = self.testDf[varKeys].values
        self.yTest = self.testDf['y'].values

        self.trained = None
        self.yTestPred = None
        self.yTrainPred = None
        self.results = None

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
                model.add_lagged_regressor(indepVar, n_lags=delay)
                
            metrics = model.fit(self.trainDf,
                                freq='M')


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
                    future = future.merge(predictor,how='outer')
            
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
                 randomState = 1337):
        """
        Initialize the model parameters
        """
        curve = preppedCountries[country].copy(deep=True)
        self.depVar = depVar
        self.country = country
        self.curve, self.indepVars = prepCurve(curve,
                                               depVar,
                                               indepVars)

        self.modelArgs = dict()
        self.features = sorted(list(self.indepVars.keys()))
        
        self.testSize = testSize
        self.randomState = randomState
        self.trainDf, self.testDf  = train_test_split(self.curve,
                                                      shuffle = False,
                                                      test_size = self.testSize)
        self.method = 'NeuralProphet future regressors'
        self.projection = projectionMethod
        self.hash = hashIt((self.curve,
                            testSize,
                            randomState,
                            self.features,
                            self.method,
                            self.projection,
                            self.modelArgs))

        self.xTrain = self.trainDf[self.features].values
        self.yTrain = self.trainDf['y'].values
        self.xTest = self.testDf[self.features].values
        self.yTest = self.testDf['y'].values

        self.trained = None
        self.yTestPred = None
        self.yTrainPred = None
        self.results = None

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
                    future = future.merge(predictor,how='outer')
        
            
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
                 randomState = 1337):
        """
        Initialize the model parameters
        """
        curve = preppedCountries[country].copy(deep=True)
        self.depVar = depVar
        self.country = country
        self.curve, self.indepVars = prepCurve(curve,
                                               depVar,
                                               indepVars)

        self.modelArgs = dict()
        self.features = sorted(list(self.indepVars.keys()))
        
        self.testSize = testSize
        self.randomState = randomState
        self.trainDf, self.testDf  = train_test_split(self.curve,
                                                      shuffle = False,
                                                      test_size = self.testSize)
        self.method = 'Scikit-learn gradient boosted regression'
        self.projection = projectionMethod
        self.hash = hashIt((self.curve,
                            testSize,
                            randomState,
                            self.features,
                            self.method,
                            self.projection,
                            self.modelArgs))

        self.xTrain = self.trainDf[self.features].values
        self.yTrain = self.trainDf['y'].values
        self.xTest = self.testDf[self.features].values
        self.yTest = self.testDf['y'].values

        self.trained = None
        self.yTestPred = None
        self.yTrainPred = None
        self.results = None

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
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    future = future.merge(predictor,how='outer')

            
            forecast = model.predict(future[self.features].values)
            
            result = {'train':self.trainDf,
                      'future':future,
                      'forecast':forecast,
                      'model':model,}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast'][-self.testSize:]
        self.yTrainPred = self.trained['forecast'][:-self.testSize]

    

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
                 modelArgs = dict()):
        """
        Initialize the model parameters
        """
        curve = preppedCountries[country].copy(deep=True)
        self.depVar = depVar
        self.country = country
        self.curve, self.indepVars = prepCurve(curve,
                                               depVar,
                                               indepVars)

        self.features = sorted(list(self.indepVars.keys()))
        
        self.testSize = testSize
        self.randomState = randomState
        self.trainDf, self.testDf  = train_test_split(self.curve,
                                                      shuffle = False,
                                                      test_size = self.testSize)
        self.method = f'Scikit-learn generic: {modelArgs["modelName"]}'
        self.modelArgs = modelArgs
        self.projection = projectionMethod
        self.hash = hashIt((self.curve,
                            testSize,
                            randomState,
                            self.features,
                            self.method,
                            self.projection,
                            self.modelArgs))

        self.xTrain = self.trainDf[self.features].values
        self.yTrain = self.trainDf['y'].values
        self.xTest = self.testDf[self.features].values
        self.yTest = self.testDf['y'].values

        self.trained = None
        self.yTestPred = None
        self.yTrainPred = None
        self.results = None

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
            print(self.modelArgs)
            model = self.modelArgs['model'](random_state=self.randomState)
            model.fit(self.xTrain, self.yTrain)


            if self.features == []:
                raise ValueError("Model cannot be trained without one or more independent variables.")
            else:
                future = self.trainDf.copy(deep=True)
                for indepVar,delay in self.indepVars.items():
                    predictor = projectPredictor(self.trainDf,
                                                 indepVar,
                                                 self.testSize,
                                                 method = self.projection,
                                                 randomState = self.randomState)
                    future = future.merge(predictor,how='outer')

            
            forecast = model.predict(future[self.features].values)
            
            result = {'train':self.trainDf,
                      'future':future,
                      'forecast':forecast,
                      'model':model,}
    
            with open(cacheFile, 'wb') as fileOut:
                pickle.dump(result, fileOut, protocol=pickle.HIGHEST_PROTOCOL)

            self.trained = result

        self.yTestPred = self.trained['forecast'][-self.testSize:]
        self.yTrainPred = self.trained['forecast'][:-self.testSize]

    

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
