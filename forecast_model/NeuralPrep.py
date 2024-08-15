###########################################################################
###   NEURALPREP.PY                                                     ###
###      * GENERAL UTILITY FUNCTIONS MODULE FOR BMGF MEASLES PROJECT    ###
###      * SUPPORTS EXPERIMENT PREP, DEBUGGING, AND MULTITHREADING      ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


import pandas as pd
import country_converter as coco

import warnings
import hashlib
import pickle
import os
import json
#import kaleido

from time import sleep
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from matplotlib import pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")


##########################################
###   VIS AND DEBUGGING METHODS        ###
##########################################


def plotCountrySurveillance(prepped,country):
    """Quick plot of country level surveillance"""
    prepped[country].plot(x='ds',
                          y='cases',
                          marker='o')
    plt.title(f'{country} Monthly New Measles Cases')
    plt.ylabel('Monthly new cases')
    plt.xlabel("Time")


##########################################
###   TRAINING RUN PREP                ###
##########################################


def hashIt(var):
    """Returns an md5 hash from an arbitrary variable"""
    
    md5_hash_obj = hashlib.md5()
    md5_hash_obj.update(str(var).encode('utf-8'))
    return md5_hash_obj.hexdigest()



def prepRun(countries,
            country,
            depVar,
            indepVars,
            confidence):
  """Preps model and curve for one run"""

  # Take the country curve, rename the dependent variable to Y, and remove all non-covariates
  curve = countries[country].copy(deep=True)
  curve.rename({depVar:'y'},
               axis=1,
               inplace = True)
  curve = curve[['ds','y']+list(indepVars.keys())]

  if confidence != 'null':
    boundaries = round((1 - confidence) / 2, 2)
    quantiles = [boundaries, confidence + boundaries]
    model = NeuralProphet(quantiles=quantiles)
  else:
    model = NeuralProphet()

  model.set_plotting_backend("plotly")

  for indepVar,delay in indepVars.items():
    model.add_lagged_regressor(indepVar, n_lags=delay)

  return model, curve



def getTrainTest(curve,
                 model,
                 method,
                 testSize):
    """Returns a train and test df of the selected size and holdout method"""
    if method == 'fraction':
            validP = testSize
            dfTrain, dfTest = model.split_df(df=curve,
                                             freq="M",
                                             valid_p=validP)
    elif method == 'count':
        testLen = 0
        nudge = 0.
        while testLen != testSize:
            validP = (testSize+nudge)/len(curve)
            dfTrain, dfTest = model.split_df(df=curve,
                                             freq="M",
                                             valid_p=validP)
            testLen = len(dfTest)
            nudge += .01
            
    return dfTrain, dfTest



def testForecast(countries,
                 country,
                 depVar = 'cases_1M',
                 indepVars = dict(),
                 method = 'fraction',
                 testSize = .2,
                 loadOnly = False,
                 retryFails = False):
    """Fits a model and returns the real and predicted max month for depVar"""

    # prep model and curve
    model,curve = prepRun(countries,
                          country,
                          depVar,
                          indepVars,
                          confidence='null')


    # base result
    results = {'country':country,
               'depVar':depVar,
               'indepVars':str(indepVars),
               'nTotal':len(curve),
               'method':method,
               'testSize':testSize,
               'success':False,
               'ran':False}

    # if indepvar does not exist in curve, return dummy dict
    if indepVars != dict():
        if len(curve[list(indepVars.keys())].dropna()) == 0:
            return results

    # prep json cache by hash of unique curve and parameters
    hashed = hashIt((curve,depVar,indepVars,method,testSize))
    cacheFile = f'store/{hashed}.json'
    print(cacheFile)

    successFound = False

    
    if os.path.exists(cacheFile):
        # load previous successful run if present
        with open(cacheFile, 'r') as fileIn:
            results = json.load(fileIn)

        succeeded = results['success']
        if succeeded or not retryFails:
            return results
        
    if loadOnly:
        # return dummy data if nothing was executed
        return results
        
    try:
        set_random_seed(0)
        
        future = model.make_future_dataframe(curve,
                                             periods=0,
                                             n_historic_predictions=len(curve))

        dfTrain,dfTest = getTrainTest(curve,
                                      model,
                                      method,
                                      testSize)

        
        cleanTestStats = lambda x: dict(x.iloc[0])
        metricsTrain = model.fit(df=dfTrain, freq="M")
        metricsTest = model.test(df=dfTest)
        metricsTrain = cleanTestStats(metricsTrain)
        metricsTest = cleanTestStats(metricsTest)

        results['nTrain'] = len(dfTrain)
        results['nTest'] = len(dfTest)
        results['success'] = True
        results['ran'] = True
        results.update(metricsTrain)
        results.update(metricsTest)
    except:
        results['ran'] = True

    with open(cacheFile, 'w') as fileOut:
        json.dump(results, fileOut)

    return results



def testForecastSub(args):
    """Wrapper function for multithreaded analyses"""
    (countries, country, depVar, indepVars, method, testSize, loadOnly, retryFails) = args
    print(country,depVar,indepVars,method,testSize)
    result = testForecast(countries,
                          country,
                          depVar,
                          indepVars,
                          method,
                          testSize,
                          loadOnly,
                          retryFails)
    return result

