###########################################################################
###   EPIPREPROCESSOR.PY                                                ###
###      * APPLIES AND LOGS PREPROCESSOR TRANSFORMATIONS TO DATA        ###
###      * USES LOCAL OR REMOTE CONFIG FILES LISTING OPS BY COLUMN      ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


import pandas as pd
import numpy as np

from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from scipy.interpolate import interp1d
from pprint import pprint
from copy import deepcopy

import warnings
import hashlib
import pickle
import os
import ast

warnings.simplefilter(action='ignore', category=FutureWarning)
set_log_level("ERROR")

tempConfigURL = 'input/tempConfig.csv'

##########################################
###   CONFIG LOADER AND HASHING METHOD ###
##########################################


def getGoogleSheetConfig(url):
    """Loads a preprocessor config from a google or directly linked remote or local csv"""
    if url.startswith('https://docs.google.com/'):
        url = url.replace('edit?','export?format=csv&')
    config = pd.read_csv(url,index_col=0)
    if url != tempConfigURL:
        config.to_csv(tempConfigURL)
    config = config.dropna(subset=['Methods'],axis=0)
    parsed = {index:row['Methods'].split() for index, row in config.iterrows()}
    return parsed


def hashIt(var):
    """Returns an md5 hash from an arbitrary variable"""    
    md5_hash_obj = hashlib.md5()
    md5_hash_obj.update(str(var).encode('utf-8'))
    return md5_hash_obj.hexdigest()

def hasNan(series):
    """Returns True if a series contains a NaN"""
    result = series.isna().any()
    return result


##########################################
###   PREPROCESSOR OPERATIONS          ###
##########################################


def zeroFillPP(dfIn,column):
    """Fills missing values with zero"""
    df = dfIn.copy(deep=True)
    df.loc[:,column] = df.loc[:,column].fillna(0)
    return df


def backFillPP(dfIn,column):
    """Fills missing entries at start of df with first valid entry"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].bfill(limit_area='outside')
    return df


def forwardFillPP(dfIn,column):
    """Fills missing entries at end of df with first valid entry"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].ffill(limit_area='outside')
    return df


def backTruncatePP(dfIn,column):
    """Drops rows containing leading nans in target column"""
    df = dfIn.copy(deep=True)
    firstValidIndex = df[column].first_valid_index()
    df = df.loc[firstValidIndex:]
    return df


def forwardTruncatePP(dfIn,column):
    """Drops rows contaiing tailing nans in target column"""
    df = dfIn.copy(deep=True)
    lastValidIndex = df[column].last_valid_index()
    df = df.loc[:lastValidIndex:]
    return df
    

def firstValidOverwritePP(dfIn,column):
    """Overwrites entire column with the first valid value"""
    df = dfIn.copy(deep=True)
    firstValidValue = df[column].dropna().iloc[0]
    df[column] = firstValidValue
    return df


def averageValidOverwritePP(dfIn,column):
    """Overwrites entire column with the average of valid values"""
    df = dfIn.copy(deep=True)
    averageValid = df[column].mean()
    df[column] = averageValid
    return df


def fillGapsPP(dfIn,column):
    """Fills gaps between two equivalent values"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].where(df[column].ffill() != df[column].bfill(), df[column].ffill())
    return df


def linearPP(dfIn,column):
    """Fills missing values in column with linear interpolation"""
    df = dfIn.copy(deep=True)
    if df[column].nunique() == 1:
        df[column] = fillGapsPP(df,column)[column]
    else:
        df[column] = df[column].interpolate(method='linear',
                                           limit_area='inside')
    return df


def timesfmPP(dfIn,column):
    # CURRENTLY UNIMPLEMENTED
    return dfIn


def passUnchangedPP(dfIn,column):
    """Returns an unaltered copy"""
    return dfIn.copy(deep=True)


def januaryOnlyPP(dfIn,column):
    """NaNs out all values in column that are not January observations"""
    df = dfIn.copy(deep=True)
    df[column] = df.apply(lambda row: row[column] if row['ds'].month == 1 else np.nan, axis=1)
    return df


def julyOnlyPP(dfIn,column):
    """NaNs out all values in column that are not January observations"""
    df = dfIn.copy(deep=True)
    df[column] = df.apply(lambda row: row[column] if row['ds'].month == 7 else np.nan, axis=1)
    return df
    

def boolToIntPP(dfIn,column):
    """Casts all bools in column to ints"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].astype(int)
    return df


def ynToIntPP(dfIn,column):
    """Casts all yesses and nos in column to ints"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].str.lower().str[:1].replace({'y':1,'n':0}).astype(int)
    return df


def flipBoolPP(dfIn,column):
    """Flips all bool values in a column to their opposite value"""
    df = dfIn.copy(deep=True)
    df[column] = ~df[column].astype(bool)
    return df


def interpolateViaPP(dfIn,column,method):
    """Performs 1D interpolation on a column via the passed method name"""
    # method must equal one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    if not hasNan(dfIn[column]):
        return dfIn
    df = dfIn.copy(deep=True)

    if df[column].nunique() == 1:
        df[column] = fillGapsPP(df,column)[column]
        return df

    x = df['ds'].view('int64') // 10**9

    mask = df[column].notna()
    xValid = x[mask]
    yValid = df.loc[mask, column]
    
    interpFx = interp1d(xValid,
                        yValid,
                        kind=method,
                        fill_value='extrapolate')

    interped = pd.Series(interpFx(x), index=df.index)
    
    df.loc[:,column] = df.loc[:,column].fillna(interped)
    return df


    

def removeFlagPP(dfIn,column,flagValue):
    """Removes a known flag value from the given column"""
    df = dfIn.copy(deep=True)
    
    if flagValue not in df[column]:
        colType = str(df[column].dtype)
        if colType.startswith('float') or colType == 'object':
            flagValue = float(flagValue)
        elif colType.startswith('int'):
            flagValue = int(flagValue)

    
    df[column] = df[column].replace({flagValue:np.nan})
    return df


def divideByPP(dfIn,column,factor):
    """Divides a column by a passed factor"""
    df = dfIn.copy(deep=True)
    df[column] = df[column] / float(factor)
    return df


def multiplyByPP(dfIn,column,factor):
    """Multiplies a column by a passed factor"""
    df = dfIn.copy(deep=True)
    df[column] = df[column] * float(factor)
    return df


def dropTailingPP(dfIn,column,n):
    """Multiplies a column by a passed factor"""
    df = dfIn[:-n].copy(deep=True)
    return df


def neuralProphetPP(dfIn,column):
    """Runs neuralprophet interpolation with fixed random seed"""
    df = neuralProphetSeedPP(dfIn,column,1337)
    return df


def neuralProphetSeedPP(dfIn,column,seed):
    """Runs neuralpophet interpolation with passed random seed"""
    if not hasNan(dfIn[column]):
        return dfIn
    df = dfIn.copy(deep=True)
    seed = int(seed)
    npDf = df[['ds',column]]

    hash = hashIt((dfIn,column,seed))
    npDf.columns = ['ds','y']

    cacheFile = f'store/{hash}IndepVar.pkl'
    
    if os.path.exists(cacheFile):
        with open(cacheFile, 'rb') as fileIn:
            df = pickle.load(fileIn)
        return df
            
    set_random_seed(seed)
    model = NeuralProphet()

    model.fit(npDf.dropna(subset=['y']), freq='M')
    forecast = model.predict(npDf).reset_index()
    forecast.index = npDf.index.tolist()
    
    df[column] = df[column].fillna(forecast['yhat1'])

    with open(cacheFile, 'wb') as fileOut:
        pickle.dump(df, fileOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    return df


def zeroMinPP(dfIn,column):
    """Replaces all values under zero with 0"""
    df = dfIn.copy(deep=True)
    df[column] = df[column].apply(lambda y: max(y, 0.))
    return df


def discardLastPP(dfIn,column,n):
    """Discards last n entries in a table"""
    df = dfIn.copy(deep=True)
    n = int(n)
    if n > 0:
        df = df[:-n]
    return df


def remapByPP(dfIn,column,json):
    """Remaps a column by str.lower().startswith() alignment"""
    df = dfIn.copy(deep=True)
    mapper = ast.literal_eval(json)
    mapper = {key.lower().replace('_',' '):value for key,value in mapper.items()}

    def mapValue(value):
        for prefix, mapped in mapper.items():
            if value.lower().startswith(prefix):
                return mapped
        return value  # Return the original value if no match is found
    
    df[column] = df[column].astype(str).apply(mapValue)

    return df


def checkCoveragePP(dfIn,column,json):
    """NaNs out a column if the coverage is below a certain percent"""
    df = dfIn.copy(deep=True)
    params = ast.literal_eval(json)

    if 'tail' in params.keys():
        testDf = df.tail(int(params['tail']))
    else:
        testDf = df

    if testDf[column].isna().mean() >= float(params['proportion']):
        df[column] = np.nan

    return df


def getLongestNanStretch(df,column):
    """Calculates the length of the longest contiguous stretch of NaN values in a pandas Series"""
    if df[column].empty:
        return 0

    isNanSeries = df[column].isnull()

    if not isNanSeries.any():
        return 0
        
    if isNanSeries.all():
        return len(df)

    grouper = (~isNanSeries).cumsum()
    nanStretchLengths = isNanSeries.groupby(grouper).cumsum()
    maxStretch = nanStretchLengths.max()
    
    return int(maxStretch)


def checkGapsPP(dfIn,column,json):
    """NaNs out a column if the coverage is below a certain percent"""
    df = dfIn.copy(deep=True)
    params = ast.literal_eval(json)

    if 'tail' in params.keys():
        testDf = df.tail(int(params['tail']))
    else:
        testDf = df

    if getLongestNanStretch(testDf,column) >= float(params['length']):
        df[column] = np.nan

    return df
   

##########################################
###   ENCODER OPERATIONS               ###
##########################################


def encodeOneHot(dfIn,encodeColumn):
    """Creates dummy variables for onehot encoding"""
    originalCols = set(dfIn.columns)
    df = pd.get_dummies(dfIn, columns=[encodeColumn])
    df.loc[:,'ID'] = dfIn.ID.copy(deep=True)
    newCols = sorted(list(set(df.columns).difference(originalCols)))
    
    return df, newCols



def encodeOrdinal(dfIn,args):
    """Adds an encoded column to a df using ascending rank from a reference column"""
    df = dfIn.copy(deep = True)
    [encodeColumn,referenceColumn] = args.split('_by_')
    
    lastValsDf = dfIn.groupby(encodeColumn).last().reset_index()
    lastValsDf = lastValsDf[[encodeColumn,referenceColumn]]
    lastValsDf = lastValsDf.sort_values(referenceColumn,axis=0)
    mapper = {encodeVal:i for i,encodeVal in enumerate(lastValsDf[encodeColumn])}

    columnOut = f'{encodeColumn}_encoded_on_{referenceColumn}'
    df.loc[:,columnOut] = df[encodeColumn].map(mapper)
    newCols = [columnOut]

    return df, newCols


##########################################
###   PP & ENC OPERATIONS ITERABLES    ###
##########################################


fixedMethods = {'zero_fill':zeroFillPP,
                'back_fill':backFillPP,
                'forward_fill':forwardFillPP,
                'back_truncate':backTruncatePP,
                'forward_truncate':forwardTruncatePP,
                'first_valid_overwrite':firstValidOverwritePP,
                'average_valid_overwrite':averageValidOverwritePP,
                'linear':linearPP,
                'neuralprophet':neuralProphetPP,
                'timesfm':timesfmPP,
                'pass_unchanged':passUnchangedPP,
                'january_only':januaryOnlyPP,
                'july_only':julyOnlyPP,
                'bool_to_int':boolToIntPP,
                'yn_to_int':ynToIntPP,
                'flip_bool':flipBoolPP,
                'zero_min':zeroMinPP}


modifiableMethods = {'interpolate_via_':interpolateViaPP,
                     'remove_flag_':removeFlagPP,
                     'divide_by_':divideByPP,
                     'multiply_by_':multiplyByPP,
                     'drop_tailing_':dropTailingPP,
                     'neuralprophet_seed_':neuralProphetSeedPP,
                     'discard_last_':discardLastPP,
                     'remap_by_':remapByPP,
                     'check_coverage_':checkCoveragePP,
                     'check_gaps_':checkGapsPP}


baseEncoders = {'encode_ordinal':encodeOrdinal,
                'encode_onehot':encodeOneHot}




##########################################
###   PRIMARY USER METHODS            ###
##########################################


def getPreprocessorMethods(column,config):
    """Given a column name and config file, returns a dict of operation names and functions"""
    names = config[column]
    methods = dict()
    for name in names:
        found = False
        try:
            methods[name] = fixedMethods[name]
            found = True
        except:
            for key in modifiableMethods.keys():
                if name.startswith(key):
                    arg = name.replace(key,'')
                    methods[name] = (lambda k=key, a=arg[:]: lambda x, y: modifiableMethods[k](x, y, a))()
                    found = True
                    break
        if not found:
            print(f'Method "{name}" not found for column "{column}", ignoring...')
            
    return methods


    
def preprocessDf(dfIn,
                 config,
                 verbose=False):
    """Given a dataframe and config file, runs all preprocessing and returns a processed df and operations log dict"""
    df = dfIn.copy(deep=True)

    selectedMethods = {column: getPreprocessorMethods(column,config) for column in config.keys() if column in df.columns}
    if verbose:
        pprint(selectedMethods)
    
    loggedTransformations = dict()
    for column,operations in selectedMethods.items():
        loggedTransformations[column] = []
        for operationName,fx in operations.items():
            if verbose:
                print(column,operationName)
            loggedTransformations[column].append(operationName)
            df = fx(df,column)


    return df, loggedTransformations



def getEncoderMethod(encoderAlignment,config):
    """Given an ML method name, returns a prepped encoder method"""
    if encoderAlignment not in config.keys():
        print(f'Country encoder not found for method: "{encoderAlignment}", using default')
        methodName = 'encode_onehot_on_ID'
    else:
        try:
            methodName = config[encoderAlignment][0]
        except:
            methodName = 'pass_unchanged'
            print("No encoding method passed, this will likely fail for most ML libraries.")
        
    if methodName == 'pass_unchanged':
        method = None
    else:
        [encoderName,args] = methodName.split('_on_')
        encoder = baseEncoders[encoderName]
        method = lambda x: baseEncoders[encoderName](x,args)

    return methodName, method



def encodeMergedDf(dfIn,
                   config,
                   encoderAlignment):
    """Encodes the grouping categorical variable from a merged tables to numeric"""
    df = dfIn.copy(deep=True)
    
    if encoderAlignment != None:
        methodName, fx = getEncoderMethod(encoderAlignment,config)
        if methodName != 'pass_unchanged':
            df, newCols = fx(df)
        else:
            newCols = []
    else:
        methodName = 'pass_unchanged'
        newCols = []

    return df, newCols, methodName
