###########################################################################
###   MEASLESDATALOADER.PY                                              ###
###      * GENERALIZED DATA LOADING AND STANDARDIZATION MODULE          ###
###      * RANKS COUNTRIES BY NUMBER OF LABELLED OUTBREAKS              ###
###      * HARDCODED TO EXPECTED DATA AND FILE LOCATIONS                ###
###                                                                     ###
###             Contact: James Schlitt ~ schlittdatasci@gmail.com       ###
###########################################################################


import pandas as pd
import country_converter as coco
import os
import hashlib
import warnings

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


validityColumn = 'cases_1M'
GEO_ID_COL = 'GEO_ID'
LEGACY_GEO_ID_COL = 'ISO3'


def ensure_geo_id_column(df):
    """Model input CSV uses GEO_ID (set at script 7 export); accept legacy ISO3 with warning."""
    if GEO_ID_COL in df.columns:
        return df
    if LEGACY_GEO_ID_COL in df.columns:
        warnings.warn(
            f'Model input uses legacy column {LEGACY_GEO_ID_COL}; '
            f're-run script 7 export or rename to {GEO_ID_COL}.',
            stacklevel=2,
        )
        return df.rename(columns={LEGACY_GEO_ID_COL: GEO_ID_COL})
    raise KeyError(
        f'Model input CSV must contain {GEO_ID_COL} (export from 7_combine_all_datasets.R).'
    )


##########################################
###   CENTRALIZED DATA LOADER          ###
##########################################


def verboseLoader(ref):
    """Loads a csv and prints it's MD5 hash"""
    md5 = hashlib.md5()
    with open(ref, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
            
    md5Hash = md5.hexdigest()
    print(f"Loading file: {ref}\t\tmd5 hash: {md5Hash}")
    df = pd.read_csv(ref)
    
    return df



def prepFilters(df,validCountries):
    """Returns a set of quick filters"""

    # Identify columns in data set with 2-9 total unique values
    totalUniqueValMatches = (df.nunique() < 12) & (df.nunique() > 1)
    uniqueValKeys = set(totalUniqueValMatches[totalUniqueValMatches].index)
    
    # Identify columns in data set with no more than one unique value per country
    onePerCountryMatches = pd.Series(df.groupby(GEO_ID_COL).nunique().max())
    onePerKeys = set(onePerCountryMatches[onePerCountryMatches == 1].index)

    # Select only keys matching both prior requirements
    quickFilterKeys = uniqueValKeys.intersection(onePerKeys)

    # Prepare the request:country list mapping dictionary
    quickFilters = {country:[country] for country in validCountries}
    for filterKey in quickFilterKeys:
        grouped = dict(df.groupby(filterKey)[GEO_ID_COL].unique())
        for groupKey,groupArray in grouped.items():
            if isinstance(groupKey, (int, float, complex)):
                if groupKey == int(groupKey):
                    groupKey = int(groupKey)
            quickFilters[f'{filterKey}:{groupKey}'] = list(groupArray)

    quickFilters['all'] = validCountries

    return quickFilters




def prepData(defaultLoc = 'input/processed_measles_model_data.csv',
             cutoffsLoc = 'input/cutoff_date_by_country.csv',
             cutoffsCol = 'cutoff_date',
             trendsLoc = 'input/CountryMeasles2010-2024.csv',
             fillMissingCutoffs = True,
             useImputation = False,
             imputeGrouping = 'date',
             imputeIgnore = {'cases',
                             'cases_1M',
                             'cuml_cases',
                             'cluster',
                             'cuml_cases_1M',
                             'Year'},
             knnNeighbors = 5,
             missingFilter = 2.):
    """Loads currently available data"""

    # Load raw measles data
    measlesData = ensure_geo_id_column(verboseLoader(defaultLoc))
    
    if useImputation:
        measlesData = imputeMissingByGroup(measlesData,
                                           knnNeighbors,
                                           missingFilter,
                                           imputeGrouping,
                                           imputeIgnore)

    # Merge google trends data
    if os.path.exists(trendsLoc):
        trendsDf = verboseLoader(trendsLoc)
        rowUniques = trendsDf.apply(lambda row: row.nunique(), axis=1)
        trendsDf = trendsDf[rowUniques >= 10]
        trendsDf.loc[:, GEO_ID_COL] = coco.convert(names=trendsDf.alpha2, to='ISO3')
        trendsDf = trendsDf.drop(['country',
                                  'alpha2',
                                  'rank',
                                  'language',
                                  'translatable',
                                  'translation'],axis=1)
        
        trendsDf = trendsDf.groupby([GEO_ID_COL]).mean().reset_index()
        trendsDf = trendsDf.melt(id_vars=GEO_ID_COL)
        trendsDf.columns = [GEO_ID_COL, 'date', 'measles_kw_trend']
        trendsDf.to_csv('CountryMeaslesMelt.csv',index=False)
        
        measlesData = measlesData.merge(trendsDf,how='left')
    measlesData.to_csv("input/preppedData.csv",index=False)

    curves = prepCountries(measlesData)
    validCountries = list(curves.keys())

    # Load cutoffs data
    try:
        cutoffsDf = verboseLoader(cutoffsLoc).dropna()
        cutoffsDf[cutoffsCol] = pd.to_datetime(cutoffsDf[cutoffsCol],
                                               format='mixed',
                                               dayfirst=False)
        cutoffs = {row['ISO3']:row[cutoffsCol] for index, row in cutoffsDf.iterrows()}
    except:
        print("No cutoff dates file found, continuing with passed int cutoff method")
        cutoffs = dict()

    if fillMissingCutoffs:
        for key, value in cutoffs.items():
            if pd.isnull(value) and key in curves.keys():
                #print(key,value)
                cutoffs[key] = curves[key]['ds'].iloc[-12]

    # Prep auto filters
    filters = prepFilters(measlesData,validCountries)

    prepped = {'curves':curves,
               'filters':filters,
               'cutoffs':cutoffs}

    return prepped


##########################################
###   COUNTRY DATA PREP                ###
##########################################


def getRankedCountries(df):
    """Sorts countries by number of outbreaks, dropping countries with no outbreaks"""
    nCountries = df[GEO_ID_COL].nunique()
    try:
        listed = df.groupby([GEO_ID_COL]).num_outbreak_20_cuml_per_M.max().sort_values(ascending=False)
    except:
        listed = df.groupby([GEO_ID_COL]).cases_1M.max().sort_values(ascending=False)
    # listed = listed[listed > 0]
    
    print(f'{len(listed)}/{nCountries} included countries found with noted outbreaks.')
    
    return listed



def getCountryCurve(df,
                    country):
    """Prepares one country data set for fitting algorithm"""
    df = df[df[GEO_ID_COL] == country].copy(deep=True)
    df.date = pd.to_datetime(df.date)
    df = df.sort_values(by='date',ascending=True)
    df = df.reset_index()
    
    df.rename({'date':'ds'},
              axis=1,
              inplace = True)
    
    lastValid = df[validityColumn].last_valid_index()
    df = df.loc[:lastValid]
    
    return df



def prepCountries(df,
                  n='all'):
  """Prepares dictionary of dataframes of prepared data for fitting algorithm"""
  countryRank = getRankedCountries(df)
  if n == 'all':
    n = len(countryRank)

  countryCurves = {country:getCountryCurve(df,country) for country in countryRank.index[:n]}

  return countryCurves
    

##########################################
###   IMPUTE ON LOAD                   ###
##########################################


def findNumericColumns(dfIn,ignore = {}):
    """Returns a list of numeric columns for KNN imputer"""
    
    df = dfIn.copy(deep=True)
    numerics = []
    ignore = {'cases',
              'cases_1M',
              'cuml_cases',
              'cluster',
              'cuml_cases_1M',
              'Year'}
    for column in df.columns:
        if df[column].nunique() > 2:
            try:
                df[column] = df[column].astype(float)
                if column not in ignore:
                    numerics.append(column)
            except:
                pass
        
    return numerics



def imputeMissing(dfIn,
                  knnNeighbors = 5,
                  missingFilter = 1.,
                  ignore = {'cases',
                            'cases_1M',
                            'cuml_cases',
                            'cluster',
                            'cuml_cases_1M',
                            'Year'}):
    """Interpolates missing numeric values in a df via knn"""
    
    df = dfIn.copy(deep=True)
    numericCols = findNumericColumns(df)
    knn_imputer = KNNImputer(n_neighbors=knnNeighbors)
    scaler = StandardScaler()

    scaledData = scaler.fit_transform(df[numericCols])
    imputedData = knn_imputer.fit_transform(scaledData)
    unscaledData = scaler.inverse_transform(imputedData)
    unscaledDf = pd.DataFrame(unscaledData,columns=numericCols)

    df.loc[:,'Percent Imputed'] = df[numericCols].isna().sum(axis=1) / len(numericCols)
    df = df[df['Percent Imputed'] < missingFilter]
    
    df[numericCols] = unscaledData
    return df



def imputeMissingByGroup(dfIn,
                         knnNeighbors = 5,
                         missingFilter  = 1.,
                         groupCol = 'date',
                         ignore = {'cases',
                                   'cases_1M',
                                   'cuml_cases',
                                   'cluster',
                                   'cuml_cases_1M',
                                   'Year'}):
    """Imputes missing numeric values via KNN imputation"""
    
    if type(groupCol) is not str:
        imputed = imputeMissing(groupDf,
                                knnNeighbors,
                                missingFilter,
                                ignore)
        return imputed
        
    else:
        results = []
        for group, groupDf in dfIn.groupby(groupCol):
            print("Interpolating missing values in",groupCol,group)
            imputed = imputeMissing(groupDf,
                                    knnNeighbors,
                                    missingFilter,
                                    ignore)
            results.append(imputed)
    
        merged = pd.concat(results)
        return merged


