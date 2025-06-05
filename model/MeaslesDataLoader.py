###########################################################################
###   MEASLESDATALOADER.PY                                              ###
###      * GENERALIZED DATA LOADING AND STANDARDIZATION MODULE          ###
###      * RANKS COUNTRIES BY NUMBER OF LABELLED OUTBREAKS              ###
###      * HARDCODED TO EXPECTED DATA AND FILE LOCATIONS                ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


import pandas as pd
import country_converter as coco
import os

validityColumn = 'cases_1M'


##########################################
###   CENTRALIZED DATA LOADER          ###
##########################################


def prepFilters(df,validCountries):
    """Returns a set of quick filters"""

    #Identify columns in data set with 2-9 total unique values
    totalUniqueValMatches = (df.nunique() < 12) & (df.nunique() > 1)
    uniqueValKeys = set(totalUniqueValMatches[totalUniqueValMatches].index)
    
    #Identify columns in data set with no more than one unique value per country
    onePerCountryMatches = pd.Series(df.groupby('ISO3').nunique().max())
    onePerKeys = set(onePerCountryMatches[onePerCountryMatches == 1].index)

    #Select only keys matching both prior requirements
    quickFilterKeys = uniqueValKeys.intersection(onePerKeys)

    #Prepare the request:country list mapping dictionary
    quickFilters = {country:[country] for country in validCountries}
    for filterKey in quickFilterKeys:
        grouped = dict(df.groupby(filterKey).ISO3.unique())
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
             fillMissingCutoffs = True):
    """Loads currently available data"""

    #Load raw measles data
    if not os.path.exists(defaultLoc):
        # NOTE: For public repository use, ensure your data files are in the input/ directory
        # The expected file is: input/processed_measles_model_data.csv
        # You can generate this file using the data_ingestion_pipeline scripts
        raise FileNotFoundError(f"""
        Required data file not found: {defaultLoc}
        
        To run this model, you need to:
        1. Run the data_ingestion_pipeline scripts (1-7) to generate the required input data
        2. Or place your measles model data file at: {defaultLoc}
        
        The file should contain processed measles case data with the following key columns:
        - ISO3, date, cases_1M, and other outbreak indicators
        """)
        
    measlesData = pd.read_csv(defaultLoc)

    #Merge google trends data
    if os.path.exists(trendsLoc):
        trendsDf = pd.read_csv(trendsLoc)
        rowUniques = trendsDf.apply(lambda row: row.nunique(), axis=1)
        trendsDf = trendsDf[rowUniques >= 10]
        trendsDf.loc[:,'ISO3'] = coco.convert(names=trendsDf.alpha2, to='ISO3')
        trendsDf = trendsDf.drop(['country',
                                  'alpha2',
                                  'rank',
                                  'language',
                                  'translatable',
                                  'translation'],axis=1)
        
        trendsDf = trendsDf.groupby(['ISO3']).mean().reset_index()
        trendsDf = trendsDf.melt(id_vars='ISO3')
        trendsDf.columns = ['ISO3','date','measles_kw_trend']
        trendsDf.to_csv('CountryMeaslesMelt.csv',index=False)
        
        measlesData = measlesData.merge(trendsDf,how='left')
    measlesData.to_csv("input/preppedData.csv",index=False)

    curves = prepCountries(measlesData)
    validCountries = list(curves.keys())

    #Load cutoffs data
    try:
        cutoffsDf = pd.read_csv(cutoffsLoc).dropna()
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

    #Prep auto filters

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
    nCountries = df['ISO3'].nunique()
    listed = df.groupby(['ISO3']).num_outbreak_20_cuml_per_M.max().sort_values(ascending=False)
    #listed = listed[listed > 0]
    
    print(f'{len(listed)}/{nCountries} included countries found with noted outbreaks.')
    
    return listed



def getCountryCurve(df,
                    country):
    """Prepares one country data set for fitting algorithm"""
    df = df[df.ISO3 == country].copy(deep=True)
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


