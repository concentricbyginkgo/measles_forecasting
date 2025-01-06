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
             trendsLoc = 'input/CountryMeasles2010-2024.csv',
             fillMissingCutoffs = True):
    """Loads currently available data"""

    #Load raw measles data    
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
        cutoffsDf = pd.read_csv(cutoffsLoc)
        cutoffsDf.cutoff_date = pd.to_datetime(cutoffsDf.cutoff_date)
        cutoffs = {row['ISO3']:row['cutoff_date'] for index, row in cutoffsDf.iterrows()}
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
    listed = listed[listed > 0]
    
    print(f'{len(listed)}/{nCountries} included countries found with noted outbreaks.')
    
    return listed



def getCountryCurve(df,
                    country):
    """Prepares one country data set for fitting algorithm"""
    df = df[df.ISO3 == country].copy(deep=True)
    df.date = pd.to_datetime(df.date)
    
    df.rename({'date':'ds'},
              axis=1,
              inplace = True)
    
    return df



def prepCountries(df,
                  n='all'):
  """Prepares dictionary of dataframes of prepared data for fitting algorithm"""
  countryRank = getRankedCountries(df)
  if n == 'all':
    n = len(countryRank)

  countryCurves = {country:getCountryCurve(df,country) for country in countryRank.index[:n]}

  return countryCurves


