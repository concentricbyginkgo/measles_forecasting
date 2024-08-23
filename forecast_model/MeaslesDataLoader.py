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
###   WIP DATA LOADER                  ###
##########################################


def prepData(defaultLoc = 'processed_measles_model_data.csv',
             trendsLoc = 'null'):
    """Loads currently available data"""

    measlesData = pd.read_csv(defaultLoc)

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

    prepped = prepCountries(measlesData)

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
