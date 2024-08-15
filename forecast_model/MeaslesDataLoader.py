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


##########################################
###   WIP DATA LOADER                  ###
##########################################


def prepData():
    """Loads currently available data"""

    measlesData = pd.read_csv('processed_measles_model_data.csv')
    
    measlesData = measlesData.merge(trendsDf,how='left')
    measlesData.outbreak_20_per_M = measlesData.outbreak_20_per_M.map({'yes':1,'no':0})
    measlesData.outbreak_2_per_M = measlesData.outbreak_2_per_M.map({'yes':1,'no':0})
    measlesData.outbreak_20_cuml_per_M = measlesData.outbreak_20_cuml_per_M.map({'yes':1,'no':0})
    measlesData['date'] = pd.to_datetime(measlesData['date'])
    measlesData = measlesData[measlesData['date'] >= '2012-01-01']
    measlesData.to_csv("preppedData.csv",index=False)

    prepped = prepCountries(measlesData)

    return prepped


##########################################
###   COUNTRY DATA PREP                ###
##########################################


def getRankedCountries(df):
  """Sorts countries by number of outbreaks, dropping countries with no outbreaks"""
  nCountries = df['Country'].nunique()
  listed = df.groupby(['Country']).num_outbreak_20_cuml_per_M.max().sort_values(ascending=False)
  listed = listed[listed > 0]

  print(f'{len(listed)}/{nCountries} included countries found with noted outbreaks.')

  return listed



def getCountryCurve(df,
                    country):
  """Prepares one country data set for fitting algorithm"""
  depVars = ['cases',
             'cases_1M']
  indepVars = ['births',
               'birth_per_1k',
               'mnths_since_outbreak_20_per_M',
               'mnths_since_outbreak_2_per_M',
               'mnths_since_outbreak_20_cuml_per_M',
               'mean_precip_mm_per_day',
               'total_precip_mm_per_day',
               'measles_kw_trend',
               'outbreak_20_per_M',
               'outbreak_2_per_M',
               'outbreak_20_cuml_per_M',
               'date',
               'MCV1']

  df = df[df.Country == country][depVars+indepVars].copy(deep=True)
  #df = df.dropna()

  df.outbreak_20_cuml_per_M = (df.outbreak_20_cuml_per_M == 'yes').astype(int)
  df.outbreak_20_per_M = (df.outbreak_20_per_M == 'yes').astype(int)
  df.outbreak_2_per_M = (df.outbreak_2_per_M == 'yes').astype(int)

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



