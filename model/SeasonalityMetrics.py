###########################################################################
###   SEASONALITYMETRICS.PY                                             ###
###      * USED TO ASSIST IN LABELLING THE SEASONAL TREND OF INPUTS     ###
###      * CALCULATES FOUR INDIVIDUAL METRICS AND RETURNS COMBINE SCORE ###
###                                                                     ###
###             Contact: James Schlitt ~ jschlitt@ginkgobioworks.com    ###
###########################################################################


import pandas as pd
import numpy as np
import MeaslesDataLoader as md
import statsmodels.api as sm
import ordpy


from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import minmax_scale

from matplotlib import pyplot as plt



##########################################
###   SEASONALITY METRICS              ###
##########################################



def getStatsModelsACF(curve):
    """Returns the autocorrelation function at 12 month intervals as a measure of the strength of the annual trend"""
    acfValues = sm.tsa.acf(curve, nlags=24)
    annualACFValue = np.abs(acfValues[12])

    return annualACFValue



def getPowerSpectrumPeakScore(curve):
    """Computes the Fourier Transform of the input time series to obtain its frequency components"""
    # Fourier Transform Analysis
    n = len(curve)  # Ensure n is defined
    fftValues = fft(curve)
    frequencies = fftfreq(n)

    # Calculates the power spectrum to understand the power distribution across different frequencies.
    powerSpectrum = np.abs(fftValues)**2

    # Identify the peak in the power spectrum corresponding to annual frequency
    annualFreq = 1 / 12
    peakIndices, _ = find_peaks(powerSpectrum)
    if len(peakIndices) == 0:
        return 0.0
    peakFreqs = frequencies[peakIndices]
    peakPowers = powerSpectrum[peakIndices]

    # Calculates the ratio of the annual peak power to the total power in the spectrum
    annualPeakIndex = np.argmin(np.abs(peakFreqs - annualFreq))
    annualPeakPower = peakPowers[annualPeakIndex]
    peakPowerScore = annualPeakPower / np.sum(powerSpectrum)

    return peakPowerScore


def getDecompSeasonalityScore(curve):
    """Measures the consistency of the seasonal component of a time series over multiple years"""
    # Seasonal Decomposition
    decomposition = seasonal_decompose(curve, model='additive', period=12)
    seasonalComponent = decomposition.seasonal

    # Calculate the standard deviation of the seasonal component for each year
    numYears = len(seasonalComponent) // 12
    seasonalityStds = []
    for year in range(numYears):
        yearlySeasonal = seasonalComponent[year*12:(year+1)*12]
        seasonalityStds.append(np.std(yearlySeasonal))
    
    # Normalize the standard deviation (lower std means more consistent seasonality)
    seasonalityConsistencyScore = (np.mean(seasonalityStds) / np.std(curve))

    return seasonalityConsistencyScore



def getDecompIntervalScore(curve):
    """Measures consistency in the intervals between detected peaks"""
    peaks, _ = find_peaks(curve, distance=10)  # Ensure peaks are sufficiently spaced
    if len(peaks) < 2:
        return 0.0
    
    peakIntervals = np.diff(peaks)
    meanInterval = np.mean(peakIntervals)
    intervalConsistencyScore = 1 - (np.std(peakIntervals) / meanInterval)

    return intervalConsistencyScore



##########################################
###   ONE SHOT ANALYSIS METHOD         ###
##########################################



def getSeasonalityMetrics(countryData,column):
    """Applies four different metrics of seasonality to a country's monthly data"""
    data = countryData.copy(deep=True)
    data.index = data.ds
    curve = data[column].fillna(0).values

    # Ensure data has enough points for decomposition
    if len(curve) < 24:  # At least two years of monthly data
        dict()
    
    smACF = getStatsModelsACF(curve)
    peakPowerScore = getPowerSpectrumPeakScore(curve)
    seasonalConsistencyScore = getDecompSeasonalityScore(curve)
    intervalConsistencyScore = getDecompIntervalScore(curve)

    result = {'Annual autocorrelation': smACF,
              'Peak power ratio': peakPowerScore,
              'Seasonal consistency': seasonalConsistencyScore,
              'Peak interval consistency': intervalConsistencyScore}

    return result



##########################################
###   DATA METRICS TABLE PREP          ###
##########################################



def scoreCountry(dataIn,
                 column = 'cases_1M',
                 withheld = 0,):
    """Score one country's data quality by coverage, contig length, and potential losses"""
    data = dataIn[column].copy(deep=True)
    if withheld != 0:
        data = data[:-withheld]
        
    count = len(data)
    data.index = range(count)
    
    # Simple percent coverage
    filled = data.notna()
    valid = filled.sum()
    coverage = valid / len(data)

    # Length of average filled section
    filledStr = str(filled.tolist())[1:-1] + ', '
    filledSplit = [i for i in filledStr.split('False, ') if i != '']
    nSegments = len(filledSplit)
    if nSegments > 0:
        meanLength = sum([i.count('True, ') for i in filledSplit])/nSegments
    else:
        meanLength = 0

    # Proportion of sum adjacent to a nan
    nanLocs = data[data.isna()].index
    adjacent = set()
    columnSum = data.sum()

    for pos in nanLocs:
        if pos - 1 >= 0:
            adjacent.add(pos - 1)
        if pos + 1 < count:
            adjacent.add(pos + 1)

    percentAdjacent = data.loc[list(adjacent)].sum()/columnSum

    permutationEntropy = ordpy.permutation_entropy(data, dx=3)


    metrics = {'coverage':coverage,
               'mean segment length':meanLength,
               'weighted percent adjacent to gaps':percentAdjacent,
               'permutation entropy':permutationEntropy}

    return metrics



def scoreCountries(prepped,
                   column,
                   withheld = 0):
    """Score all countries' data quality by coverage, contig length, and potential losses"""
    countries = sorted(list(prepped['curves'].keys()))
    results = []
    for country in countries:
        scored = scoreCountry(prepped['curves'][country],
                              column,
                              withheld = withheld)
        scored['country'] = country
        results.append(scored)
    merged = pd.DataFrame(results)
    merged = merged.set_index(['country'])
    merged = merged.sort_index()

    return merged



def scoreDepVIndepVars(prepped,
                       depVar,
                       indepVars,
                       withheld = 0):
    """Scores the dependent and independent variables of an experiment, returning mean quality metrics for the independent vars"""
    depVarDf = scoreCountries(prepped,depVar,withheld)
    indepVarsDfs = {indepVar:scoreCountries(prepped,indepVar,withheld) for indepVar in indepVars.keys()}
    meanIndepVars = pd.concat(indepVarsDfs).groupby(level=1).mean()

    depVarDf.columns = ['depvar ' + column for column in depVarDf.columns]
    meanIndepVars.columns = ['indepvar ' + column for column in meanIndepVars.columns]

    mergedQualityMetrics = pd.concat([depVarDf,meanIndepVars],axis = 1)
    
    return mergedQualityMetrics

    
def assessSeasonality(countryData,
                     column,
                     withheld = 0):
    """Calculates seasonality metrics for all countries within prepped country data"""
    results = []
    for country, table in countryData['curves'].items():
        if withheld != 0:
            curve = table[:-withheld]
        else:
            curve = table
        try:
            metrics = getSeasonalityMetrics(curve,column)
            metrics['ISO3'] = country
            results.append(metrics)
        except:
            pass

    results = pd.DataFrame(results)
    results = results.set_index('ISO3')
    
    normedResults = (results-results.min())/(results.max()-results.min())
    results.loc[:,'Combined seasonality score'] = normedResults.mean(axis=1)
    
    return results


##########################################
### ONE SHOT DATA METRICS TABLE PREP   ###
##########################################


def getDataMetrics(preppedData,
                   depVar,
                   indepVars,
                   withheld = 0):
    """Gets the data seasonality and quality metrics for a given prepped data set with a fixed witholding period"""
    qualityMetrics = scoreDepVIndepVars(preppedData,
                                        depVar,
                                        indepVars,
                                        withheld = withheld)
    
    seasonalityMetrics = assessSeasonality(preppedData,
                                           depVar,
                                           withheld = withheld)
    
    mergedDataMetrics = seasonalityMetrics.merge(qualityMetrics,
                                                 left_index=True,
                                                 right_index=True)
    
    return mergedDataMetrics
