import pandas as pd
import MeaslesDataLoader as md

countryCurves = md.prepData()['curves']
populations = {country:table['population_jan'].dropna().iloc[-1] for country, table in countryCurves.items()}
recentCases = {country:table['cases'].iloc[-36:].sum() for country, table in countryCurves.items()}
totalOutbreaks = {country:table['outbreak_2_per_M'].iloc[-36:].sum() for country, table in countryCurves.items()}

failScore = 10**6

def regressorPrioritizeF1PValid(df,mlRuns):
    """
    Metric returns a fail score if less than 75% of countries ran.
    Otherwise it attempts to achieve a balance of increasing F1 score, reducing MSE, and running more countries.
    Non MSE terms are weighted highly as MSE range is inifite.
    """
    
    runnableN = df['runnable country count'].iloc[0]
    initialN = df['initial country count'].iloc[0]
    validP = runnableN / initialN

    if validP < .75:
        return failScore

    f1Loss = 1 - df['F1 Score'].mean()
    mseLoss = df['Test MSE'].median()
    validLoss = 1 - validP

    score = 1000*f1Loss + 1000*validLoss + mseLoss

    return score


def classifierSumF1MSEPValid(df,mlRuns):
    """
    Metric returns a fail score if less than 75% of countries ran.
    Otherwise it attempts to achieve a balance of increasing F1 score, reducing MSE, and running more countries.
    Non MSE terms are weighted highly as MSE range is inifite.
    """
    
    runnableN = df['runnable country count'].iloc[0]
    initialN = df['initial country count'].iloc[0]
    validP = runnableN / initialN

    if validP < .75:
        return failScore

    f1Loss = 1 - df['F1 Score'].mean()
    mseLoss = df['Test MSE'].mean()
    validLoss = 1 - validP

    score = f1Loss + validLoss + mseLoss

    return score


def regressorRMSEFlatLinePopWeight(df,mlRuns):
    """
    Metric penalizes erroneous flat line predictions and weights predictions by population
    A total modelled population score divider encourages completeness with an emphasis on larger nations
    """

    runnableN = df['runnable country count'].iloc[0]
    initialN = df['initial country count'].iloc[0]
    validP = runnableN / initialN

    if validP < .75:
        return failScore

    flatPenaltyFactor = 2
    
    df.loc[:,'RMSE'] = df['Test MSE'] ** .5
    df.loc[:,'weight'] = df.index.map(populations)
    df.loc[:,'true outbreaks'] = df['True positives'] + df['False negatives']
    df.loc[:,'predicted outbreaks'] = df['True positives'] + df['False positives']
    df.loc[:,'flat line penalty'] = ((df['predicted outbreaks'] == 0) & (df['true outbreaks'] != 0)).astype(int)
    df.loc[:,'penalty'] =  (df['RMSE'] + df['flat line penalty'] * flatPenaltyFactor) * df['weight']

    nCountries = df.index.nunique()
    modelledPopulation = df.weight.sum() / nCountries

    score = df['penalty'].mean() / modelledPopulation

    return score


def regressorRMSEFlatLineByWeight(df,
                                  mlRuns,
                                  weightsDict,
                                  inclusionBonusPow = 1.2):
    """
    Metric penalizes erroneous flat line predictions and weights predictions by flexible weights
    A total modelled population score divider encourages completeness with an emphasis on larger nations
    Global local runs with < 75% completion are dropped as failures.
    """

    runnableN = df['runnable country count'].iloc[0]
    initialN = df['initial country count'].iloc[0]
    validP = runnableN / initialN

    if validP < .70:
        return failScore

    flatPenaltyFactor = 2
    
    df.loc[:,'Test RMSE'] = df['Test MSE'] ** .5
    df.loc[:,'Train RMSE'] = df['Train MSE'] ** .5
    df.loc[:,'weight'] = df.index.map(weightsDict)
    df.loc[:,'true outbreaks'] = df['True positives'] + df['False negatives']
    df.loc[:,'predicted outbreaks'] = df['True positives'] + df['False positives']
    df.loc[:,'flat line penalty'] = ((df['predicted outbreaks'] == 0) & (df['true outbreaks'] != 0)).astype(int)
    df.loc[:,'penalty'] =  (df['Test RMSE'] + df['Train RMSE']/2 + df['flat line penalty'] * flatPenaltyFactor) * df['weight']

    nCountries = df.index.nunique()
    nReplicates = len(df) / nCountries
    totalWeight = df.weight.sum() / nReplicates

    score = df['penalty'].mean() / totalWeight ** inclusionBonusPow

    return score



def regressorRMSEFlatLinePopWeight(df,mlRuns):
    """
    Flatline penalty by population.
    """
    weightsDict = populations
    score = regressorRMSEFlatLineByWeight(df,mlRuns,weightsDict)

    return score



def regressorRMSEFlatLineCasesWeight(df,mlRuns):
    """
    Flatline penalty by recent cases.
    """
    weightsDict = recentCases
    score = regressorRMSEFlatLineByWeight(df,mlRuns,weightsDict)

    return score


def regressorRMSEFlatLineOutbreaksWeight(df,mlRuns):
    """
    Flatline penalty by total outbreak months.
    """
    weightsDict = recentCases
    score = regressorRMSEFlatLineByWeight(df,mlRuns,weightsDict)

    return score



    


def classifierF1RecallByWeight(df,
                               mlRuns,
                               weightsDict,
                               alpha=0.6,
                               recallWeight=0.7,
                               inclusionBonusPow = 1.2):
    """
    Metric calculates a combined Macro-F1 and weighted recall-specificity score
    for outbreak predictions across countries. Penalizes incomplete runs (<75%).
    
    Parameters:
    - df: DataFrame with true/false positives and negatives for each country
    - mlRuns: Number of machine learning runs (for logging, optional)
    - weightsDict: Dictionary to weight each country's contribution
    - alpha: Weighting factor between Macro-F1 and recall-specificity score (default: 0.6)
    - recall_weight: Weight for recall vs specificity in recall-specificity score (default: 0.7)
    """

    runnableN = df['runnable country count'].iloc[0]
    initialN = df['initial country count'].iloc[0]
    validP = runnableN / initialN
    df['weight'] = df.index.map(weightsDict)

    if validP < 0.30:
        return failScore

    df['Recall score'] = df['True positives'] / (df['True positives'] + df['False negatives'] + 1e-9)
    df['Specificity score'] = df['True negatives'] / (df['True negatives'] + df['False positives'] + 1e-9)
    df['Precision score'] = df['True positives'] / (df['True positives'] + df['False positives'] + 1e-9)
    df['F1 score'] = 2 * (df['Precision score'] * df['Recall score']) / (df['Precision score'] + df['Recall score'] + 1e-9)
    df['Recall-Specificity'] = (recallWeight * df['Recall score']) + ((1 - recallWeight) * df['Specificity score'])

    totalWeight = df['weight'].sum() ** inclusionBonusPow
    macroF1Score = (df['F1 score'] * df['weight']).sum() / totalWeight
    weightedRecallSpecScore = (df['Recall-Specificity'] * df['weight']).sum() / totalWeight
    trainMAEPenalty = 2 * (df['Train MAE'] * df['weight']).sum() / totalWeight

    score = trainMAEPenalty - (alpha * macroF1Score + (1 - alpha) * weightedRecallSpecScore)

    return score




def classifierF1RecallCasesWeight(df,mlRuns):
    """
    Flatline penalty by recent cases.
    """
    weightsDict = recentCases
    score = classifierF1RecallByWeight(df,mlRuns,weightsDict)

    return score


regressorsLabelMetric = lambda x: x >= 5
classifiersLabelMetric = lambda x: int(x) >= 1
regressorsSeriesMetric = regressorsLabelMetric
classifiersSeriesMetric = lambda x: x.astype(int) >= 1


lossFunctions = {'Classifier_Sum_F1_MSE_PValid': classifierSumF1MSEPValid,
                 'Regressor_Prioritize_F1_PValid': regressorPrioritizeF1PValid,
                 'Regressor_Popweighted_Flatline_RMSE': regressorRMSEFlatLinePopWeight,
                 'Regressor_Caseweighted_Flatline_TestTrainRMSE': regressorRMSEFlatLineCasesWeight,
                 'Regressor_Outbreakweighted_Flatline_TestTrainRMSE': regressorRMSEFlatLineOutbreaksWeight,
                 'Classifier_CaseWeighted_MacroF1_TrainMAE_Recall': classifierF1RecallCasesWeight}
               
