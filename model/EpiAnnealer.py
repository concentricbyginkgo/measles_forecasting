import json
import pickle
import os
import sys

import pandas as pd
import numpy as np
import multiprocess as mp

from matplotlib import pyplot as plt
from IPython.display import clear_output
from random import seed, shuffle, randint, choice, random, uniform
from ast import literal_eval
from copy import deepcopy

import MeaslesDataLoader as md
import MeaslesModelEval as mm
import ModelSweeps as ms
import EpiPreprocessor as ep
import LossFunctions as lf

print(sys.argv)

try:
    (_,method,skipWBIndicators,offset,skip) = sys.argv
    offset = int(offset)
    skip = int(skip)
except:
    (_,method,skipWBIndicators) = sys.argv
    offset = 0
    skip = 1


if method.lower().startswith('class'):
    depVar = 'outbreak_5_per_M'
    models = ms.classifierObjects
    
elif method.lower().startswith('regress'):
    depVar = 'cases_1M'
    models = ms.regressorObjects


skipWorldBank = skipWBIndicators[0].lower() in {'t','y'}


preppedCountries = md.prepData()
defaultConfigURL = 'https://docs.google.com/spreadsheets/d/1zn0e2-hi-p9hcOWiMk03v4TH_zq64BfD9pUgbAf1uRs/edit?gid=1118061464#gid=1118061464'
preprocessorConfig = ep.getGoogleSheetConfig(defaultConfigURL)


rawData = pd.read_csv('input/processed_measles_model_data.csv')
usableColumns = set(rawData.columns)
worldBankColumns = {column for column in usableColumns if len(column.split('.')[0]) == 2}

ignoredColumns = {'ISO3','Year','Region','Country',
                  'mnth','Month','cases','date',
                  'births','migrations','total_precip_mm_per_day','unicef_region',
                  'cases_1M','cluster'}
if skipWorldBank:
    ignoredColumns = ignoredColumns.union(worldBankColumns)
usableColumns = sorted(list(usableColumns.difference(ignoredColumns)))
failScore = 10**6



def sortDict(dictIn):
    """Returns dict contents in predictable order"""
    return dict(sorted(dictIn.items()))


def scriptToIndepVars(script):
    """Takes a run script and returns a sorted indepVars dict"""
    indepVars = dict()
    for line in script:
        if line != '{}':
            indepVars.update(literal_eval(line))
    indepVars = sortDict(indepVars)
    return indepVars



class epiAnnealer:
    def __init__(self,
                 geography,
                 metric,
                 models = ms.regressorObjects,
                 usableColumns = usableColumns,
                 r = 1,
                 depVar = 'cases_1M',
                 maxLag = 9,
                 scriptLength = 60,
                 percentBlank = .9,
                 changeRate = .25,
                 stagnantLimit = 500):
        """Initializes the model, loading a past instance if available"""

        self.geography = geography
        self.depVar = depVar
        self.name = metric
        self.r = r
        self.seeds = [1337*i for i in range(r)]
        self.metric = lf.lossFunctions[metric]

        if metric.lower().startswith('class'):
            self.binaryLabelMetric = lf.classifiersLabelMetric
        elif metric.lower().startswith('regr'):
            self.binaryLabelMetric = lf.regressorsLabelMetric  
        
        self.scriptLength = scriptLength
        self.percentBlank = percentBlank
        self.changeRate = changeRate
        self.modelObjects = models
        self.modelNames = list(self.modelObjects.keys())
        self.usableColumns = usableColumns
        self.lags = list(range(maxLag+1))
        self.stagnantLimit = stagnantLimit

        firstScript = self.getFirstScript()

        self.bestScript = firstScript
        self.bestIndepVars = scriptToIndepVars(self.bestScript)
        self.bestScore = failScore
        self.bestModel = choice(self.modelNames)

        self.filePrefix = f"output/annealing/{geography.replace(':','-')}_{self.name}_"
        self.logFile = f'{self.filePrefix}Log.txt'
        self.bestVarsFile = f'{self.filePrefix}BestVars.txt'
        self.bestScriptFile = f'{self.filePrefix}BestScript.txt'

        self.runNumber = 0
        self.lastImproved = 0
        self.stagnant = False

        stripLines = lambda x: [line.rstrip('\n') for line in x]
        
        if os.path.exists(self.bestVarsFile):
            print("Loading past instance from",self.bestVarsFile)
            
            with open(self.bestScriptFile,'r') as fileIn:
                bestScript = fileIn.readlines()
            
            bestScript = stripLines(bestScript)
            self.bestScript = bestScript
                
            with open(self.bestVarsFile,'r') as fileIn:
                bestVarLines = fileIn.readlines()
            
            bestVarLine = stripLines(bestVarLines)[-1]
            bestVars = json.loads(bestVarLine)
            self.bestModel = bestVars['model']
            self.bestIndepVars = {key:var for key,var in bestVars['indepVars'].items() if not key.startswith("ID_")}
            self.runNumber = bestVars['runNumber']
            self.lastImproved = bestVars['runNumber']
            self.bestScore = bestVars['score']

            _,self.bestScore,_ = self.runOne(self.bestScript,self.bestModel)
            clear_output()
                


    def getIndepVar(self):
        """Generates json text for one indepvar"""
        text = '{"'+choice(self.usableColumns)+'":'+str(choice(self.lags))+'}'
        return text
    
    
    def getFirstScript(self):
        """Generates a script from scratch"""
        success = False
        while not success:
            script = []
            for i in range(self.scriptLength-2):
                if random() < self.percentBlank:
                    script.append('{}')
                else:
                    script.append(self.getIndepVar())
            script.append('{"mean_precip_mm_per_day":3}')
            script.append('{"mean_max_temp":3}')
            success = len(set(script)) > 3
        return script

    
    def annealScript(self):
        """Randomly alters the best known script"""
        script = deepcopy(self.bestScript)
        for i, row in enumerate(script):
            if random() < self.changeRate:
                if random() < self.percentBlank:
                    script[i] = '{}'
                else:
                    script[i] = self.getIndepVar()
            else:
               script[i] = row
        return script
                
    
    def annealModel(self):
        """Changes the model selection randomly"""
        if random() < self.changeRate:
            modelName = choice(self.modelNames)
        else:
            modelName = self.bestModel
        return modelName

    
    def annealOnce(self):
        """Runs the annealer once, updating the best run tracking if an improvement is noted"""
        if self.runNumber - self.lastImproved > self.stagnantLimit:
            self.stagnant = True
            return
        
        if self.runNumber == 0:
            script = self.bestScript
            modelName = self.bestModel
        else:
            done = False
            indepVarsIn = scriptToIndepVars(self.bestScript)
            while not done:
                script = self.annealScript()
                indepVarsOut = scriptToIndepVars(script)
                modelName = self.annealModel()
                if modelName != self.bestModel or indepVarsOut != indepVarsIn:
                    done = True
                
        tables, score, varsRan = self.runOne(script,modelName)
        if score != score and self.bestScore == failScore:
            self.bestScript = self.getFirstScript()
            self.bestModel = choice(self.modelNames)

        if score < self.bestScore and score == score:
            self.bestScript = script
            self.bestIndepVars = varsRan
            self.bestScore = score
            self.bestModel = modelName
            self.lastImproved = self.runNumber
            bestParams = {'model':modelName,
                          'indepVars':self.bestIndepVars,
                          'runNumber':self.runNumber,
                          'score':self.bestScore}
            with open(self.bestVarsFile,'a+') as fileOut:
                json.dump(bestParams,fileOut)
                fileOut.write('\n')
            with open(self.bestScriptFile,'w') as fileOut:
                fileOut.write('\n'.join(self.bestScript).replace('\n\n','\n'))

        clear_output()

        logRow = f'Run: {self.runNumber} \tScore: {score} \tBest: {self.bestScore} \tRows: {len(tables)/self.r}\n'

        with open(self.logFile,'a+') as fileOut:
            fileOut.write(logRow)

        self.runNumber += 1        
        return

            

    def runOne(self,
               script,
               modelName):
        """Runs one annealer script"""
        
        indepVars = scriptToIndepVars(script)
        model = self.modelObjects[modelName]
        tables = []
        runs = []
        
        print('Run:',model,'\n',indepVars)
        try:
            for seed in self.seeds:    
                if type(model) is not dict:
                    mlRun = model(self.geography,
                                  self.depVar,
                                  indepVars = indepVars,
                                  binaryLabelMetric = self.binaryLabelMetric,
                                  randomState = seed,
                                  useCache = False)
                
                elif type(model) is dict:
                    mlRun = mm.sklGeneric(self.geography,
                                          self.depVar,
                                          indepVars = indepVars,
                                          modelArgs = model,
                                          binaryLabelMetric = self.binaryLabelMetric,
                                          randomState = seed,
                                          useCache = False)
    
                mlRun.train()
                tables.append(mlRun.evaluate())
                #runs.append(mlRun)
                
            tables = pd.concat(tables)
            score = self.metric(tables,runs)
            varsRan = mlRun.indepVars
            varsRan = {key:var for key,var in varsRan.items() if not key.startswith("ID_")}
        except:
            tables = pd.DataFrame()
            score =  np.nan
            varsRan = dict()


        return tables,score,varsRan
        





def sweepAnnealers(annealers):
    """Runs every annealer in a cluster of annealers until stagnant""" 
    for annealer in annealers:
        gen = 0
        stagnant = False
        while not stagnant:
            gen += 1
            annealer.annealOnce()
            stagnant = annealer.stagnant
            if gen % 10 == 0:
                annealer.changeRate = uniform(.05,.5)
                annealer.percentBlank = uniform(.5,1.0)




geographies = list(preppedCountries['curves'].keys()) + [key for key in preppedCountries['filters'].keys() if key.startswith('cluster')] + ['all']
geographies = geographies[offset::skip]


annealers = [epiAnnealer(geography,
                         method,
                         depVar = depVar,
                         models = models) for geography in geographies]

sweepAnnealers(annealers)
        
