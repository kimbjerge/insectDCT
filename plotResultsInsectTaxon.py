# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:15:07 2025

@author: Kim Bjerge
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from numpy.linalg import norm
import datetime
#from scipy.stats import expon
from idac.configreader.configreader import readconfig
from idac.predictions.predictions import Predictions
from scipy.stats import pearsonr

ignoreInsectNames = ["Vegetation", "Unsure"]

config_filename = './Taxon_config.json'

"""
labelInsectsPlot = ["Apidae", "Apis mellifera","Bombus terrestris", "Bombus lapidarius", "Bombus pascuorum", 
                    "Apoidea", "Apoidea small", "Hymenoptera_nobees", "Vespula vulgaris", "Syrphidae", 
                    "Eristalis",  "Eupeodes", "Episyrphus balteatus", "Sphaerophoria scripta-complex", "Aglais urticae"]

projectPath ="/RTNI/"
trackPath = projectPath + "tracks_V6/"
detectionPath = projectPath + "detections_V6/"
plotsDir = projectPath + "plots_V6/"
"""

projectPath ="/MAMBO/"
selectedYear = 2024
trackPath = projectPath + "tracks_V6/"
detectionPath = projectPath + "detections_V6/"
plotsDir = projectPath + "detections_V6/plots/"

class timedate:
    
    def __init__(self):
        print('timeDate')
        
    # Functions to get seconds, minutes and hours from time in predictions
    def getSeconds(self, recTime):
        return int(recTime%100)
    
    def getMinutes(self, recTime):
        minSec = recTime%10000
        return int(minSec/100)
    
    def getHours(self, recTime):
        return int(recTime/10000)
    
    def getTimesec(self, recTime):
        
        timestamp = self.getSeconds(recTime)
        timestamp += self.getMinutes(recTime)*60
        timestamp += self.getHours(recTime)*3600
        return timestamp
    
    # Functions to get day, month and year from date in predictions
    def getDay(self, recDate):
        return int(recDate%100)
    
    def getMonthDay(self, recDate):
        return int(recDate%10000)
    
    def getMonth(self, recDate):
        return int(self.getMonthDay(recDate)/100)
    
    def getYear(self, recDate):
        return int(recDate/10000)
    
    def strMonthDay(self, recDate):
        text = str(self.getDay(recDate)) + '/' + str(self.getMonth(recDate))
        return text
    
    def getDayOfYear(self, recDate):
        date = datetime.datetime(self.getYear(recDate), self.getMonth(recDate), self.getDay(recDate))
        return date.strftime('%j')
    
    def formatTime(self, recTime):
        minutes = self.getMinutes(recTime)
        seconds = self.getSeconds(recTime)
        text = str(minutes).zfill(2) + ":" + str(seconds).zfill(2)
        return text
    
def createDatelist(dataset, useDetections=False):

    td = timedate()
    currentDate = 0
    nextDate = 0
    dates = []
    dayOfYears = []
    
    if useDetections:
        dateKey = 'date'
    else:        
        dateKey = 'startdate'
        
    for i, obj in dataset.iterrows():
        if currentDate != obj[dateKey]:
            if nextDate != obj[dateKey]:
                print("NextDate", nextDate)
            currentDate = obj[dateKey]
            nextDate = currentDate + 1
            dates.append(currentDate)
            dayOfYear = td.getDayOfYear(currentDate)
            #print(dayOfYear)
            dayOfYears.append(int(dayOfYear))

    return dates, dayOfYears

def countAbundance(dataset, dates, useDetections=False):

    if useDetections:
        dateKey = 'date'
    else:        
        dateKey = 'startdate'
        
    abundance = np.zeros(len(dates)).tolist()
    for i, obj in dataset.iterrows():
        dateIdx = dates.index(obj[dateKey])
        abundance[dateIdx] += 1

    return abundance
  
def countSnapAbundance(predicted, dates, labelName, valid=True):

    abundance = np.zeros(len(dates)).tolist()
    for insect in predicted:
        if insect["taxaLabel"] == labelName:
            if valid == False or insect['taxaSure'] == True:
                if insect['date'] in dates:
                    dateIdx = dates.index(insect['date'])
                    abundance[dateIdx] += 1
                else:
                    print("Date did not exist", insect['date'])

    return abundance    

def loadTrackFiles(trap, countsTh, percentageTh, trackPath=trackPath):

    #trackFiles = trackPath + trap + '/'  
    trackFiles = trackPath 
    dataframes = []
    for fileName in sorted(os.listdir(trackFiles)):
        if "TR.csv" in fileName:
            #print(trap, trackFiles + fileName)
            #data_df = pd.read_json(trackFiles + fileName)
            print(fileName)
            data_df = pd.read_csv(trackFiles + fileName)
            dataframes.append(data_df)                
    dataset = pd.concat(dataframes)
    print(trap, len(dataset))
    dateList, dayOfYear = createDatelist(dataset)
    selDataset1 = dataset.loc[dataset['confidence'] >= percentageTh]
    selDataset2 = selDataset1.loc[selDataset1['counts'] >= countsTh]
    
    return dateList, dayOfYear, selDataset2

def loadDetectionFiles(trap, year="", detectionPath=detectionPath):
    
    detectionFiles = detectionPath + trap + '/'
    dataframes = []
    for fileName in sorted(os.listdir(detectionFiles)):
        if "CL.csv" in fileName:
            #print(trap, trackFiles + fileName)
            #data_df = pd.read_json(trackFiles + fileName)
            print(fileName)
            data_df = pd.read_csv(detectionFiles + fileName)
            dataframes.append(data_df)                
    dataset = pd.concat(dataframes)
    dataset = dataset.sort_values(by=['date', 'time'])
    print("Raw dataset", trap, len(dataset))
    
    selDataset1 = dataset.loc[dataset['taxaSure'] == True] # Skip unsure classifications
    if year != "": # Select only detections from specific year
        selDataset1 = selDataset1.loc[selDataset1['year'] == year]
    selDataset2 = selDataset1.loc[selDataset1['taxaLabel'] != 'Vegetation'] # Skip vegetation
    print("Filtered dataset", trap, len(selDataset2))

    dateList, dayOfYear = createDatelist(selDataset2, useDetections=True)
    
    return dateList, dayOfYear, selDataset2

def loadSnapFiles(trap):
    
    conf = readconfig(config_filename)
    predict = Predictions(conf)
    trackSnapFile = detectionPath + trap + ".csv"
    threshold=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    print("Load taxa predictions", trackSnapFile)
    predicted = predict.load_predictions_taxon(trackSnapFile, filterTime=0, threshold=threshold)
    
    return predicted

def findInsectSpecies(dataframe, numSpecies, useDetections=False):
    
    insectSpecies = {}
    for index, row in dataframe.iterrows():
        if useDetections:
            className = row['taxaLabel']
        else:
            className = row['class']
        if className not in ignoreInsectNames:
            #print(className)
            if className in insectSpecies.keys():
                insectSpecies[className] += 1
            else:
                insectSpecies[className] = 1
    
    insectSpeciesSorted = dict(sorted(insectSpecies.items(), key=lambda item: item[1], reverse=True))
    insectSpeciesNames = []
    for i, key in enumerate(insectSpeciesSorted):
        insectSpeciesNames.append(key)
        if i >= numSpecies-1: 
            break;
   
    return insectSpeciesSorted, insectSpeciesNames

def plotInsectSpecies(trap, insectSpecies, resultFileName, numSpecies, selectedYear="", useDetections=False):
    
    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
    plt.rcParams.update({'font.size': 18})
    ax = figure.add_subplot(1,1,1)
    
    species = []
    abundance = []
    numSpeciesFive = 0
    numSpeciesAll = 0
    for i, key in enumerate(insectSpecies):
        if i < numSpecies:
            species.append(key)
            abundance.append(insectSpecies[key])
        if insectSpecies[key] > 4:
            numSpeciesFive += 1
        numSpeciesAll += 1
            
    print(trap, numSpeciesFive, numSpeciesAll)    

    #bar_labels = ['red', 'blue', 'red', 'orange']
    #bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
    
    ax.barh(species, abundance)
    
    if useDetections:
        ax.set_xlabel('Detections')
    else:
        ax.set_xlabel('Tracks')
    ax.set_title('Abundance of insect taxon ' + trap + ' ' + selectedYear)
    plt.tight_layout(pad=2.0)
    plt.savefig(plotsDir + resultFileName + "LT" + selectedYear + ".png")
    
    plt.show()

def plotAbundanceAllClasses(trap, countsTh, percentageTh, resultFileName, useDetections=False, useSnapImages=False):
  
    firstYear = True
    trackFiles = trackPath
    yearTrackPath = trackFiles
    
    if useDetections:
        dateList, dayOfYear, selDataset2 = loadDetectionFiles(trap, year=selectedYear, detectionPath=detectionPath)        
    else:
        dateList, dayOfYear, selDataset2 = loadTrackFiles(trap, countsTh, percentageTh, trackPath=yearTrackPath)
    
    insectSpecies, insectSpeciesNames = findInsectSpecies(selDataset2, 15, useDetections=useDetections)
    plotInsectSpecies(trap, insectSpecies, resultFileName, numSpecies=50, useDetections=useDetections)
  
    td = timedate()
    subtitle = trap + " (" + td.strMonthDay(dateList[0]) + '-' + td.strMonthDay((dateList[-1])) + ")"
    
    print(subtitle)
 
    if useSnapImages:    
        predicted = loadSnapFiles(trap)
    
    figure = plt.figure(figsize=(20,20))
    figure.tight_layout(pad=1.0)
    plt.rcParams.update({'font.size': 20})
                          
    idxFig = 1
    if firstYear == True:
        firstYear = False
        labelNamesPlot = insectSpeciesNames 
        #labelNamesPlot = labelInsectsPlot
        
    for labelName in labelNamesPlot:

        if "ax" in locals():
            ax = figure.add_subplot(5, 3, idxFig, sharex = ax) #, sharey = ax) 
        else:
            ax = figure.add_subplot(5, 3, idxFig) 
             
        title = labelName
        colorIdx = 0
        if useSnapImages:
            colors = ["green", "cyan",  "orange", 
                      "orange","green", "cyan", 
                      "green", "cyan",  "orange", 
                      "orange","green", "cyan",  
                      "green", "cyan",  "orange"]
        else:
            colors = ["green", "red", "purple", 
                      "brown", "brown", "purple", 
                      "olive",  "cyan", "orange", 
                      "red",   "green", "blue", 
                      "cyan", "orange", "olive"]
            
                  
        #labelNamesPlot = ["Araneae", "Coleoptera", "Diptera Brachycera", "Diptera Nematocera", "Diptera Tipulidae", 
        #                  "Diptera Trichocera", "Ephemeroptera", "Hemiptera", "Hymenoptera Other", "Hymenoptera Vespidae", 
        #                  "Lepidoptera Macros", "Lepidoptera Micros", "Neuroptera", "Opiliones", "Trichoptera"]
        if useDetections:
            selDataset = selDataset2.loc[selDataset2['taxaLabel'].str.contains(labelName)]
        else: 
            selDataset = selDataset2.loc[selDataset2['class'].str.contains(labelName)]
            
        abundance = countAbundance(selDataset, dateList, useDetections=useDetections)
        print(trap, labelName, len(selDataset), sum(abundance))

        labelText = labelName #+ ' ' + str(countsTh*2) + 's'
        colorIdx = labelNamesPlot.index(labelName)
        
        if useDetections:
            ax.plot(dayOfYear, abundance, label="Detections", color=colors[colorIdx])
        else:
            ax.plot(dayOfYear, abundance, label="Tracks", color=colors[colorIdx])
        
        if useSnapImages:    
            abundanceSnap = countSnapAbundance(predicted, dateList, labelName)
            ax.plot(dayOfYear, abundanceSnap, label="TL", color="black")
            correlation, _ = pearsonr(abundance, abundanceSnap)
            correlation = np.round(correlation * 100)/100
            title += r" ($\rho$=" + str(correlation) + ")"
  
        ax.set_title(title)
        if useSnapImages and idxFig == 1:
            ax.legend()  
        if useSnapImages:
            ax.set_yscale('log')
        if idxFig in [13, 14, 15]: 
            ax.set_xlabel('Day of Year')
        ax.set_xlim(dayOfYear[0], dayOfYear[-1]) # NB adjust for days operational
        if idxFig in [1, 4, 7, 10, 13]: 
            if useSnapImages or useDetections:
                ax.set_ylabel('Detections')
            else:
                ax.set_ylabel('Tracks')
        #ax.set_xlim(130, 310)
        #ax.set_xlim(205, 250)
        
        idxFig += 1
  
    plt.suptitle(subtitle)
    plt.tight_layout(pad=2.0)
    plt.savefig(plotsDir + resultFileName + ".png")
    plt.show() 

if __name__ == '__main__':

    countsTh = 3 # 4 sec or three detections in one track
    percentageTh = 50  
         
    plt.rcParams.update({'font.size': 14})
    
    """
    traps = ['RTNI']
    for trap in traps:
        plotAbundanceAllClasses(trap, countsTh, percentageTh, "abundance")
    """
    
    traps = ['au', 'cirad', 'ecoinn', 'ufz', 'ukceh', 'uva']
    for trap in traps:
        plotAbundanceAllClasses(trap, countsTh, percentageTh, trap + "_abundance", useDetections=True)

