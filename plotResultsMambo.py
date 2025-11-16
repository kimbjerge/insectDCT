# -*- coding: utf-8 -*-
"""
Created on Sun May 25 18:58:31 2025

@author: Kim Bjerge
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
#import json
import datetime
from scipy.stats import expon
from idac.configreader.configreader import readconfig
from idac.predictions.predictions import Predictions
from scipy.stats import pearsonr

save_dir = "/MAMBO/detections/plots/"

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
         
def createDatelist(dataset):

    td = timedate()
    currentDate = 0
    nextDate = 0
    dates = []
    dayOfYears = []
    for i, obj in dataset.iterrows():
        if currentDate != obj['date']:
            #if nextDate != obj['date']:
            #    print("NextDate", nextDate)
            currentDate = obj['date']
            nextDate = currentDate + 1
            dates.append(currentDate)
            dayOfYear = td.getDayOfYear(currentDate)
            #print(dayOfYear)
            dayOfYears.append(int(dayOfYear))

    return dates, dayOfYears


def loadDetectionFiles(trapPath, trapName, taxaSure=True, vegetation=False):

    detectFiles = trapPath + '/'  
    dataframes = []
    for fileName in sorted(os.listdir(detectFiles)):
        if trapName in fileName and "CL.csv" in fileName:
            # print(trapName, detectFiles + fileName)
            #data_df = pd.read_json(trackFiles + fileName)
            data_df = pd.read_csv(detectFiles + fileName)
            dataframes.append(data_df) 
    dataset = pd.concat(dataframes)
    # print(trapPath, len(dataset))
    dateList, dayOfYear = createDatelist(dataset)
    if taxaSure:
        selDataset1 = dataset.loc[dataset['taxaSure'] == True]
    else:
        selDataset1 = dataset
    if not vegetation:
        selDataset2 = selDataset1.loc[selDataset1['taxaLabel'] != "Vegetation"]
    else:
        selDataset2 = selDataset1
    
    return dateList, dayOfYear, selDataset2


def getTaxaList(selDataset):
    
    taxons = {}
    for i, obj in selDataset.iterrows():
        taxa = obj['taxaLabel']
        if taxa in taxons.keys():
            taxons[taxa] += 1
        else:
            taxons[taxa] = 1
            
    taxonsSorted = dict(sorted(taxons.items(), key=lambda item: item[1], reverse=True))              
    return taxonsSorted
    
def plotHistogram(trapName, selDataset, figsize=(10,10)):
    
    taxons = getTaxaList(selDataset)
    print(trapName, taxons)
    
    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(1, 1, 1) 
    
    taxonsKey = list(taxons.keys())
    taxonsValue = list(taxons.values())
    ax.barh(taxonsKey, taxonsValue)
    ax.plot()
    ax.set_xlabel('Detections')
    ax.set_xscale('log')
    ax.set_title(trapName)
    plt.tight_layout(pad=2.0)
    plt.savefig(save_dir + trapName + ".png")
    plt.show()

def plotAllTraps(trapPath, trapNames, country='', yearStr="2024"):

    for trapName in trapNames:
        dateList, dayOfYear, selDataset = loadDetectionFiles(trapPath, trapName, taxaSure=False, vegetation=False)
        plotHistogram(country + trapName + '-' + yearStr, selDataset)

def loadAllDetectionFiles(trapPathAll, trapNameAll, taxaSure=True, vegetation=False):

    dataframes = []
    for idx in range(len(trapPathAll)):
        trapPath = trapPathAll[idx]
        trapNames = trapNameAll[idx]
        for trapName in trapNames:
            detectFiles = trapPath + '/'  
            for fileName in sorted(os.listdir(detectFiles)):
                if trapName in fileName and "CL.csv" in fileName:
                    print(trapName, detectFiles + fileName)
                    data_df = pd.read_csv(detectFiles + fileName)
                    dataframes.append(data_df) 

    dataset = pd.concat(dataframes)
    if taxaSure:
        selDataset1 = dataset.loc[dataset['taxaSure'] == True]
    else:
        selDataset1 = dataset
    if not vegetation:
        selDataset2 = selDataset1.loc[selDataset1['taxaLabel'] != "Vegetation"]
    else:
        selDataset2 = selDataset1
    
    return selDataset2    

# %% Insect plots
if __name__ == '__main__':

    trapPathAll = []
    trapNamesAll = {}
    
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/uva/detectionsTH3"
    trapPath = "/MAMBO/detections/uva"
    trapNames = ["uva-NL11", "uva-NL21", "uva-NL22"]
    plotAllTraps(trapPath, trapNames)
    trapPathAll.append(trapPath)
    trapNamesAll[0] = trapNames

    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/ukceh/detectionsTH3"
    trapPath = "/MAMBO/detections/ukceh"
    trapNames = ["ukceh-UK_1_1", "ukceh-UK_1_2", "ukceh-UK_2_1", "ukceh-UK_2_2"]
    plotAllTraps(trapPath, trapNames)
    trapPathAll.append(trapPath)
    trapNamesAll[1] = trapNames
    
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/ecoin/errorsn/detectionsTH3err"
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/ecoinn/detectionsTH3"
    trapPath = "/MAMBO/detections/ecoinn"
    trapNames = ["ecoinn-MT_1_1", "ecoinn-MT_1_2", "ecoinn-MT_2_1", "ecoinn-MT_2_2"]
    plotAllTraps(trapPath, trapNames)
    trapPathAll.append(trapPath)
    trapNamesAll[2] = trapNames
        
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/cirad/errors/detectionsTH3err"
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/cirad/detectionsTH3"
    trapPath = "/MAMBO/detections/cirad"
    trapNames = ["FR02_Bagnas-Cam.A", "FR02_Bagnas-Cam.B", "FR03_Parpalhon-Cam.A", "FR03_Parpalhon-Cam.B"]
    plotAllTraps(trapPath, trapNames)
    trapPathAll.append(trapPath)
    trapNamesAll[3] = trapNames
        
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/ufz/errors/detectionsTH2err"
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/ufz/detectionsTH3"
    trapPath = "/MAMBO/detections/ufz"
    trapNames = ["DE_1_1", "DE_1_2", "DE_1_2", "DE_2_2"]
    plotAllTraps(trapPath, trapNames, country="ufz-")
    trapPathAll.append(trapPath)
    trapNamesAll[4] = trapNames
    
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/au/errors/detectionsTH3err"
    #trapPath = "O:/Tech_TTH-KBE/MAMBO/2024/au/detectionsTH3"
    trapPath = "/MAMBO/detections/au"
    trapNames = ["-01", "-02", "-03", "-05", "-06", "DK_1_1", "DK_1_2", "DK_2_1", "DK_2_2",  "DK_3_1", "DK_3_2"]
    plotAllTraps(trapPath, trapNames, country="au")
    trapPathAll.append(trapPath)
    trapNamesAll[5] = trapNames
    
    selDataset = loadAllDetectionFiles(trapPathAll, trapNamesAll)
    plotHistogram("All MAMBO demonstration sites 2024", selDataset, figsize=(14,14))
    