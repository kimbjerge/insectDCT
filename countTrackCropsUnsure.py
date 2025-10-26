# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:47:08 2025

@author: Kim Bjerge
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

    
trueIdx = 0 # True positive crops (Correct predictions)
falseAIdx = 1 # False positive arthropods (Wrong prediction)
falseBIdx = 2 # False positive background (Plant parts - wrong prediction)

def printStat(classes, path, below=True):
    
    
    fileName = 'resultsTrackCrops.csv'
    if os.path.exists(fileName):
        csvFile = open(fileName, 'a', newline = '\n')
    else:
        csvFile = open(fileName, 'w', newline = '\n')
        csvFile.write("checked,animal,total,truePositive,falsePositive,precision\n")

    csv_writer = csv.writer(csvFile, delimiter = ',')
    
    insects = []
    precisions = []
    truePs = []
    falseAs = []
    falseBs = []
    classes = dict(sorted(classes.items()))
    for keyName in classes.keys():
        #if keyName == "Unsure":
        #    continue
        total = sum(classes[keyName])
        trueP = classes[keyName][trueIdx]
        falseA = classes[keyName][falseAIdx]
        falseB = classes[keyName][falseBIdx]
        precision = round(trueP/total*1000)/10
        
        checked = path.split('/')[2]
        checked = checked.split('_')[1]
        input_variable = [checked, keyName, total, trueP, falseA, precision]
        csv_writer.writerow(input_variable)
        csvFile.flush()

        #if (below and precision < 75 and total > 10) or (not below and precision >= 75 and total > 10):
        print(keyName, total, trueP, falseA, precision)
        insects.append(keyName)
        precisions.append(precision)
        truePs.append(trueP)
        falseAs.append(falseA)
        falseBs.append(falseB)
    
    csvFile.close()
    
    plt.figure(figsize=(12,8))
    p1 = plt.barh(insects, truePs)
    bottom = np.zeros(len(insects))
    bottom += truePs
    p2 = plt.barh(insects, falseAs, left=bottom)
    bottom += falseAs
    p3 = plt.barh(insects, falseBs, left=bottom)
    
    cropsName = path.split('/')[-2] 
    
    if below:
        titleName = 'Arthropods ' + cropsName + ' (P<75%,N>10)'
        fName='Below'
    else:
        #titleName = 'Arthropods ' + cropsName + ' (P>75%,N>10)'
        titleName = 'Arthropods ' + cropsName
        fName='Above'
        
    plt.title(titleName)
    
    plt.ylabel('Taxa')
    plt.xlabel('Tracks')
    #if not below:
    plt.xscale('log') # Only log scale above
    #plt.legend((p1[0], p2[0], p3[0]), ('True Positive', 'FP-Arthropods', 'FP-Background'), loc="lower right")
    plt.tight_layout()
    plt.savefig('/UFZ/plots/UFZtracks_'+fName+'.png')
    plt.show()


# %% Insect plots
if __name__ == '__main__':
        
    cropPaths = ["/Orchard/trackCrops_CheckTaxa/",
                 "/Orchard/trackCrops_NoCheckTaxa/"
                 ]
    
    cropPaths = ["/RTNI/trackCrops_CheckTaxa/",
                 "/RTNI/trackCrops_NoCheckTaxa/"
                 ]

    cropPaths = ["/UFZ/trackCrops_AllV5/",
                 "/UFZ/trackCrops_UnsureV5/"
                 ]
    
    classes = {}
    for cropPath in cropPaths:    
        for dirName in sorted(os.listdir(cropPath)):
            if dirName != '.gitignore':
                if dirName == "Unsure" and ("AllV5" in cropPath):
                    continue

                if not dirName in classes.keys():   
                    classes[dirName] = [0, 0, 0]
                                    
                pathTrue = cropPath + dirName
                files = os.listdir(pathTrue)
                files = [f for f in files if os.path.isfile(pathTrue + '/' + f)]
                
                if "AllV5" in cropPath:
                    classes[dirName][trueIdx] += len(files)
                else:
                    classes[dirName][falseAIdx] = len(files)
                
                    
                pathFalseB = cropPath + dirName + '/FalseB'
                if os.path.exists(pathFalseB):
                    files = os.listdir(pathFalseB)
                    classes[dirName][falseBIdx] += len(files)
        
    print("===================================================")        
    print(cropPath)
    #printStat(classes, cropPath, below=True)
    printStat(classes, cropPath, below=False)                