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
    
    cropsName = path.split('/')[-2] 

    fileName = cropsName + '.csv'
    if os.path.exists(fileName):
        csvFile = open(fileName, 'a', newline = '\n')
    else:
        csvFile = open(fileName, 'w', newline = '\n')
        csvFile.write("animal,total,truePositive,falseAnimal,falseBackground,precision\n")

    csv_writer = csv.writer(csvFile, delimiter = ',')
    
    insects = []
    precisions = []
    truePs = []
    falseAs = []
    falseBs = []
    totalTrue = 0
    totalFalseA = 0
    totalFalseB = 0
    totalAll = 0
    macroPrecision = 0
    numTaxa = 0
    
    for keyName in classes.keys():
        total = sum(classes[keyName])
        trueP = classes[keyName][trueIdx]
        falseA = classes[keyName][falseAIdx]
        falseB = classes[keyName][falseBIdx]
        precision = round(trueP/total*1000)/10
        if not keyName in ["Usure", "Vegetation"]: 
            totalTrue += trueP
            totalFalseA += falseA
            totalFalseB += falseB
            totalAll += total
            macroPrecision += precision
            numTaxa += 1
        
        input_variable = [keyName, total, trueP, falseA, falseB, precision]
        csv_writer.writerow(input_variable)
        csvFile.flush()

        if (below and precision < 75 and total > 10) or (not below and precision >= 75 and total > 10):
            print(keyName, total, trueP, falseA, falseB, precision)
            insects.append(keyName)
            precisions.append(precision)
            truePs.append(trueP)
            falseAs.append(falseA)
            falseBs.append(falseB)
    
    microPrecision = round(totalTrue/totalAll*1000)/10
    input_variable = ["Micro", totalAll, totalTrue, totalFalseA, totalFalseB, microPrecision]
    csv_writer.writerow(input_variable)
    input_variable = ["Macro", totalAll, totalTrue, totalFalseA, totalFalseB, macroPrecision/numTaxa]
    csv_writer.writerow(input_variable)
    csvFile.flush()
    
    csvFile.close()
    
    plt.figure(figsize=(12,8))
    p1 = plt.barh(insects, truePs)
    bottom = np.zeros(len(insects))
    bottom += truePs
    p2 = plt.barh(insects, falseAs, left=bottom)
    bottom += falseAs
    p3 = plt.barh(insects, falseBs, left=bottom)
        
    if below:
        titleName = 'Arthropods ' + cropsName + ' (P<75%,N>10)'
        fName='Below'
    else:
        titleName = 'Arthropods ' + cropsName + ' (P>75%,N>10)'
        fName='Above'
        
    plt.title(titleName)
    
    plt.ylabel('Taxa')
    plt.xlabel('Observations')
    if not below:
        plt.xscale('log') # Only log scale above
    plt.legend((p1[0], p2[0], p3[0]), ('True Positive', 'FP-Arthropods', 'FP-Background'))
    plt.tight_layout()
    plt.savefig('/UFZ_O2/' + cropsName + '/' + cropsName+'_'+fName+'.png')
    plt.show()


# %% Insect plots
if __name__ == '__main__':

    version = "V3"
    
    if version == "V3":
        cropPath = "/UFZ_O2/Test-HierarchicalClassifierV3/"
        samples = "cropsV3_SAMPLE"
        falseA = "cropsV3_FAULTS/"
        falseB = "cropsV3_BACKGROUND/"    
    else:
        cropPath = "/UFZ_O2/Test-HierarchicalClassifierV4/"
        samples = "cropsV4_SAMPLE"
        falseA = "cropsV4_FAULTS/"
        falseB = "cropsV4_BACKGROUND/"
    
    cropPathSamples = cropPath + samples + '/'
    classes = {}
    for dirName in sorted(os.listdir(cropPathSamples)):
        if dirName != '.gitignore' and not dirName.endswith('txt'):
            if not dirName in classes.keys():   
                classes[dirName] = [0, 0, 0]
            
            pathTrue = cropPathSamples + dirName
            print(pathTrue)
            files = os.listdir(pathTrue)
            files = [f for f in files if os.path.isfile(pathTrue + '/' + f)]
            classes[dirName][trueIdx] = len(files)
            
            pathFalseA = cropPath + falseA + dirName
            if os.path.exists(pathFalseA):
                files = os.listdir(pathFalseA)
                classes[dirName][falseAIdx] = len(files)
            
            pathFalseB = cropPath + falseB + dirName
            if os.path.exists(pathFalseB):
                files = os.listdir(pathFalseB)
                classes[dirName][falseBIdx] = len(files)
        
    print("===================================================")        
    print(cropPath)
    printStat(classes, cropPath, below=True)
    printStat(classes, cropPath, below=False)
                
            
                