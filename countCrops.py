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
    
    
    fileName = 'results.csv'
    if os.path.exists(fileName):
        csvFile = open('results.csv', 'a', newline = '\n')
    else:
        csvFile = open('results.csv', 'w', newline = '\n')
        csvFile.write("partner,animal,total,truePositive,falseAnimal,falseBackground,precision\n")

    csv_writer = csv.writer(csvFile, delimiter = ',')
    
    insects = []
    precisions = []
    truePs = []
    falseAs = []
    falseBs = []
    for keyName in classes.keys():
        total = sum(classes[keyName])
        trueP = classes[keyName][trueIdx]
        falseA = classes[keyName][falseAIdx]
        falseB = classes[keyName][falseBIdx]
        precision = round(trueP/total*1000)/10
        
        partner = path.split('/')[-2]
        partner = partner.split('_')[1]
        input_variable = [partner, keyName, total, trueP, falseA, falseB, precision]
        csv_writer.writerow(input_variable)
        csvFile.flush()

        if (below and precision < 75 and total > 10) or (not below and precision >= 75 and total > 10):
            print(keyName, total, trueP, falseA, falseB, precision)
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
        titleName = 'Arthropods ' + cropsName + ' (P>75%,N>10)'
        fName='Above'
        
    plt.title(titleName)
    
    plt.ylabel('Taxa')
    plt.xlabel('Observations')
    if not below:
        plt.xscale('log') # Only log scale above
    plt.legend((p1[0], p2[0], p3[0]), ('True Positive', 'FP-Arthropods', 'FP-Background'))
    plt.tight_layout()
    plt.savefig('/ArthropodsCrops/plots/'+cropsName+'_'+fName+'.png')
    plt.show()


# %% Insect plots
if __name__ == '__main__':
        
    cropPaths = ["/ArthropodsCrops/crops_au/",
                 "/ArthropodsCrops/crops_cirad/",
                 "/ArthropodsCrops/crops_ecoinn/",
                 "/ArthropodsCrops/crops_ufz/",
                 "/ArthropodsCrops/crops_ukceh/",
                 "/ArthropodsCrops/crops_uva/",
                 #"/ArthropodsCrops/crops_ni2_1/",
                 #"/ArthropodsCrops/crops_ni2_2/"
                 ]
    
    for cropPath in cropPaths:    
        classes = {}
        for dirName in sorted(os.listdir(cropPath)):
            if dirName != '.gitignore':
                if not dirName in classes.keys():   
                    classes[dirName] = [0, 0, 0]
                
                pathTrue = cropPath + dirName
                files = os.listdir(pathTrue)
                files = [f for f in files if os.path.isfile(pathTrue + '/' + f)]
                classes[dirName][trueIdx] = len(files)
                
                pathFalseA = cropPath + dirName + '/FalseA'
                if os.path.exists(pathFalseA):
                    files = os.listdir(pathFalseA)
                    classes[dirName][falseAIdx] = len(files)
                
                pathFalseB = cropPath + dirName + '/FalseB'
                if os.path.exists(pathFalseB):
                    files = os.listdir(pathFalseB)
                    classes[dirName][falseBIdx] = len(files)
        
        print("===================================================")        
        print(cropPath)
        printStat(classes, cropPath, below=True)
        printStat(classes, cropPath, below=False)
                
            
                