# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:47:08 2025

@author: Kim Bjerge
"""
import os

    
trueIdx = 0 # True positive crops (Correct predictions)
falseAIdx = 1 # False positive arthropods (Wrong prediction)
falseBIdx = 2 # False positive background (Plant parts - wrong prediction)

def printStat(classes):
    
    for keyName in classes.keys():
        total = sum(classes[keyName])
        trueP = classes[keyName][trueIdx]
        falseA = classes[keyName][falseAIdx]
        falseB = classes[keyName][falseBIdx]
        precision = round(trueP/total*1000)/10
        print(keyName, sum(classes[keyName]), trueP, falseA, falseB, precision)
        

# %% Insect plots
if __name__ == '__main__':
        
    cropPaths = [#"/ArthropodsCrops/crops_au/",
                 #"/ArthropodsCrops/crops_ufz/",
                 #"/ArthropodsCrops/crops_ni2_1/",
                 "/ArthropodsCrops/crops_ni2_2/"]
    
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
        printStat(classes)
                
            
                