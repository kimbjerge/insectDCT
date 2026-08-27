# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 21:47:08 2025

@author: Kim Bjerge
"""
import os
import numpy as np
import shutil

    
trueIdx = 0 # True positive crops (Correct predictions)
falseAIdx = 1 # False positive arthropods (Wrong prediction)
falseBIdx = 2 # False positive background (Plant parts - wrong prediction)


# %% Insect plots
if __name__ == '__main__':
        
    cropPaths = ["/ArthropodsCrops/crops_au/",
                 "/ArthropodsCrops/crops_cirad/",
                 "/ArthropodsCrops/crops_ecoinn/",
                 "/ArthropodsCrops/crops_ufz/",
                 "/ArthropodsCrops/crops_ukceh/",
                 "/ArthropodsCrops/crops_uva/",
                 "/ArthropodsCrops/crops_ni2_1/",
                 "/ArthropodsCrops/crops_ni2_2/"
                 ]
    
    copySelectedFolder = "FalseB"
    #copySelectedFolder = "F"
    
    pathDest = "/ArthropodsDataset/NEW_classifier/NI2_MAMBO/"
    pathVegetation = pathDest + "Vegetation/"
    
    foldersToCopy = ["Birds",
                     "Coleoptera",
                     "Diptera Syrphidae Myathropa florea",
                     "Diptera Syrphidae Chrysotoxum",
                     "Diptera Syrphidae Helophilus",
                     "Diptera Syrphidae Meliscaeva cinctella",
                     "Diptera Syrphidae Platycheirus",
                     "Diptera Syrphidae Scaeva pyrastri",
                     "Diptera Syrphidae Scaeva selenitica",
                     "Diptera Syrphidae Syritta pipiens",
                     "Diptera Syrphidae Xanthogramma dives", #GBIF?
                     "Feathers",
                     "Frogs",
                     "Hemiptera",
                     "Hymenoptera Crabronidae",
                     "Hymenoptera Vespidae Ancistrocerus nigricornis",
                     "Hymenoptera Vespidae Eumenes coronatus",
                     "Larvae",
                     "Lepidoptera Lycenidae",
                     "Lepidoptera Nymphalidae Melitaea cinxia",
                     "Lepidoptera Nymphalidae Lasiommata megera", #GBIF?
                     "Lizards",
                     "Milipedes",
                     "Orthoptera Tetrigidae Tetrix",
                     "Orthoptera Tettigoniidae Decticus verrucivorus",
                     "Snails"]
    
    #foldersToCopy = ["Vegetation"]
    #skipImages = 5 or 2
    skipImages = 0
    
    classes = {} # All MAMBO sites
    filesToCopy = 0
    countCrops = np.zeros(len(foldersToCopy))
    skipCount = 0
    
    for cropPath in cropPaths:    
        for dirName in sorted(os.listdir(cropPath)):
            if dirName != '.gitignore':
                if not dirName in classes.keys():   
                    classes[dirName] = [0, 0, 0]
                
                pathTrue = cropPath + dirName
                files = os.listdir(pathTrue)
                files = [f for f in files if os.path.isfile(pathTrue + '/' + f)]
                classes[dirName][trueIdx] += len(files)
                
                if dirName in foldersToCopy:
                    idx = foldersToCopy.index(dirName)
                    countCrops[idx] += len(files)
                    print(dirName, len(files))
                    for file in files:
                        fileName = pathTrue + '/' + file
                        destinationPath = pathDest + dirName
                        if not os.path.exists(destinationPath):
                            os.mkdir(destinationPath)
                        destination = destinationPath + '/' + file
                        if skipCount % skipImages == 0:
                            print(fileName, destination)
                            #shutil.copy(fileName, destination)
                        skipCount += 1
                
                pathFalseA = cropPath + dirName + '/FalseA'
                if os.path.exists(pathFalseA):
                    files = os.listdir(pathFalseA)
                    classes[dirName][falseAIdx] += len(files)
                
                pathFalseB = cropPath + dirName + '/FalseB'
                if os.path.exists(pathFalseB):
                    files = os.listdir(pathFalseB)
                    classes[dirName][falseBIdx] += len(files)
                    
                    if copySelectedFolder == "FalseB":
                        for file in files:
                            fileName = pathFalseB + '/' + file
                            #print(fileName)
                            filesToCopy += 1
                            destination = pathVegetation + file
                            if not os.path.exists(pathVegetation):
                                os.mkdir(pathVegetation)
                            #shutil.copy(fileName, destination)
                            
    print("=================================================")
    for idx in range(len(foldersToCopy)):
        print(foldersToCopy[idx], int(countCrops[idx]))
        
    print("Vegetation", filesToCopy)
                