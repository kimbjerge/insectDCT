# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 09:24:10 2025

@author: Kim Bjerge
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#%% MAIN
if __name__=='__main__':
    
    # Classification dataset V5
    image_path_list = ['/home/don/data/Arthropods/NI2classesV2',
                       '/home/don/data/Arthropods/NI2sortedV2',
                       '/home/don/data/Arthropods/NItrainV2',
                       '/home/don/data/Arthropods/NIvalV2',
                       '/home/don/data/Arthropods/sorted_orchard_crops',
                       '/home/don/data/Arthropods/Orchard_cropsV2',
                       '/home/don/data/Arthropods/NI2_MAMBOV2',
                       '/home/don/data/Arthropods/GBIF_cropsV2'
                      ]
    
    # Classification dataset V5 - GBIF
    image_path_list = ['D:/insectsDCT_datasets/classifier/NI2classesV2',
                       'D:/insectsDCT_datasets/classifier/NI2sortedV2',
                       'D:/insectsDCT_datasets/classifier/NItrainV2',
                       'D:/insectsDCT_datasets/classifier/NIvalV2',
                       'D:/insectsDCT_datasets/classifier/sorted_orchard_crops',
                       'D:/insectsDCT_datasets/classifier/Orchard_cropsV2',
                       'D:/insectsDCT_datasets/classifier/NI2_MAMBOV2',
                       'D:/insectsDCT_datasets/classifier/GBIF_cropsV2'
                      ]
    
    boxesWidth = []
    boxesHeight = []
    boxesArea = []
    boxesMaxSize = []
    taxaCrops = {}
    totalImages = 0
    for path in image_path_list:
        print(path, np.mean(boxesMaxSize))
        
        for taxaName in sorted(os.listdir(path)):
            taxaPath = path + '/' + taxaName + '/'

            countImages = 0
            for imageName in sorted(os.listdir(taxaPath)):
                if imageName.endswith('.jpg') or imageName.endswith('.JPG'):
                    image = Image.open(taxaPath+'/'+imageName)
                    w, h = image.size
                    boxesWidth.append(w)
                    boxesHeight.append(h)
                    boxesArea.append(w*h)
                    s = h
                    if w > h:
                        s = w
                    boxesMaxSize.append(s)
                    totalImages += 1
                    countImages += 1
                    
                    if taxaName in taxaCrops.keys():
                        taxaCrops[taxaName] += 1
                    else:
                        taxaCrops[taxaName] = 1
                    
            print(taxaPath, countImages)

    print("==========================================================================")
    file = open("datasetV5_taxaList.csv", 'w')
    header = "taxa,crops\n"
    file.write(header)
    for key in sorted(taxaCrops.keys()):
        line = key + ',' + str(taxaCrops[key]) + '\n'
        file.write(line)
    file.close()
        
    print("Total images:", totalImages)
    print("Average labels width:", np.mean(boxesWidth), "STD:", np.std(boxesWidth))
    print("Average labels height:", np.mean(boxesHeight), "STD:", np.std(boxesHeight))
    print("Average labels area:", np.mean(boxesArea), "STD:", np.std(boxesArea))
    
    plt.hist(boxesMaxSize, bins=200, density=True, alpha=0.6, color='b')
    modelSize = 128 # Image size for classifiation
    steps = 20
    step = 0.015/steps
    listTh = [modelSize for i in range(steps)]
    listPb = [step*i for i in range(steps)]
    plt.plot(listTh, listPb, 'r')
    
    
    #plt.title("Insect pixel size (datasetV5 without GBIF)")
    plt.title("Insect pixel size (datasetCV5)")
    plt.xlabel('Pixel size (max.[w|h])')
    plt.ylabel('Probability')
    #plt.xlim(0, 500)
    plt.xlim(0, 1000)
    plt.savefig("./plots/classificationDatasetV5.png")
    plt.show()
    
    longTail = sorted(taxaCrops.values(), reverse=True)
    plt.plot(longTail)
    plt.title("Class distribution CV5")
    plt.xlabel('Class Id')
    plt.ylabel('Images')
    plt.savefig("./plots/distributionDatasetV5.png")
    plt.show()
    
    plt.plot(longTail)
    plt.yscale('log')
    plt.title("Class distribution CV5")
    plt.xlabel('Class Id')
    plt.ylabel('Images')
    plt.savefig("./plots/distributionLogDatasetV5.png")
    plt.show()
                                        