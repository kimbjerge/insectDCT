# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 09:24:10 2025

@author: Kim Bjerge
"""
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


useTestDataset = True


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
    
    # Classification dataset V5 with GBIF
    image_path_list = ['E:/insectsDCT_datasets/classifier/datasetV5/NI2classesV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/NI2sortedV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/NItrainV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/NIvalV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/sorted_orchard_crops',
                       'E:/insectsDCT_datasets/classifier/datasetV5/Orchard_cropsV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/NI2_MAMBOV2',
                       'E:/insectsDCT_datasets/classifier/datasetV5/GBIF_cropsV2'
                      ]
    
    if useTestDataset:
        # Test dataset - new names and reorganization
        image_path_list = [
                           'E:/insectsDCT_datasets/classifier/dataset_test/NI2_MAMBO',
                           'E:/insectsDCT_datasets/classifier/dataset_test/RTNI',
                           'E:/insectsDCT_datasets/classifier/dataset_test/Orchard'
                          ]
    else:
        # Classification dataset V6 - new names and reorganization
        image_path_list = ['E:/insectsDCT_datasets/classifier/datasetV6/NI2',
                           'E:/insectsDCT_datasets/classifier/datasetV6/NI2_MAMBO',
                           'E:/insectsDCT_datasets/classifier/datasetV6/NI',
                           'E:/insectsDCT_datasets/classifier/datasetV6/Orchard',
                           'E:/insectsDCT_datasets/classifier/datasetV6/GBIF'
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
    if useTestDataset:
        file = open("datasetTest_taxaList.csv", 'w')
    else:
        file = open("datasetV6_taxaList.csv", 'w')
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
    
    
    if useTestDataset:
        plt.title("Insect pixel size (Test dataset)")
        plt.xlabel('Pixel size (max.[w|h])')
        plt.ylabel('Probability')
        #plt.xlim(0, 500)
        plt.xlim(0, 1000)
        plt.savefig("./plots/classificationDatasetTest.png")
        plt.show()
        
        longTail = sorted(taxaCrops.values(), reverse=True)
        plt.plot(longTail)
        plt.title("Class distribution test dataset")
        plt.xlabel('Class Id')
        plt.ylabel('Images')
        plt.savefig("./plots/distributionDatasetTest.png")
        plt.show()
        
        plt.plot(longTail)
        plt.yscale('log')
        plt.title("Class distribution test dataset")
        plt.xlabel('Class Id')
        plt.ylabel('Images')
        plt.savefig("./plots/distributionLogDatasetTest.png")
        plt.show()
    else:
        #plt.title("Insect pixel size (datasetV5 without GBIF)")
        plt.title("Insect pixel size (CV6)")
        plt.xlabel('Pixel size (max.[w|h])')
        plt.ylabel('Probability')
        #plt.xlim(0, 500)
        plt.xlim(0, 1000)
        plt.savefig("./plots/classificationDatasetV6.png")
        plt.show()
        
        longTail = sorted(taxaCrops.values(), reverse=True)
        plt.plot(longTail)
        plt.title("Class distribution (CV6)")
        plt.xlabel('Class Id')
        plt.ylabel('Images')
        plt.savefig("./plots/distributionDatasetV6.png")
        plt.show()
        
        plt.plot(longTail)
        plt.yscale('log')
        plt.title("Class distribution (CV6)")
        plt.xlabel('Class Id')
        plt.ylabel('Images')
        plt.savefig("./plots/distributionLogDatasetV6.png")
        plt.show()
                                            