# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 20:34:57 2025

@author: Kim Bjerge
"""

import os
import cv2
import pandas as pd
import argparse
import pickle

from common.hierarchical_classifier import HierarchicalClassifier

def createHierarchicalClassifier(weights_file, label_file, threshold_file, img_size=128):
    
    with open(label_file, 'rb') as f:
        _, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, _, _, _, _, _, _ = pickle.load(f)
        print("Labels and hierarchy dependency loaded from ", label_file)
        print("=============================================================================================")
        print("L1 classes", labelsL1, len(labelsL1))
        print("=============================================================================================")
        print("L2 classes", labelsL2, len(labelsL2))
        print("=============================================================================================")
        print("L3 classes", labelsL3, len(labelsL3))
        print("=============================================================================================")
        print("L2 -> L1 dependency", hierarchyL1)
        print("=============================================================================================")
        print("L3 -> L2 dependency", hierarchyL2)
        print("=============================================================================================")
    
    classifier = HierarchicalClassifier(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, img_size=img_size, device='cpu')
    classifier.loadmodel(weights_file, threshold_file)
    
    return classifier


def saveCrop(x1, y1, x2, y2, frameId, imagePath, cropDirName, dstPath, csvName, border=1):
    
    print(imagePath, dstPath+cropDirName)
    image = cv2.imread(imagePath)
    height, width, channels = image.shape

    x1_str = str(x1)
    y1_str = str(y1)
    
    # Compute center of object, width and height    
    w = (x2-x1)
    xc = x1 + int(round(w/2))

    h = (y2-y1)
    yc = y1 + int(round(h/2))

    # Use square crops with border
    w += border*2
    h += border*2
    if h > w: 
        wh2 = int(round(h/2))
    else:
        wh2 = int(round(w/2))
        
    x1 = xc-wh2
    if x1 < 0:
       x1 = 0
    x2 = xc+wh2
    if x2 >= width:
        x2 = width-1
    y1 = yc-wh2
    if y1 < 0: 
        y1 = 0
    y2 = yc+wh2
    if y2 >= height:
        y2 = height-1
    
    # get image crop of object
    imgCrop = image[y1:y2, x1:x2,  :].copy()
    
    if os.path.exists(dstPath + cropDirName) == False:
        print("Create directory:", dstPath + cropDirName)
        os.mkdir(dstPath + cropDirName)
    imgNameCrop = csvName +'-' + str(frameId) + '-' + x1_str + '-' + y1_str + '.jpg'
    print(dstPath + cropDirName + '/' + imgNameCrop)
    cv2.imwrite(dstPath + cropDirName + '/' + imgNameCrop, imgCrop)   

def createCropDirName(level, labelL1, labelL2, labelL3):
    
    cropDirName = ''
    
    if level == 1:
        cropDirName = labelL1
        
    if level == 2:
        if labelL2 == labelL1:
            cropDirName = labelL2
        else:
            cropDirName = labelL1 + ' ' + labelL2
    
    if level == 3:
        if labelL3 == labelL2:
            if labelL2 == labelL1:    
                cropDirName = labelL1
            else:
                cropDirName = labelL1 + ' ' + labelL2
        else:
            cropDirName = labelL1 + ' ' + labelL2 + ' ' + labelL3
    
    return cropDirName
        
def createCrops(csvName, imgPath, dstPath, dataset, hierarchicalClassifier):
    
    for i, obj in dataset.iterrows():
        if obj['taxaLabel'] != "Unsure":
            labelL1, labelL2, labelL3 = hierarchicalClassifier.getLabels(obj['taxaLevel'], obj['taxaLabel'])
            cropDirName = createCropDirName(obj['taxaLevel'], labelL1, labelL2, labelL3)
            #print(obj['taxaLabel'], obj['taxaLevel'], labelL1, labelL2, labelL3, cropDirName)
        else:
            cropDirName = "Unsure"
            
        x1 = obj['x1']
        y1 = obj['y1']
        x2 = obj['x2']
        y2 = obj['y2']
        
        if 'trapDir' in obj.keys():
            trapDir = obj['trapDir']
        else:
            trapDir = ''
        
        imagePath = imgPath + trapDir + '/' + obj['fileName']
        saveCrop(x1, y1, x2, y2, obj['frameId'], imagePath, cropDirName, dstPath, csvName)       
            
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--CSVfiles', default='./detections/') # Directory that contains CSV files
    parser.add_argument('--imagesPath', default='./images/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='O:/Tech_TTH-KBE/MAMBO/2024/au/detectionsV1/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/MAMBO/2024/au/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='O:/Tech_TTH-KBE/MAMBO/2024/ufz/detectionsV1/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/MAMBO/2024/ufz/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='O:/Tech_TTH-KBE/MAMBO/2024/cirad/detectionsV1/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/MAMBO/2024/cirad/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='O:/Tech_TTH-KBE/MAMBO/2024/ecoinn/detectionsV1/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/MAMBO/2024/') # Directory that contains images
    
    parser.add_argument('--cropsPath', default='./crops/') # Directory to save images crops
    
    parser.add_argument('--dataset', default="V5") # Support for dataset "V3" (Wingscapes, Logitech, Pi3, GBIF) or "V4" without GBIF data or "V5" with GBIF and additional data
    parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_RES_V3_05092025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_RES_V3_05092025.pkl')
    parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds3S_RES_V3_05092025.csv') # Use thresholds below = mean-3*std

    #parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_13052025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    #parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_13052025.pkl')
    #parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds_13052025_TH3.csv') # Use thresholds below mean-3*std
    
    args = parser.parse_args() 
    print(args)
    
    hierarchicalWeights = args.hierachical
    hierarchicalLabels = args.labels
    hierarchicalThresholds = args.thresholds
    if args.dataset != 'V3': # Select model weights trained on dataset V3 or V4f
        hierarchicalWeights = hierarchicalWeights.replace('V3', args.dataset)
        hierarchicalLabels = hierarchicalLabels.replace('V3', args.dataset)
        hierarchicalThresholds = hierarchicalThresholds.replace('V3', args.dataset)
            
    print("Loading hierarchical insect classifier model", hierarchicalWeights)
    hierarchicalClassifier = createHierarchicalClassifier(hierarchicalWeights, hierarchicalLabels, hierarchicalThresholds, 128)
    
    for filename in sorted(os.listdir(args.CSVfiles)):
        if filename.endswith('.csv') and '-CL' in filename:
            #Header: year,trapDir,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId
            data_df = pd.read_csv(args.CSVfiles + filename)
            #print(args.CSVfiles + filename)
            createCrops(filename.replace('.csv', ''), args.imagesPath, args.cropsPath, data_df, hierarchicalClassifier)