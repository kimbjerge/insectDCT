# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:04:04 2025

@author: Kim Bjerge
"""

import os
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

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--CSVfiles', default='O:/Tech_TTH-KBE/MAMBO/2024/au/detectionsV1-4/') # Directory that contains CSV files
    parser.add_argument('--cropsPath', default='./crops_au4/') # Directory to save images crops

    parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_13052025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_13052025.pkl')
    parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds_13052025_TH3.csv') # Use thresholds below mean-3*std
    
    args = parser.parse_args() 
    print(args)
    
    
    args = parser.parse_args() 
    print(args)
    
    print("Loading hierarchical insect classifier model", args.hierachical)
    hierarchicalClassifier = createHierarchicalClassifier(args.hierachical, args.labels, args.thresholds, 128)
    
    taxaDirLabels = {}
    totalDirCrops = 0
    for dirname in sorted(os.listdir(args.cropsPath)):
        if dirname != ".gitignore":
            cropsInDir = sorted(os.listdir(args.cropsPath + dirname))
            taxaDirLabels[dirname] = [0, len(cropsInDir)]
            totalDirCrops += len(cropsInDir)
       
    totalLabels = 0
    for filename in sorted(os.listdir(args.CSVfiles)):
        if filename.endswith('.csv') and '-CL' in filename:
            
            dataset = pd.read_csv(args.CSVfiles + filename)
            
            for i, obj in dataset.iterrows():
                if obj['taxaLabel'] != "Unsure":
                    labelL1, labelL2, labelL3 = hierarchicalClassifier.getLabels(obj['taxaLevel'], obj['taxaLabel'])
                    cropDirName = createCropDirName(obj['taxaLevel'], labelL1, labelL2, labelL3)
                    #print(obj['taxaLabel'], obj['taxaLevel'], labelL1, labelL2, labelL3, cropDirName)
                else:
                    cropDirName = obj['taxaLabel']
                totalLabels += 1
                taxaDirLabels[cropDirName][0] += 1
        
    print(taxaDirLabels)
    print(totalDirCrops, totalLabels)
