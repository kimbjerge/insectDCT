# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 20:34:57 2025

@author: Kim Bjerge
"""

import os
import sys
import cv2
import time
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


def saveCrop(x1, y1, x2, y2, prevImage, frame_count, frameId, imagePath, videoCap, cropDirName, dstPath, csvName, border=1):
    
    success = True
    image = []
    if videoCap == None:
        image = cv2.imread(imagePath)
    else:
        #videoCap.set(cv2.CAP_PROP_POS_FRAMES, frameId-1) # Not working wrong frame???
        #success, image = videoCap.read()
        if (frame_count == frameId - 1): # Crop in same frame as last called
            image = prevImage
            print("More crops in same frame", frame_count)
        else:
            #while success and (frame_count < frameId - 1): # Why offset needed KBE??? frame_stride = 3, Scarbosa
            while success and (frame_count < frameId + 1): # Why offset needed KBE??? frame_stride = 1, PAU
                success, image = videoCap.read()
                frame_count += 1
                #sys.stdout.write("FrameId %d \r" % (frame_count))
                #sys.stdout.flush()

    if success == False or len(image) == 0:
        print("Error reading image or video")
        return frame_count, image, success
    
    height, width, channels = image.shape
    #print(dstPath+cropDirName, height, width)
        
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
    
    #if cropDirName != "Vegetation" and cropDirName != "Unsure":
    print(dstPath + cropDirName + '/' + imgNameCrop)
    cv2.imwrite(dstPath + cropDirName + '/' + imgNameCrop, imgCrop) 
    
    return frame_count, image, success

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
        
def createCrops(csvName, imgPath, videoPath, dstPath, dataset, hierarchicalClassifier):
    
    videoCap = None
    frame_count = 0
    image = []
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
        #elif 'trap' in obj.keys():
        #    trapDir = obj['trap']   
        else:
            trapDir = ''
        
        imagePath = imgPath + trapDir + '/' + obj['fileName']
        
        if videoPath != '' and videoCap == None:
            videoFile = videoPath + obj['fileName']
            videoCap = cv2.VideoCapture(videoFile)

        if videoCap == None:
            print("Used image file", imagePath, "FrameId", obj['frameId'])
        else:
            print("Uses video recording", videoFile, "FrameId", obj['frameId'])

        frame_count, image, success = saveCrop(x1, y1, x2, y2, image, frame_count, obj['frameId'], imagePath, videoCap, cropDirName, dstPath, csvName)       
        
        if success == False: 
            return
            
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
    #parser.add_argument('--CSVfiles', default='/UFZ/detectionsAllV5/') # Directory that contains CSV files
    #parser.add_argument('--CSVfiles', default='/Orchard/detections_V6/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/UFZ/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='/RTNI/detections_V6/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/NI/RT/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='D:/MINIMON/detections/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='D:/MINIMON/') # Directory that contains images
    #parser.add_argument('--CSVfiles', default='D:/PAU/detections_L/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='D:/PAU/videos/') # Directory that contains images

    #parser.add_argument('--CSVfiles', default='/PollNI/S2/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='O:/Tech_TTH-KBE/PollinatorWatch/FIN/S2/') # Directory that contains images
    
    #parser.add_argument('--CSVfiles', default='D:/UFZ_BOS_STR/detections/') # Directory that contains CSV files
    #parser.add_argument('--imagesPath', default='D:/UFZ_BOS_STR/') # Directory that contains images
    #parser.add_argument('--videoPath', default="D:/UFZ_BOS_STR/") # Directory that contains video, if empty then imagesPath is used
    
    parser.add_argument('--videoPath', default="") # Directory that contains video, if empty then imagesPath is used
    #parser.add_argument('--videoPath', default="D:/PAU/videos/") # Directory that contains video, if empty then imagesPath is used
    
    parser.add_argument('--cropsPath', default='./crops/') # Directory to save images crops
    #parser.add_argument('--cropsPath', default='/RTNI/crops_V6/') # Directory to save images crops
    #parser.add_argument('--cropsPath', default='J:/PollNI/crops/') # Directory to save images crops
    #parser.add_argument('--cropsPath', default='D:/PAU/crops/') # Directory to save images crops
    
    parser.add_argument('--dataset', default="V6") # Support for dataset "V3" (Wingscapes, Logitech, Pi3, GBIF) or "V4" without GBIF data 
                                                   # or "V5" with GBIF and additional data, "V6" with more data and reorganized hierarchy
    parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_RES_V6.pth') # 128x128
    parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_RES_V6.pkl')
    parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds3S_RES_V6.csv') # Use thresholds below = mean-3.5*std

    #parser.add_argument('--hierachical', default='./models_save/HierarchicalClassifier_13052025.pth') # 128x128 F1: L1 0.93, L2 0.76, L3 0.68
    #parser.add_argument('--labels', default='./models_save/HierarchicalLabels3L_13052025.pkl')
    #parser.add_argument('--thresholds', default='./models_save/HierarchicalThresholds_13052025_TH3.csv') # Use thresholds below mean-3*std

    print("createCrops.py version: 1.0")

    args = parser.parse_args() 
    
    hierarchicalWeights = args.hierachical
    hierarchicalLabels = args.labels
    hierarchicalThresholds = args.thresholds
    if args.dataset != 'V6': # Select model weights trained on dataset V3, V4, V5 of V6
        oldVersionsDate = "_05092025" # V3-V5 used date in file names
        hierarchicalWeights = hierarchicalWeights.replace('V6', args.dataset + oldVersionsDate)
        hierarchicalLabels = hierarchicalLabels.replace('V6', args.dataset + oldVersionsDate)
        hierarchicalThresholds = hierarchicalThresholds.replace('V6', args.dataset + oldVersionsDate)
            
    print("Loading hierarchical insect classifier model", hierarchicalWeights)
    hierarchicalClassifier = createHierarchicalClassifier(hierarchicalWeights, hierarchicalLabels, hierarchicalThresholds, 128)
    
    print(args)
    time.sleep(5)
    
    for filename in sorted(os.listdir(args.CSVfiles)):
        if filename.endswith('.csv') and '-CL' in filename:
            #Header: year,trapDir,date,time,detectConf,detectId,x1,y1,x2,y2,fileName,taxaLabel,taxaId,taxaLevel,taxaConf,taxaSure,frameId
            data_df = pd.read_csv(args.CSVfiles + filename)
            #print(args.CSVfiles + filename)
            createCrops(filename.replace('.csv', ''), args.imagesPath, args.videoPath, args.cropsPath, data_df, hierarchicalClassifier)
            
    print("Finished creating insect crops from combinded detection and classification");