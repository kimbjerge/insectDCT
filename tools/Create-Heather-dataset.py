# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 19:01:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from images recorded in KU Heather project
Use camera systems from sites situated with Heather vegetation

@author: Kim Bjerge
"""

import os
import fnmatch
import cv2
import shutil
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

# Heather image size (Wingscapes)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def createLabelsAndImages(selDataset, data_df, pathToRecordedFiles, pathToDestDataset, pathToDestDatasetMIE, split):
    
    mie = MotionEnhancement()
    
    skip = int(100/split)
    count = 0
    prevPathToImages = ""
    fileList = []
    for idx, row in selDataset.iterrows():
        detections_df = data_df.loc[data_df['fileName'] == row['fileName']]
      
        imageFilePath = row['fileName'].split('/')[0]
        imageFileName = row['fileName'].split('/')[1]
        labelFileName = imageFileName.replace('.JPG', '.txt')
        cameraId = "HE_" + imageFilePath + '_'
        count += 1
        if count % skip == 0: # Save to test dataset
            pathToDest = pathToDestDataset.replace("train", "test")
            pathToDestMIE = pathToDestDatasetMIE.replace("train", "test")
            print("Test image", cameraId+labelFileName)
        else: # save to train dataset
            pathToDest = pathToDestDataset
            pathToDestMIE = pathToDestDatasetMIE
            print("Train image", cameraId+labelFileName)
            
        labelFile = open(pathToDest+cameraId+labelFileName, "w")
        print(pathToDest+cameraId+labelFileName)
        for i, detection in detections_df.iterrows():
            #print(detection['fileName'], detection['x1'], detection['y1'], detection['x2'], detection['y2'])
            w = detection['x2'] - detection['x1']
            h = detection['y2'] - detection['y1']
            xc = detection['x1'] + 0.5*w
            yc = detection['y1'] + 0.5*h
            line = "0 " + str(xc/IMG_WIDTH) + " " + str(yc/IMG_HEIGHT) + " " + str(w/IMG_WIDTH) + " " + str(h/IMG_HEIGHT)
            print(line)
            labelFile.write(line + "\n")
        labelFile.close()
        
        pathToImageFile = pathToRecordedFiles+imageFilePath+'/'+imageFileName
        shutil.copyfile(pathToImageFile, pathToDest+cameraId+imageFileName)
        
        pathToImages = pathToRecordedFiles+imageFilePath+'/'
        if prevPathToImages != pathToImages:
            fileList = fnmatch.filter(sorted(os.listdir(pathToImages)), '*.JPG')
            prevPathToImages = pathToImages
        fileIdx = fileList.index(imageFileName)
        filePrevIdx = fileIdx
        fileNextIdx = fileIdx
        if fileIdx > 0:
            filePrevIdx = fileIdx - 1
        if fileIdx < len(fileList) - 1:
            fileNextIdx = fileIdx + 1
        print(fileList[filePrevIdx], imageFileName, fileList[fileNextIdx])
        print(pathToImages, pathToDestMIE+cameraId+imageFileName)
        mieImage = mie.motion_three_images(pathToImages, fileList[filePrevIdx], imageFileName, fileList[fileNextIdx])
        cv2.imwrite(pathToDestMIE+cameraId+imageFileName, mieImage)
    
if __name__=='__main__':
    
    numInsects = 697
    numUnsure = 250
    numVegetation = 250
    splitPercentage = 100 # Only for test dataset
    pathToSrcDataset = 'J:/KUHeather/HA_SD_c1/'
    pathToRecordData = 'O:/Tech_Insects/KUBier/HA_SD_c1/'
    pathToDestDatasetMIE = 'J:/KUHeather/trainHeatherm/'
    pathToDestDataset = 'J:/KUHeather/trainHeather/'
        
    firstTime = True
    pathToSrcDetections = pathToSrcDataset
    for filename in sorted(os.listdir(pathToSrcDetections)):
        if (filename.endswith('.csv')):
            if "-CL.csv" in filename:
                print("Reading", filename)
                data_df = pd.read_csv(pathToSrcDetections+filename)
                if firstTime:
                    data_frames = data_df.copy()
                    firstTime = False
                else:    
                    data_frames = pd.concat([data_frames, data_df])
        
    # Select only images where insects has been detected - many are false positive detections
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] != "Vegetation"]
    selDataset2 = selDataset1.loc[selDataset1['taxaLabel'] != "Unsure"]
    selDataset2 = selDataset2.sample(n=numInsects, random_state=37)
    selDataset3 = selDataset2.sort_values(by=['trapDir', 'fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Vegetation"]
    selDataset2 = selDataset1.sample(n=numVegetation, random_state=65)
    selDataset3 = selDataset2.sort_values(by=['trapDir', 'fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Unsure"]
    selDataset2 = selDataset1.sample(n=numUnsure, random_state=43)
    selDataset3 = selDataset2.sort_values(by=['trapDir', 'fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)


    