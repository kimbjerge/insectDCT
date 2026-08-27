# -*- coding: utf-8 -*-
"""
Created on Sun Oct 05 19:01:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from images recorded in KU Heather project
Use camera systems from sites situated with Heather vegetation

@author: Kim Bjerge
"""

import os
import shutil
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

# Heather image size (Wingscapes)
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def createLabelsAndImages(selDataset, data_df, pathToRecordedFiles, pathToDestDataset, pathToDestDatasetMIE, split):
    
    
    skip = int(100/split)
    count = 0
    for idx, row in selDataset.iterrows():
        detections_df = data_df.loc[data_df['fileName'] == row['fileName']]
      
        imageFilePath = row['fileName'].split('/')[0]
        imageFileName = row['fileName'].split('/')[1]
        labelFileName = imageFileName.replace('.jpg', '.txt')
        #cameraId = "" + imageFilePath + '_'
        cameraId = ""
        count += 1
        if count % skip == 0: # Save to test dataset
            pathToDest = pathToDestDataset.replace("train", "test")
            print("Test image", cameraId+labelFileName)
        else: # save to train dataset
            pathToDest = pathToDestDataset
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
        
    
if __name__=='__main__':
    
    splitPercentage = 100 # Only for test dataset
    pathToSrcDataset = 'D:/OTAR/detections/'
    pathToRecordData = 'D:/OTAR/images/'
    pathToDestDatasetMIE = 'D:/OTAR/trainAnn/'
    pathToDestDataset = 'D:/OTAR/trainAnn/'
        
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
    selDataset1 = data_frames
    createLabelsAndImages(selDataset1, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    



    