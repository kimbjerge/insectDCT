# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:38:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from PollinatorWatch images 
Use camera systems S1, S2, S3, S4 and S5 (Each system with two Logitech cameras)

@author: Kim Bjerge
"""

import os
import cv2
import shutil
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def createLabelsAndImages(selDataset, data_df, pathToRecordedFiles, pathToDestDataset, pathToDestDatasetMIE):
    
    mie = MotionEnhancement()
    
    for idx, row in selDataset.iterrows():
        detections_df = data_df.loc[data_df['fileName'] == row['fileName']]
        imageFilePath =  row['fileName'].split('/')[0] + '/'
        imageFileName = row['fileName'].split('/')[1]
        labelFileName = imageFileName.replace('.jpg', '.txt')
        print(labelFileName)
        labelFile = open(pathToDestDataset+labelFileName, "w")
        for i, detection in detections_df.iterrows():
            print(detection['fileName'], detection['x1'], detection['y1'], detection['x2'], detection['y2'])
            w = detection['x2'] - detection['x1']
            h = detection['y2'] - detection['y1']
            xc = detection['x1'] + 0.5*w
            yc = detection['y1'] + 0.5*h
            line = "0 " + str(xc/IMG_WIDTH) + " " + str(yc/IMG_HEIGHT) + " " + str(w/IMG_WIDTH) + " " + str(h/IMG_HEIGHT)
            print(line)
            labelFile.write(line + "\n")
        labelFile.close()
        shutil.copyfile(pathToRecordedFiles+row['fileName'], pathToDestDataset+imageFileName)
        mieImage = mie.motion_enhanced_image2(pathToRecordedFiles+imageFilePath, imageFileName)
        cv2.imwrite(pathToDestDatasetMIE+imageFileName, mieImage)
    
if __name__=='__main__':
    
    numImages = 100
    cameraSystem = "S3"
    pathToSrcDataset = 'J:/PollNI/' + cameraSystem + '/'
    pathToRecordData = 'O:/Tech_TTH-KBE/PollinatorWatch/FIN/' + cameraSystem + '/'
    pathToDestDatasetMIE = 'J:/PollNI/trainPollWm/'
    pathToDestDataset = 'J:/PollNI/trainPollW/'
    
    firstTime = True
    # File format: S2_123-Aug09_1_88-20190808104930.jpg
    for filename in sorted(os.listdir(pathToSrcDataset)):
        if (filename.endswith('.csv')):
            if "-CL" in filename:
                print("Reading", filename)
                data_df = pd.read_csv(pathToSrcDataset+filename)
                if firstTime:
                    data_frames = data_df.copy()
                    firstTime = False
                else:    
                    data_frames = pd.concat([data_frames, data_df])
                
    # Select only images where insects has been detected - many are false positive detections
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] != "Vegetation"]
    selDataset2 = selDataset1.loc[selDataset1['taxaLabel'] != "Unsure"]
    selDataset3 = selDataset2.sample(n=numImages, random_state=42)
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE)
    
    
    