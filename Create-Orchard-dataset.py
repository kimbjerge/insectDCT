# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:38:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from UFZ images 
Use camera systems Pi8-pi28 (Each system with one Pi model 3 cameras)

@author: Kim Bjerge
"""

import os
import cv2
import shutil
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

# UFZ image size
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
        imageFilePath =  row['fileName'].split('/')[0] 
        imageFileName = row['fileName'].split('/')[1]
        labelFileName = imageFileName.replace('.jpg', '.txt')
        cameraId = "OC_"
        imageFilePath += '/'
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
        pathToImageFile = pathToRecordedFiles+row['trap']+'/'+row['fileName']
        shutil.copyfile(pathToImageFile, pathToDest+cameraId+imageFileName)
        
        imageFile = row['fileName'].split('/')[1]
        pathToImages = pathToRecordedFiles+row['trap']+'/'+row['fileName'].split('/')[0]+'/'
        if prevPathToImages != pathToImages:
            fileList = sorted(os.listdir(pathToImages))
            prevPathToImages = pathToImages
        fileIdx = fileList.index(imageFile)
        filePrevIdx = fileIdx
        fileNextIdx = fileIdx
        if fileIdx > 0:
            filePrevIdx = fileIdx - 1
        if fileIdx < len(fileList) - 1:
            fileNextIdx = fileIdx + 1
        print(fileList[filePrevIdx], imageFile, fileList[fileNextIdx])
        mieImage = mie.motion_three_images(pathToImages, fileList[filePrevIdx], imageFile, fileList[fileNextIdx])
        cv2.imwrite(pathToDestMIE+cameraId+imageFileName, mieImage)
    
if __name__=='__main__':
    
    numInsects = 3000
    numUnsure = 1000
    numVegetation = 1000
    splitPercentage = 20 # Percentage of image used for test
    
    pathToSrcDataset = '/UFZ/detectionsTrain/'
    pathToRecordData = 'O:/Tech_TTH-KBE/UFZ/'
    pathToDestDatasetMIE = '/UFZ/trainOrchardm/'
    pathToDestDataset = '/UFZ/trainOrchard/'
    
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
    selDataset2 = selDataset2.sample(n=numInsects, random_state=37)
    selDataset3 = selDataset2.sort_values(by=['fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Vegetation"]
    selDataset2 = selDataset1.sample(n=numUnsure, random_state=65)
    selDataset3 = selDataset2.sort_values(by=['fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Unsure"]
    selDataset2 = selDataset1.sample(n=numVegetation, random_state=43)
    selDataset3 = selDataset2.sort_values(by=['fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)


    