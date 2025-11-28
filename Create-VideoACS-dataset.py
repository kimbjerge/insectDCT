# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:38:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from Real-time Nature Impact images 
Use camera systems wiht Jetson Nano and Logitech cameras (Each system with one Logitech Camreas)
Paper:https://doi.org/10.1002/rse2.245
    
@author: Kim Bjerge
"""

import os
import cv2
import shutil
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

# UFZ image size, Pi model 3 camera HD resolution
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def createLabelsAndImages(selDataset, data_df, pathToRecordedFiles, pathToDestDataset, pathToDestDatasetMIE, split):
    
    mie = MotionEnhancement()
    
    skip = int(100/split)
    count = 0
    frame_count = 0
    currentVideoFileName = ""
    videoCap = None
    imageRGB = None
    imageNext = None

    for idx, row in selDataset.iterrows():
        
        if currentVideoFileName != row['fileName']: # Check for new video
            currentVideoFileName = row['fileName']
            pathToVideoFile = pathToRecordedFiles + currentVideoFileName
            print("Uses video recording", pathToVideoFile)
            if videoCap != None:
                videoCap.release()
            videoCap = cv2.VideoCapture(pathToVideoFile) # Opens new video file to capture frames
            frame_count = 0
        
        frameId = row['frameId']
        framePos = frameId + 2
        success = True
        if frameId < 3: # Ignore first frames 
            success = False
        #while success and (frame_count < insect['frameId']): # Why offset needed KBE??? frame_stride = 3
        while success and (frame_count < framePos): # Reading next image to create MIE
            imageRGB = imageNext
            success, imageNext = videoCap.read()
            if success and (frame_count > framePos - 3):
                imageMIE, imgPrev = mie.motion_image(imageNext)
            frame_count += 1
        
        print("Video frame", frame_count, "detected frame", frameId, success)

        if success:
            
            detections_df = data_df.loc[data_df['fileName'] == currentVideoFileName]
            detections_df = detections_df.loc[detections_df['frameId'] == frameId]
            
            frameIdTxt = '_' + str(frameId) + '.jpg'
            imageFileName = row['fileName'].replace('.mp4', frameIdTxt)
            labelFileName = imageFileName.replace('.jpg', '.txt')
            cameraId = "ACS_"

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
                # Label = 0 (insect)
                line = "0 " + str(xc/IMG_WIDTH) + " " + str(yc/IMG_HEIGHT) + " " + str(w/IMG_WIDTH) + " " + str(h/IMG_HEIGHT)
                print(line)
                labelFile.write(line + "\n")
            labelFile.close()
            
            cv2.imwrite(pathToDest+cameraId+imageFileName, imageRGB)
            cv2.imwrite(pathToDestMIE+cameraId+imageFileName, imageMIE)
            
            
if __name__=='__main__':
    
    TrainDataset = False
    if TrainDataset == False: # Not used for training
        numInsects = 2000
        numUnsure = 250
        numVegetation = 250
        splitPercentage = 20 # Percentage of image used for test
        pathToSrcDataset = 'D:/PAU/detections_A/'

    pathToRecordData = 'D:/PAU/videos/'
    pathToDestDatasetMIE = 'D:/PAU/trainACSHQm/'
    pathToDestDataset = 'D:/PAU/trainACSHQ/'
                
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
    selDataset2 = selDataset1.sample(n=numVegetation, random_state=65)
    selDataset3 = selDataset2.sort_values(by=['fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Unsure"]
    selDataset2 = selDataset1.sample(n=numUnsure, random_state=43)
    selDataset3 = selDataset2.sort_values(by=['fileName'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)


    