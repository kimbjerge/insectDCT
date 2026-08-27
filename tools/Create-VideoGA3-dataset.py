# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:38:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from the paper:
Serra-Martin et. al "Comparative assessment of automated and manual monitoring in comprehensive plant–pollinator communities"
Use systems with ACS-HQ cameras to create 10 sec. videos each 1 minute
Paper:https://doi.org/10.1111/2041-210X.70165
    
@author: Kim Bjerge
"""

import os
import cv2
import pandas 
import pandas as pd
from common.motionEnhancement import MotionEnhancement

# Save an empty frame with offset of last detected insects 
save_empty_frame_offset = 2 # If 0 then no frames are saved
#save_empty_frame_offset = 0 # If 0 then no frames are saved
frame_id_offset = 2 # frame_stride = 1 during detection
#frame_id_offset = 0 # frame_stride = 3 during detection

# Identification of camera sites
cameraId = "GA_"

videoNames = [ # Training and validation
    #"cam1-did_2026_08_22_13_26_02.avi", # HD size
    "cam2-did_2026_08_01_14_12_37.avi", # 4K size
    "cam3-zoo_2026_07_31_11_08_53.avi",
    "cam3-zoo_2026_07_31_12_18_08.avi",
    "cam4-did_2026_08_01_15_48_58.avi",
    "cam4-did_2026_08_08_11_40_28.avi",
    "cam4-did_2026_08_08_14_36_20.avi",
    "cam5-did_2026_08_01_14_18_08.avi",
    "cam6-did_2026_08_01_14_50_16.avi",
    "cam6-did_2026_08_08_11_55_34.avi",
    "cam8-did_2026_08_01_11_19_56.avi",
    "cam9-did_2026_08_01_16_20_22.avi",
    "cam10-did_2026_08_08_12_15_43.avi",
    "cam10-did_2026_08_08_12_50_16.avi",
    "cam10-did_2026_08_08_14_58_43.avi",
    "cam10-zoo_2026_07_31_11_52_24.avi",
    "cam5-did_2026_08_08_15_26_34.avi", # GA4
    "cam5-did_2026_08_08_17_10_36.avi" # GA4
    ]


# UFZ image size, Pi model 3 camera HD resolution
#IMG_WIDTH = 1920 # GA4
#IMG_HEIGHT = 1080
IMG_WIDTH = 2688
IMG_HEIGHT = 1520

def saveImages(frameId, srcFilename, count, skip, imageRGB, imageMIE, text="Insect"):
    
    frameIdTxt = '_' + str(frameId) + '.jpg'
    fileId = videoNames.index(srcFilename)
    #imageFileName = srcFilename.replace('.mp4', frameIdTxt)
    imageFileName = 'S' + str(fileId) + frameIdTxt
    labelFileName = imageFileName.replace('.jpg', '.txt')

    count += 1
    if count % skip == 0: # Save to test dataset
        # pathToDest = pathToDestDataset.replace("train", "val")
        # pathToDestMIE = pathToDestDatasetMIE.replace("train", "val")
        pathToDest = pathToDestDataset.replace("train", "test")
        pathToDestMIE = pathToDestDatasetMIE.replace("train", "test")
        print(text + " Val   image", cameraId+labelFileName)
    else: # save to train dataset
        pathToDest = pathToDestDataset
        pathToDestMIE = pathToDestDatasetMIE
        print(text + " Train image", cameraId+labelFileName)
    
    cv2.imwrite(pathToDest+cameraId+imageFileName, imageRGB)
    cv2.imwrite(pathToDestMIE+cameraId+imageFileName, imageMIE)
    
    return count, labelFileName, pathToDest
            
def createLabelsAndImages(selDataset, data_df, pathToRecordedFiles, pathToDestDataset, pathToDestDatasetMIE, split):
    
    mie = MotionEnhancement()
    
    skip = int(100/split)
    count = 0
    frame_count = 0
    frameId = 0
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
        
        frameIdLast = frameId
        frameId = row['frameId']

        print("Video frame", frame_count, "detected frame", frameId)
        
        # Handle saving empty image with save_empty_frame_offset of last frame
        if (frameId > frameIdLast + save_empty_frame_offset) and (save_empty_frame_offset > 0) and (frameIdLast > 0):
            framePosEmpty = frameIdLast + save_empty_frame_offset + frame_id_offset
            success = True
            while success and (frame_count < framePosEmpty): # Reading next image to create MIE
                imageRGB = imageNext
                success, imageNext = videoCap.read()
                if success and (frame_count > framePosEmpty - 3):
                    imageMIE, imgPrev = mie.motion_image(imageNext)
                frame_count += 1
                
            if success:
                count, labelFileName, pathToDest = saveImages(frameIdLast + save_empty_frame_offset, 
                                         row['fileName'], count, skip, imageRGB, imageMIE, text="Empty ")
                
                labelFile = open(pathToDest+cameraId+labelFileName, "w") # Create empty label file
                labelFile.close()
                
        # Handle saving image with detections
        framePos = frameId + frame_id_offset
        success = True
        if frameId < 3: # Ignore first frames 
            success = False
        while success and (frame_count < framePos): # Reading next image to create MIE
            imageRGB = imageNext
            success, imageNext = videoCap.read()
            if success and (frame_count > framePos - 3):
                imageMIE, imgPrev = mie.motion_image(imageNext)
            frame_count += 1
        
        if success:
        
            count, labelFileName, pathToDest = saveImages(frameId, row['fileName'], count, skip, imageRGB, imageMIE)
            
            detections_df = data_df.loc[data_df['fileName'] == currentVideoFileName]
            detections_df = detections_df.loc[detections_df['frameId'] == frameId]
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
            
            
if __name__=='__main__':
    
    TrainDataset = False
    if TrainDataset == False: # Not used for training

        # Below should write to trainGA2
        # Selecting videos with large insects and not detected
        
        pathToSrcDataset = 'D:/OTAR2/detectionsAugTrain2/'
        numInsects = 1300 # dataset
        numUnsure = 200 # dataset
        numVegetation = 300 # dataset
        
        #pathToSrcDataset = 'D:/OTAR2/detectionsAugTrain/' # GA4
        #numInsects = 50 # dataset
        #numUnsure = 50 # dataset
        #numVegetation = 50 # dataset

        save_empty_frame_offset = 2
        splitPercentage = 20 # Percentage of image used for validation
        
        # Below should write to testGA2
        # Selecting videos with large insects and not detected
        #numInsects = 2000 # dataset
        #numUnsure = 300 # dataset
        #numVegetation = 5 # dataset
        #save_empty_frame_offset = 0
        #splitPercentage = 100 # Percentage of image used for test
        #pathToSrcDataset = 'D:/OTAR2/detectionsJulyTest/'
        

    #pathToRecordData = 'D:/OTAR2/videoJulyTrain/'
    pathToRecordData = 'D:/OTAR2/videoAug2/'
    
    pathToDestDatasetMIE = 'D:/OTAR2/trainGA3m/'  # GA4
    pathToDestDataset = 'D:/OTAR2/trainGA3/' #GA4
    
                
    #for filename in sorted(os.listdir(pathToRecordData)):
    #    print(f"\"{filename}\",")
        
    firstTime = True
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
    selDataset2 = selDataset2.sample(n=numInsects, random_state=13)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Vegetation"]
    selDataset2 = selDataset1.sample(n=numVegetation, random_state=42)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Unsure"]
    selDataset2 = selDataset1.sample(n=numUnsure, random_state=114)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)


    