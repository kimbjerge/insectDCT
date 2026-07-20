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

videoNames = [
    "ACS-HQ111_2024_04_14_07_01_19.avi",
    "ACS-HQ200_2024_04_14_07_01_19.avi",
    # "ACS-HQ222_2024_04_14_07_01_19.AVI", 1920x1080
    "ACS-HQ333_2024_04_14_07_01_19.avi",
    "ACS-HQ444_2024_04_14_07_01_19.avi",
    "ACS-HQ555_2024_04_14_07_01_19.avi",
    "ACS-HQ666_2024_04_14_07_01_19.avi",
    "ACS-HQ777_2024_04_14_07_01_19.avi",
    "cam-div_2026_04_22_13_59_34.avi",
    "cam-div_2026_04_22_14_15_08.avi",
    "cam-div_2026_04_22_14_41_46.avi",
    "cam-div_2026_04_22_15_12_22.avi",
    "cam-div_2026_04_22_16_00_38.avi",
    "cam-div_2026_04_25_14_47_12.avi",
    "cam-vilg_2026_05_09_15_13_02.avi",
    "cam-vilg_2026_05_09_15_30_16.avi",
    "cam-vilg_2026_05_09_16_05_08.avi",
    "cam-wknt_2026_06_19_11_08_39.avi",
    "cam-wknt_2026_06_19_11_28_53.avi",
    "cam-wknt_2026_06_19_14_36_47.avi",
    "cam-wknt_2026_06_19_14_47_40.avi",
    "cam-wknt_2026_06_19_15_10_45.avi",
    "cam-wknt_2026_06_19_15_33_48.avi",
    "cam-zoo_2026_04_26_11_42_30.avi",
    "cam-zoo_2026_04_26_13_03_30.avi",
    "cam-zoo_2026_04_26_13_50_38.avi",
    "cam-zoo_2026_04_26_14_10_12.avi",
    "cam-zoo_2026_04_26_14_27_26.avi",
    "cam-zoo_2026_06_20_11_43_48.avi",
    "cam-zoo_2026_06_20_12_09_00.avi",
    "cam-zoo_2026_06_20_12_28_34.avi",
    "cam-zoo_2026_06_20_12_38_58.avi",
    "cam-zoo_2026_06_20_12_45_41.avi",
    "cam-zoo_2026_06_20_12_56_10.avi",
    "cam-zoo_2026_06_20_13_03_22.avi",
    "cam-zoo_2026_06_20_13_17_14.avi",
    "cam-zoo_2026_06_20_13_17_54.avi",
    "cam-zoo_2026_06_20_13_20_29.avi",
    "cam-zoo_2026_06_20_13_53_28.avi",
    "cam-zoo_2026_06_20_14_20_32.avi",
    "cam-zoo_2026_06_20_14_42_20.avi",
    "cam-zoo_2026_06_20_14_48_47.avi",
    "cam-zoo_2026_06_20_15_33_00.avi",
    "cam10-wknt_2026_06_19_10_49_42.avi",
    "cam10-wknt_2026_06_19_12_57_11.avi",
    "cam4-wknt_2026_06_19_10_43_11.avi",
    "cam8-wknt_2026_06_19_11_44_54.avi",
    "cam8-wknt_2026_06_19_14_49_54.avi"
    ]

# UFZ image size, Pi model 3 camera HD resolution
#IMG_WIDTH = 1920
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
        numInsects = 2100 # dataset
        numUnsure = 300 # dataset
        numVegetation = 200 # dataset
        save_empty_frame_offset = 2
        splitPercentage = 20 # Percentage of image used for test
        pathToSrcDataset = 'D:/OTAR2/detectionsJulyTrain/'
        

    pathToRecordData = 'D:/OTAR2/videoJulyTrain/'
    
    pathToDestDatasetMIE = 'D:/OTAR2/trainGA2m/'
    pathToDestDataset = 'D:/OTAR2/trainGA2/'
    
                
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
    selDataset2 = selDataset2.sample(n=numInsects, random_state=37)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Vegetation"]
    selDataset2 = selDataset1.sample(n=numVegetation, random_state=65)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)
    
    selDataset1 = data_frames.loc[data_frames['taxaLabel'] == "Unsure"]
    selDataset2 = selDataset1.sample(n=numUnsure, random_state=43)
    selDataset3 = selDataset2.sort_values(by=['fileName', 'frameId'])
    createLabelsAndImages(selDataset3, data_frames, pathToRecordData, pathToDestDataset, pathToDestDatasetMIE, splitPercentage)


    