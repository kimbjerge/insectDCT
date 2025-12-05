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
cameraId = "PP_"

videoNames = [
    "ACS-HQ10_2024_04_13_18_51_16_rotated.mp4",
    "ACS-HQ10_2024_04_17_13_52_49.mp4",
    "ACS-HQ11_2024_04_17_13_52_49_NI_ipomoea.mp4",
    "ACS-HQ12_2024_04_17_13_52_49.mp4",
    "ACS-HQ13_2024_04_17_13_52_49_GS_ipomoea.mp4",
    "ACS-HQ14_2024_04_17_13_52_49_GS_asystacia.mp4",
    "ACS-HQ14_2024_04_17_13_52_49_GS_scaevola.mp4",
    "ACS-HQ16_2024_02_19_15_54_49_apis.mp4",
    "ACS-HQ16_2024_04_17_13_52_49.mp4",
    "ACS-HQ16_2024_04_17_13_52_49_apis.mp4",
    "ACS-HQ17_2024_04_17_13_52_49_FE_anacardium.mp4",
    "ACS-HQ17_2024_04_17_13_52_49_GS_premna.mp4",
    "ACS-HQ17_2024_04_17_13_52_49_GS_stachytarpeta.mp4",
    "ACS-HQ19_2024_04_17_13_52_49_GS_chrysobalanus.mp4",
    "ACS-HQ19_2024_04_17_13_52_49_NI_scaevola.mp4",
    "ACS-HQ19_2024_04_17_13_52_49_thespesia_GS.mp4",
    "ACS-HQ1_2024_02_19_15_24_42_lasio_xylocopa.mp4",
    "ACS-HQ1_2024_04_17_13_52_49_AR_scaevola.mp4",
    "ACS-HQ22_2024_04_17_13_52_49_bee.mp4",
    "ACS-HQ25_2024_04_17_13_52_49_GS_morinda.mp4",
    "ACS-HQ27_2024_04_17_13_52_49_GS_chrysobalanus.mp4",
    "ACS-HQ27_2024_04_17_13_52_49_NI_anacardium.mp4",
    "ACS-HQ28_2024_04_17_13_52_49_NI_chrysobalanus.mp4",
    "ACS-HQ29_2024_04_17_13_52_49.mp4",
    "ACS-HQ29_2024_04_17_13_52_49_GS_premna.mp4",
    "ACS-HQ2_2024_02_19_15_24_42.mp4",
    "ACS-HQ2_2024_04_17_13_52_49_GS_chrysobalanus.mp4",
    "ACS-HQ2_2024_04_17_13_52_49_GS_ipomoea.mp4",
    "ACS-HQ30_2024_04_17_13_52_49_apis_lasio.mp4",
    "ACS-HQ34_2024_02_19_15_54_49_megachile.mp4",
    "ACS-HQ34_2024_04_17_13_52_49_megachile.mp4",
    "ACS-HQ38_2024_04_17_13_52_49_apis.mp4",
    "ACS-HQ3_2024_04_14_00_50-23_lasio_sarcophaga.mp4",
    "ACS-HQ3_2024_04_17_13_52_49.mp4",
    "ACS-HQ3_2024_04_17_13_52_49_NI_premna.mp4",
    "ACS-HQ45_2024_04_17_13_52_49.mp4",
    "ACS-HQ45_2024_04_17_13_52_49_apis.mp4",
    "ACS-HQ47_2024_04_17_13_52_49_apis_formiga.mp4",
    "ACS-HQ4_2024_02_19_15_54_49_lasioglossum.mp4",
    "ACS-HQ4_2024_04_17_13_52_49.mp4",
    "ACS-HQ4_2024_04_17_13_52_49_AR_scaevola.mp4",
    "ACS-HQ4_2024_04_17_13_52_49_NI_stachytarpeta.mp4",
    "ACS-HQ56_2024_04_17_13_52_49.mp4",
    "ACS-HQ5_2024_04_13_12_19_21_Sirfid.mp4",
    "ACS-HQ5_2024_04_17_13_52_49_AR_scaevola.mp4",
    "ACS-HQ5_2024_04_17_13_52_49_GS_premna.mp4",
    "ACS-HQ5_2024_04_17_13_52_49_NI_stachytarpeta.mp4",
    "ACS-HQ60_2024_03_04_19_14_44_sarcophaga.mp4",
    "ACS-HQ61_2024_04_13_17_52_49_nomioides.mp4",
    "ACS-HQ62_2024_04_13_19_51_17_anthophora_bona.mp4",
    "ACS-HQ6_2024_03_03_20_33_09_Anthophora.mp4",
    "ACS-HQ6_2024_04_17_13_52_49_GS_premna.mp4",
    "ACS-HQ70_2024_04_13_22_55_32.mp4",
    "ACS-HQ71_2024_04_13_23_05_14.mp4",
    "ACS-HQ72_2024_04_13_23_35_34.mp4",
    "ACS-HQ73_2024_04_14_01_01_09.mp4",
    "ACS-HQ7_2024_03_04_13_35_53_rotated.mp4",
    "ACS-HQ7_2024_04_17_13_52_49_NI_coco.mp4",
    "ACS-HQ8_2024_02_20_08_34_20_anthophora.mp4",
    "ACS-HQ8_2024_04_17_13_52_49_NI_morinda.mp4",
    "ACS-HQ8_2024_04_17_13_52_49_apis_party.mp4",
    "ACS-HQ9_2024_02_19_15_54_49_rotated.mp4",
    "ACS-HQ9_2024_04_17_13_52_49_NI_scaevola.mp4"
    ]

# UFZ image size, Pi model 3 camera HD resolution
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def saveImages(frameId, srcFilename, count, skip, imageRGB, imageMIE, text="Insect"):
    
    frameIdTxt = '_' + str(frameId) + '.jpg'
    fileId = videoNames.index(srcFilename)
    #imageFileName = srcFilename.replace('.mp4', frameIdTxt)
    imageFileName = 'S' + str(fileId) + frameIdTxt
    labelFileName = imageFileName.replace('.jpg', '.txt')

    count += 1
    if count % skip == 0: # Save to test dataset
        pathToDest = pathToDestDataset.replace("train", "val")
        pathToDestMIE = pathToDestDatasetMIE.replace("train", "val")
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
    
        # Selecting videos with large insects and not detected
        #numInsects = 500 # dataset_L
        #numUnsure = 200 # dataset_L
        #numVegetation = 2 # dataset_L
        #splitPercentage = 20 # Percentage of image used for test
        #save_empty_frame_offset = 2
        #pathToSrcDataset = 'D:/PAU/detections_L/'
        
        # Selecting videos with normal sized insects and not detected
        numInsects = 1250 # dataset
        numUnsure = 350 # dataset
        numVegetation = 5 # dataset
        #save_empty_frame_offset = 2
        splitPercentage = 20 # Percentage of image used for test
        pathToSrcDataset = 'D:/PAU/detections/'
        
        # Selecting videos with false positive background detections
        #numInsects = 200 # dataset
        #numUnsure = 300 # dataset
        #numVegetation = 50 # dataset
        #save_empty_frame_offset = 0
        #splitPercentage = 20 # Percentage of image used for test
        #pathToSrcDataset = 'D:/PAU/detections_FP/'

    pathToRecordData = 'D:/PAU/videos/'
    pathToDestDatasetMIE = 'D:/PAU/trainACSHQm/'
    pathToDestDataset = 'D:/PAU/trainACSHQ/'
                
    #for filename in sorted(os.listdir(pathToRecordData)):
    #    print(f"\"{filename}\",")
        
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


    