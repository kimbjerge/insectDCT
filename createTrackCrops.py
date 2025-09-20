# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 10:07:04 2025

@author: Kim Bjerge
"""
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

validTrackLength = 3
validTrackConfidence = 50

def plotTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, taxa, confidence, trackRows):
    
    # Define grid size
    rows, cols = 4, 4
    
    # Create a figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    plt.rcParams.update({'font.size': 20})
    axes = axes.flatten()

    i = 0
    for row in trackRows:

        if i < rows*cols:                
            imageFilePath = pathToRecordData + str(trackDate) + '/' + row['fileName']
            print(imageFilePath)
            image = cv2.imread(imageFilePath)
    
            height, width, channels = image.shape
            
            w = row['width']
            h = row['height']
            xc = row['xc']
            yc = row['yc']
    
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
            if taxa == row['taxaLabel']:
                color = [0, 0, 255]
            else:
                color = [255, 0, 0]
                plt.title(row['taxaLabel'])
            for r in range(wh2*2):
                imgCrop[r, 0, :] = color 
                imgCrop[r, 1, :] = color 
            axes[i].imshow(imgCrop)
            axes[i].axis('off')  # Hide axes

        i += 1
        
    plt.suptitle(taxa + " Id " + str(trackId) + " Len " + str(i) + " Conf " + str(confidence) + " Date " + str(trackDate))
    plt.tight_layout(pad=1.0)

    plt.savefig(pathToDestCrops + taxa + "_" + str(trackId) + "_" + row['fileName'].replace(".jpg", ".png"))
    plt.show()
        
    
def calcConfidence(trackRows):

    labels = {}
    counts = 0
    first = True
    for row in trackRows:   
        cnt = 1
        if first:
            cnt = 0
            first = False 
            
        taxa = row['taxaLabel']
        counts += cnt
        if taxa not in labels.keys():
            labels[taxa] = cnt
        else:
            labels[taxa] += cnt

    maximum = 0
    taxa = ""
    for key, value in labels.items():
        if value > maximum:
            maximum = value
            taxa = key
   
    confidence = int(1000*maximum/counts) # Newer version not +1
    confidence = confidence/10
    
    return confidence, taxa, counts 
    
def saveTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, trackRows):
    
    
    confidence, taxa, counts = calcConfidence(trackRows)
    
    print(trackDate, trackId, taxa,  counts, f"{confidence:.1f}")
    
    if confidence >= validTrackConfidence:
        plotTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, taxa, confidence, trackRows)
    
def analyseTracks(data_fp, pathToRecordData, pathToDestCrops):
    
    trackId = -1
    elmCnt = 0
    validTrack = False
    trackDate = ""
    trackRows = []
    for index, row in data_fp.iterrows():
        currTrackId = row["id"]
        if currTrackId == trackId: # Append to current track
            trackRows.append(row)
            elmCnt += 1
            if elmCnt >= validTrackLength: # Mark track valid
                validTrack = True
                
        else: #New track
            if validTrack: # Save valid track
                saveTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, trackRows)
            trackRows = []
            trackId = currTrackId
            elmCnt = 1
            validTrack = False
            trackRows.append(row)
            trackDate = row['date']
             
if __name__=='__main__':
            
    pathToSrcDataset = '/RTNI/tracks/'
    pathToRecordData = 'O:/Tech_TTH-KBE/NI/RT/'
    pathToDestCrops = '/RTNI/trackCrops/'
    
    firstTime = True
    # File format: S2_123-Aug09_1_88-20190808104930.jpg
    for filename in sorted(os.listdir(pathToSrcDataset)):
        if (filename.endswith('.csv')):
            if "-TRS" in filename:
                print("Reading", filename)
                data_df = pd.read_csv(pathToSrcDataset+filename)
                if firstTime:
                    data_frames = data_df.copy()
                    firstTime = False
                else:    
                    data_frames = pd.concat([data_frames, data_df])
                    
    analyseTracks(data_frames, pathToRecordData, pathToDestCrops)