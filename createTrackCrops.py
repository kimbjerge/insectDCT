# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 10:07:04 2025

Creates plots of tracks with crops of detected insect taxa in a sequence of images 
Used the *-TRS.csv files as inputs from the insect tracker

@author: Kim Bjerge
"""
import os
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt

smallInsects = True # UFZ is small, MAMBO is not
showAxis = 'off' # Or 'on'

validTrackLength = 3 # Overwritten by args parameter validNum
validTrackConfidence = 50 # Overwritten by args parameter validConfTH

def plotTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, taxa, confidence, trackRows):
    
    # Define grid size
    rows, cols = 3, 4
    
    if smallInsects:        
        # Create a figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=(9, 7))
        plt.rcParams.update({'font.size': 10})
    else:
        # Create a figure and axes
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        plt.rcParams.update({'font.size': 20})
        
    axes = axes.flatten()

    i = 0
    length = 0
    for row in trackRows:

        if i < rows*cols:                
            imageFilePath = pathToRecordData + row['fileName']
            #print(imageFilePath)
            image = cv2.imread(imageFilePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
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
                color = [0, 0, 255] # Blue
            else:
                if row['taxaLabel'] == "Unsure":
                    color = [255, 255, 0] # Yellow
                else:
                    color = [255, 0, 0] # Red
                axes[i].set_title(row['taxaLabel'])
            for r in range(imgCrop.shape[0]):
                imgCrop[r, 0, :] = color 
                imgCrop[r, 1, :] = color 
            axes[i].imshow(imgCrop)
            axes[i].axis(showAxis)  # Hide axes
     
            del image 
            del imgCrop

        i += 1
        length += 1
     
    while i < rows*cols:
        axes[i].axis(showAxis)  # Hide axes
        i += 1
        
    plt.suptitle(taxa + " Id " + str(trackId) + " Len " + str(length) + " Conf " + str(confidence) + " Date " + str(trackDate))
    plt.tight_layout(pad=1.0)

    if os.path.exists(pathToDestCrops + taxa) == False:
        print("Create directory:", pathToDestCrops + taxa)
        os.mkdir(pathToDestCrops + taxa)
    fileName = row['fileName'].replace('/', '_').replace(".jpg", "") + "-" + str(trackId) + ".png"
    plt.savefig(pathToDestCrops + taxa + "/" + fileName)
    #plt.show()
    plt.close('all')
    
def calcConfidence(trackRows):

    labels = {}
    counts = 0
    for row in trackRows:   
        taxa = row['taxaLabel']
        counts += 1
        if taxa not in labels.keys():
            labels[taxa] = 1
        else:
            labels[taxa] += 1

    maximum = 0
    taxa = ""
    for key, value in labels.items():
        if value > maximum:
            maximum = value
            taxa = key
   
    confidence = int(1000*maximum/counts)
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
            del trackRows
            trackRows = []
            trackId = currTrackId
            elmCnt = 1
            validTrack = False
            trackRows.append(row)
            trackDate = row['date']

    if validTrack: # Save last valid track
        saveTrackCrops(pathToRecordData, pathToDestCrops, trackDate, trackId, trackRows)
             
if __name__=='__main__':
            
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tracks', default='./tracks') #Directory that contains CSV files of tracks
    parser.add_argument('--images', default='./images') #Directory that contains the image files
    parser.add_argument('--resultsDir', default='./trackCrops') # Optimized for embedded processing (ncnn)

    #parser.add_argument('--tracks', default='/UFZ/tracks') #Directory that contains CSV files of tracks
    #parser.add_argument('--images', default='O:/Tech_TTH-KBE/UFZ') #Directory that contains the image files
    #parser.add_argument('--resultsDir', default='/UFZ/trackCrops') # Optimized for embedded processing (ncnn)

    #parser.add_argument('--tracks', default='/RTNI/tracks') #Directory that contains CSV files of tracks
    #parser.add_argument('--images', default='O:/Tech_TTH-KBE/NI/RT') #Directory that contains the image files
    #parser.add_argument('--resultsDir', default='/RTNI/trackCrops') # Optimized for embedded processing (ncnn)
    parser.add_argument('--date', default="") # if date specified then only create track crops for specified date (YYYY_MM_DD or YYYMMDD)

    parser.add_argument('--validNum', default='3', type=int) # Number of detections used to define valid track
    #parser.add_argument('--validConfTH', default='20', type=int) # Confidence threshold used to define valid track
    parser.add_argument('--validConfTH', default='50', type=int) # Confidence threshold used to define valid track

    args = parser.parse_args() 
    print(args)
    
    validTrackLength = args.validNum 
    validTrackConfidence = args.validConfTH

    pathToSrcDataset = args.tracks + '/'
    pathToRecordData = args.images + '/'
    pathToDestCrops = args.resultsDir + '/'
    
    firstTime = True
    for filename in sorted(os.listdir(pathToSrcDataset)):
        if (filename.endswith('.csv')):

            if args.date == "":
                validDate = True
            else:
                validDate = False
                if args.date in filename:
                    validDate = True

            if validDate and "-TRS" in filename:
                print("Reading", filename)
                data_df = pd.read_csv(pathToSrcDataset+filename)
                data_df = data_df.sort_values(by=['id', 'time'])  
                subDir = filename.split('-')[0]
                pathToRecordDataSubDir = pathToRecordData + subDir + '/'
                #pathToRecordDataSubDir = pathToRecordData + subDir.split('_')[0] + '/' + subDir + '/'
                analyseTracks(data_df, pathToRecordDataSubDir, pathToDestCrops)