# -*- coding: utf-8 -*-
"""
Created on Mon June 5 08:35:16 2025

@author: Kim Bjerge
         Aarhus University
"""
import os
import cv2
import argparse
import datetime
import time
from skimage import io
from idac.configreader.configreader import readconfig
from idac.datareader.data_reader import DataReader
from idac.tracker.tracker import Tracker
from idac.imagemod.image_mod import Imagemod
from idac.moviemaker.movie_maker import MovieMaker
from idac.stats.stats import Stats
from idac.predictions.predictions import Predictions
from idac.tracksSave.tracksSave import TracksSave
#from PyQt5.QtGui import QImage
import pickle

#%% if --checkTaxa is True, then the below class is used to 
# test if hierarchical classifications is the same insect in track
class TaxaHierarchy():
    
    def __init__(self, hierachyL1, hierachyL2, labelsL1, labelsL2, labelsL3):
        self.hierachyL1 = hierachyL1
        self.hierachyL2 = hierachyL2
        self.labelsL1 = labelsL1
        self.labelsL2 = labelsL2
        self.labelsL3 = labelsL3
        self.logFile = open("trackLog.txt", "a")
        
    def __del__(self):
        self.logFile.close()
    
    def log(self, text):
        print(text)
        self.logFile.write(text + '\n')
        self.logFile.flush()

    def checkSameInsect(self, nameX, levelX, nameY, levelY):
        
        #logStr = "Check insects  :" + nameX + str(levelX) + nameY + str(levelY)
        #self.log(logStr)
        
        if (nameX == "Unsure") or (nameY == "Unsure"): # One of the insects are "Unsure"
            return True

        # Find the correct highest level
        if nameX in self.labelsL3:
            levelX = 3
        if nameX in self.labelsL2:
            levelX = 2
        if nameX in self.labelsL1:
            levelX = 1
            
        if nameY in self.labelsL3:
            levelY = 3
        if nameY in self.labelsL2:
            levelY = 2
        if nameY in self.labelsL1:
            levelY = 1
  
        if levelX == levelY: # Classification at the same taxonomic rank (level)
            if nameX == nameY:
                return True
            else:
                # Check same genus (For level 3)
                if nameX in nameY:
                    return True
                if nameY in nameX:
                    return True
                
                # If level3 different but same level2 then OK???
                
                logStr = f"Different insects A: {nameX} L{levelX} - {nameY} L{levelY}"
                self.log(logStr)
                
                return False # True if same at higher rank?
        
        if levelX < levelY: # Insect X at higher rank than insect Y
            levelA = levelX
            nameA = nameX
            levelB = levelY
            nameB = nameY
        else: # Insect Y at higher rank than insect X
            levelA = levelY
            nameA = nameY
            levelB = levelX
            nameB = nameX
            
        if levelA == 1 and levelB == 2:
            if nameA in self.hierachyL1.keys():
                if nameB in self.hierachyL1[nameA]: # Check hierarchy L1 -> L2
                    return True
        if levelA == 2 and levelB == 3:
            if nameA in self.hierachyL2.keys():
                if nameB in self.hierachyL2[nameA]: # Check hierarchy L2 -> L3
                    return True
        if levelA == 1 and levelB == 3:
            if nameA in self.hierachyL1.keys():
                for nameL2 in self.hierachyL1[nameA]:
                    if nameB in self.hierachyL2[nameL2]: # Chech hierarchy L1 -> L2 -> L3
                        return True
                            
        logStr = f"Different insects B: {nameX} L{levelX} - {nameY} L{levelY}"
        self.log(logStr)
        
        return False # Not same insect
    
        
    def validate(self):
        
        assert(self.checkSameInsect("Apis mellifera", 3, "Bombus lapidarius", 3) == False)
        assert(self.checkSameInsect("Vespula vulgaris", 3, "Bombus lapidarius", 3) == False)
        assert(self.checkSameInsect("Bombus lapidarius", 3, "Bombus lapidarius", 3) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Unsure", 0) == True)
        assert(self.checkSameInsect("Unsure", 0, "Apidea", 2) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Apidae", 2) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Apidae", 3) == True)
        assert(self.checkSameInsect("Apidae", 2, "Apis mellifera", 3) == True)
        assert(self.checkSameInsect("Apidae", 3, "Apis mellifera", 3) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Apoidea", 1) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Apoidea", 2) == True)
        assert(self.checkSameInsect("Apis mellifera", 3, "Apoidea", 3) == True)
        assert(self.checkSameInsect("Apoidea", 1, "Apis mellifera", 3) == True)
        assert(self.checkSameInsect("Apoidea", 2, "Apis mellifera", 3) == True)
        assert(self.checkSameInsect("Apoidea", 3, "Apis mellifera", 3) == True)
        assert(self.checkSameInsect("Syrphidae", 2, "Eupeodes", 3) == True)
        assert(self.checkSameInsect("Syrphidae", 3, "Eupeodes", 3) == True)
        assert(self.checkSameInsect("Episyrphus balteatus", 3, "Syrphidae", 2) == True)
        assert(self.checkSameInsect("Episyrphus balteatus", 3, "Syrphidae", 3) == True)
    
    
#%% Convert class hierarchy to a flat structure with labels : L1, L2 and L3 and removed dublicates
def createFlatSpeciesList(label_file):
    
    with open(label_file, 'rb') as f:
        _, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, _, _, _, _, _, _ = pickle.load(f)
        print("Labels and hierarchy dependency loaded from ", label_file)
        print("=============================================================================================")
        print("L1 classes", labelsL1, len(labelsL1))
        print("=============================================================================================")
        print("L2 classes", labelsL2, len(labelsL2))
        print("=============================================================================================")
        print("L3 classes", labelsL3, len(labelsL3))
        print("=============================================================================================")
        print("L2 -> L1 dependency", hierarchyL1)
        print("=============================================================================================")
        print("L3 -> L2 dependency", hierarchyL2)
        print("=============================================================================================")
        
    speciesList = []

    for label in labelsL1:
        speciesList.append(label)
    for label in labelsL2:
        if label not in speciesList:
            speciesList.append(label)
    for label in labelsL3:
        if label not in speciesList:
            speciesList.append(label)
            
    speciesList.append("Unsure") # Unsure label if below threshold or wrong hierarchy
    
    # Create class to check taxaHierarchy
    taxaHierarchy = TaxaHierarchy(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3)
    
    return speciesList, taxaHierarchy

        
#%% Return datetime based on image filename with the format: camera_YYYY_MM_DD_HH_MM_SS.jpg
"""
def getDateTime(image_filename): 

    if args.dateFormat == 'YYYY_MM_DD':
        nameSplit = image_filename.split('.')[0].split('_')
        dateTimeStr = nameSplit[1] + nameSplit[2] + nameSplit[3] + nameSplit[4] + nameSplit[5] + nameSplit[6] # Format: YYYYMMDDHHMMSS
    else: # Format 'YYYYMMMDD'
        dateTimeStr = image_filename.split('.')[0]
    
    return dateTimeStr
"""

#%% Run tracking of time-lapse images in CSV file specified by dirName       
def run(trackName, imagePath, detectPath, trackPath, conf, taxaHierarchy, ignoreVegetation, videoCap=None):
    
    writemovie = conf['moviemaker']['writemovie']
    reader = DataReader(conf)
    gen = reader.getimage()
    print(type(gen))

    tr = Tracker(conf, taxaHierarchy)
    imod = Imagemod()
    conf['moviemaker']['resultdir'] = trackPath # Overwrite resultDir in configuration with trackPath
    mm = MovieMaker(conf, name=trackName+'-TR.avi')

    stat = Stats(conf)
    predict = Predictions(conf)
    tracksFilename = trackPath+trackName+'-TRS.csv'
    print(tracksFilename)
    tracks = TracksSave(tracksFilename)

    csvFilename = detectPath+trackName+'-CL.csv'

    if ignoreVegetation:
        predicted = predict.load_predictionsTaxon(csvFilename, filterTime=0, ignoreLabels=['Vegetation']) # Skip if not moved within filterTime in seconds
    else:
        predicted = predict.load_predictionsTaxon(csvFilename, filterTime=0, ignoreLabels=[]) 
        
    totPredictions, totFilteredPredictions = predict.getPredictions()
    total = len(predicted)
    startid = 0

    iterCount = 0
    firstTime = 1
    oldFile = ""
    frame_count = 0
    
    for insect in predicted:
        file = insect['image']
        
        #if file.lower().endswith(('.jpg', '.png')):
        #    filedatetime = getDateTime(file)
        #else:
        filedatetime = insect['datetimeStr']
            
        if args.trapFilePath == True: # Trap (system) part of file path - used in project Orchard
            filepath = insect['system'] + '/' + insect['pathimage']
        else:
            filepath = insect['pathimage']
            
        if oldFile == file:
            oldFile = file
            continue
        else:
            oldFile = file
            
        iterCount += 1
        print('Image nr. ' + str(iterCount) + '/' + str(total), file)
        #time1 = time.time()
        
        if firstTime == 1:
            firstTime = 0
            count, ooi1 = predict.findboxes(file, predicted)
            for oi in ooi1:
                oi.id = startid
                startid = startid + 1            
        
        count2, ooi2 = predict.findboxes(file, predicted)
        if count2 > 0:
            goods, startid = tr.track_boxes(ooi1, ooi2, count2, startid)
            ooi1 = goods
            tracks.save(insect, goods)
            stat.update_stats(goods, filedatetime)
            #print(stat.count)

            if writemovie:
                success = True
                if videoCap != None: # Use video recording
                    if insect['frameId'] < 2: # Ignore first 5 frames KBE???
                        success = False
                    while success and (frame_count < insect['frameId'] + 1): # Why offset needed KBE??? frame_skip = 1
                    #while success and (frame_count < insect['frameId'] - 1): # Why offset needed KBE??? frame_skip = 3
                        success, im = videoCap.read()
                        if ".mp4" in args.video:
                            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        frame_count += 1
                    
                    #success = False
                    #if insect['frameId'] > 3:
                    #    videoCap.set(cv2.CAP_PROP_POS_FRAMES, insect['frameId']) # Not working wrong frame???
                    #    frame_count = insect['frameId']
                    #    success, im = videoCap.read()
                        
                    print("Video frame", frame_count, "detected frame", insect['frameId'], success)
                else:
                    file_name = imagePath+filepath
                    im = io.imread(file_name)
                    

                if success:
                    image = imod.drawoois(im, goods)
                    height, width, channel = image.shape
                    #print("Image shape", width, height, channel)
                    if width != 1920 and height != 1080:
                        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
                    #bytesPerLine = 3 * width
                    #qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            
                    # Write frame
                    mm.writeframe(image, filedatetime)

            #time2 = time.time()
            #print('Processing image took {:.3f} ms'.format((time2 - time1) * 1000.0))

    if writemovie:
        mm.releasemovie()
        
    tracks.close()
    stat.writedetails(trackPath+trackName+'-TR')

    return stat, iterCount, totPredictions, totFilteredPredictions

def print_totals(date, stat, resultdir):
    
    record = str(date) + ','
    for spec in stat.species:
        print(spec, stat.count[spec])
        record += str(stat.count[spec]) + ','
    print('Tracks total', stat.count['total'])
    record += str(stat.count['total']) + '\n'

    file = open(resultdir + 'statistics.csv', 'a')
    file.write(record)
    file.close()

    stat.count['date'] = date
    file = open(resultdir + 'statistics.json', 'a')
    file.write(str(stat.count) + '\n')
    file.close()

if __name__ == '__main__':

    version = "pipeTrackInsectsTaxon.py version: 1.1.1\n" # Updated for models trained on datasetV6

    print('Tracking insects based on detection files *-DL.csv')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='./images/') #Path to images used with fileName in *-CL.csv files
    parser.add_argument('--video', default='') # Path to video recording (If video used instead of single images)
    parser.add_argument('--detections', default='./detections/') #Directory that contains detections in *-CL.csv files
    parser.add_argument('--tracks', default='./tracks/') #Directory where track results are stored
    #parser.add_argument('--dateFormat', default='YYYY_MM_DD') #Filename data format or 'YYYYMMDD', not used anymore
    parser.add_argument('--dataset', default='V6') #dataset V2 (ResNet), dataset V3 or V4 (ResNet or ConvNextBase), dataset V4
    parser.add_argument('--checkTaxa', default='', type=bool) # Use hierarchy to check if same insect in track, empty = False 
    parser.add_argument('--trapFilePath', default='', type=bool) # Is trap (system) part of file path to images used by project Orchard
    parser.add_argument('--ignoreVegetation', default='True', type=bool) # Do not use classified vegetation part of tracking, empty = False
    args = parser.parse_args() 
    
    # Read configuration file
    config_filename = './config/Taxon_config.json'
    conf = readconfig(config_filename)
    
    # Convert hierarchical labels to flat list of labels
    if args.dataset == 'V2':
        conf["classifier"]['species'], taxaHierarchy = createFlatSpeciesList(conf["classifier"]["labelFile"])
    if args.dataset == 'V3' or args.dataset == 'V4':
        conf["classifier"]['species'], taxaHierarchy = createFlatSpeciesList(conf["classifier"]["labelFileV3"])
    if args.dataset == 'V5':
        conf["classifier"]['species'], taxaHierarchy = createFlatSpeciesList(conf["classifier"]["labelFileV5"])
    if args.dataset == 'V6':
        conf["classifier"]['species'], taxaHierarchy = createFlatSpeciesList(conf["classifier"]["labelFileV6"])
        
    print(version, args)
    with open(args.tracks+"/pipeTrackInsectsTaxon.txt", "a") as f:
        f.write(version)
        f.write("Arguments: " + str(args) + '\n')
        f.write("Processing time start: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write("================================================================================================================\n")
        f.close()
    time.sleep(5)
        
    # For testing only 
    #taxaHierarchy.validate()
    
    if args.checkTaxa == False:
        taxaHierarchy = None # Disable using hierarchy to check if same insect in track - see TaxaHierarchy class
    else:
        print("Use hierarchy to check if same insect in track")
    
    # Open the input video file if specified
    video_path = args.video
    videoFiles = []
    if video_path != '': # Process video file
        videoFiles = os.listdir(video_path)
        
    imageCounts = 0
    totalPredictions = 0
    totalFilteredPredictions = 0
    for fileName in sorted(os.listdir(args.detections)): # fileName must be in format <trapId>_YYYY_MM_DD-CL.csv
        if '-CL.csv' in fileName:
            trackName = fileName.split('-')[0] 
            
            videoCap = None
            videoFile = ""
            if video_path != "":
                videoFile = [file for file in videoFiles if trackName in file]
                if len(videoFile) == 1:
                    videoFile = videoFile[0]
                else:
                    videoFile = ""
            if videoFile != "": 
                print("Uses video recording", video_path + videoFile)
                videoCap = cv2.VideoCapture(video_path + videoFile)
                
            print(fileName, trackName)
            stat, counts, totPred, totFiltered = run(trackName, args.images, args.detections, args.tracks, conf, taxaHierarchy, args.ignoreVegetation, videoCap)
            totalPredictions += totPred
            totalFilteredPredictions += totFiltered
            imageCounts += counts

            if videoCap != None:
                videoCap.release()

            """ Not used 
            if args.dateFormat == 'YYYY_MM_DD':
                trackNameSplit = trackName.split('_')
                date = int(trackNameSplit[1] + trackNameSplit[2] + trackNameSplit[3])  # format YYYYMMDD
            else: # Assumed 'YYYYMMDD'
                date = int(trackName)  # format YYYYMMDD
             
            print_totals(date, stat, args.tracks)
            """
         
    print("Images", imageCounts, "Predictions", totalPredictions) #, "Filtered", totalFilteredPredictions)
