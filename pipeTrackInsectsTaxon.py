# -*- coding: utf-8 -*-
"""
Created on Mon June 5 08:35:16 2025

@author: Kim Bjerge
         Aarhus University
"""
import time
import os
import argparse
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
            
    return speciesList

        
#%% Return datetime based on image filename with the format: camera_YYYY_MM_DD_HH_MM_SS.jpg
def getDateTime(image_filename): 

    nameSplit = image_filename.split('.')[0].split('_')
    dateTimeStr = nameSplit[1] + nameSplit[2] + nameSplit[3] + nameSplit[4] + nameSplit[5] + nameSplit[6] # Format: YYYYMMDDHHMMSS
    return dateTimeStr


#%% Run tracking of time-lapse images in CSV file specified by dirName       
def run(trackName, imagePath, detectPath, trackPath, conf):
    
    writemovie = conf['moviemaker']['writemovie']
    reader = DataReader(conf)
    gen = reader.getimage()
    print(type(gen))

    tr = Tracker(conf)
    imod = Imagemod()
    mm = MovieMaker(conf, name=trackName+'-TR.avi')

    stat = Stats(conf)
    predict = Predictions(conf)
    tracksFilename = trackPath+trackName+'-TRS.csv'
    print(tracksFilename)
    tracks = TracksSave(tracksFilename)

    csvFilename = detectPath+trackName+'-CL.csv'

    predicted = predict.load_predictionsTaxon(csvFilename, filterTime=0) # Skip if not moved within filterTime in seconds
    totPredictions, totFilteredPredictions = predict.getPredictions()
    total = len(predicted)
    startid = 0

    iterCount = 0
    firstTime = 1
    oldFile = ""

    for insect in predicted:
        file = insect['image']
        filedatetime = getDateTime(file)
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
                file_name = imagePath+filepath
                im = io.imread(file_name)
                image = imod.drawoois(im, goods)
                height, width, channel = image.shape
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

    print('Tracking insects based on detection files *-DL.csv')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='./images/') #Path to images used with fileName in *-CL.csv files
    parser.add_argument('--detections', default='./detections/') #Directory that contains detections in *-CL.csv files
    parser.add_argument('--tracks', default='./tracks/') #Directory where track results are stored
    parser.add_argument('--dateFormat', default='YYYY_MM_DD') #Filename data format or 'YYYYMMDD'
    parser.add_argument('--dataset', default="V2") #dataset V2 (ResNet), dataset V3 (ResNet or ConvNextBase)
    args = parser.parse_args() 
    print(args)
    
    # Read configuration file
    config_filename = './config/Taxon_config.json'
    conf = readconfig(config_filename)
    
    # Convert hierarchical labels to flat list of labels
    if args.dataset == 'V2':
        conf["classifier"]['species'] = createFlatSpeciesList(conf["classifier"]["labelFile"])
    else:
        conf["classifier"]['species'] = createFlatSpeciesList(conf["classifier"]["labelFileV3"])
        
    
    imageCounts = 0
    totalPredictions = 0
    totalFilteredPredictions = 0
    for fileName in sorted(os.listdir(args.detections)): # fileName must be in format <trapId>_YYYY_MM_DD-CL.csv
        if '-CL.csv' in fileName:
            trackName = fileName.split('-')[0] 
            print(fileName, trackName)
            stat, counts, totPred, totFiltered = run(trackName, args.images, args.detections, args.tracks, conf)
            totalPredictions += totPred
            totalFilteredPredictions += totFiltered
            imageCounts += counts

            if args.dateFormat == 'YYYY_MM_DD':
                trackNameSplit = trackName.split('_')
                date = int(trackNameSplit[1] + trackNameSplit[2] + trackNameSplit[3])  # format YYYYMMDD
            else: # Assumed 'YYYYMMDD'
                date = int(trackName)  # format YYYYMMDD
             
            print_totals(date, stat, args.tracks)
        
    print("Images", imageCounts, "Predictions", totalPredictions) #, "Filtered", totalFilteredPredictions)
