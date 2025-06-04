# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:10:16 2020

@author: Kim Bjerge
         Aarhus University
"""
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
from PyQt5.QtGui import QImage
import pickle

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
def run(dirName):
    config_filename = './config/ITC_config.json'
    conf = readconfig(config_filename)
    conf['datareader']['datapath']
    print(conf['datareader']['datapath'])
    # Convert hierarchical labels to flat list of labels
    conf["classifier"]['species'] = createFlatSpeciesList(conf["classifier"]["labelFile"])
    print(conf['moviemaker']['resultdir'])
    writemovie = conf['moviemaker']['writemovie']
    reader = DataReader(conf)
    gen = reader.getimage()
    print(type(gen))

    tr = Tracker(conf)
    imod = Imagemod()
    if dirName == '':
        dirName = 'tracks'
    mm = MovieMaker(conf, name=dirName + 'TR.avi')

    stat = Stats(conf)
    predict = Predictions(conf)
    tracksFilename = conf['moviemaker']['resultdir']+'/'+dirName+'TRS.csv'
    print(tracksFilename)
    tracks = TracksSave(tracksFilename)

    csvFilename = './detections/'+dirName+'-CL.csv'

    predicted = predict.load_predictionsTaxon(csvFilename, filterTime=0) # Skip if not moved within 5 minutes
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
        time1 = time.time()
        
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
                file_name = conf['datareader']['datapath'] + '/' + filepath
                im = io.imread(file_name)
                image = imod.drawoois(im, goods)
                height, width, channel = image.shape
                bytesPerLine = 3 * width
                qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        
                # Write frame
                mm.writeframe(image, filedatetime)

            time2 = time.time()
            #print('Processing image took {:.3f} ms'.format((time2 - time1) * 1000.0))

    if writemovie:
        mm.releasemovie()
        
    tracks.close()
    resultdir = conf['moviemaker']['resultdir'] + '/'
    stat.writedetails(resultdir + dirName + 'TR')

    return stat, resultdir, iterCount, totPredictions, totFilteredPredictions

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

    print('STARTING NOW. Please wait.....')

    dirNames = [ 
                'pi1_2025_02_21'
                ]
    
    imageCounts = 0
    totalPredictions = 0
    totalFilteredPredictions = 0
    for dirName in dirNames:
        print(dirName)
        stat, resultdir, counts, totPred, totFiltered = run(dirName)
        totalPredictions += totPred
        totalFilteredPredictions += totFiltered
        imageCounts += counts
        if dirName == '':
            date = 20250221
        else:    
            dirNameSplit = dirName.split('_')
            date = int(dirNameSplit[1] + dirNameSplit[2] + dirNameSplit[3])  # format YYYYMMDD
        print_totals(date, stat, resultdir)
        
    print("Images", imageCounts, "Predictions", totalPredictions) #, "Filtered", totalFilteredPredictions)
