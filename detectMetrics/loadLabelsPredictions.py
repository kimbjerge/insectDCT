# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:10:22 2022

Loads YOLO label files and result CSV files
CSV file Format:
system, camera, date, time, confidence, class, x1, y1, x2. y2. path/image.jpg
system: S1-4 or 1-4
camera: 0 or 1
date: YYYYMMDD
time: HMMSS     
confidence : 1-100
class: 1-N
x1, y1, x2, y2: pixel position of bounding box

Eg:
S1,0,20220612,062330,28,1,1646,594,1688,627,S1-20220619_0/20220612062330.jpg
@author: Kim Bjerge
"""
import os
import math
import numpy as np
from PIL import Image

# Functions to get seconds, minutes and hours from time in predictions
def getSeconds(recTime):
    return int(recTime%100)

def getMinutes(recTime):
    minSec = recTime%10000
    return int(minSec/100)

def getHours(recTime):
    return int(recTime/10000)

# Functions to get day, month and year from date in predictions
def getDay(recDate):
    return int(recDate%100)

def getMonthDay(recDate):
    return int(recDate%10000)

def getMonth(recDate):
    return int(getMonthDay(recDate)/100)

def getYear(recDate):
    return int(recDate/10000)


# Substract filterTime in minutes from recTime, do not handle 00:00:00
def substractMinutes(recTime, filterTime):
    
    minute = getMinutes(recTime)
    
    newRecTime = recTime - int(filterTime)*100
    if minute < filterTime: # No space to substract filterTime
        newRecTime = newRecTime - 4000 # Adjust minutes
    
    return newRecTime

# Filter predictions - if the positions are very close and of same class
# Checked within filterTime in minutes (must be a natural number 0,1,2..60)
# It is is assumed that the new prediction belong to the same object
def filter_prediction(lastPredictions, newPrediction, filterTime):
    
    newObject = True
    
    # Filter disabled
    if filterTime == 0:
        return lastPredictions, newObject
    
    # Update last predictions within filterTime window
    timeWindow = substractMinutes(newPrediction['time'], filterTime)
    newLastPredictions = []
    for lastPredict in lastPredictions:
        # Check if newPredition is the same date as last predictions and within time window
        if (lastPredict['date'] == newPrediction['date']) and (lastPredict['time'] > timeWindow):
            newLastPredictions.append(lastPredict)
    
    # Check if new predition is found in last Preditions - nearly same position and class
    for lastPredict in newLastPredictions:
        # Check if new prediction is of same class
        if lastPredict['class'] == newPrediction['class']:
            xlen = lastPredict['xc'] - newPrediction['xc']
            ylen = lastPredict['yc'] - newPrediction['yc']
            # Compute distance between predictions
            dist = math.sqrt(xlen*xlen + ylen*ylen)
            #print(dist)
            if dist < 25: # NB adjusted for no object movement
                # Distance between center of boxes are very close
                # Then we assume it is not a new object
                newObject = False
    
    # Append new prediction to last preditions
    newLastPredictions.append(newPrediction)
    
    return newLastPredictions, newObject
    
# Load prediction CSV file
# filterTime specifies in minutes how long time window used
# to decide if predictions belongs to the same object
# probability threshold for each class, default above 50%
def load_predictions(filename, selection = 'All', filterTime=0, threshold=[50,50,50,50,50,50]):
    
    file = open(filename, 'r')
    content = file.read()
    file.close()
    splitted = content.split('\n')
    lines = len(splitted)
    foundObjects = []
    lastObjects = []
    firstLine = True
    for line in range(lines):
        if firstLine:
            #print(splitted[line]) # Print header of file 
            firstLine = False
            continue
        subsplit = splitted[line].split(',')
        if len(subsplit) == 17: # required 17 data values
            prob = int(subsplit[4])
            objClass = int(subsplit[5])
            # Check selection 
            if prob >= threshold[objClass-1]:
                x1 = int(subsplit[6])
                y1 = int(subsplit[7])
                x2 = int(subsplit[8])
                y2 = int(subsplit[9])

                # Convert points of box to YOLO format: center point and w/h
                width = x2-x1
                height = y2-y1
                xc = x2 - round(width/2)
                if xc < 0: xc = 0
                yc = y2 - round(height/2)
                if yc < 0: yc = 0
                
                imgName = subsplit[10]
                imgNameSplit = subsplit[10].split('/')
                name = imgNameSplit[1].split('.')[0]
                record = {'system': subsplit[0], # Year
                'camera': subsplit[1], # trapDir
                'date' : int(subsplit[2]),
                'time' : int(subsplit[3]),
                'prob' : prob, # Class probability 0-100%
                'class' : objClass+1, # Classes 1-6
                # Box position and size
                'x1' : x1,
                'y1' : y1,
                'x2' : x2,
                'y2' : y2,
                'xc' : xc,
                'yc' : yc,
                'w' : width,
                'h' : height,
                'dic' : imgNameSplit[0],
                'image' : imgNameSplit[1],
                'name' : name,                
                'imagePath' : imgName,
                'label' : 0, # Class label (Unknown = 0)
                'tpfpfn' : ''} 
                
                lastObjects, newObject =  filter_prediction(lastObjects, record, filterTime)
                if newObject:
                    foundObjects.append(record)
            
    return foundObjects

# Finds filename in predictions
def findObjectsInImage(filename, predictions):
    
    found = False
    objectsFound = []
    for object in predictions:
        if object['image'] == filename:
            objectsFound.append(object)
            found = True
    
    return found, objectsFound


def getImageSize(imageFileName):
    
    im = Image.open(imageFileName)
    (width,height) = im.size # (width,height) tuple
    
    print(imageFileName, width, height)
    return width, height
    
# Load labled images txt files with number of classes    
def load_labels(dirname, classes=6):
    
    labelCounts = []
    for i in range(classes):
        labelCounts.append(0)
    
    dirnameSplit = dirname.split('/')
    dic = dirnameSplit[len(dirnameSplit)-2]
    labledObjects = []
    for filename in os.listdir(dirname):
        if filename == "classes.txt":
            continue
        if (filename.endswith('.txt')):
            file = open(dirname+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            
            if lines > 0:
                imageFilePath = dirname.replace('labels', 'images') + filename.replace('txt', 'jpg')
                width, height = getImageSize(imageFilePath)
                
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    if index < classes:
                        imagename = filename.split('.')
                        xc = float(subsplit[1])*width;
                        yc = float(subsplit[2])*height;
                        w = float(subsplit[3])*width;
                        h = float(subsplit[4])*height;
                        x1 = xc - w/2;
                        y1 = yc - h/2;
                        x2 = xc + w/2;
                        y2 = yc + h/2;
                        record = {
                            'prob' : -1,
                            'class' : int(subsplit[0])+1,
                            'xc' : int(round(xc)),
                            'yc' : int(round(yc)),
                            'w' : int(round(w)),
                            'h' : int(round(h)),
                            'x1' : int(round(x1)),
                            'y1' : int(round(y1)),
                            'x2' : int(round(x2)),
                            'y2' : int(round(y2)),
                            'dic' : dic,
                            'image' : imagename[0]+'.jpg',
                            'name' : imagename[0],
                            'found' : False,
                            'tpfpfn' : '', # TP, FP, FN
                            'width' : width, 
                            'height' : height} 
                        labledObjects.append(record)
                        labelCounts[index] +=1
                    
    return labledObjects, labelCounts

# This function takes the predicted bounding box and ground truth bounding box and 
# return the IoU ratio
def calc_iou( gt_bbox, pred_bbox):

    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)   
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox 
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0

    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

# Check if label exist for predicted object
def exist_label(predObject, labels, iou_used=True, iou_thr=0.2):
    
    found = False
    for idx in range(len(labels)):
        if labels[idx]["image"] == predObject["image"]:
            if labels[idx]["found"] == False:
                if iou_used == False:
                    xlen = labels[idx].get("xc") - predObject["xc"]
                    ylen = labels[idx].get("yc") - predObject["yc"]
                    dist = math.sqrt(xlen*xlen + ylen*ylen)
                    #print(dist)
                    if dist < 250: # Distance between center of boxes
                        labels[idx]["found"] = True
                        found = True
                        break
                else: # Using IoU
                    gt_bbox = [labels[idx]["x1"], labels[idx]["y1"], labels[idx]["x2"], labels[idx]["y2"]]
                    pred_bbox = [predObject["x1"], predObject["y1"], predObject["x2"], predObject["y2"]]
                    IoU = calc_iou(gt_bbox, pred_bbox)
                    if IoU >= iou_thr:
                        labels[idx]["found"] = True
                        found = True
                        break
            
    return found, labels[idx]

