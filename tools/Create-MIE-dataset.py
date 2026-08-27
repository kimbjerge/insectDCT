# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:38:53 2025

Used to create dataset for detection using YOLO11 based on RGB and MIE images selected from images recorded in Greenhouses
Use camera systems wiht Raspberry Pi and Logitech cameras (Each system with two Logitech Camreas)
Paper: https://doi.org/10.3390/s23167242

@author: Kim Bjerge
"""

import os
import cv2
import shutil
from common.motionEnhancement import MotionEnhancement

# Object Detection of Small Insects image size, Logitech camera HD resolution
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

mie = MotionEnhancement()

def createLabelsAndImages(filePath, camera, filename, pathToDestDataset, pathToDestDatasetMIE):
        
    labelFileName = "GH_" + camera + '-' + filename
    imageFileName = labelFileName.replace('.txt', '.jpg')
    mieImage = mie.motion_enhanced_image2(filePath, filename.replace('.txt', '.jpg'))
    
    if len(mieImage) > 0:
        #print(filename, imageFileName, labelFileName)
        cv2.imwrite(pathToDestDatasetMIE+imageFileName, mieImage)        
        shutil.copyfile(filePath+filename.replace('.txt', '.jpg'), pathToDestDataset+imageFileName)
        shutil.copyfile(filePath+filename, pathToDestDataset+labelFileName)
        shutil.copyfile(filePath+filename, pathToDestDatasetMIE+labelFileName)
    
if __name__=='__main__':
    
    pathToRecordData = 'O:/Tech_Insects/Bees/Dataset/images/'
    pathToDestDatasetMIE = 'J:/Bees/testGreenHm/'
    pathToDestDataset = 'J:/Bees/testGreenH/'
    
    testCameras = ['S1-20220619_0', 'S1-20220724_1', 'S2-20220619_0', 'S2-20220703_1', 'S3-20220717_0', 'S4-20220710_0', 'S4-20220731_1']
                
    firstTime = True
    
    countTotal = 0
    for camera in testCameras:
        filePath = pathToRecordData + camera + '/'
        count = 0
        print("Processing images for", camera)
        for filename in sorted(os.listdir(filePath)):
            if (filename != 'classes.txt') and filename.endswith('.txt'):
                count += 1
                countTotal += 1
                # Select only images where insects has been detected - many are false positive detections
                createLabelsAndImages(filePath, camera, filename, pathToDestDataset, pathToDestDatasetMIE)
        print("Number of files", camera, count)
                            
    print("Total number of files", countTotal)
    
 

    