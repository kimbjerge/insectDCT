# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:41:20 2025

@author: Kim Bjerge
"""

import os
import cv2
import datetime

class MotionEnhancement:

    def __init__(self, motion='none'):
        
        self.imgPrevGray = None
        self.imgPrevDiff = None
        self.imgPrev = None
        self.motionMethod = motion
        
    def motion_image(self, im, method='RGB'): #'RGB' or 'HSV'

        # Convert to gray scale blur image
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)
        if self.imgPrevGray is None:
            self.imgPrevGray = img_gray.copy()
        imgDiff = cv2.absdiff(self.imgPrevGray, img_gray) #** 2
        if self.imgPrevDiff is None:
            self.imgPrevDiff = imgDiff.copy()
        imgSumDiff = cv2.add(self.imgPrevDiff, imgDiff)
        if self.imgPrev is None:
            self.imgPrev = im.copy()
    
        if method == 'RGB':
            # Create motion image 
            imgMotion = self.imgPrev.copy() 
            imgMotion[:,:,0] = self.imgPrev[:,:,0]/2 + self.imgPrev[:,:,2]/2
            imgMotion[:,:,2] = imgSumDiff
    
        if method == 'HSV':
            # HSV replace channel saturation with motion information
            imgHSV = cv2.cvtColor(self.imgPrev, cv2.COLOR_BGR2HSV)
            value = imgHSV[:,:,1]
            alfa = 0.95
            imgHSV[:,:,1] = cv2.add(alfa*imgSumDiff, (1-alfa)*value) 
            imgMotion = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
    
        # Copy history
        self.imgPrevGray = img_gray.copy()
        self.imgPrevDiff = imgDiff.copy()
        imgPrev = self.imgPrev
        self.imgPrev = im.copy()
    
        return imgMotion, imgPrev
    
    def motion_three_images(self, path, prevFilename, filename, nextFilename):
        
        img_color_prev = cv2.imread(path+prevFilename)
        img_gray_prev = cv2.cvtColor(img_color_prev, cv2.COLOR_BGR2GRAY)
        img_gray_prev = cv2.GaussianBlur(img_gray_prev, ksize=(5,5), sigmaX=0)
 
        img_color = cv2.imread(path+filename)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, ksize=(5,5), sigmaX=0)

        img_color_next = cv2.imread(path+nextFilename)
        img_gray_next = cv2.cvtColor(img_color_next, cv2.COLOR_BGR2GRAY)
        img_gray_next = cv2.GaussianBlur(img_gray_next, ksize=(5,5), sigmaX=0)
        
        imgDiffPrev = cv2.absdiff(img_gray_prev, img_gray) #** 2
        imgDiffNext = cv2.absdiff(img_gray, img_gray_next) #** 2
        imgSumDiff = cv2.add(imgDiffPrev, imgDiffNext)

        imgMotion = img_color.copy() 
        imgMotion[:,:,0] = img_color[:,:,0]/2 + img_color[:,:,2]/2 #BGR
        imgMotion[:,:,2] = imgSumDiff
        
        return imgMotion
    
    # Filename format: SSS-YYYYMMDDHHSS-snapshot.jpg
    def motion_enhanced_image(self, path, filename, time_lapse_seconds=30):
        
        print(path)
        filenameSplit = filename.split('-')
        dateTime = filenameSplit[1]
        curr_time = datetime.datetime.strptime(dateTime[0:14], "%Y%m%d%H%M%S")
        prev_time = curr_time - datetime.timedelta(seconds=time_lapse_seconds)
        prevFilename = filenameSplit[0] + '-' + prev_time.strftime("%Y%m%d%H%M%S") + '-snapshot.jpg'
        next_time = curr_time + datetime.timedelta(seconds=time_lapse_seconds)
        nextFilename = filenameSplit[0] + '-' + next_time.strftime("%Y%m%d%H%M%S") + '-snapshot.jpg'
        currFilename = filename + '-snapshot.jpg'
        print(prevFilename)
        print(currFilename)
        print(nextFilename)
        
        if os.path.exists(path + currFilename) and os.path.exists(path + prevFilename) and os.path.exists(path + nextFilename):
            img_motion = self.motion_three_images(path, prevFilename,  currFilename, nextFilename)
            return img_motion
        
        return []
    
    # Filename format: YYYYMMDDHHSS.jpg
    def motion_enhanced_image2(self, path, filename, time_lapse_seconds=30):
        
        print(path)
        filenameSplit = filename.split('.')
        dateTime = filenameSplit[0]
        curr_time = datetime.datetime.strptime(dateTime[0:14], "%Y%m%d%H%M%S")
        prev_time = curr_time - datetime.timedelta(seconds=time_lapse_seconds)
        prevFilename = prev_time.strftime("%Y%m%d%H%M%S") + '.jpg'
        next_time = curr_time + datetime.timedelta(seconds=time_lapse_seconds)
        nextFilename = next_time.strftime("%Y%m%d%H%M%S") + '.jpg'
        currFilename = filename
        print(prevFilename)
        print(currFilename)
        print(nextFilename)
        
        if os.path.exists(path + currFilename) and os.path.exists(path + prevFilename) and os.path.exists(path + nextFilename):
            img_motion = self.motion_three_images(path, prevFilename,  currFilename, nextFilename)
            return img_motion
        
        return []