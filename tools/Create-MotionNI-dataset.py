# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:54:00 2025

@author: Kim Bjerge
"""

import os
import cv2
import datetime
import shutil

showImage = False
saveImage = True
time_lapse_seconds = 30

# Train background images
image_dic_labels = ['E:/NatureImpact/train0709/', #10 Labels and background 
                    'E:/NatureRoof/S5_187/Juli08_0/', #11 Background images only
                    'E:/NatureRoof/S1_146/Aug09_0/', #12 Background images only
                    'E:/NatureRoof/S4_199/Juli15_0/', #13 Background images only
                    'E:/NatureRoof/S1_146/Sep06_0/', #14 Labels and background
                    'E:/NatureRoof/S5_187/Aug09_0/', #15 Background images only
                    'E:/NatureRoof/S5_187/Juli08_1/', #16 Background images only
                    'E:/NatureRoof/S1_146/Aug02_1/', #17 Background images only
                    'E:/NatureRoof/S1_146/Juli15_0/', #18 Labels and background
                    'E:/NatureRoof/S4_199/Aug30_1/', #19 Background images only                        
                    'E:/NatureRoof/S5_187/Juli15_0/', #20 Background images only
                    'E:/NatureRoof/S4_199/Aug16_1/', #21 Background images only
                    'E:/NatureRoof/S1_146/Aug23_0/', #22 Labels and background
                    'E:/NatureRoof/S4_199/Juni1/', #23 Background images only
                    'E:/NatureRoof/S5_187/Aug16_1/', #24 Background images only
                    'E:/NatureRoof/S4_199/Aug02_1/', #25 Background images only
                    'E:/NatureRoof/S1_146/Juli25_1/', #26 Labels and background
                    'E:/NatureRoof/S5_187/Aug23_0/', #27 Background images only
                    'E:/NatureRoof/S1_146/Juli08_0/', #28 Background images only
                    'E:/NatureRoof/S4_199/Aug16_0/', #29 Background images only
                    'E:/NatureRoof/S1_146/Aug23_1/', #30 Labels and background
                    'E:/NatureRoof/S4_199/Juli08_0/', #31 Background images only
                    'E:/NatureRoof/S1_146/Sep06_1/', #32 Background images only
                    'E:/NatureRoof/S4_199/Aug23_0/', #33 Background images only
                    'E:/NatureRoof/S1_146/Juli25_0/', #34 Background images only
                    'E:/NatureRoof/S1_146/Juli15_1/', #35 Labels and background
                    'E:/NatureRoof/S5_187/Aug23_1/', #36 Background images only
                    'E:/NatureRoof/S1_146/Juni0/', #37 Background images only
                    'E:/NatureRoof/S1_146/Aug02_0/'] #38 Background images only

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
        self.imgPrev = im.copy()
    
        return imgMotion
    
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
    
    def motion_enhanced_image(self, path, filename):
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

def copyLabelFile(srcFilename, dstFilename):
    
    srcFile = open(srcFilename, 'r')
    dstFile = open(dstFilename, 'w')
    srcText = srcFile.readlines()
    
    for line in srcText:
        dstLine = '0' + line[1:] # Change label to "insect" (0)
        #print(dstLine)
        dstFile.write(dstLine)
        
    dstFile.close()
    srcFile.close()
 
        
if __name__=='__main__':
    
    #pathToSrcDataset = '/home/don/yolov5r/datasets/insects/labels/train1201/'
    #pathToRecordData = '/mnt/Dfs/Tech_Insects/NatureRoof/'
    #pathToDestDataset = '/home/don/yolo11/datasets/train1201m/'

    #pathToSrcDataset = 'O:/Tech_TTH-KBE/NI/Dataset/train1201/'
    pathToSrcDataset = 'J:/AccurateDetection/train1201/'
    pathToRecordData = 'E:/NatureRoof/'
    #pathToDestDataset = 'O:/Tech_Insects/NatureRoof/datasets/train1201m/'
    pathToDestDataset = 'J:/AccurateDetection/train1201m/'
    
    MIE = MotionEnhancement()
    
    # File format: S2_123-Aug09_1_88-20190808104930.jpg
    for filename in os.listdir(pathToSrcDataset):
        if(filename.endswith('.jpg')):
            if filename[0] == 'S':
                filenameSplit = filename.split('-')
                system = filenameSplit[0]
                monthCameraSplit = filenameSplit[1].split('_')
                dateTime = filenameSplit[2].split('.')[0]
                pathToRecordedFile = pathToRecordData + system + '/' + monthCameraSplit[0] + '_' + monthCameraSplit[1] + '/'
                if len(monthCameraSplit) == 3:
                    srcFileName = monthCameraSplit[2] + '-' + dateTime
                else:
                    srcFileName = monthCameraSplit[1] + '-' + dateTime
                image = MIE.motion_enhanced_image(pathToRecordedFile, srcFileName)
                if len(image) > 0:
                    
                    if saveImage:
                        print(pathToDestDataset + filename)
                        cv2.imwrite(pathToDestDataset + filename, image)
                        labelSrcFilename = filename.replace('.jpg', '.txt')
                        copyLabelFile(pathToSrcDataset + labelSrcFilename, pathToDestDataset + labelSrcFilename)
                    
                    if showImage: 
                        cv2.imshow("MIE image", image)
                        # waits for user to press any key
                        # (this is necessary to avoid Python kernel form crashing)
                        cv2.waitKey(0)                        
                        # closing all open windows
                        cv2.destroyAllWindows()
                        
            # File format: 31-55-20190707092300-snapshot.jpg
            else: # Background images numbered 10 - 38, using index to image_dic_labels
                filenameSplit = filename.split('-') # '_' val
                index = int(filenameSplit[0])
                if index > 10 and index < 39:
                    index = index - 10
                    pathToRecordedFile = image_dic_labels[index]
                    #srcFileNameSplit = filenameSplit[1].split('-') # val
                    #srcFileName = srcFileNameSplit[0] + '-' + srcFileNameSplit[1] 
                    srcFileName = filenameSplit[1] + '-' + filenameSplit[2] 
                    image = MIE.motion_enhanced_image(pathToRecordedFile, srcFileName)
                    if len(image) > 0:
                        if saveImage:
                            print(pathToDestDataset + filename)
                            cv2.imwrite(pathToDestDataset + filename, image)
                            labelSrcFilename = filename.replace('.jpg', '.txt')
                            copyLabelFile(pathToSrcDataset + labelSrcFilename, pathToDestDataset + labelSrcFilename)
            
            
            
            