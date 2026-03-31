# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:03:05 2026

@author: Kim Bjerge
"""
import os
import shutil

# %% Insect plots
if __name__ == '__main__':
    
    cropPath = "D:/PAU/corrected_crops/"
    #dstPath = "D:/PAU/train_crops/"
    dstPath = "D:/PAU/test_crops/"
    
    #totalImages = 0 # train_crops
    totalImages = 2 # test_crops
    
    selectedImages = 0
    skipImages = 5
    for dirName in sorted(os.listdir(cropPath)):
        if dirName != '.DS_Store' and dirName != 'Unsure':
            filesPath = cropPath+dirName
            imageFiles = sorted(os.listdir(filesPath))
            numImages = len(imageFiles) - 1
            if numImages > 0:
                print(dirName, numImages)
                #totalImages += numImages
                
            for imageName in imageFiles:
                if imageName != '.DS_Store':
                    totalImages += 1
                    if totalImages % skipImages == 0:
                        dstImagePath = dstPath + dirName
                        if not os.path.exists(dstImagePath):
                            os.mkdir(dstImagePath)
                            print("Created destination directory", dstImagePath)
                        dstImageFile = dstImagePath + '/' + imageName
                        srcImageFile = filesPath + '/' + imageName
                        print(srcImageFile, dstImageFile)
                        shutil.copy(srcImageFile, dstImageFile)
                        selectedImages += 1
                    
    
    print(selectedImages, totalImages)