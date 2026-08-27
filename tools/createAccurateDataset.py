"""
Created on Sat Sep  7 09:32:22 2019

Python script to count number of labels

@author: Kim Bjerge
"""
import os
import shutil

def createValidateDataset(image_dic):
    
    destColorPath = "J:/AccurateDetection/accurateColor/val/"
    destMotionPath = "J:/AccurateDetection/accurateMotion/val/"
    srcMotion = "J:/AccurateDetection/val1201m/"
    
    for filename in os.listdir(image_dic):
        if (filename.endswith('.txt')):

            labelFilename = filename    
            filename = filename.replace('.txt', '.jpg')
            
            srcFilenameColor = image_dic + filename
            dstFilenameColor = destColorPath + 'images/' + filename        
            srcFilenameMotion = srcMotion + filename
            dstFilenameMotion = destMotionPath + 'images/' + filename
            
            srcFilenameLabels = srcMotion + labelFilename
            dstFilenameColorLabels = destColorPath + 'labels/' + labelFilename
            dstFilenameMotionLabels = destMotionPath + 'labels/' + labelFilename
            
            if os.path.exists(srcFilenameLabels): # Does motion label exist
                shutil.copyfile(srcFilenameLabels, dstFilenameColorLabels)
                shutil.copyfile(srcFilenameColor, dstFilenameColor)
                
                shutil.copyfile(srcFilenameLabels, dstFilenameMotionLabels)
                shutil.copyfile(srcFilenameMotion, dstFilenameMotion)
                #print(srcFilenameLabels, dstFilenameColorLabels)
                #print(srcFilenameColor, dstFilenameColor)
                
                #print(srcFilenameLabels, dstFilenameMotionLabels)
                print(srcFilenameMotion, dstFilenameMotion)
            
            
def createDataset(image_dic, filenames):
    
    destColorPath = "J:/AccurateDetection/accurateColor/train/"
    destMotionPath = "J:/AccurateDetection/accurateMotion/train/"
    srcMotion = "J:/AccurateDetection/train1201m/"
    
    for filename in filenames:

        labelFilename = filename    
        filename = filename.replace('.txt', '.jpg')
        
        srcFilenameColor = image_dic + filename
        dstFilenameColor = destColorPath + 'images/' + "AC-" + filename        
        srcFilenameMotion = srcMotion + filename
        dstFilenameMotion = destMotionPath + 'images/' + "AC-" + filename
        
        srcFilenameLabels = srcMotion + labelFilename
        dstFilenameColorLabels = destColorPath + 'labels/' + "AC-" + labelFilename
        dstFilenameMotionLabels = destMotionPath + 'labels/' + "AC-" + labelFilename
        
        if os.path.exists(srcFilenameLabels): # Does motion label exist
            shutil.copyfile(srcFilenameLabels, dstFilenameColorLabels)
            shutil.copyfile(srcFilenameColor, dstFilenameColor)
            
            shutil.copyfile(srcFilenameLabels, dstFilenameMotionLabels)
            shutil.copyfile(srcFilenameMotion, dstFilenameMotion)
            #print(srcFilenameLabels, dstFilenameColorLabels)
            #print(srcFilenameColor, dstFilenameColor)
            
            #print(srcFilenameLabels, dstFilenameMotionLabels)
            print(srcFilenameMotion, dstFilenameMotion)
            


if __name__=='__main__':

    #image_dic = '/home/don/yolov5r/datasets/insects/labels/train1201/'
    #image_dic = '/home/don/yolov5r/datasets/insects/labels/val1201/'
    #image_dic = '/home/don/yolov5r/datasets/insects/labels/test1201/'
    image_dic = 'J:/AccurateDetection/train1201/'
    image_dic_val = 'J:/AccurateDetection/val1201/'
    
    # Coccinellidae septempunctata  0
    # Apis mellifera                1
    # Bombus lapidarius             2
    # Bombus terrestris             3
    # Eupeodes corolla              4
    # Episyrphus balteatus          5
    # Aglais urticae                6   
    # Vespula vulgaris              7
    # Eristalis tenax               8
    
    maxInsects = 400
    numberInsects = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    indexCounts = []
    for i in range(9):
        indexCounts.append(0)
        
    numLadybirds = 0
    numBombusTer = 0
    numEupeodes = 0
    
    backgrounds = 0
    filenames = {}
    for filename in os.listdir(image_dic):
        if(filename.endswith('.txt')):
            file = open(image_dic+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            #print('Lines', lines)
            if lines == 1:
                backgrounds = backgrounds + 1
                filenames[filename] = 1
            joined = '';
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    if index != 1: # Only images with insects other than Honeybees
                        if index == 0:
                            numLadybirds += 1
                            if numLadybirds % 4 != 0:
                                break
                        if index == 3:
                            numBombusTer += 1
                            if numBombusTer % 2 != 0:
                                break
                        if index == 4:
                            numEupeodes += 1
                            if numEupeodes % 2 != 0:
                                break
                        
                        #if index < 9:
                        #   indexCounts[index] +=1
                        if filename in filenames.keys():
                            filenames[filename] += 1
                        else:
                            filenames[filename] = 1

    backgrounds = 0
    for filename in os.listdir(image_dic):
        if (filename.endswith('.txt')) and filename in filenames.keys():
            file = open(image_dic+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            #print('Lines', lines)
            if lines == 1:
                backgrounds = backgrounds + 1
            joined = '';
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    if index < 9:
                       indexCounts[index] +=1
            
    print(indexCounts, ' labels total:', sum(indexCounts))
    print("Background images:", backgrounds)
    # [1629, 588, 1536, 1480, 1839, 1267, 282, 952, 285]  labels total: 9858
    # Background images: 1153
    # Images total: 9959
    #print(filenames.keys())
    print("Images total:", len(filenames.keys()))
  
    createDataset(image_dic, filenames.keys())
    #createValidateDataset(image_dic_val)
            
    
    
 

