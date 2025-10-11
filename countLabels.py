"""
Created on Sat Sep  7 09:32:22 2019

Python script to count number of labels

@author: Kim Bjerge
"""
import os
import numpy as np

IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def countLabels(image_dic, name):

    indexCounts = []
    for i in range(9):
        indexCounts.append(0)
        
    backgrounds = 0
    totalImages = 0
    boxesWidth = []
    boxesHeight = []
    boxesArea = []
    for filename in os.listdir(image_dic):
        if (filename.endswith('.txt')) and filename != "classes.txt":
            file = open(image_dic+filename, 'r')
            content = file.read()
            file.close()
            splitted = content.split('\n')
            lines = len(splitted)
            totalImages += 1
            #print('Lines', lines)
            if lines == 1:
                backgrounds = backgrounds + 1
            joined = '';
            for line in range(lines):
                subsplit = splitted[line].split(' ')
                if len(subsplit) == 5: # required: index x y w h
                    index = int(subsplit[0])
                    xc = float(subsplit[1])*IMG_WIDTH
                    yc = float(subsplit[2])*IMG_HEIGHT
                    w = float(subsplit[3])*IMG_WIDTH
                    h = float(subsplit[4])*IMG_HEIGHT
                    if index < 9:
                       indexCounts[index] +=1
                    boxesWidth.append(w)
                    boxesHeight.append(h)
                    boxesArea.append(w*h)
            
    print(name, indexCounts, ' total:', sum(indexCounts))
    print("Total images:", totalImages, "background:", backgrounds)
    print("Average labels width:", np.mean(boxesWidth), "STD:", np.std(boxesWidth))
    print("Average labels height:", np.mean(boxesHeight), "STD:", np.std(boxesHeight))
    print("Average labels area:", np.mean(boxesArea), "STD:", np.std(boxesArea))
    
    return boxesWidth, boxesHeight
    
    
if __name__=='__main__':

    
    # Coccinellidae septempunctata  0 or insect
    # Apis mellifera                1
    # Bombus lapidarius             2
    # Bombus terrestris             3
    # Eupeodes corolla              4
    # Episyrphus balteatus          5
    # Aglais urticae                6   
    # Vespula vulgaris              7
    # Eristalis tenax               8

    #image_dic = '/home/don/yolov5r/datasets/insects/labels/train1201/'
    #image_dic = '/home/don/yolov5r/datasets/insects/labels/val1201/'
    boxesWidth = []
    boxesHeight = []
    
    image_path = 'D:/insectsDCT_datasets/detectTestDataset/'
    image_dic = image_path + 'testGreenH/'
    boxW, boxH = countLabels(image_dic, "Green house Logitech test: ")
    boxesWidth += boxW
    boxesHeight += boxH
    
    image_dic = image_path + 'testHeather/'
    boxW, boxH = countLabels(image_dic, "KU Heather Wingscapes test: ")
    boxesWidth += boxW
    boxesHeight += boxH

    image_dic = image_path + 'testMBO/'
    boxW, boxH = countLabels(image_dic, "MAMBO Wingscapes test: ")
    boxesWidth += boxW
    boxesHeight += boxH

    image_dic = image_path + 'testOrchard/'
    boxW, boxH = countLabels(image_dic, "Orchard Pi3Cam test: ")
    boxesWidth += boxW
    boxesHeight += boxH

    image_dic = image_path + 'testPollW/'
    boxW, boxH = countLabels(image_dic, "Pollinator Watch Logitech test: ")
    boxesWidth += boxW
    boxesHeight += boxH

    image_dic = image_path + 'testRTNI/'
    boxW, boxH = countLabels(image_dic, "Real-time Nature Impact Logitech test: ")
    boxesWidth += boxW
    boxesHeight += boxH
    
    print("Average all labels width:", np.mean(boxesWidth), "STD:", np.std(boxesWidth))
    print("Average all labels height:", np.mean(boxesHeight), "STD:", np.std(boxesHeight))
 
    """
    image_dic = 'D:/MAMBO/trainMBO/'
    countLabels(image_dic, "MAMBO train: ")
    image_dic = 'D:/MAMBO/testMBO/'
    countLabels(image_dic, "MAMBO val: ")
    
    image_dic = '/AccurateDetection/trainOrchard/'
    countLabels(image_dic, "UFZ Orchard train: ")
    image_dic = '/AccurateDetection/testOrchard/'
    countLabels(image_dic, "UFZ Orchard val: ")
    
    image_dic = '/AccurateDetection/trainPollW/'
    countLabels(image_dic, "Pollinator Watch train: ")
    image_dic = '/AccurateDetection/testPollW/'
    countLabels(image_dic, "Pollinator Watch val: ")

    image_dic = '/home/don/yolo11/datasets/accurateColor/train/labels/'
    countLabels(image_dic, "Accurate train: ")
    image_dic = '/home/don/yolo11/datasets/beesColor/trainColor/'
    countLabels(image_dic, "Bees train: ")
    image_dic = '/home/don/yolov5r/datasets/insects/labels/trainI21/'
    countLabels(image_dic, "Arthropods train: ")
    image_dic = '/home/don/yolo11/datasets/insects2Color/train/labels/'
    countLabels(image_dic, "Insects2 train: ")
    image_dic = '/home/don/yolo11/datasets/insects3Color/train/labels/'
    countLabels(image_dic, "Insects3 train: ")
    
    image_dic = '/home/don/yolo11/datasets/accurateColor/val/labels/'
    countLabels(image_dic, "Accurate val: ")
    image_dic = '/home/don/yolo11/datasets/beesColor/valColor/'
    countLabels(image_dic, "Bees val: ")
    image_dic = '/home/don/yolov5r/datasets/insects/labels/testI21/'
    countLabels(image_dic, "Arthropods val: ")
    image_dic = '/home/don/yolo11/datasets/insects2Color/val/labels/'
    countLabels(image_dic, "Insects2 val: ")
    image_dic = '/home/don/yolo11/datasets/insects3Color/val/labels/'
    countLabels(image_dic, "Insects3 val: ")
    """
 

