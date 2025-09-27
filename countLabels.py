"""
Created on Sat Sep  7 09:32:22 2019

Python script to count number of labels

@author: Kim Bjerge
"""
import os


def countLabels(image_dic, name):

    indexCounts = []
    for i in range(9):
        indexCounts.append(0)
        
    backgrounds = 0
    totalImages = 0
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
                    if index < 9:
                       indexCounts[index] +=1
            
    print(name, indexCounts, ' total:', sum(indexCounts))
    print("Total images:", totalImages, "background:", backgrounds)
    
    
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
    
    image_dic = '/AccurateDetection/trainPollW/'
    countLabels(image_dic, "Pollinator Watch train: ")
    image_dic = '/AccurateDetection/testPollW/'
    countLabels(image_dic, "Pollinator Watch val: ")
    """
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
 

