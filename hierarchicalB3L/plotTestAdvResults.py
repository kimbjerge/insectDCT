# -*- coding: utf-8 -*-
"""
Modified on Wen November 05 13:19:03 2025

@author: Kim Bjerge
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
from hierarchical_loss import HierarchicalLossNetwork

graph_folder = './graph_folder/'
saved_folder = "./models_saved/saved_128_ConvNextV6_3_apoidae2L/"

thresholdSTD = 0.0

label_file = saved_folder+"labelsAdv3L.pkl"
# Load labels shared by several functions
with open(label_file, 'rb') as f:
    image_path_list, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, trainL1, trainL2, trainL3, valL1, valL2, valL3 = pickle.load(f)
    print("Labels and hierarchy dependency loaded from ", label_file)
    print("L1", labelsL1)
    print("L2", labelsL2)
    print("L3", labelsL3)
    
def renameUnknownLabels(labelsL, level_label):
    
    unknownIdx = len(labelsL)
    for idx in range(len(level_label)):
        if level_label[idx] == -1:
            level_label[idx] = unknownIdx
            
    labelsL = labelsL + ['Unsure']
    
    return labelsL, level_label


# Class to handle loading thresholds and marking unsure predictions
class Thresholds:

    def __init__(self, threshold_file, thresholdSTD=0):
        self.thresholdSTD = thresholdSTD
        self.loadThresholds(threshold_file)
    
    def loadThresholds(self, threshold_file):
        
        data_thresholds = pd.read_csv(threshold_file)
        self.levels = data_thresholds["Level"].to_list() 
        self.labels = data_thresholds["ClassName"].to_list()
        self.thresholds = data_thresholds["Threshold"].to_list()
        self.means = data_thresholds["Mean"].to_list()
        self.stds = data_thresholds["Std"].to_list()
        
        self.level1Thresholds = []
        self.level2Thresholds = []
        self.level3Thresholds = []
        for idx in range(len(self.levels)):
                       
            if self.thresholdSTD == 0:
                threshold = self.thresholds[idx]
            else:
                threshold = self.means[idx] - self.thresholdSTD*self.stds[idx]
            
            if self.levels[idx] == 1:
                self.level1Thresholds.append(threshold)
            if self.levels[idx] == 2:
                self.level2Thresholds.append(threshold)
            if self.levels[idx] == 3:
                self.level3Thresholds.append(threshold)
                
    def unsurePredictions(self, level, labels, level_pred):
        
        if level == 1:
            levelThredsholds = self.level1Thresholds
        if level == 2:
            levelThredsholds = self.level2Thresholds
        if level == 3:
            levelThredsholds = self.level3Thresholds
        
        unknownIdx = len(labels)-1
        level_p = np.argmax(level_pred, axis=1)
        max_p = np.max(level_pred, axis=1)
        for idx in range(len(level_p)):
            #if max_p[idx] < 3.8: # KBE?? level to be found based on training dataset 3.8 - 9.1
            if max_p[idx] < levelThredsholds[level_p[idx]]:
                level_p[idx] = unknownIdx
        
        return level_p
    

def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, thredsholds, normalize=False, font_size=14):

    matrixAll = np.zeros((len(labels), len(labels))).astype('int')

    for i in range(len(level_predict)):
        matrixAll[level_label[i], level_predict[i]] += 1
        
    matrixSumTrue = matrixAll.sum(axis=1)[:, np.newaxis]
    matrixSumPredict = matrixAll.sum(axis=0)[:, np.newaxis] 
    
    labelsTrue = []
    labelsPredict = []
    for i in range(len(labels)-1): # Skip unsure
        #if matrixSumTrue[i] > 0:
        labelsTrue.append(labels[i])
    for i in range(len(labels)):
        #if matrixSumPredict[i] > 0:
        labelsPredict.append(labels[i])
            
    matrix = np.zeros((len(labelsTrue), len(labelsPredict))).astype('int') 
    x = 0;
    for i in range(len(labelsTrue)):
        y = 0;
        #if matrixSumTrue[i] > 0:
        for j in range(len(labelsPredict)):
            #if matrixSumPredict[j] > 0:
            matrix[x, y] = matrixAll[i, j] 
            y += 1               
        x += 1
    
    if normalize:
        matrixSum = matrix.sum(axis=1)[:, np.newaxis]
        #matrix = matrix.astype('float') / (matrixSum+0.001)
        matrix = np.round(100 * (matrix.astype('float') / (matrixSum+0.001)))
        matrix = matrix.astype('int')
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix, matrixSum)        
    
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.imshow(matrix, cmap='Greens')
    
    yLabels = []
    for idx in range(len(labelsTrue)):
        yLabels.append(labelsTrue[idx] + ' (' + str(int(matrixSum[idx,0])) + ')')
    
    ax.set(xticks=np.arange(len(labelsPredict)),
           yticks=np.arange(len(yLabels)),
           # ... and label them with the respective list entries
           xticklabels=labelsPredict, yticklabels=yLabels,
           title=levelName,
           ylabel='True label',
           xlabel='Predicted label')
    
   
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    #fmt = '.2f' if normalize else 'd'
    fmt = 'd'
    for i in range(len(labelsTrue)):
        for j in range(len(labelsPredict)):
            color = 'k'
            if matrix[i, j] > 69:
            #if matrix[i, j] > 0.495:
                color = 'w'
            if matrix[i,j] > 0:
            #if matrix[i,j] > 0.005:
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=color)
    
    #ax.set_title(levelName)
    fig.tight_layout()
    plt.savefig(graph_folder+levelName +'ConfTest.png')
    plt.show()   
    
def plotLevelConfusion(level, level_pred, level_label, labels, level_name, thredsholds):

    #level_p = np.argmax(level_pred, axis=1)
    level_p = thredsholds.unsurePredictions(level, labels, level_pred)
    
    font_size = 28
    if level == 2:
        font_size = 20        
    if level == 3:
        font_size = 14
    plotConfusionMatrixLevel('L' + str(level) + ' ' + level_name, level_p, level_label, labels, thresholds, normalize=True, font_size=font_size)
       
    precision = metrics.precision_score(level_label, level_p, average='macro')
    print("Level %d (macro) precision %.4f" % (level, precision))
    recall = metrics.recall_score(level_label, level_p, average='macro')
    print("Level %d (macro) recall    %.4f" % (level, recall))
    f1score = metrics.f1_score(level_label, level_p, average='macro')
    print("Level %d (macro) f1-score  %.4f" % (level, f1score))
    
    precision = metrics.precision_score(level_label, level_p, average='micro')
    print("Level %d (micro) precision %.4f" % (level, precision))
    recall = metrics.recall_score(level_label, level_p, average='micro')
    print("Level %d (micro) recall    %.4f" % (level, recall))
    f1score = metrics.f1_score(level_label, level_p, average='micro')
    print("Level %d (micro) f1-score  %.4f" % (level, f1score))
    
    
def plotConfusionMatrix(resultFile, thresholds):

    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
    
    labelsrL1, level1_label = renameUnknownLabels(labelsL1, level1_label)
    labelsrL2, level2_label = renameUnknownLabels(labelsL2, level2_label)
    labelsrL3, level3_label = renameUnknownLabels(labelsL3, level3_label)
    plotLevelConfusion(1, level1_pred, level1_label, labelsrL1, "Order", thresholds)
    plotLevelConfusion(2, level2_pred, level2_label, labelsrL2, "Family", thresholds)
    plotLevelConfusion(3, level3_pred, level3_label, labelsrL3, "Species", thresholds)
    
    level1p = np.argmax(level1_pred, axis=1)
    level2p = np.argmax(level2_pred, axis=1)
    level3p = np.argmax(level3_pred, axis=1)
    level3False = 0
    for idx in range(len(level3p)):
        if level3p[idx] != level3_label[idx]:
            #print(labelsrL1[level1p[idx]], labelsrL2[level2p[idx]], labelsrL3[level3p[idx]])
            level3False += 1
    
    return level3False
            
    
def checkHierarcy(resultFile):    
    
    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
        
    HLN = HierarchicalLossNetwork(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, 
                                  [None, None, None], 
                                  total_level=3, device="cpu", simple=True) 
        
    level1p = np.argmax(level1_pred, axis=1)
    level2p = np.argmax(level2_pred, axis=1)
    level3p = np.argmax(level3_pred, axis=1)
    checkL2 = HLN.check_hierarchy_list(level=1, current_level=level2p, previous_level=level1p)
    checkL3 = HLN.check_hierarchy_list(level=2, current_level=level3p, previous_level=level2p)
    
    checkList = checkL3 
    
    for idx in range(len(checkList)):
        if checkL3[idx] == False or checkL2[idx] == False:
            checkList[idx] = False
            #print(labelsL1[level1p[idx]], labelsL2[level2p[idx]], labelsL3[level3p[idx]])
    
    return checkList    

#%% MAIN
if __name__=='__main__':
        
    thredsholdFile = saved_folder + "thresholds.csv"
    #resultFile = saved_folder + "predictLabels3Lval.pkl"
    resultFile = saved_folder + "predictLabels3Ltest.pkl"

    thresholds = Thresholds(thredsholdFile, thresholdSTD=thresholdSTD)
    level3False = plotConfusionMatrix(resultFile, thresholds)
    checkList = checkHierarcy(resultFile)
    
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
    
