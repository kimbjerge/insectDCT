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
#saved_folder = "./models_saved/saved_128_ConvNextV6_3_apoidae2L/"
saved_folder = "./models_saved/saved_128_ConvNextV6/"

thresholdSTD = 0.0

label_file = saved_folder+"labelsAdv3L.pkl"
# Load labels shared by several functions
with open(label_file, 'rb') as f:
    image_path_list, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, trainL1, trainL2, trainL3, valL1, valL2, valL3 = pickle.load(f)
    print("Labels and hierarchy dependency loaded from ", label_file)
    #print("L1", labelsL1)
    #print("L2", labelsL2)
    #print("L3", labelsL3)
    
def renameUnsureLabels(labelsL, level_label):
    
    unsureIdx = len(labelsL)
    for idx in range(len(level_label)):
        if level_label[idx] == -1:
            level_label[idx] = unsureIdx
            
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
        
        unsurePredictions = 0
        unknownIdx = len(labels)-1
        level_p = np.argmax(level_pred, axis=1)
        max_p = np.max(level_pred, axis=1)
        for idx in range(len(level_p)):
            #if max_p[idx] < 3.8: # KBE?? level to be found based on training dataset 3.8 - 9.1
            if max_p[idx] < levelThredsholds[level_p[idx]]:
                level_p[idx] = unknownIdx
                unsurePredictions += 1
        
        return level_p, unsurePredictions

# Computes macro and micro recall, precision and f1scores for each class in the confusion matrix
def computeClassScores(level, level_predict, level_label, labels, thresholds, clearUnsure=True):
    
    total_predictions = len(level_predict)
    level_predict, unsurePredictions = thresholds.unsurePredictions(level, labels, level_predict)
    
    confMatrix = np.zeros((len(labels), len(labels))).astype('int')
    
    unsureIdx = len(labels)-1 # Unsure last entry in matrix
    for i in range(len(level_predict)):
        addValue = 1
        if clearUnsure and level_predict[i] == unsureIdx:
            addValue = 0   
        confMatrix[level_label[i], level_predict[i]] += addValue
            
    matrixSumTrue = confMatrix.sum(axis=1)[:] # Number of labeled samples for each class
    matrixSumPredicted = confMatrix.T.sum(axis=1)[:] # Number of predicted samples for each class
    
    recalls = np.zeros(len(labels))
    precisions = np.zeros(len(labels))
    f1scores = np.zeros(len(labels))
    matrixSumTP = np.zeros(len(labels)).astype('int')
       
    for idx in range(len((labels))):
        matrixSumTP[idx] = confMatrix[idx][idx]
        
        if matrixSumTrue[idx] == 0: # No labeled samples for class
            recalls[idx] = 1.0 # KBE??? not done in standard computations
        else:
            recalls[idx] = matrixSumTP[idx] /matrixSumTrue[idx]
            
        if matrixSumPredicted[idx] == 0: # No predicitons for class
            if matrixSumTrue[idx] == 0: #  No labeled samples for class
                precisions[idx] = 1.0 # Found all non existing labels
            else:
                precisions[idx] = 0.0 # Predicted samples for class without labeled samples
        else:
            precisions[idx] = matrixSumTP[idx] /matrixSumPredicted[idx]
        
        if (recalls[idx] + precisions[idx]) == 0:
            if  matrixSumTrue[idx] == 0: # KBE??? not done in standard computations
                f1scores[idx] = 1.0 # No predictions or labels for class
            else:
                f1scores[idx] = 0.0
        else:
            f1scores[idx] = 2*recalls[idx]*precisions[idx]/(recalls[idx] + precisions[idx])
            
    macroRecall = np.mean(recalls)
    macroPrecision = np.mean(precisions)
    macroF1score = np.mean(f1scores)

    microRecall = sum(matrixSumTP)/sum(matrixSumTrue)
    microPrecision = sum(matrixSumTP)/sum(matrixSumPredicted)
    microF1score =  2*microRecall*microPrecision/(microRecall + microPrecision)
    
    percentageUnsure = (unsurePredictions/total_predictions)*100
    print(f"Number of unsure predictions {unsurePredictions} {percentageUnsure:.2f}%")
    print(f"Performance level L{level}:")
    print(f"            Recall {macroRecall:.3f} Precision {macroPrecision:.3f}")
    print(f"            F1-score (Macro) {macroF1score:.3f} F1-score (Micro) {microF1score:.3f}")
    
    #return macroRecall, macroPrecision, macroF1score, microF1score, unsurePredictions, percentageUnsure      
    return macroF1score, microF1score, unsurePredictions, percentageUnsure      

def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, thredsholds, normalize=False, font_size=14):

    matrixAll = np.zeros((len(labels), len(labels))).astype('int')

    for i in range(len(level_predict)):
        matrixAll[level_label[i], level_predict[i]] += 1
        
    #matrixSumTrue = matrixAll.sum(axis=1)[:, np.newaxis]
    #matrixSumPredict = matrixAll.sum(axis=0)[:, np.newaxis] 
    
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
    plt.savefig(graph_folder+levelName +'ConfidenceUnsure.png')
    plt.show()   
    
def plotLevelConfusion(level, level_pred, level_label, labels, level_name, thresholds):

    #level_p = np.argmax(level_pred, axis=1)
    total_predictions = len(level_pred)
    level_p, unsurePredictions = thresholds.unsurePredictions(level, labels, level_pred)
    
    font_size = 28
    if level == 2:
        font_size = 20        
    if level == 3:
        font_size = 14
    plotConfusionMatrixLevel('L' + str(level) + ' ' + level_name, level_p, level_label, labels, thresholds, normalize=True, font_size=font_size)
    
    percentageUnsure = (unsurePredictions/total_predictions)*100
    print(f"Number of unsure predictions {unsurePredictions} {percentageUnsure:.2f}%")
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
    
    labelsrL1, level1_label = renameUnsureLabels(labelsL1, level1_label)
    labelsrL2, level2_label = renameUnsureLabels(labelsL2, level2_label)
    labelsrL3, level3_label = renameUnsureLabels(labelsL3, level3_label)
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

def printPerformanceMetrics(resultFile, thredsholdFile, clearUnsure=True):

    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
    
    labelsrL1, level1_label = renameUnsureLabels(labelsL1, level1_label)
    labelsrL2, level2_label = renameUnsureLabels(labelsL2, level2_label)
    labelsrL3, level3_label = renameUnsureLabels(labelsL3, level3_label)

    pctL1 = []
    pctL2 = []
    pctL3 = []
    f1L1 =  []
    f1L2 =  []
    f1L3 =  []
    
    thresholdIdx = 8 # Default used 6.0
    thresholdSTDs = [10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]
    for thresh in thresholdSTDs:
        print(f"Using threshold of {thresh}*STD")
        thresholds = Thresholds(thredsholdFile, thresholdSTD=thresh)
        
        macroF1score, microF1score, unsurePredictions, percentageUnsure = computeClassScores(1, level1_pred, level1_label, labelsrL1, thresholds, clearUnsure=clearUnsure)
        pctL1.append(percentageUnsure)
        f1L1.append(macroF1score)
        
        macroF1score, microF1score, unsurePredictions, percentageUnsure = computeClassScores(2, level2_pred, level2_label, labelsrL2, thresholds, clearUnsure=clearUnsure)
        pctL2.append(percentageUnsure)
        f1L2.append(macroF1score)
        
        macroF1score, microF1score, unsurePredictions, percentageUnsure = computeClassScores(3, level3_pred, level3_label, labelsrL3, thresholds, clearUnsure=clearUnsure)
        pctL3.append(percentageUnsure)
        f1L3.append(macroF1score)
    
    plt.plot(pctL1, f1L1, "b--", label="L1")
    plt.scatter(pctL1[thresholdIdx], f1L1[thresholdIdx], color="k", marker="s")
    plt.plot(pctL2, f1L2, "m--", label="L2")
    plt.scatter(pctL2[thresholdIdx], f1L2[thresholdIdx], color="k", marker="s")
    plt.plot(pctL3, f1L3, "g--", label="L3")
    plt.scatter(pctL3[thresholdIdx], f1L3[thresholdIdx], color="k", marker="s")
    title = "Unsure threshold (10-1)"
    plt.title(title)
    plt.xlabel('Unsure (%)')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(graph_folder+'UsureThresholds.png')
    plt.show()
      
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
    resultFile = saved_folder + "predictLabels3Lval.pkl"
    
    printPerformanceMetrics(resultFile, thredsholdFile)
    
    thresholds = Thresholds(thredsholdFile, thresholdSTD=thresholdSTD)
    level3False = plotConfusionMatrix(resultFile, thresholds)
    checkList = checkHierarcy(resultFile)
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Validation dataset - number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
    
    graph_folder += "test/"
    resultFile = saved_folder + "predictLabels3Ltest.pkl"

    printPerformanceMetrics(resultFile, thredsholdFile)

    thresholds = Thresholds(thredsholdFile, thresholdSTD=thresholdSTD)
    level3False = plotConfusionMatrix(resultFile, thresholds)
    checkList = checkHierarcy(resultFile)
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Test dataset - number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
    
