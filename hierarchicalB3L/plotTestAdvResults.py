# -*- coding: utf-8 -*-
"""
Modified on Sat April 10 18:19:03 2025

@author: Kim Bjerge
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#from level_testNI_dict import labelsL1, labelsL2, labelsL3
from level_NI_dict import labelsL1, labelsL2, labelsL3
from hierarchical_loss import HierarchicalLossNetwork

graph_folder = './graph_folder/'

def renameUnknownLabels(labelsL, level_label):
    
    unknownIdx = len(labelsL)
    for idx in range(len(level_label)):
        if level_label[idx] == -1:
            level_label[idx] = unknownIdx
            
    labelsL = labelsL + ['Unsure']
    
    return labelsL, level_label

# Level thresholds for unknown idx found on train dataset as mean - 2xstd
# Trained model dhc_save1_00_best.pth
#level1Thresholds = [7.4, 4.4, 4.2, 6.9]
#level2Thresholds = [8.6, 5.1, 3.7, 3.9, 8.1]
#level3Thresholds = [9.6, 6.3, 6.3, 3.9, 4.5, 5.9, 7.1, 4.5, 8.5]

# Level thresholds for unknown idx found on train dataset as mean - 2xstd, dhc0_20_best.pth
level1Thresholds = [6.4, 2.5, 2.8, 4.5] # Min 2.5
level2Thresholds = [6.6, 2.5, 2.5, 2.9, 5.1] # Min 2.5
level3Thresholds = [8.2, 2.6, 3.1, 3.4, 3.5, 5.0, 4.1, 2.6, 5.0]

# Level thresholds for unknown idx found on train dataset as mean - 2xstd, dhc_GBIFNI.pth - MIX
#level1Thresholds = [5.1, 3.5, 3.1, 3.0] # Min 2.5
#level2Thresholds = [5.7, 4.1, 3.9, 2.5, 3.5] # Min 2.5
#level3Thresholds = [6.7, 3.9, 5.4, 4.4, 4.1, 5.3, 6.4, 3.4, 4.1]

def unknownPredictions(level, labels, level_pred):
    if level == 1:
        levelThredsholds = level1Thresholds
    if level == 2:
        levelThredsholds = level2Thresholds
    if level == 3:
        levelThredsholds = level3Thresholds
    unknownIdx = len(labels)-1
    level_p = np.argmax(level_pred, axis=1)
    max_p = np.max(level_pred, axis=1)
    for idx in range(len(level_p)):
        #if max_p[idx] < 3.8: # KBE?? level to be found based on training dataset 3.8 - 9.1
        if max_p[idx] < levelThredsholds[level_p[idx]]:
            level_p[idx] = unknownIdx
    return level_p
    

def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, normalize=False):

    matrixAll = np.zeros((len(labels), len(labels))).astype('int')

    for i in range(len(level_predict)):
        matrixAll[level_label[i], level_predict[i]] += 1
        
    matrixSumTrue = matrixAll.sum(axis=1)[:, np.newaxis]
    matrixSumPredict = matrixAll.sum(axis=0)[:, np.newaxis] 
    
    labelsTrue = []
    labelsPredict = []
    for i in range(len(labels)):
        if matrixSumTrue[i] > 0:
            labelsTrue.append(labels[i])
    for i in range(len(labels)):
        if matrixSumPredict[i] > 0:
            labelsPredict.append(labels[i])
            
    matrix = np.zeros((len(labelsTrue), len(labelsPredict))).astype('int') 
    x = 0;
    for i in range(len(labels)):
        y = 0;
        if matrixSumTrue[i] > 0:
            for j in range(len(labels)):
                if matrixSumPredict[j] > 0:
                    matrix[x, y] = matrixAll[i, j] 
                    y += 1               
            x += 1
         
    if normalize:
        matrixSum = matrix.sum(axis=1)[:, np.newaxis]
        matrix = matrix.astype('float') / (matrixSum+0.001)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix, matrixSum)        
        
    fig, ax = plt.subplots(figsize=(8, 6))
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
    fmt = '.2f' if normalize else 'd'
    for i in range(len(labelsTrue)):
        for j in range(len(labelsPredict)):
            color = 'k'
            if matrix[i, j] > 0.495:
                color = 'w'
            if matrix[i,j] > 0.005:
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=color)
    
    #ax.set_title(levelName)
    fig.tight_layout()
    plt.savefig(graph_folder+levelName +'ConfTest.png')
    plt.show()   
    
def plotLevelConfusion(level, level_pred, level_label, labels, level_name):

    #level_p = np.argmax(level_pred, axis=1)
    level_p = unknownPredictions(level, labels, level_pred)
    
    plotConfusionMatrixLevel('L' + str(level) + ' ' + level_name, level_p, level_label, labels, normalize=True)
    
    """
    #confusion_matrix = metrics.confusion_matrix(level_label, level_p)
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics.ConfusionMatrixDisplay.from_predictions(level_label, level_p, normalize='true', display_labels=labels, #Normalized labels
    #metrics.ConfusionMatrixDisplay.from_predictions(level_label, level_p, normalize='pred', display_labels=labels, #Normalized predictions
                                                    xticks_rotation='vertical', values_format='.2g', cmap='Greens', ax=ax)
    #metrics.ConfusionMatrixDisplay.from_predictions(level_label, level_p, normalize=None, display_labels=labels, 
    #                                                xticks_rotation='vertical', values_format='.d', cmap='Greens', ax=ax)
    plt.title('L' + str(level) + ' ' + level_name)
    plt.savefig( graph_folder+'levelTest'+ str(level) + '.png')
    plt.show()
    """
    
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
    
    
def plotConfusionMatrix(resultFile):

    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
    
    labelsrL1, level1_label = renameUnknownLabels(labelsL1, level1_label)
    labelsrL2, level2_label = renameUnknownLabels(labelsL2, level2_label)
    labelsrL3, level3_label = renameUnknownLabels(labelsL3, level3_label)
    plotLevelConfusion(1, level1_pred, level1_label, labelsrL1, "Order")
    plotLevelConfusion(2, level2_pred, level2_label, labelsrL2, "Family")
    plotLevelConfusion(3, level3_pred, level3_label, labelsrL3, "Species")
    
    level1p = np.argmax(level1_pred, axis=1)
    level2p = np.argmax(level2_pred, axis=1)
    level3p = np.argmax(level3_pred, axis=1)
    level3False = 0
    for idx in range(len(level3p)):
        if level3p[idx] != level3_label[idx]:
            print(labelsrL1[level1p[idx]], labelsrL2[level2p[idx]], labelsrL3[level3p[idx]])
            level3False += 1
    
    return level3False
            
    
def checkHierarcy(resultFile):    
    
    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
        
    HLN = HierarchicalLossNetwork(total_level=3, device="cpu", simple=True)

    level1p = np.argmax(level1_pred, axis=1)
    level2p = np.argmax(level2_pred, axis=1)
    level3p = np.argmax(level3_pred, axis=1)
    checkL2 = HLN.check_hierarchy_list(level=1, current_level=level2p, previous_level=level1p)
    checkL3 = HLN.check_hierarchy_list(level=2, current_level=level3p, previous_level=level2p)
    
    checkList = checkL3 
    
    for idx in range(len(checkList)):
        if checkL3[idx] == False or checkL2[idx] == False:
            checkList[idx] = False
            print(labelsL1[level1p[idx]], labelsL2[level2p[idx]], labelsL3[level3p[idx]])
    
    return checkList
    

#%% MAIN
if __name__=='__main__':
    
    
    #resultFile = './saved/predictTestLabels3L_alpha1_e17.pkl'
    #resultFile = './saved/predictTestLabels3L_alpha0_2.pkl'
    #resultFile = './saved/predictTestLabels3L_alpha0_5_T4.pkl'
    resultFile = './saved/predictLabels3Lval.pkl'
    # Trained mix model
    #resultFile = './saved/predictTestLabels3L_GBIFNIMix.pkl'
    level3False = plotConfusionMatrix(resultFile)
    checkList = checkHierarcy(resultFile)
    
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
    
