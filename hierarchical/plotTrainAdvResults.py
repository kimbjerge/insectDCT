# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 20:19:03 2022

@author: Kim Bjerge
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from sklearn import metrics
from plot import plot_loss_acc
from level_NI_dict import labelsL1, labelsL2, labelsL3
from hierarchical_loss import HierarchicalLossNetwork
graph_folder = './graph_folder/'

def plotAccuracy(resultFile):

    with open(resultFile, 'rb') as f:
        epoch_idx, best_epoch_idx, train_epoch_level1_accuracy, train_epoch_level2_accuracy, train_epoch_level3_accuracy, train_epoch_loss, test_epoch_level1_accuracy, test_epoch_level2_accuracy, test_epoch_level3_accuracy, test_epoch_loss, test_epoch_countWrongHierarchy, test_epoch_countWrongPercentage, testSize = pickle.load(f)
        print("Train and test accuracy and loss loaded from ", resultFile)

    #plot accuracy and loss graph
    plot_loss_acc(graph_folder, num_epoch=epoch_idx, 
                    train_accuracies_level1=train_epoch_level1_accuracy, train_accuracies_level2=train_epoch_level2_accuracy, 
                    train_accuracies_level3=train_epoch_level3_accuracy, train_losses=train_epoch_loss,
                    test_accuracies_level1=test_epoch_level1_accuracy, test_accuracies_level2=test_epoch_level2_accuracy,
                    test_accuracies_level3=test_epoch_level3_accuracy, test_losses=test_epoch_loss)

    epochs = [x+1 for x in range(epoch_idx+1)]

    #test_wrong_hierarchy_df = pd.DataFrame({"Epochs":epochs, "Wrong predictions":test_epoch_countWrongPercentage})
    test_wrong_hierarchy_df = pd.DataFrame({"Epochs":epochs, "Predictions":test_epoch_countWrongHierarchy})

    sns.lineplot(data=test_wrong_hierarchy_df, x='Epochs', y='Predictions')
    plt.title('Predicted wrongly in hierarchy (' + str(testSize) + ')')
    plt.savefig(graph_folder+'wrong_hierarchy.png')
    plt.show()
    print("===========================================================================")
    print("Result file", resultFile)
    print("Minium loss at epochs", best_epoch_idx)
    print("Accuracy level1", test_epoch_level1_accuracy[best_epoch_idx-1])
    print("Accuracy level2", test_epoch_level2_accuracy[best_epoch_idx-1])
    print("Accuracy level3", test_epoch_level3_accuracy[best_epoch_idx-1])
    print("Predicted wrong in hierarchy", test_epoch_countWrongHierarchy[best_epoch_idx-1])
    
    return test_epoch_level1_accuracy[best_epoch_idx-1], test_epoch_level2_accuracy[best_epoch_idx-1], test_epoch_level3_accuracy[best_epoch_idx-1], test_epoch_countWrongHierarchy[best_epoch_idx-1], best_epoch_idx
    

def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, normalize=False, font_size=14):

    matrix = np.zeros((len(labels), len(labels))).astype('int')

    for i in range(len(level_predict)):
        matrix[level_label[i], level_predict[i]] += 1
              
    matrixSum = matrix.sum(axis=1)[:, np.newaxis]
    matrixAvg = matrix.astype('float') / (matrixSum+0.001)
    
    if normalize:
        matrix = matrixAvg
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix, matrixSum)        
        
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(matrixAvg, cmap='Greens')
    
    #yLabels = []
    #for idx in range(len(labelsTrue)):
    #    yLabels.append(labelsTrue[idx] + ' (' + str(int(matrixSum[idx,0])) + ')')
    
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=levelName,
           ylabel='True label',
           xlabel='Predicted label')    
   
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'k'
            if i == j:
                color = 'w'
            if matrix[i, j] > 0:
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=color)
    
    #ax.set_title(levelName)
    fig.tight_layout()
    plt.savefig(graph_folder+levelName +'ConfTest.png')
    plt.show()   

def plotLevelConfusion(level, level_pred, level_label, labels, level_name):

    
    level_p = np.argmax(level_pred, axis=1)
    font_size = 16
    if level == 3:
        font_size = 12
    plotConfusionMatrixLevel('L' + str(level) + ' ' + level_name, level_p, level_label, labels, normalize=False, font_size=font_size)
  
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
    
    plotLevelConfusion(1, level1_pred, level1_label, labelsL1, "Order")
    plotLevelConfusion(2, level2_pred, level2_label, labelsL2, "Family")
    plotLevelConfusion(3, level3_pred, level3_label, labelsL3, "Species")
    
    level1p = np.argmax(level1_pred, axis=1)
    level2p = np.argmax(level2_pred, axis=1)
    level3p = np.argmax(level3_pred, axis=1)
    level3False = 0
    for idx in range(len(level3p)):
        if level3p[idx] != level3_label[idx]:
            #print(labelsL1[level1p[idx]], labelsL2[level2p[idx]], labelsL3[level3p[idx]])
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


def createScore(labels, level_pred, level_label):
    
    predictions = np.argmax(level_pred, axis=1)
    size_labels = len(labels)
    scores = []
    for idx in range(size_labels):
        scores.append([])
    for idx in range(len(predictions)):
        if predictions[idx] == level_label[idx]: # Correct prediction
            predScore = level_pred[idx][predictions[idx]]
            scores[predictions[idx]].append(predScore)
            
    return scores
        
def plotHist(level, labelsL, classIdx, score):
    
    mu, std = norm.fit(score)
    classThreshold = round((mu - 2*std)*10)/10
    print(mu, std, classThreshold)
    # Plot histogram
    plt.rcParams.update({'font.size': 16})
    plt.hist(score, bins=50, density=True, alpha=0.6, color='b')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)      
    plt.plot(x, p, 'k', linewidth=2)
    steps = 20
    step = max(p)/steps
    listTh = [classThreshold for i in range(steps)]
    listPb = [step*i for i in range(steps)]
    plt.plot(listTh, listPb, 'r')
    plt.title('L' + str(level) + ' ' + labelsL[classIdx])
    plt.xlabel('Class score (Th=' + str(classThreshold) + ')')
    plt.ylabel('Probability')
    plt.savefig(graph_folder + 'histL'+ str(level) + labelsL[classIdx] + '.png')
    plt.show()
    
def plotHistogram(resultFile):

    with open(resultFile, 'rb') as f:
        level1_pred, level2_pred, level3_pred, level1_label, level2_label, level3_label = pickle.load(f)
        print("Predictions and labels and loss loaded from ", resultFile)
        
    level1_score = createScore(labelsL1, level1_pred, level1_label)
    print("Level 1")
    for classIdx in range(len(level1_score)):
        score = level1_score[classIdx]
        plotHist(1, labelsL1, classIdx, score)
        
    level2_score = createScore(labelsL2, level2_pred, level2_label)
    print("Level 2")
    for classIdx in range(len(level2_score)):
        score = level2_score[classIdx]
        plotHist(2, labelsL2, classIdx, score)

    level3_score = createScore(labelsL3, level3_pred, level3_label)
    print("Level 3")
    for classIdx in range(len(level3_score)):
        score = level3_score[classIdx]
        plotHist(3, labelsL3, classIdx, score)

    
#%% MAIN
if __name__=='__main__':
    
    acc_level1 = []
    acc_level2 = []
    acc_level3 = []
    acc_avg_levels = []
    wrong_hierarchy = []
    alpha_values = []
    best_epoch = []
    counts = []
    

    resultFiles = [
        # Result file                  , alpha value
        ['./saved/resultsAdv3L.pkl',     0.5]
        ]  
    
    """
    resultFiles = [
        # Result file                  , alpha value
        ['./saved/resultsAdv3L0_01.pkl',     0.01],
        ['./saved/resultsAdv3L0_10.pkl',     0.1],
        ['./saved/resultsAdv3L0_20.pkl',     0.2],
        ['./saved/resultsAdv3L0_30.pkl',     0.3],
        ['./saved/resultsAdv3L0_40.pkl',     0.4],
        ['./saved/resultsAdv3L0_50.pkl',     0.5],
        ['./saved/resultsAdv3L0_60.pkl',     0.6],
        ['./saved/resultsAdv3L0_70.pkl',     0.7],
        ['./saved/resultsAdv3L0_80.pkl',     0.8],
        ['./saved/resultsAdv3L0_90.pkl',     0.9],
        ['./saved/resultsAdv3L0_99.pkl',     0.99]
        ] 
    """
    
    count = 1
    for resultFile, alpha in resultFiles:
        print(resultFile, alpha)
        accLevel1, accLevel2, accLevel3, wrongHierarchy, best_epoch_idx = plotAccuracy(resultFile)
        acc_avg_levels.append((accLevel1+accLevel2+accLevel3)/3)
        acc_level1.append(accLevel1)
        acc_level2.append(accLevel2)
        acc_level3.append(accLevel3)
        wrong_hierarchy.append(wrongHierarchy)
        alpha_values.append(alpha)
        best_epoch.append(best_epoch_idx)
        counts.append(count)
        count += 1

    acc_level1_df =  pd.DataFrame({"Alpha":alpha_values, "Accuracy":acc_level1, "Level":'L1'})
    acc_level2_df =  pd.DataFrame({"Alpha":alpha_values, "Accuracy":acc_level2, "Level":'L2'})
    acc_level3_df =  pd.DataFrame({"Alpha":alpha_values, "Accuracy":acc_level3, "Level":'L3'})
    average_accuracy_df = pd.DataFrame({"Alpha":alpha_values, "Accuracy":acc_avg_levels, "Level":'Avg'})
    data_to_plot = pd.concat([acc_level1_df, acc_level2_df, acc_level3_df, average_accuracy_df])
    g = sns.lineplot(data=data_to_plot, x='Alpha', y='Accuracy', hue='Level', style='Level', markers=True)
    #g.set(xscale='log')
    g.set(ylim=(96.5,99.5))
    plt.title('Accuracy vs. alpha')
    plt.savefig(graph_folder+'accuracy_alpha_A9.png')
    plt.show()    
    print(alpha_values, best_epoch, acc_avg_levels)

    acc_level1_df =  pd.DataFrame({"Times":counts, "Accuracy":acc_level1, "Level":'L1'})
    acc_level2_df =  pd.DataFrame({"Times":counts, "Accuracy":acc_level2, "Level":'L2'})
    acc_level3_df =  pd.DataFrame({"Times":counts, "Accuracy":acc_level3, "Level":'L3'})
    average_accuracy_df = pd.DataFrame({"Times":counts, "Accuracy":acc_avg_levels, "Level":'Avg'})
    data_to_plot = pd.concat([acc_level1_df, acc_level2_df, acc_level3_df, average_accuracy_df])
    g = sns.lineplot(data=data_to_plot, x='Times', y='Accuracy', hue='Level', style='Level', markers=True)
    g.set(ylim=(96.5,99.5))
    plt.title('Accuracy vs. train times')
    plt.savefig(graph_folder+'accuracy_train_alpha_A9.png')
    plt.show()    
    print(alpha_values, best_epoch, acc_avg_levels)

    resultFile = './saved/predictLabels3Ltrain.pkl'    
    plotHistogram(resultFile)
    
    resultFile = './saved/predictLabels3Lval.pkl'
    level3False = plotConfusionMatrix(resultFile)
    checkList = checkHierarcy(resultFile)
    
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
