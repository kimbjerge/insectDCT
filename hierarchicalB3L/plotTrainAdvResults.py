# -*- coding: utf-8 -*-
"""
Modified on Sat April 10 18:19:03 2025

@author: Kim Bjerge
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from sklearn import metrics
from plot import plot_loss_acc
#from level_NI_dict import labelsL1, labelsL2, labelsL3
from hierarchical_loss import HierarchicalLossNetwork

#saved_folder = "./saved_128_ConvNextV4/"
#saved_folder = "./saved_128_finalV4/"
#saved_folder = "./models_saved/saved_128_ResNetV5/"
#saved_folder = "./models_saved/saved_128_ConvNextV6_1/"
saved_folder = "./models_saved/saved_128_ConvNextV5/"
graph_folder = "./graph_folder/"

# Check taxon prediction correct in hiearachy when plotting confusion matrix L2, L3 and saving scores
checkHierarchy = False
checkedName = ""

label_file = saved_folder+"labelsAdv3L.pkl"
# Load labels shared by several functions
with open(label_file, 'rb') as f:
    image_path_list, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, trainL1, trainL2, trainL3, valL1, valL2, valL3 = pickle.load(f)
    print("Labels and hierarchy dependency loaded from ", label_file)
    print("L1", labelsL1)
    print("L2", labelsL2)
    print("L3", labelsL3)
        
    
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


# Move predictions to diagonal in confusion matrix for true positive predictions in hierarchy
def moveTruePositives(x, y, checkedMatrix, labels, labelLx):

    if checkedMatrix[x][y] > 0:
        print(labels[x], "same as", labelLx, checkedMatrix[x][y])
    checkedMatrix[x][x] += checkedMatrix[x][y] # Accept predictions by moving to diagnonal in matrix
    checkedMatrix[x][y] = 0
    if checkedMatrix[y][x] > 0:
        print(labelLx, "same as", labels[x], checkedMatrix[y][x])
    checkedMatrix[y][y] += checkedMatrix[y][x] # Accept predictions by moving to diagnonal in matrix
    checkedMatrix[y][x] = 0

    return checkedMatrix    


# Check and create cleaned confusion matrix for 
def checkValidHiearchyMatrix(levelName, matrix, labels):
    
    checkedMatrix = matrix.copy()
    
    if ("L2" in levelName) or ("L3" in levelName):
        
        # Check hierarchy L1 -> L2
        for x in range(len(labels)):
            if labels[x] in labelsL1:
                labelsL2x = hierarchyL1[labels[x]]
                for labelL2 in labelsL2x:
                    if labelL2 in labels:
                        y = labels.index(labelL2)
                        if x != y:
                            checkedMatrix = moveTruePositives(x, y, checkedMatrix, labels, labelL2)
                            
    if "L3" in levelName:
        
        # Check for L3 if species rank of genus
        for x in range(len(labels)):
            if labels[x] in labelsL3:
                genus = labels[x].split(' ')[0]
                if len(labels[x].split(' ')) == 1: # Genus
                    for y in range(len(labels)):
                        if labels[y] in labelsL3:
                            species = labels[y]
                            if len(labels[y].split(' ')) == 2: # Species
                                if genus in species:
                                    if x != y:
                                        checkedMatrix = moveTruePositives(x, y, checkedMatrix, labels, species)                                  
                
                    
        
        # Check hierarchy L2 -> L3
        for x in range(len(labels)):
            if labels[x] in labelsL2:
                labelsL3x = hierarchyL2[labels[x]]
                for labelL3 in labelsL3x:
                    if labelL3 in labels:
                        y = labels.index(labelL3)
                        if x != y:
                            checkedMatrix = moveTruePositives(x, y, checkedMatrix, labels, labelL3)

        # Check hierarchy L1 -> L2 -> L3
        for x in range(len(labels)):
            if labels[x] in labelsL1:
                labelsL2x = hierarchyL1[labels[x]]
                for labelL2 in labelsL2x:
                    labelsL3x = hierarchyL2[labels[x]]
                    for labelL3 in labelsL3x:
                        if labelL3 in labels:
                            y = labels.index(labelL3)
                            if x != y:
                                checkedMatrix = moveTruePositives(x, y, checkedMatrix, labels, labelL3)       
    
    return checkedMatrix


# Computes macro and micro recall, precision and f1scores for each class in the confusion matrix
def saveClassScores(levelName, labels, confMatrix):
    
    matrixSumTrue = confMatrix.sum(axis=1)[:] # Number of labeled samples for each class
    matrixSumPredicted = confMatrix.T.sum(axis=1)[:] # Number of predicted samples for each class
    
    recalls = np.zeros(len(labels))
    precisions = np.zeros(len(labels))
    f1scores = np.zeros(len(labels))
    matrixSumTP = np.zeros(len(labels)).astype('int')
    
    if "L1" in levelName:
        file = open(graph_folder + checkedName + "ClassScores.csv", "w")
        file.write("level,label,TP,TP_FP,TP_FN,precision,recall,f1score\n")
    else:
        file = open(graph_folder + checkedName + "ClassScores.csv", "a")
        
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
        line = f"{levelName},{labels[idx]},{matrixSumTP[idx]},{matrixSumPredicted[idx]},{matrixSumTrue[idx]},{precisions[idx]},{recalls[idx]},{f1scores[idx]}\n"
        
        #if f1scores[idx] < 0.3:
        #    print(line)
        file.write(line)
            
    macroRecall = np.mean(recalls)
    macroPrecision = np.mean(precisions)
    macroF1score = np.mean(f1scores)
    line = f"{levelName},Macro,{np.mean(matrixSumTP)},{np.mean(matrixSumPredicted)},{np.mean(matrixSumTrue)},{macroPrecision},{macroRecall},{macroF1score}\n"
    file.write(line)

    microRecall = sum(matrixSumTP)/sum(matrixSumTrue)
    microPrecision = sum(matrixSumTP)/sum(matrixSumPredicted)
    microF1score =  2*microRecall*microPrecision/(microRecall + microPrecision)
    line = f"{levelName},Micro,{sum(matrixSumTP)},{sum(matrixSumPredicted)},{sum(matrixSumTrue)},{microPrecision},{microRecall},{microF1score}\n"
    file.write(line)
    
    file.close()
    
    plt.rcParams.update({'font.size': 12})
    plt.scatter(matrixSumTrue, f1scores)
    title = f"{levelName} Macro: P {macroPrecision:.3f} R {macroRecall:.3f} F1 {macroF1score:.3f}"
    plt.title(title)
    plt.xlabel('Samples in class')
    plt.ylabel('F1-score')
    plt.xscale('log', base=10)
    plt.savefig(graph_folder+checkedName+levelName+'F1scores.png')
    plt.show()
    

def plotConfusionMatrixLevel(levelName, level_predict, level_label, labels, normalize=False, font_size=14):

    matrix = np.zeros((len(labels), len(labels))).astype('int')

    count = 0
    for i in range(len(level_predict)):
        if level_label[i] > -1: # ????
            matrix[level_label[i], level_predict[i]] += 1
        else:
            count += 1
            print("Invalid label", levelName, level_label[i], i, count)
    
    if checkHierarchy:
        matrix = checkValidHiearchyMatrix(levelName, matrix, labels)
    saveClassScores(levelName, labels, matrix)
    
    matrixSum = matrix.sum(axis=1)[:, np.newaxis]
    matrixAvg = matrix.astype('float') / (matrixSum+0.001)
    
    ylabels = []
    for idx in range(len((labels))):
        ylabels.append(labels[idx] + " (" + str(matrixSum[idx][0]) + ")")
        
    if normalize:
        matrix = matrixAvg
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(matrix, matrixSum)        
        
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.imshow(matrixAvg, cmap='Greens')
    
    #yLabels = []
    #for idx in range(len(labelsTrue)):
    #    yLabels.append(labelsTrue[idx] + ' (' + str(int(matrixSum[idx,0])) + ')')
    
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=ylabels,
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
    plt.savefig(graph_folder + checkedName + levelName +'ConfTest.png')
    plt.show()   


def plotLevelConfusion(level, level_pred, level_label, labels, level_name):
    
    level_p = np.argmax(level_pred, axis=1)
    font_size = 28
    if level == 2:
        font_size = 20        
    if level == 3:
        font_size = 14
    plotConfusionMatrixLevel('L' + str(level) + ' ' + level_name, level_p, level_label, labels, normalize=False, font_size=font_size)
  
    if level == 1:
        f = open(graph_folder+"Results.txt", "w")
    else:
        f = open(graph_folder+"Results.txt", "a")
        
    precision = metrics.precision_score(level_label, level_p, average='macro')
    text = f"Level {level} (macro) precision {precision:.4f}"
    print(text)
    f.write(text+"\n")

    recall = metrics.recall_score(level_label, level_p, average='macro')
    text = f"Level {level} (macro) recall {recall:.4f}"
    print(text)
    f.write(text+"\n")

    #f1score = 2*recall*precision/(recall+precision) #
    f1score = metrics.f1_score(level_label, level_p, average='macro')
    text = f"Level {level} (macro) f1-score {f1score:.4f}"
    print(text)
    f.write(text+"\n")
    
    precision = metrics.precision_score(level_label, level_p, average='micro')
    text = f"Level {level} (micro) recall {precision:.4f}"
    print(text)
    f.write(text+"\n")

    recall = metrics.recall_score(level_label, level_p, average='micro')
    text = f"Level {level} (micro) precision {recall:.4f}"
    print(text)
    f.write(text+"\n")

    f1score = metrics.f1_score(level_label, level_p, average='micro')
    text = f"Level {level} (micro) f1-score {f1score:.4f}"
    print(text)
    f.write(text+"\n")
    f.close()
    
    
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
    
    
    HLN = HierarchicalLossNetwork(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, 
                                  [0, 0, 0],
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
        

def plotHist(level, labelsL, classIdx, score, file):
    
    if len(score) > 0:
        mu, std = norm.fit(score)
    else:
        mu = 0
        std = 0
    
    #print(labelsL[classIdx], mu, std)
    #classThreshold = round((mu - 2*std)*10)/10 # Used in paper
    classThreshold = round((mu - 3*std)*100)/100 # Less number of unsure
    if std == 0:
        classThreshold = -1.0 # No threshold when standard deviation is zero 
    print(level, classIdx, labelsL[classIdx], mu, std, classThreshold)
    line = str(level) + ',' + str(classIdx) + ',' + labelsL[classIdx] + ',' + str(mu) + ',' + str(std) + ',' + str(classThreshold) + "\n" 
    file.write(line)
    
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

    file = open(saved_folder+"thresholds.csv", "w")
    file.write("Level,ClassIdx,ClassName,Mean,Std,Threshold\n")
    level1_score = createScore(labelsL1, level1_pred, level1_label)
    print("Level 1")
    for classIdx in range(len(level1_score)):
        score = level1_score[classIdx]
        plotHist(1, labelsL1, classIdx, score, file)
        
    level2_score = createScore(labelsL2, level2_pred, level2_label)
    print("Level 2")
    for classIdx in range(len(level2_score)):
        score = level2_score[classIdx]
        plotHist(2, labelsL2, classIdx, score, file)

    level3_score = createScore(labelsL3, level3_pred, level3_label)
    print("Level 3")
    for classIdx in range(len(level3_score)):
        score = level3_score[classIdx]
        plotHist(3, labelsL3, classIdx, score, file)
    
    file.close()
    
    
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
        [saved_folder+'resultsAdv3L.pkl',     0.5]
        ]  
    
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
    plt.savefig(graph_folder+'accuracy_alpha.png')
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
    plt.savefig(graph_folder+'accuracy_train_alpha.png')
    plt.show()    
    print(alpha_values, best_epoch, acc_avg_levels)

    resultFile = saved_folder+'predictLabels3Ltrainval.pkl'    
    plotHistogram(resultFile)
    
    # Evaluate model on validations datasets
    resultFile = saved_folder+'predictLabels3Lval.pkl'
    level3False = plotConfusionMatrix(resultFile)

    checkList = checkHierarcy(resultFile)  
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    print("Number of wrong predictions in hierarchy", countWrongHierarchy, level3False, 100*countWrongHierarchy/level3False)
    
    # Evaluate model on test datasets
    resultFile = saved_folder+'predictLabels3Ltest.pkl'
    graph_folder += "test/"
    if not os.path.exists(graph_folder):
        os.mkdir(graph_folder)        
    level3False = plotConfusionMatrix(resultFile)
    
    checkHierarchy = True
    checkedName = "Taxonomy"
    level3False = plotConfusionMatrix(resultFile)
    

