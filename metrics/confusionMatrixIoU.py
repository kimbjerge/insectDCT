"""
Created on Mon Sep  17 09:32:22 2019
Modified on Sat Oc   4 19:45:00 2025

Python script to calculate precision, recall and confusion matrix
based on YOLO label files and result CSV files (insectsDCT)

@author: Kim Bjerge (Made from scratch)
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from loadLabelsPredictions import load_predictions, load_labels, exist_label

#missed/false   0    
#honningbi      1
labelNames = ["Missed", "Insect"]
iou_used = True
iou_thr = 0.2

# Calculate precision, recall and F1-score as micro average on all testsets
def calculateMicroScores(truepositive, falsepositive, falsenegative):

    # Calculate precision
    sumtruefalse = [truepositive[i] + falsepositive[i] for i in range(len(truepositive))]
    precisionTotal = sum(truepositive) /sum(sumtruefalse)
    
    # Calculate recall
    sumtruefalse = [truepositive[i] + falsenegative[i] for i in range(len(truepositive))]
    recallTotal = sum(truepositive) /sum(sumtruefalse)
    f1scoreTotal = 2*precisionTotal*recallTotal/(precisionTotal+recallTotal)
    
    return precisionTotal, recallTotal, f1scoreTotal
    
# Calculation of precision and recall
def calc_precision_recall(cm):
    
    truepositive = []
    falsepositive = []
    falsenegative = []
    ridx = 0
    for row in cm:
        fn = 0
        cidx = 0
        for cnt in row:
            if ridx == 0 and cidx > 0:
                falsepositive.append(cnt) # First row is false positive
            if ridx > 0 and cidx == 0:
                fn = fn + cnt # First coloum is false negative
            if ridx > 0 and cidx > 0:
                if ridx == cidx: # Diagonal
                    truepositive.append(cnt) # True positive
                else:
                    fn = fn + cnt # False classifications
            cidx = cidx + 1
        if ridx > 0: falsenegative.append(fn)
        ridx = ridx + 1
    
    # Calculate precision
    sumtruefalse = [truepositive[i] + falsepositive[i] for i in range(len(truepositive))]
    precision = [truepositive[i] / sumtruefalse[i] for i in range(len(truepositive))]
    precisionTotal = sum(truepositive) /sum(sumtruefalse)
    
    # Calculate recall
    sumtruefalse = [truepositive[i] + falsenegative[i] for i in range(len(truepositive))]
    recall = [truepositive[i] / sumtruefalse[i] for i in range(len(truepositive))]
    recallTotal = sum(truepositive) /sum(sumtruefalse)
    
    f1scores = [2*precision[i]*recall[i]/(precision[i]+recall[i]) for i in range(len(precision))]
    
    print("TP       :", truepositive)
    print("FP       :", falsepositive)
    print("FN       :", falsenegative)
    for i in range(len(precision)):
        precision[i] = np.round(precision[i]*1000)/1000
        recall[i] = np.round(recall[i]*1000)/1000
        f1scores[i] = np.round(f1scores[i]*1000)/1000    
    print("Recall    TP/(TP+FN):", recall)
    print("Precision TP/(TP+FP):", precision)
    print("F1-score 2*P*R/(P+R):", f1scores);
    
    return precisionTotal, recallTotal, truepositive, falsepositive, falsenegative

# Plots confusion matrix and calculate precission and recall
# Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    precision, recall, tp, fp, fn = calc_precision_recall(cm)
    
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    #fig.tight_layout()
    return precision, recall, tp, fp, fn
        
# Create true and pred lists based on predictions and labels
def update_predictions(predictObjects, labelObjects):
    
    y_true = []
    y_pred = []
 
    for predObject in predictObjects:
        found, labelObject = exist_label(predObject, labelObjects, iou_used=iou_used, iou_thr=iou_thr)
        if found:
            y_true.append(labelObject['class'])
        else:
            y_true.append(0) # Label not found
        y_pred.append(predObject['class'])
    
    for labelObject in labelObjects:
        if labelObject["found"] == False:
            y_true.append(labelObject['class'])
            y_pred.append(0)
 
    # Clear labels found
    for idx in range(len(labelObjects)):
            labelObjects[idx]["found"] = False
    
    return y_true, y_pred

def create_confusion_matrix(labels, result_file, labelNames, modelName, threshold=[50,50,50,50,50,50]):
    
    predictions = load_predictions(result_file, filterTime=0, threshold=threshold)
    y_true, y_pred = update_predictions(predictions, labels)
    print("-------------------------------------------------------------------------------------------------------")
    print("Confusion matrix with probability thresholds:", result_file)
    precision, recall, tp, fp, fn = plot_confusion_matrix(y_true, y_pred, labelNames, normalize=False, title="Nature Impact Confusion Matrix "+modelName, cmap=plt.cm.Greens)
    #print("from CSV file:", result_file)
    #print("Precision (all): %.3f" % precision)
    #print("Recall (all)   : %.3f" % recall)
    f1score = 2*precision*recall/(precision+recall)
    #print("F1-score (all) : %.3f" % f1score)
    #plot_confusion_matrix(y_true, y_pred, labelNames, normalize=False, title="Nature Impact Confusion Matrix", cmap=plt.cm.Greens)
    return precision, recall, f1score, tp, fp, fn

# Create true and score lists based on predictions and labels for classId
def true_score_predictions(predictObjects, labelObjects, classId):
    
    y_true = []
    y_scores = []
 
    for predObject in predictObjects:
        if predObject['class'] == classId:
            found, labelObject = exist_label(predObject, labelObjects, iou_used=iou_used, iou_thr=iou_thr)
            if found:
                y_true.append(1) # True positive
            else:
                y_true.append(0) # Label not found, false positive
            y_scores.append(predObject['prob']/100)
    
    for labelObject in labelObjects:
        if labelObject["found"] == False:
            if labelObject["class"] == classId:
                y_true.append(1) # Label found, false negative
                y_scores.append(0.0) # with no score (0.0)
 
    # Clear labels found
    for idx in range(len(labelObjects)):
            labelObjects[idx]["found"] = False
    
    return y_true, y_scores

# Calculate mean Average Precision using sklearn
def calc_mAP(image_dic, result_file, classId):

    labels, count = load_labels(image_dic)
    predictions = load_predictions(result_file, threshold=[0])
    y_true, y_scores = true_score_predictions(predictions, labels, classId)
    #print(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.step(recall, precision, where='post')
    AP = average_precision_score(y_true, y_scores)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-recall curve (test2110) for '+ labelNames[classId] )
    plt.show()
    return AP

def predictTestData(label_path, label_dir, result_file, threshold):
    
    precisionAll = []
    recallAll = []
    f1scoreAll = []
    tpAll = []
    fpAll = []
    fnAll= []

    print("--------------------------------------------------------------------------------------------------------")
    print("Evaluation based on labels and predictions ", label_path, result_file)
    
    savedLabels = label_dir+"_labels.npy"        
    if os.path.exists(savedLabels):
        print("Uses saved labels from", savedLabels)
        labels = np.load(savedLabels, allow_pickle=True)
    else:   
        print("Reading labels from ", label_path)
        labels, counts = load_labels(label_path, classes=1)
        np.save(savedLabels, labels)   
        print("Label saved to file", savedLabels)
        
    precision, recall, f1score, tp, fp, fn = create_confusion_matrix(labels, result_file, labelNames, 'Model', threshold=threshold) 
    precisionAll.append(precision)
    recallAll.append(recall)
    f1scoreAll.append(f1score)
    tpAll.append(tp[0]) # Only one class
    fpAll.append(fp[0])
    fnAll.append(fn[0])
    
    """
    print("--------------------------------------------------------------------------------------------------------")
    
    print("Dataset     : %s" % result_file)
    print("Recalls     : %.3f" % recallAll[0])
    print("Precisions  : %.3f" % precisionAll[0])
    print("F1-scores   : %.3f" % f1scoreAll[0])
    
    print("Macro Avg Recall   : %.3f" % np.mean(recallAll))
    print("Macro Avg Precision: %.3f" % np.mean(precisionAll))
    print("Macro Avg F1-score : %.3f" % np.mean(f1scoreAll))
    precisionMicro, recallMicro, f1scoreMicro = calculateMicroScores(tpAll, fpAll, fnAll)
    print("Micro Avg Recall   : %.3f" % recallMicro)
    print("Micro Avg Precision: %.3f" % precisionMicro)
    print("Micro Avg F1-score : %.3f" % f1scoreMicro)
    #plot_confusion_matrix(y_true, y_pred, l
    """
    precisionMicro, recallMicro, f1scoreMicro = calculateMicroScores(tpAll, fpAll, fnAll)
    
    print("========================================================================================================")
    
    return recallAll, precisionAll, f1scoreAll, recallMicro, precisionMicro, f1scoreMicro


def printStat(result, test_dics):

    recallDirs = np.zeros( (len(result), len(test_dics)) )
    precisionDirs = np.zeros( (len(result), len(test_dics)) )
    f1scoreDirs = np.zeros( (len(result), len(test_dics)) )
    recallMicroDirs = np.zeros( len(result) )
    precisionMicroDirs = np.zeros( len(result) )
    f1scoreMicroDirs = np.zeros( len(result) )
    idx = 0
    for test_dir, recall, precision, f1score, recallMicro, precisionMicro, f1scoreMicro in result:
        recallDirs[idx, :] = recall
        precisionDirs[idx, :] = precision
        f1scoreDirs[idx, :] = f1score
        recallMicroDirs[idx] = recallMicro
        precisionMicroDirs[idx] = precisionMicro
        f1scoreMicroDirs[idx] = f1scoreMicro
        idx += 1
  
    for idx in range(len(test_dics)):
        print(test_dics[idx])
        print('Avg Recall  : %.3f' % np.mean(recallDirs[:, idx]))
        print('Avg Precision  : %.3f' % np.mean(precisionDirs[:, idx]))
        print('Avg F1-score  : %.3f' % np.mean(f1scoreDirs[:, idx]))
    
    print("Macro Avg Recall   : %.3f" % np.mean(np.mean(recallDirs)))
    print("Macro Avg Precision: %.3f" % np.mean(np.mean(precisionDirs)))
    print("Macro Avg F1-score : %.3f" % np.mean(np.mean(f1scoreDirs)))
    
    print("Micro Avg Recall   : %.3f" % np.mean(recallMicroDirs))
    print("Micro Avg Precision: %.3f" % np.mean(precisionMicroDirs))
    print("Micro Avg F1-score : %.3f" % np.mean(f1scoreMicroDirs))       
    
if __name__=='__main__':

    threshold = [50]   #YOLOv5, Test with minimum 25% 
    

    test_files = [ 
                  #['insects5Color-train-images-CL.csv', 'insects5Color', 37],
                  ['insects5Motion-train-images-CL.csv', 'insects5Motion', 35]
                  #['insects3Color-train-images-CL.csv', 'insects5Color', 37],
                  #['insects3Motion-train-images-CL.csv', 'insects5Motion', 40]
                ]
 
    result = []
    idx = 0
    for csvFile, label_dir, conf in test_files:
        print(csvFile, label_dir)
        result_path = 'D:/insectsDCT_datasets/insectsModelCSV/' + csvFile
        threshold[0] = conf
        if 'train' in csvFile:
            label_path = 'D:/insectsDCT_datasets/' + label_dir + '/train/labels/'
        else:
            label_path = 'D:/insectsDCT_datasets/' + label_dir + '/val/labels/'
            
        recallAll, precisionAll, f1scoreAll, recallMicro, precisionMicro, f1scoreMicro = predictTestData(label_path, label_dir, result_path, threshold)
        result.append([csvFile, recallAll, precisionAll, f1scoreAll, recallMicro, precisionMicro, f1scoreMicro])
        idx += 1
        
    #printStat(result, test_dics)

"""
TH:50, YOLOv5
Macro Avg Recall   : 0.299
Macro Avg Precision: 0.573
Macro Avg F1-score : 0.388
Micro Avg Recall   : 0.387
Micro Avg Precision: 0.774
Micro Avg F1-score : 0.516
TH:50, MRGBv5c2
Macro Avg Recall   : 0.577
Macro Avg Precision: 0.690
Macro Avg F1-score : 0.624
Micro Avg Recall   : 0.649
Micro Avg Precision: 0.784
Micro Avg F1-score : 0.710
"""