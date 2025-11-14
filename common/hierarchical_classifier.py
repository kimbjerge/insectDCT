
import os
import cv2
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
from scipy.stats import norm
#from skimage import io
#from skimage.transform import resize
from common.resnet50tf import ResNet50
from common.convNext import ConvNextBase
from common.efficientNet import EfficientNet
#from resnet50tf import ResNet50

class HierarchicalClassifier:

    def __init__(self, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, 
                  img_size=128, stdThreshold=0.0, img_depth=3, device='cpu'):
        self.img_size = img_size
        self.img_depth = img_depth
        self.device = device
        self.stdThreshold = stdThreshold
        self.hierarchical_labelsL1 = hierarchyL1
        self.hierarchical_labelsL2 = hierarchyL2
        self.labelsL1 = labelsL1
        self.labelsL2 = labelsL2
        self.labelsL3 = labelsL3
        self.numeric_hierarchyL1, self.numeric_hierarchyL2 = self.words_to_indices()
    
    def words_to_indices(self):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchyL1 = {}
        for k, v in self.hierarchical_labelsL1.items():
            numeric_hierarchyL1[self.labelsL1.index(k)] = [self.labelsL2.index(i) for i in v]

        numeric_hierarchyL2 = {}
        for k, v in self.hierarchical_labelsL2.items():
            numeric_hierarchyL2[self.labelsL2.index(k)] = [self.labelsL3.index(i) for i in v]

        return numeric_hierarchyL1, numeric_hierarchyL2 
    
    def check_hierarchy_list(self, level, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        if level == 1:
            bool_tensor = [current_level[i] in self.numeric_hierarchyL1[previous_level[i]] for i in range(len(previous_level))]
        if level == 2:
            bool_tensor = [current_level[i] in self.numeric_hierarchyL2[previous_level[i]] for i in range(len(previous_level))]

        return bool_tensor
    
    def checkHierarhcy(self, level1p, level2p, level3p):    
            
        checkL2 = self.check_hierarchy_list(level=1, current_level=level2p, previous_level=level1p)
        checkL3 = self.check_hierarchy_list(level=2, current_level=level3p, previous_level=level2p)
        
        checkList = checkL3 
        
        for idx in range(len(checkList)):
            if checkL3[idx] == False or checkL2[idx] == False:
                checkList[idx] = False
                #print(self.labelsL1[level1p[idx]], self.labelsL2[level2p[idx]], self.labelsL3[level3p[idx]])
        
        return checkList
    
    def getLabels(self, level, labelName):
        
        labelL3 = ''
        labelL2 = ''
        labelL1 = ''
        if level == 1:
            labelL1 = labelName
        if level == 2:
            labelL2 = labelName
            for i, (k, v) in enumerate(self.hierarchical_labelsL1.items()):
                if labelL2 in v:
                    labelL1 = k
        if level == 3:
            labelL3 = labelName
            for i, (k, v) in enumerate(self.hierarchical_labelsL2.items()):
                if labelL3 in v:
                    labelL2 = k
            for i, (k, v) in enumerate(self.hierarchical_labelsL1.items()):
                if labelL2 in v:
                    labelL1 = k                  
            
        return labelL1, labelL2, labelL3
        
    def findLabelIndex(self, level, label):

        for idx in range(len(self.labels)):
            if self.levels[idx] == level and self.labels[idx] == label:
                return idx
            
        print("Error not found label", label, "at level", level)
        return -1

    def checkAboveThreshold(self, level, output_score, predicted_label):
        
        if level == 1:
            label = self.labelsL1[predicted_label]
        if level == 2:
            label = self.labelsL2[predicted_label]
        if level == 3:
            label = self.labelsL3[predicted_label]
        
        predicted_index = self.findLabelIndex(level, label)
        
        if self.stdThreshold != 0.0: # Check if different threshold should be used
            mu = self.means[predicted_index]
            std = self.stds[predicted_index]
            classThreshold = round((mu - self.stdThreshold*std)*100)/100 # Rounded 0.01
        else:
            classThreshold = self.thresholds[predicted_index] # Use CSV file with thresholds
        
        if output_score >= classThreshold:
            sure_label = True
        else:
            sure_label = False
        
        if self.stds[predicted_index] > 0: # Standard deviation different from zero
            confidence_value = norm.cdf(output_score, self.means[predicted_index], self.stds[predicted_index])
        else:
            sure_label = True # Typically the case with one sample in the training dataset
            confidence_value  = norm.cdf(output_score, self.means[predicted_index], 10.0) # Setting a high std value
        
        return confidence_value, sure_label
        
    def resize_image(self, img, size=(128,128)):
    
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1 # Check for colored images
    
        if h == w: # Squared image
            if h > size[0]:
                return cv2.resize(img, size, cv2.INTER_AREA) # Shrink image
            else:
                return cv2.resize(img, size, cv2.INTER_CUBIC) # Enlarge image             
    
        dif = h if h > w else w
    
        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
    
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2
    
        if len(img.shape) == 2: # BW image
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else: # Colored image
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    
        return cv2.resize(mask, size, interpolation)
 
    # Loading model with model weights model names: ResNet50, ConvNextLarge, ConvNextBase, ConvNextSmall, ConvNextTiny, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
    def loadmodel(self, model_weights, threshold_file, modelName="ResNet50"):
        
        print("Load model", modelName)
        if modelName == "ResNet50":
            self.model = ResNet50(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True) 
        if modelName == "ConvNextLarge":
            self.model = ConvNextBase(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, ConvNext="Large") 
        if modelName == "ConvNextBase":
            self.model = ConvNextBase(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, ConvNext="Base") 
        if modelName == "ConvNextSmall":
            self.model = ConvNextBase(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, ConvNext="Small") 
        if modelName == "ConvNextTiny":
            self.model = ConvNextBase(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, ConvNext="Tiny")
        if modelName == "EfficientNetV2S":
            self.model = EfficientNet(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, effNet="small")
        if modelName == "EfficientNetV2M":
            self.model = EfficientNet(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, effNet="medium")
        if modelName == "EfficientNetV2L":
            self.model = EfficientNet(num_classes=[len(self.labelsL1), len(self.labelsL2), len(self.labelsL3)], simple=True, effNet="large")
            
        
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device)) 
        print('Loaded model: ', model_weights)   
        self.model.eval() # Very important! - else error in predictions
        self.model = self.model.to(self.device)

        data_thresholds = pd.read_csv(threshold_file)
        self.levels = data_thresholds["Level"].to_list() 
        self.labels = data_thresholds["ClassName"].to_list()
        self.thresholds = data_thresholds["Threshold"].to_list()
        self.means = data_thresholds["Mean"].to_list()
        self.stds = data_thresholds["Std"].to_list()
        
    def setmodel(self, model):
        
        self.model = model
        
    def createBatch(self, batch_size):
        self.batch_idx = 0
        self.batch_size = batch_size
        self.imagesInBatch = torch.FloatTensor(batch_size, self.img_depth, self.img_size, self.img_size) 
        #print("Batch created of size", self.batch_size)
        
    def appendToBatch(self, imageCrop):
        if self.batch_idx < self.batch_size:
            image = self.resize_image(imageCrop, (self.img_size, self.img_size))
            #image = cv2.resize(imageCrop, (self.img_size, self.img_size), cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.ToTensor()(image)
            self.imagesInBatch[self.batch_idx] = image
            self.batch_idx += 1
            return True
    
        return False
    
    def appendBatchToBatch(self, batch): 
        self.imagesInBatch = batch
    

    def classifyBatch(self, useSoftMax=True):       
        """
        Classify batch of images with hierarchal model without a following outlier and hierachy checking
        """
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        level1class_pred, level2class_pred, level3class_pred = self.model(self.imagesInBatch)
        
        if useSoftMax:
            level1class_pred = nn.Softmax(dim=1)(level1class_pred)
            level2class_pred = nn.Softmax(dim=1)(level2class_pred)
            level3class_pred = nn.Softmax(dim=1)(level3class_pred)
        
        predicted_labels1 = np.argmax(level1class_pred.cpu().detach().numpy(), axis=1)
        predicted_labels2 = np.argmax(level2class_pred.cpu().detach().numpy(), axis=1)
        predicted_labels3 = np.argmax(level3class_pred.cpu().detach().numpy(), axis=1)
        
        lines = []
        for idx in range(len(predicted_labels1)):
            predicted_label1 = predicted_labels1[idx]
            predicted_label2 = predicted_labels2[idx]
            predicted_label3 = predicted_labels3[idx]

            line = f"{self.labelsL1[predicted_label1]},{predicted_label1},{self.labelsL2[predicted_label2]},{predicted_label2},{self.labelsL3[predicted_label3]},{predicted_label3}"
            print(line)
            lines.append(line)
            
        return lines

    def makePrediction(self, imageCrop):
        """
        Classify single a image with the hierarchal model following outlier and hierachy checking
        """
        
        self.createBatch(1)
        self.appendToBatch(imageCrop)
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        
        level1class_pred, level2class_pred, level3class_pred = self.model(self.imagesInBatch)
                
        level1class_pred = level1class_pred.cpu().detach().numpy()
        level2class_pred = level2class_pred.cpu().detach().numpy()
        level3class_pred = level3class_pred.cpu().detach().numpy()
        
        predicted_labels1 = np.argmax(level1class_pred, axis=1)
        predicted_labels2 = np.argmax(level2class_pred, axis=1)
        predicted_labels3 = np.argmax(level3class_pred, axis=1)
        
        predicted_label1 = predicted_labels1[0]
        predicted_label2 = predicted_labels2[0]
        predicted_label3 = predicted_labels3[0]
  
        predicted_score1 = level1class_pred[0][predicted_label1]
        predicted_score2 = level2class_pred[0][predicted_label2]
        predicted_score3 = level3class_pred[0][predicted_label3]
 
        # Checking that predictions are correct according to the hierachical taxonomic levels
        checkList = self.checkHierarhcy(predicted_labels1, predicted_labels2, predicted_labels3)

        # Labeling predictions unsure if ouput scores are below outlier thredsholds
        conf1, sure1 = self.checkAboveThreshold(1, predicted_score1, predicted_label1)
        conf2, sure2 = self.checkAboveThreshold(2, predicted_score2, predicted_label2)
        conf3, sure3 = self.checkAboveThreshold(3, predicted_score3, predicted_label3)
 
        label_name = "Unsure" 
        predicted_label = -1 # Unsure label index - outlier detection
        confidence = 0.0
        level = 0
        
        # Get label from sure levels
        if sure2 and sure3:
            label_name = self.labelsL3[predicted_label3]
            predicted_label = predicted_label3
            confidence = conf3
            level = 3
        else:
            if sure1 and sure2:
                label_name = self.labelsL2[predicted_label2]
                predicted_label = predicted_label2
                confidence = conf2
                level = 2
            else:
                if sure1:
                    label_name = self.labelsL1[predicted_label1]
                    predicted_label = predicted_label1
                    confidence = conf1
                    level = 1
        
        if checkList[0] == False: # Wrong hierarchy - then unsure prediction
           predicted_label = -1 # Unsure label index, wrong hierarchy (-2)
           label_name =  "Unsure" 
           confidence = 0.0
           level = 0

        # Label1,LabelId1,Logit1,Conf1,Above1,Label2,LabelId2,Logit2,Conf2,Above2,Label3,LabelId3,Logit3,Conf3,Above3,Checked
        line1 = f"{self.labelsL1[predicted_label1]},{predicted_label1},{predicted_score1},{conf1},{sure1},"
        line2 = f"{self.labelsL2[predicted_label2]},{predicted_label2},{predicted_score2},{conf2},{sure2},"
        line3 = f"{self.labelsL3[predicted_label3]},{predicted_label3},{predicted_score3},{conf3},{sure3},"
        line = line1 + line2 + line3 + str(checkList[0])
        
        return line, level, predicted_label, label_name, confidence 
    
#%% MAIN for testing HierarchicalClassifier class
if __name__=='__main__':
    
    device = 'cuda:0'
    
    label_file = "../models_save/HierarchicalLabels3L_13052025.pkl"
    
    with open(label_file, 'rb') as f:
        _, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, _, _, _, _, _, _ = pickle.load(f)
        print("Labels and hierarchy dependency loaded from ", label_file)
        print("=============================================================================================")
        print("L1 classes", labelsL1, len(labelsL1))
        print("=============================================================================================")
        print("L2 classes", labelsL2, len(labelsL2))
        print("=============================================================================================")
        print("L3 classes", labelsL3, len(labelsL3))
        print("=============================================================================================")
        print("L2 -> L1 dependency", hierarchyL1)
        print("=============================================================================================")
        print("L3 -> L2 dependency", hierarchyL2)
        print("=============================================================================================")
    
    """
    count = 1
    for label in labelsL1:
        print("\t", str(count), " ", label)
        count += 1
    count = 1
    for label in labelsL2:
        print("\t", str(count), " ", label)       
        count += 1
    count = 1
    for label in labelsL3:
        print("\t", str(count), " ", label)
        count += 1
    """
    
    #classifier = HierarchicalClassifier(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, img_size=128, device='cpu')
    classifier = HierarchicalClassifier(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, img_size=128, device='cuda:0')
    classifier.loadmodel("../models_save/HierarchicalClassifier_13052025.pth", 
                         "../models_save/HierarchicalThresholds_13052025_TH2.csv")

    #dataset_path = "/ArthropodsDataset/NI2classes/"
    dataset_path = "/home/don/data/Arthropods/NI2classes/"
    
    count = 0
    batch_size = 10
    classifier.createBatch(batch_size)
    for class_dir in os.listdir(dataset_path):
        class_dir_path = dataset_path + class_dir
        for file_name in os.listdir(class_dir_path):
            if file_name.endswith(".jpg"):
                count += 1
                file_name_path = class_dir_path + '/' + file_name
                print(file_name_path)
                image = cv2.imread(file_name_path)
                # Test processing single images
                line, level, predicted_label, label_name, confidence = classifier.makePrediction(image)
                print("Predicted:", label_name, level, predicted_label, confidence)
                print(line)
                """ Test processing images in batch
                classifier.appendToBatch(image)
                if count % batch_size == 0:
                    lines = classifier.classifyBatch()
                    classifier.createBatch(batch_size)
                """
    