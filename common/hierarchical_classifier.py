import os
import pickle
import cv2
import pandas as pd
from scipy.stats import norm
from resnet50tf import ResNet50
from skimage import io
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

label_file = "../models_save/HierarchicalLabels3L_15052025.pkl"

with open(label_file, 'rb') as f:
    image_path_list, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, trainL1, trainL2, trainL3, valL1, valL2, valL3 = pickle.load(f)
    print("Labels and hierarchy dependency loaded from ", label_file)
    print("=============================================================================================")
    print("L1 classes", labelsL1)
    print("=============================================================================================")
    print("L2 classes", labelsL2)
    print("=============================================================================================")
    print("L3 classes", labelsL3)
    print("=============================================================================================")
    print("L2 -> L1 dependency", hierarchyL1)
    print("=============================================================================================")
    print("L3 -> L2 dependency", hierarchyL2)
    print("=============================================================================================")

class HierarchicalClassifier:

    def __init__(self, img_size=128, img_depth=3, device='cpu'):
        self.img_size = img_size
        self.img_depth = img_depth
        self.device = device
 
    def loadmodel(self, model_weights, threshold_file):
            
        model = ResNet50(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
        model.load_state_dict(torch.load(model_weights, map_location=self.device))
        print('Loaded model: ', model_weights)    
        model = model.to(self.device)

        data_thresholds = pd.read_csv(threshold_file)
        self.levels = data_thresholds["Level"].to_list() 
        self.labels = data_thresholds["ClassName"].to_list()
        self.thresholds = data_thresholds["Threshold"].to_list()
        self.means = data_thresholds["Mean"].to_list()
        self.stds = data_thresholds["Std"].to_list()
        #print(self.labels)                   
        self.hierarchyL1 = hierarchyL1
        self.hierarchyL1 = hierarchyL2
        self.labelsL1 = labelsL1
        self.labelsL2 = labelsL2
        self.labelsL3 = labelsL3
        self.model = model
        
        return model
    
    def resize_image(self, img, size=(28,28)):
    
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
    
    def createBatch(self, batch_size):
        self.batch_idx = 0
        self.batch_size = batch_size
        self.imagesInBatch = torch.FloatTensor(batch_size, self.img_depth, self.img_size, self.img_size) 
        #print("Batch created of size", self.batch_size)
        
    def appendToBatch(self, imageCrop):
        if self.batch_idx < self.batch_size:
            image = self.resize_image(imageCrop, (self.img_size, self.img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = transforms.ToTensor()(image) #.unsqueeze_(0)
            self.imagesInBatch[self.batch_idx] = image/255.0
            self.batch_idx += 1
            return True
    
        return False
    
    def classifyBatch(self):            
        self.imagesInBatch = self.imagesInBatch.to(self.device)
        level1class_pred, level2class_pred, level3class_pred = self.model(self.imagesInBatch)
        
        #level1class_pred = nn.Softmax(dim=1)(level1class_pred)
        #level2class_pred = nn.Softmax(dim=1)(level2class_pred)
        #level3class_pred = nn.Softmax(dim=1)(level3class_pred)
        
        level1class_pred = level1class_pred.cpu().detach().numpy()
        level2class_pred = level2class_pred.cpu().detach().numpy()
        level3class_pred = level3class_pred.cpu().detach().numpy()
        predicted_labels1 = np.argmax(level1class_pred, axis=1)
        predicted_labels2 = np.argmax(level2class_pred, axis=1)
        predicted_labels3 = np.argmax(level3class_pred, axis=1)
        
        lines = []
        for idx in range(len(predicted_labels1)):
            predicted_label1 = predicted_labels1[idx]
            predicted_label2 = predicted_labels2[idx]
            predicted_label3 = predicted_labels3[idx]
            #confidence_value = norm.cdf(predictions[idx][predicted_label], self.means[predicted_label], self.stds[predicted_label])
            #confidence_value = round(confidence_value*10000)/100
            #if predictions[idx][predicted_label] >= self.thresholds[predicted_label]:
            #    sure_label = True
            #else:
            #    sure_label = False
            line = f"{self.labelsL1[predicted_label1]},{predicted_label1},{self.labelsL2[predicted_label2]},{predicted_label2},{self.labelsL3[predicted_label3]},{predicted_label3}"
            print(line)
            lines.append(line)
            
        return lines
    
    def makePrediction(self, im, number=1):
        #resizedImg = cv2.resize(im, self.dim, interpolation = cv2.INTER_AREA)
        #print('Resized Dimensions : ', resizedImg.shape)
        #rgbImg = cv2.cvtColor(resizedImg, cv2.COLOR_RGB2BGR)
        #img = tf.keras.preprocessing.image.img_to_array(rgbImg)
        img = np.expand_dims(im, axis = 0)
        
        predictions = self.model.predict(img, verbose=1)
        predictions = predictions[0]
        probability = np.amax(predictions)
        index = np.where(predictions == probability)
        idx = index[0][0]
        print(predictions, probability, idx, self.species[idx])
        return idx, self.species[idx], probability 


    def makePredictionFile(self, imgFile):
        img = io.imread(imgFile)
        img = resize(img, self.dim, anti_aliasing = True)
        #img = tf.keras.preprocessing.image.img_to_array(img)
        img_pre = np.expand_dims(img, axis = 0)
        
        predictions = self.model.predict(img_pre)
        #print(decode_predictions(predictions, top=2)[0])
        predictions = predictions[0]
        probability = np.amax(predictions)
        index = np.where(predictions == probability)
        idx = index[0][0]
        #print(predictions, probability, idx, self.species[idx])
        print(predictions, self.species[idx])
        return idx, self.species[idx]
    
#%% MAIN for testing HierarchicalClassifier class
if __name__=='__main__':
    
    classifier = HierarchicalClassifier(img_size=128, device='cuda:0')
    classifier.loadmodel("../models_save/HierarchicalClassifier_15052025.pth", 
                         "../models_save/HierarchicalThresholds_15052025.csv")
    
    dataset_path = "/ArthropodsDataset/NI2classes/"
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
                classifier.appendToBatch(image)
                if count % batch_size == 0:
                    lines = classifier.classifyBatch()
                    classifier.createBatch(batch_size)
        
    