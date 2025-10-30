# -*- coding: utf-8 -*-
"""
Modified on Sat April 10 18:19:03 2025

@author: Kim Bjerge
"""

import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torchsummary import summary
from torchvision import transforms

from runtime_args import args
from load_dataset_hierarchical import LoadDataset
from hierarchical_loader import HierarchicalDatasetLoader
from hierarchical_loss import HierarchicalLossNetwork
from resnet50tf import ResNet50
from convNext import ConvNextBase
from helper import calculate_accuracy
#from plot import plot_loss_acc
from balanced_softmax_loss import BalancedSoftmaxLoss

#%% MAIN
if __name__=='__main__':
   
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    
    print(args)

    if not os.path.exists(args.model_test_path) : 
        print("Model to test do not exist", args.model_test_path)
        exit(1)

    image_path_list = []
    for subdir in args.path_list_test.split(','): # Scan subdirectories with datasets
        image_path_list.append(args.data_path_test+subdir)
    
    hierarchicalDataset = HierarchicalDatasetLoader(image_path_list, split_validate=100) # default 100% used for validation
    #hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3 = hierarchicalDataset.get_hierarchy_labels()
    
    # Load training labels
    with open(args.model_test_path+args.label_file, 'rb') as f:
        _, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, _, _, _, _, _, _ = pickle.load(f)
        print("Labels and hierarchy dependency loaded from ", args.model_test_path+args.label_file)
    
    # Overwrites hierarchy with training labels
    hierarchicalDataset.hierarchyL1 = hierarchyL1
    hierarchicalDataset.hierarchyL2 = hierarchyL2
    hierarchicalDataset.labelsL1 = labelsL1
    hierarchicalDataset.labelsL2 = labelsL2
    hierarchicalDataset.labelsL3 = labelsL3
    
    test_dataset = LoadDataset(hierarchicalDataset, image_size=args.img_size, image_depth=args.img_depth, 
                               transform=transforms.ToTensor(), validate=True)

    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.model == 'ConvNextBase':
        model = ConvNextBase(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
        print("Training ConvNext-Base model")
    else:        
        model = ResNet50(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
        print("Training ResNet50 model")
        
    model.load_state_dict(torch.load(args.model_test_path + args.weights, map_location=device))
    
    model = model.to(device)
    
    if args.loss_function == "Balanced":
        cls_num_L1 = test_dataset.get_cls_num_list(0)
        lossFnL1 = BalancedSoftmaxLoss(cls_num_list=cls_num_L1, reduction='mean') # Best loss function for LT datasets
        cls_num_L2 = test_dataset.get_cls_num_list(1)
        lossFnL2 = BalancedSoftmaxLoss(cls_num_list=cls_num_L2, reduction='mean') # Best loss function for LT datasets
        cls_num_L3 = test_dataset.get_cls_num_list(2)
        lossFnL3 = BalancedSoftmaxLoss(cls_num_list=cls_num_L3, reduction='mean') # Best loss function for LT datasets
        print("Using Balanced Softmax Loss Function")
        print("=====================================================================================")
        print("Class list L1:", labelsL1, cls_num_L1, sum(cls_num_L1))    
        print("=====================================================================================")
        print("Class list L2:", labelsL2, cls_num_L2, sum(cls_num_L2))    
        print("=====================================================================================")
        print("Class list L3:", labelsL3, cls_num_L3, sum(cls_num_L3))    
        print("=====================================================================================")
    else:
        lossFnL1 = nn.CrossEntropyLoss() # Standard cross-entropy loss function
        lossFnL2 = nn.CrossEntropyLoss() # Standard cross-entropy loss function
        lossFnL3 = nn.CrossEntropyLoss() # Standard cross-entropy loss function
        print("Using Softmax Cross-entropy Loss Function")
        
    HLN = HierarchicalLossNetwork(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3,
                                  [lossFnL1, lossFnL2, lossFnL3],
                                  total_level=3, device=device, simple=True)
     
    test_epoch_loss = []
    test_epoch_level1class_accuracy = []
    test_epoch_level2class_accuracy = []
    test_epoch_level3class_accuracy = []
     
    j = 0

    epoch_loss = []
    epoch_level1class_accuracy = []
    epoch_level2class_accuracy = []
    epoch_level3class_accuracy = []
    
    level1_pred = []
    level2_pred = []
    level3_pred = []
    level1_label = []
    level2_label = []
    level3_label = []
    
    model.eval()
    with torch.set_grad_enabled(False):
        for j, sample in tqdm(enumerate(test_generator)):

            batch_x, batch_y1, batch_y2, batch_y3 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device)

            level1class_pred, level2class_pred, level3class_pred = model(batch_x)
            prediction = [level1class_pred, level2class_pred, level3class_pred]
            dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2, batch_y3])
            lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2, batch_y3])

            total_loss = lloss + dloss

            epoch_loss.append(total_loss.item())
            epoch_level1class_accuracy.append(calculate_accuracy(predictions=prediction[0], labels=batch_y1))
            epoch_level2class_accuracy.append(calculate_accuracy(predictions=prediction[1], labels=batch_y2))
            epoch_level3class_accuracy.append(calculate_accuracy(predictions=prediction[2], labels=batch_y3))
            
            level1_pred = level1_pred + level1class_pred.tolist()
            level2_pred = level2_pred + level2class_pred.tolist()
            level3_pred = level3_pred + level3class_pred.tolist()
            level1_label = level1_label + batch_y1.tolist()
            level2_label = level2_label + batch_y2.tolist()
            level3_label = level3_label + batch_y3.tolist()


    test_epoch_loss.append(sum(epoch_loss)/(j+1))
    test_epoch_level1class_accuracy.append(sum(epoch_level1class_accuracy)/(j+1))
    test_epoch_level2class_accuracy.append(sum(epoch_level2class_accuracy)/(j+1))
    test_epoch_level3class_accuracy.append(sum(epoch_level3class_accuracy)/(j+1))

    #plot accuracy and loss graph
    #plot_loss_acc('./graph_folder/', num_epoch=[0], 
    #                train_accuracies_level1=test_epoch_level1class_accuracy, train_accuracies_level2=test_epoch_level2class_accuracy, 
    #                train_accuracies_level3=test_epoch_level3class_accuracy, train_losses=test_epoch_loss,
    #                test_accuracies_level1=test_epoch_level1class_accuracy, test_accuracies_level2=test_epoch_level2class_accuracy,
    #                test_accuracies_level3=test_epoch_level3class_accuracy, test_losses=test_epoch_loss)

    print(f'Testing Loss : {sum(epoch_loss)/(j+1)}')
    print(f'Testing level1class accuracy : {sum(epoch_level1class_accuracy)/(j+1)}')
    print(f'Testing level2class accuracy : {sum(epoch_level2class_accuracy)/(j+1)}')
    print(f'Testing level3class accuracy : {sum(epoch_level3class_accuracy)/(j+1)}')
    print('-------------------------------------------------------------------------------------------')
    
    predictedLablesFile = args.model_test_path + 'predictLabels3Ltest.pkl'
        
    with open(predictedLablesFile, 'wb') as f:
        objs = [level1_pred, 
                level2_pred,
                level3_pred,
                level1_label,
                level2_label,
                level3_label]
        pickle.dump(objs, f)
        print("Predictions and labels", predictedLablesFile)

    
