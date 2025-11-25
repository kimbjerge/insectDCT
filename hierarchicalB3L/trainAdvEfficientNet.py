'''Train script.
'''

import os
import pickle
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
#from torchsummary import summary
from torchvision import transforms

from runtime_args_EfficientNet import args
from hierarchical_loader import HierarchicalDatasetLoader
from hierarchical_loss import HierarchicalLossNetwork
from load_dataset_hierarchical import LoadDataset
from resnet50tf import ResNet50
from convNext import ConvNextBase
from efficientNet import EfficientNet
from helper import calculate_accuracy
from balanced_softmax_loss import BalancedSoftmaxLoss

def checkHierarcy(HLN, level1_pred, level2_pred, level3_pred):    
         
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
    
    countWrongHierarchy = sum(map(lambda x : x == False, checkList))
    
    return countWrongHierarchy, len(level3p)


def trainModel(alpha, save_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
    
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
        print("Directory created", save_path)

    print(args)
            
    image_path_list = []
    for subdir in args.path_list.split(','): # Scan subdirectories with datasets
        image_path_list.append(args.data_path+subdir)
        
    hierarchicalDataset = HierarchicalDatasetLoader(image_path_list, split_validate=args.split) # default 10% used for validation
    
    hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3 = hierarchicalDataset.get_hierarchy_labels()
    trainL1 = hierarchicalDataset.get_cls_num_list(0, countIdx=0)
    trainL2 = hierarchicalDataset.get_cls_num_list(1, countIdx=0)
    trainL3 = hierarchicalDataset.get_cls_num_list(2, countIdx=0)
    valL1 = hierarchicalDataset.get_cls_num_list(0, countIdx=1)
    valL2 = hierarchicalDataset.get_cls_num_list(1, countIdx=1)
    valL3 = hierarchicalDataset.get_cls_num_list(2, countIdx=1)
    with open(save_path + 'labelsAdv3L.pkl', 'wb') as f:
            objs = [image_path_list,
                    hierarchyL1, # Hierarchy level dependency L2 -> L1
                    hierarchyL2, # Hierarchy level dependency L3 -> L2
                    labelsL1, # Labels at hierarchical level 1 (order)
                    labelsL2, # Labels at hierarchical level 2 (family)
                    labelsL3, # Labels at hierarchical level 3 (genus/species)
                    trainL1, # Number of training images for labels at level 1
                    trainL2, # Number of training images for labels at level 2
                    trainL3, # Number of training images for labels at level 3
                    valL1, # Number of validation images for labels at level 1
                    valL2, # Number of validation images for labels at level 2
                    valL3, # Number of validation images for labels at level 3
                    ]
            pickle.dump(objs, f)
            print("Hierarchical labels labelsAdv3L.pkl saved in", save_path)
            
    train_dataset = LoadDataset(hierarchicalDataset, image_size=args.img_size, image_depth=args.img_depth, 
                                transform=transforms.Compose([transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomPerspective(distortion_scale=0.2),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.ToTensor()]))
    test_dataset = LoadDataset(hierarchicalDataset, image_size=args.img_size, image_depth=args.img_depth, 
                               transform=transforms.ToTensor(),
                               validate=True)
    
    train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.model == 'ConvNextBase':
        model = ConvNextBase(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
        print("Training ConvNext-Base model")
    else: 
        if args.model == "EfficientNet":
            model = EfficientNet(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
            print("Training EfficientNet_v2 model")
        else:        
            model = ResNet50(num_classes=[len(labelsL1), len(labelsL2), len(labelsL3)], simple=True) 
            print("Training ResNet50 model")

    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        print('Adam learning rate', args.learning_rate)
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        print('SGD learning rate', args.learning_rate)
        print('SGD momentum', args.momentum)
        print('SGD weight decay', args.weight_decay)
    if optimizer is None:
        print('Wrong optimizer specified', args.optimizer)
    
    if args.loss_function == "Balanced":
        cls_num_L1 = train_dataset.get_cls_num_list(0)
        lossFnL1 = BalancedSoftmaxLoss(cls_num_list=cls_num_L1, reduction='mean') # Best loss function for LT datasets
        cls_num_L2 = train_dataset.get_cls_num_list(1)
        lossFnL2 = BalancedSoftmaxLoss(cls_num_list=cls_num_L2, reduction='mean') # Best loss function for LT datasets
        cls_num_L3 = train_dataset.get_cls_num_list(2)
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
        
    model = model.to(device)
    HLN = HierarchicalLossNetwork(hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, 
                                  [lossFnL1, lossFnL2, lossFnL3], 
                                  total_level=3, alpha=alpha, device=device, simple=True) 
    
    train_epoch_loss = []
    train_epoch_level1class_accuracy = []
    train_epoch_level2class_accuracy = []
    train_epoch_level3class_accuracy = []
    
    test_epoch_loss = []
    test_epoch_level1class_accuracy = []
    test_epoch_level2class_accuracy = []
    test_epoch_level3class_accuracy = []
    test_epoch_countWrongHierarchy = []
    test_epoch_countWrongPercentage = []
    test_minimum_loss = 1000
    best_epoch_idx = 0
    
    for epoch_idx in range(args.epoch):
    
        i = 0
    
        epoch_loss = []
        epoch_level1class_accuracy = []
        epoch_level2class_accuracy = []
        epoch_level3class_accuracy = []
    
        model.train()
        for i, sample in tqdm(enumerate(train_generator)):
    
    
            batch_x, batch_y1, batch_y2, batch_y3 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device)
            optimizer.zero_grad()
    
            leve1class_pred, level2class_pred, level3class_pred = model(batch_x)
            prediction = [leve1class_pred, level2class_pred, level3class_pred]
            dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2, batch_y3])
            lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2, batch_y3])
    
            total_loss = lloss + dloss
    
            total_loss.backward()
            optimizer.step()
            epoch_loss.append(total_loss.item())
            epoch_level1class_accuracy.append(calculate_accuracy(predictions=prediction[0].detach(), labels=batch_y1))
            epoch_level2class_accuracy.append(calculate_accuracy(predictions=prediction[1].detach(), labels=batch_y2))
            epoch_level3class_accuracy.append(calculate_accuracy(predictions=prediction[2].detach(), labels=batch_y3))
    
    
        train_epoch_loss.append(sum(epoch_loss)/(i+1))
        train_epoch_level1class_accuracy.append(sum(epoch_level1class_accuracy)/(i+1))
        train_epoch_level2class_accuracy.append(sum(epoch_level2class_accuracy)/(i+1))
        train_epoch_level3class_accuracy.append(sum(epoch_level3class_accuracy)/(i+1))
     
    
        print(f'Training Loss at epoch {epoch_idx+1} : {sum(epoch_loss)/(i+1)}')
        print(f'Training level 1 accuracy at epoch {epoch_idx+1} : {sum(epoch_level1class_accuracy)/(i+1)}')
        print(f'Training level 2 accuracy at epoch {epoch_idx+1} : {sum(epoch_level2class_accuracy)/(i+1)}')
        print(f'Training level 3 accuracy at epoch {epoch_idx+1} : {sum(epoch_level3class_accuracy)/(i+1)}')
    
        j = 0
    
        epoch_loss = []
        epoch_level1class_accuracy = []
        epoch_level2class_accuracy = []
        epoch_level3class_accuracy = []
        level1_pred = []
        level2_pred = []
        level3_pred = []
    
        model.eval()
        with torch.set_grad_enabled(False):
            for j, sample in tqdm(enumerate(test_generator)):
    
    
                batch_x, batch_y1, batch_y2, batch_y3 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device)
    
                level1class_pred, level2class_pred, level3class_pred = model(batch_x)
                
                level1_pred = level1_pred + level1class_pred.tolist()
                level2_pred = level2_pred + level2class_pred.tolist()
                level3_pred = level3_pred + level3class_pred.tolist()
                
                prediction = [level1class_pred, level2class_pred, level3class_pred]
                dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2, batch_y3])
                lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2, batch_y3])
    
                total_loss = lloss + dloss
    
                epoch_loss.append(total_loss.item())
                epoch_level1class_accuracy.append(calculate_accuracy(predictions=prediction[0], labels=batch_y1))
                epoch_level2class_accuracy.append(calculate_accuracy(predictions=prediction[1], labels=batch_y2))
                epoch_level3class_accuracy.append(calculate_accuracy(predictions=prediction[2], labels=batch_y3))
    
    
        test_loss_avg = sum(epoch_loss)/(j+1)
        test_epoch_loss.append(test_loss_avg)
        test_epoch_level1class_accuracy.append(sum(epoch_level1class_accuracy)/(j+1))
        test_epoch_level2class_accuracy.append(sum(epoch_level2class_accuracy)/(j+1))
        test_epoch_level3class_accuracy.append(sum(epoch_level3class_accuracy)/(j+1))
    
        countWrongHierarchy, testSize = checkHierarcy(HLN, level1_pred, level2_pred, level3_pred)
        countWrongPercentage = countWrongHierarchy/testSize     
        test_epoch_countWrongHierarchy.append(countWrongHierarchy)
        test_epoch_countWrongPercentage.append(countWrongPercentage)
     
   
        print(f'Testing Loss at epoch {epoch_idx+1} : {sum(epoch_loss)/(j+1)}')
        print(f'Testing level1class accuracy at epoch {epoch_idx+1} : {sum(epoch_level1class_accuracy)/(j+1)}')
        print(f'Testing level2class accuracy at epoch {epoch_idx+1} : {sum(epoch_level2class_accuracy)/(j+1)}')
        print(f'Testing level3class accuracy at epoch {epoch_idx+1} : {sum(epoch_level3class_accuracy)/(j+1)}')
        print("Testing wrong hierarchy predictions %d/%d (%.5f)" % (countWrongHierarchy, testSize, countWrongPercentage))
        print('-------------------------------------------------------------------------------------------')
       
        src_model_file_old  = save_path + 'dhc' + str(epoch_idx) + '.pth'
        if os.path.exists(src_model_file_old):
            os.remove(src_model_file_old)
            
        src_model_file = save_path + 'dhc' + str(epoch_idx+1) + '.pth'
        torch.save(model.state_dict(), src_model_file)
        print("Model saved in ", src_model_file)

        if test_minimum_loss > test_loss_avg:
            best_epoch_idx = epoch_idx+1
            test_minimum_loss = test_loss_avg
            best_model_file = save_path + 'dhc_best.pth'
            shutil.copyfile(src_model_file, best_model_file)
         
        with open(save_path + 'resultsAdv3L.pkl', 'wb') as f:
            objs = [epoch_idx,
                    best_epoch_idx,
                    train_epoch_level1class_accuracy, 
                    train_epoch_level2class_accuracy,
                    train_epoch_level3class_accuracy,
                    train_epoch_loss,
                    test_epoch_level1class_accuracy,
                    test_epoch_level2class_accuracy,
                    test_epoch_level3class_accuracy,
                    test_epoch_loss,
                    test_epoch_countWrongHierarchy,
                    test_epoch_countWrongPercentage,
                    testSize
                    ]
            pickle.dump(objs, f)
            print("Train and test accuracy and loss resultsAdv3L.pkl saved in", save_path)
                    

        with open(save_path + 'optimizerAdv3L.pkl', 'wb') as f:
            objs = [args.optimizer,
                    args.learning_rate,
                    args.momentum,
                    args.weight_decay,
                    alpha]
            pickle.dump(objs, f)
            print("Optimizer optimizerAdv3L.pkl settings saved in", save_path)
        
        
#%% MAIN
if __name__=='__main__':
    
    trainModel(alpha=0.5, save_path=args.model_save_path)

