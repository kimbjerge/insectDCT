# -*- coding: utf-8 -*-
"""
Created on Sat May 10 08:57:57 2025

@author: Kim Bjerge

Analysis hierarchical dataset specified distributed in different directories 
specified by directory paths in image_path_list

The HierarchicalDatasetLoader creates labels on three levels of taxon
including: order, family, genus/species

"""
import os

class HierarchicalDatasetLoader():
    '''Reads the directory of image files and create hierarchy labels
    '''

    def __init__(self, image_path_list, split_validate): # Percentage used for validation
        
        self.split_validate = int(100/split_validate)
        self.data_path_list = image_path_list
        self.path_names = self.create_labels_file_list()
        self.create_hierarchy_labels()
    
    
    def create_labels_file_list(self):
        
        path_names = {}
        for file_path in self.data_path_list:
            for path_name in sorted(os.listdir(file_path)):
                full_path_name = file_path + '/' + path_name
                if not path_name in path_names.keys():                
                    path_names[path_name] = []
                path_names[path_name].append(full_path_name)
        
        return path_names
    
    
    def get_path_names(self):
        
        return self.path_names
    
    
    def create_hierarchy_labels(self):
        
        # Labels on levels L1 (order), L2 (family), L3 (genus/species)
        self.labelsL1 = [] 
        self.labelsL2 = []
        self.labelsL3 = []
        # Hierarchy label dependcy L2 -> L1 and L3 -> L2
        self.hierarchyL1 = {} 
        self.hierarchyL2 = {}
        # List of filenames with path and level labels [filename, L1, L2, L2]
        self.data_list = []
        self.data_list_val = []
        # Directories with label keys and number of images in each class
        classListL1 = {}
        classListL2 = {}
        classListL3 = {}
        
        count = 0
        countHymen = 0
        for dirName in self.path_names.keys():
            dirSplit = dirName.split(' ')
            lenSplit = len(dirSplit) # File name syntax: "<order> <family> <genus> <species>"
            if lenSplit > 0:
                labelL1 = dirSplit[0] # Order
                labelL2 = labelL1
                labelL3 = labelL2
            if lenSplit > 1:
                labelL2 = dirSplit[1] # Family
                labelL3 = labelL2
            if lenSplit == 3:
                labelL3 = dirSplit[2] # Genus
            if lenSplit == 4:
                labelL3 = dirSplit[2] + ' ' + dirSplit[3] # Genius + species (Same level L3)

            if not labelL1 in self.hierarchyL1.keys():
                self.hierarchyL1[labelL1] = []
            if not labelL2 in self.hierarchyL1[labelL1]:
                self.hierarchyL1[labelL1].append(labelL2)
            if not labelL2 in self.hierarchyL2.keys():
                self.hierarchyL2[labelL2] = []
            if not labelL3 in self.hierarchyL2[labelL2]:
                self.hierarchyL2[labelL2].append(labelL3)
                
            if not labelL1 in self.labelsL1:
                self.labelsL1.append(labelL1)
                #print(f"L1 {labelL1}")
            if not labelL2 in self.labelsL2:
                self.labelsL2.append(labelL2)
                #print(f"L2 {labelL2}")
            if not labelL3 in self.labelsL3:
                self.labelsL3.append(labelL3)
                #print(f"L3 {labelL3}")
                
            filePaths = self.path_names[dirName]
            for filePath in filePaths:
                for fileName in sorted(os.listdir(filePath)):
                    if fileName.endswith('.jpg') or fileName.endswith('.JPG'):
                        record = [filePath + '/' + fileName, labelL1, labelL2, labelL3]
                        
                        """
                        if labelL2 == "Hymenoptera":
                            countHymen += 1
                            if countHymen == 1:
                                print(countHymen, labelL1, labelL2, labelL3)
                            if countHymen == 42:
                                print(countHymen, labelL1, labelL2, labelL3)
                        """
                         
                        count += 1
                        if count % self.split_validate == 0:
                            self.data_list_val.append(record) # Use image for validation
                            num_images = [0, 1] # Validate image
                        else:   
                            self.data_list.append(record) # Use image for training
                            num_images = [1, 0] # Train image
                        
                        # Counting number of samples for each class at each level
                        if labelL1 in classListL1.keys():
                            classListL1[labelL1][0] += num_images[0] # Train images
                            classListL1[labelL1][1] += num_images[1] # Validate images
                        else:
                            classListL1[labelL1] = num_images.copy()
                            
                        if labelL2 in classListL2.keys():
                            classListL2[labelL2][0] += num_images[0]
                            classListL2[labelL2][1] += num_images[1]
                        else:
                            classListL2[labelL2] = num_images.copy()
                            
                        if labelL3 in classListL3.keys():
                            classListL3[labelL3][0] += num_images[0]
                            classListL3[labelL3][1] += num_images[1]
                        else:
                            classListL3[labelL3] = num_images.copy() 
                
        self.labelsL1 = sorted(self.labelsL1)
        self.labelsL2 = sorted(self.labelsL2)
        self.labelsL3 = sorted(self.labelsL3)
        self.classListL1 = classListL1
        self.classListL2 = classListL2
        self.classListL3 = classListL3
    
    def get_hierarchy_labels(self):
        return self.hierarchyL1, self.hierarchyL2, self.labelsL1, self.labelsL2, self.labelsL3


    def get_data_list(self, validate):
        if validate:
            return self.data_list_val
        else:
            return self.data_list
    
    
    def get_cls_num_list(self, level, countIdx=0):# 0 Train images, 1 Validate images
        
        cls_num_list = []
        if level == 0:
            cls_num_list = [self.classListL1[labelName][countIdx] for labelName in self.labelsL1]
        if level == 1:
            cls_num_list = [self.classListL2[labelName][countIdx] for labelName in self.labelsL2]
        if level == 2:
            cls_num_list = [self.classListL3[labelName][countIdx] for labelName in self.labelsL3]
            
        return cls_num_list
        
    
#%% MAIN for testing HierarchicalDatasetLoader class
if __name__=='__main__':
    
    image_path_list = ['/ArthropodsDataset/NI2classes',
                       '/ArthropodsDataset/NI2sorted',
                       '/ArthropodsDataset/NItrain',
                       '/ArthropodsDataset/NIval'
                       ]
    image_path_listV1 = ['/home/don/data/Arthropods/NI2classes',
                       '/home/don/data/Arthropods/NI2sorted',
                       '/home/don/data/Arthropods/NItrain',
                       '/home/don/data/Arthropods/NIval'
                       ]
    image_path_listV2 = ['/home/don/data/Arthropods/NI2classesV2',
                       '/home/don/data/Arthropods/NI2sortedV2',
                       '/home/don/data/Arthropods/NItrainV2',
                       '/home/don/data/Arthropods/NIvalV2'
                       '/home/don/data/Arthropods/sorted_orchard_crops'
                       ]
    image_path_list = ['/home/don/data/Arthropods/NI2classesV2',
                       '/home/don/data/Arthropods/NI2sortedV2',
                       '/home/don/data/Arthropods/NItrainV2',
                       '/home/don/data/Arthropods/NIvalV2',
                       '/home/don/data/Arthropods/sorted_orchard_crops',
                       '/home/don/data/Arthropods/NI2_MAMBO',
                       '/home/don/data/Arthropods/GBIF_crops'
                      ]

    datasetLoader = HierarchicalDatasetLoader(image_path_list, 10)
    path_names = datasetLoader.get_path_names()
    print("=============================================================================================")
    print(path_names)
    print("=============================================================================================")
    print("Class directory hierarchy with 3 levels: order - family - genus/species")
    print("-----------------------------------------------------------------------")
    for name in sorted(path_names.keys()):
        print(name)
    hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3 = datasetLoader.get_hierarchy_labels()
    data_set = datasetLoader.get_data_list(validate=False)
    print("=============================================================================================")
    #print("Labels L1", labelsL1, datasetLoader.get_cls_num_list(0), datasetLoader.get_cls_num_list(0, 1))
    print("Level 1               train validate")
    [print(f'{l:<20} {t:>6} {v:>6}') for l, t, v in zip(labelsL1, datasetLoader.get_cls_num_list(0), datasetLoader.get_cls_num_list(0, 1))]
    print("=============================================================================================")
    print("Level 2               train validate")
    [print(f'{l:<20} {t:>6} {v:>6}') for l, t, v in zip(labelsL2, datasetLoader.get_cls_num_list(1), datasetLoader.get_cls_num_list(1, 1))]
    #print("Labels L2", labelsL2, datasetLoader.get_cls_num_list(1), datasetLoader.get_cls_num_list(1, 1))
    print("=============================================================================================")
    print("Level 3                         train validate")
    [print(f'{l:<30} {t:>6} {v:>6}') for l, t, v in zip(labelsL3, datasetLoader.get_cls_num_list(2), datasetLoader.get_cls_num_list(2, 1))]
    #print("Labels L3", labelsL3, datasetLoader.get_cls_num_list(2), datasetLoader.get_cls_num_list(2, 1))
    
