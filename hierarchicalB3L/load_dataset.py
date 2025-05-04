'''Pytorch dataset loading script from directories.
'''

import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from level_NI_dict import hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3

class LoadDataset(Dataset):
    '''Reads the directory of image files and loads the data.
    '''

    def __init__(self, image_path, image_size=112, image_depth=3, return_label=True, transform=None):
        '''Init param.
        '''

        assert os.path.exists(image_path), 'The given image path must be valid!'

        self.data_path = image_path
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.transform = transform
        self.labelsL1 = labelsL1
        self.labelsL2 = labelsL2
        self.labelsL3 = labelsL3
        self.hierarchyL1 = hierarchyL1
        self.hierarchyL2 = hierarchyL2
        self.unspecified = 'Unspecified'
        self.data_list = self.path_to_list()
        
        #check if the hierarchy dictionary is consistent
        for k,v in self.hierarchyL1.items():
            if not k in self.labelsL1:
                print(f"Level1 class missing! {k}")
            for subclass in v:
                if not subclass in self.labelsL2:
                    print(f"Level2 class missing! {subclass}")
                    
        #check if the hierarchy dictionary is consistent
        for k,v in self.hierarchyL2.items():
            if not k in self.labelsL2:
                print(f"Level2 class missing! {k}")
            for subclass in v:
                if not subclass in self.labelsL3:
                    print(f"Level3 class missing! {subclass}")


    def path_to_list(self):
        '''Reads the path of the file and its corresponding label
        '''
        data = []
        for dirName in os.listdir(self.data_path):
            labelL1 = self.unspecified
            labelL2 = self.unspecified
            labelL3 = self.unspecified
            dirSplit = dirName.split(' ')
            lenSplit = len(dirSplit)
            if lenSplit > 0:
                labelL1 = dirSplit[0] # Order
                if not labelL1 in self.labelsL1:
                    print(f"Level1 class order wrong {labelL1}")
            if lenSplit > 1:
                labelL2 = dirSplit[1] # Family
                if not labelL2 in self.labelsL2:
                    print(f"Level2 class family wrong {labelL2}")
            if lenSplit == 3:
                labelL3 = dirSplit[2] # Genius
                if not labelL3 in self.labelsL3:
                    print(f"Level3 class genius wrong {labelL3}")
            if lenSplit == 4:
                labelL3 = dirSplit[2] + ' ' + dirSplit[3] # Genius + species
                if not labelL3 in self.labelsL3:
                    print(f"Level3 class species wrong {labelL3}")

            filePath = self.data_path + '/' + dirName
            for fileName in os.listdir(filePath):
                if fileName.endswith('.jpg') or fileName.endswith('.JPG'):
                    record = [filePath + '/' + fileName, labelL1, labelL2, labelL3]
                    data.append(record)
            
        return data


    def getLabelIndex(self, classL, labels):
        if classL in labels:
            return labels.index(classL)
        else:
            return -1 # Unknown class


    def __len__(self):
        '''Returns the total amount of data.
        '''
        return len(self.data_list)

    def __getitem__(self, idx):
        '''Returns a single item.
        '''
        image_path, classL1, classL2, classL3 = self.data_list[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
            image = np.zeros((self.image_size,self.image_size,self.image_depth), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
 
        labelL1index = self.getLabelIndex(classL1, self.labelsL1)
        labelL2index = self.getLabelIndex(classL2, self.labelsL2)
        labelL3index = self.getLabelIndex(classL3, self.labelsL3)
        
        return {
            'image':image/255.0,
            'label_1': labelL1index,
            'label_2': labelL2index,
            'label_3': labelL3index
        }
        
    # NOT USED KBE???
    def getitemFlexible(self, idx):
        '''Returns a single item.
        '''
        image_path, image, classL1, classL2, classL3 = None, None, None, None, None
        if self.return_label:
            image_path, classL1, classL2, classL3 = self.data_list[idx]
        else:
            image_path = self.data_list[idx]

        if self.image_depth == 1:
            image = cv2.imread(image_path, 0)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size != 32:
            image = cv2.resize(image, (self.image_size, self.image_size))


        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.return_label:
            return {
                'image':image/255.0,
                'label_1': self.labelsL1.index(classL1),
                'label_2': self.labelsL2.index(classL2),
                'label_3': self.labelsL3.index(classL3)
            }
        else:
            return {
                'image':image
            }

