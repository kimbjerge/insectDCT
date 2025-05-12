'''Pytorch dataset loading script from directories.
'''

#import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from hierarchical_loader import HierarchicalDatasetLoader

class LoadDataset(Dataset):
    '''Reads the directory of image files and loads the data.
    '''

    def __init__(self, hierarchicalDataset: HierarchicalDatasetLoader, image_size=112, image_depth=3, return_label=True, transform=None, validate=False):
        '''Init param.
        '''
        self.image_size = image_size
        self.image_depth = image_depth
        self.return_label = return_label
        self.transform = transform
        self.hierarchicalDataset = hierarchicalDataset
        self.hierarchyL1, self.hierarchyL2, self.labelsL1, self.labelsL2, self.labelsL3 = self.hierarchicalDataset.get_hierarchy_labels()
        self.unspecified = 'Unspecified'
        self.data_list = self.hierarchicalDataset.get_data_list(validate)
        
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

 
    def getLabelIndex(self, classL, labels):
        if classL in labels:
            return labels.index(classL)
        else:
            return -1 # Unknown class


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
            # Handling of non squared images to avoid distortion
            image = self.resize_image(image, size=(self.image_size, self.image_size))
            #image = cv2.resize(image, (self.image_size, self.image_size)) # ,cv2.INTER_LINEAR default
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

    
    def get_cls_num_list(self, level):
        
        return self.hierarchicalDataset.get_cls_num_list(level)      

    
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

