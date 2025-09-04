'''Deep Hierarchical Classifier using different ConvNext models
'''

import torch.nn as nn

from torchvision.models import convnext_large, ConvNeXt_Large_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

class ConvNextBase(nn.Module):
    '''ConvNext-Base Architecture with pretrained weights - default configuration
    '''

    def __init__(self, image_depth=3, num_classes=[20,100], simple=True, ConvNext="Base"): # Large, Base, Small, Tiny
        '''Params init and build arch.
        '''
        super(ConvNextBase, self).__init__()

        self.simple = simple
        self.out_channels = 256
        
        if ConvNext == "Large":
            self.model_ft = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1) 
            print("ConvNext_Large")
        if ConvNext == "Base":
            self.model_ft = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1) 
            print("ConvNext_Base")
        if ConvNext == "Small":
            self.model_ft = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1) 
            print("ConvNext_Small")
        if ConvNext == "Tiny":
            self.model_ft = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1) 
            print("ConvNext_Tiny")
           
    
        # overwrite the 'fc' layer
        num_in_features = self.model_ft.classifier[2].in_features
        print("In features", num_in_features)
        self.model_ft.classifier[2] =  nn.Identity() # Do nothing just pass input to output
 
        self.layers = len(num_classes)
        
        # At least one layer
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(num_in_features, self.out_channels)
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes[0])
        
        if self.simple:
            print("Simple: Dropout -> Linear (256) -> ReLU -> Linear (num_classe layer Lx)")
            print("Layer 1", num_classes[0])
 
            if self.layers > 1:
                self.linear_lvl2 = nn.Linear(num_in_features, self.out_channels)
                self.relu_lv2 = nn.ReLU(inplace=False)
                self.softmax_reg2 = nn.Linear(self.out_channels, num_classes[1])
                print("Layer 2", num_classes[1])
                
            if self.layers > 2:
                self.linear_lvl3 = nn.Linear(num_in_features, self.out_channels)
                self.relu_lv3 = nn.ReLU(inplace=False)
                self.softmax_reg3 = nn.Linear(self.out_channels, num_classes[2])
                print("layer 3", num_classes[2])
                
        else:
            self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
            print("Original")
         
        
    def forward(self, x):
        '''Forward propagation of pretrained ConvNext model
        '''

        x = self.model_ft(x)
        
        x = self.drop(x) # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        level_2 = self.softmax_reg2(self.relu_lv2(self.linear_lvl2(x)))
        level_3 = self.softmax_reg3(self.relu_lv3(self.linear_lvl3(x)))
                
        return level_1, level_2, level_3
    