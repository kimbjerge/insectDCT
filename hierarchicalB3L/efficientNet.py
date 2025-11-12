'''Deep Hierarchical Classifier using different ConvNext models
'''

import torch.nn as nn

from torchvision.models.efficientnet import efficientnet_v2_s
from torchvision.models.efficientnet import efficientnet_v2_m
from torchvision.models.efficientnet import efficientnet_v2_l
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models.efficientnet import EfficientNet_V2_M_Weights
from torchvision.models.efficientnet import EfficientNet_V2_L_Weights

class EfficientNet(nn.Module):
    '''ConvNext-Base Architecture with pretrained weights - default configuration
    '''

    def __init__(self, image_depth=3, num_classes=[20,100], simple=True, effNet="medium"): # large, medium, small
        '''Params init and build arch.
        '''
        super(EfficientNet, self).__init__()

        self.simple = simple
        self.out_channels = 256
        
        if effNet == "large":
            self.model_ft = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1) 
            print("EfficientNet_v2_l")
        if effNet == "medium":
            self.model_ft = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1) 
            print("EfficientNet_v2_m")
        if effNet == "small":
            self.model_ft = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) 
            print("EfficientNet_v2_s")

           
        # overwrite the 'fc' layer
        num_in_features = self.model_ft.classifier[1].in_features
        print("In features", num_in_features)
        self.model_ft.classifier[1] =  nn.Identity() # Do nothing just pass input to output
 
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
    