'''Deep Hierarchical Classifier using ConvNext-Base
'''

import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class ConvNextBase(nn.Module):
    '''ConvNext-Base Architecture with pretrained weights
    '''

    def __init__(self, image_depth=3, num_classes=[20,100], simple=True):
        '''Params init and build arch.
        '''
        super(ConvNextBase, self).__init__()

        self.simple = simple
        self.out_channels = 512
        
        self.model_ft = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1) # 80.86, 25.6M
    
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
            print("ConvNext-Base Simple: Dropout + ReLU")

            if self.layers > 1:
                self.linear_lvl2 = nn.Linear(num_in_features, self.out_channels)
                self.relu_lv2 = nn.ReLU(inplace=False)
                self.softmax_reg2 = nn.Linear(self.out_channels, num_classes[1])
                print("Layer 2")
                
            if self.layers > 2:
                self.linear_lvl3 = nn.Linear(num_in_features, self.out_channels)
                self.relu_lv3 = nn.ReLU(inplace=False)
                self.softmax_reg3 = nn.Linear(self.out_channels, num_classes[2])
                print("layer 3")
                
        else:
            self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
            print("Original")
         
        
    def forward(self, x):
        '''Forward propagation of pretrained ConvNext-Base.
        '''

        x = self.model_ft(x)
        
        x = self.drop(x) # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        level_2 = self.softmax_reg2(self.relu_lv2(self.linear_lvl2(x)))
        level_3 = self.softmax_reg3(self.relu_lv3(self.linear_lvl3(x)))
                
        return level_1, level_2, level_3
    