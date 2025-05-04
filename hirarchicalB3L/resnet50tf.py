'''Deep Hierarchical Classifier using resnet50 with cbam as the base.
'''

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50(nn.Module):
    '''ResNet-50 Architecture with pretrained weights
    '''

    def __init__(self, use_cbam=True, image_depth=3, num_classes=[20,100], simple=True):
        '''Params init and build arch.
        '''
        super(ResNet50, self).__init__()

        #self.in_channels = 64 ??
        self.expansion = 4
        self.simple = simple
        self.out_channels = 512
        
        self.model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
        
        # optionally freeze parameters
        #for p in self.model_ft.parameters():
        #  p.requires_grad = False
        
        # fintune layer4
        #for p in self.model_ft.layer4.parameters():
        #  p.requires_grad = True
        
        # overwrite the 'fc' layer
        print("In features", self.model_ft.fc.in_features)
        #self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 512*self.expansion) 
        self.model_ft.fc = nn.Identity() # Do nothing just pass input to output
 
        self.layers = len(num_classes)
        
        # At least one layer
        #self.linear_lvl1 = nn.Linear(512*self.expansion, num_classes[0])
        #self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes[0])
        
        if self.simple:
            print("RestNet50tf Simple: Dropout + ReLU")

            if self.layers > 1:
                #self.linear_lvl2 = nn.Linear(512*self.expansion, num_classes[1])
                #self.softmax_reg2 = nn.Linear(num_classes[1], num_classes[1])
                self.linear_lvl2 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
                self.relu_lv2 = nn.ReLU(inplace=False)
                self.softmax_reg2 = nn.Linear(self.out_channels, num_classes[1])
                print("Layer 2")
                
            if self.layers > 2:
                #self.linear_lvl3 = nn.Linear(512*self.expansion, num_classes[2])
                #self.softmax_reg3 = nn.Linear(num_classes[2], num_classes[2])
                self.linear_lvl3 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
                self.relu_lv3 = nn.ReLU(inplace=False)
                self.softmax_reg3 = nn.Linear(self.out_channels, num_classes[2])
                print("layer 3")
                
        else:
            self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
            print("Original")
         
        """
        self.linear_lvl1 = nn.Linear(512*self.expansion, num_classes[0])
        self.linear_lvl2 = nn.Linear(512*self.expansion, num_classes[1])

        self.softmax_reg1 = nn.Linear(num_classes[0], num_classes[0])
        if self.simple:
            self.softmax_reg2 = nn.Linear(num_classes[1], num_classes[1])
            print("Simple ResNet50tf")
        else:
            self.softmax_reg2 = nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
            print("Original ResNet50tf")
        """
        
    def forward(self, x):
        '''Forward propagation of pretrained ResNet-18.
        '''

        x = self.model_ft(x)
        
        x = self.drop(x) # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        level_2 = self.softmax_reg2(self.relu_lv2(self.linear_lvl2(x)))
        level_3 = self.softmax_reg3(self.relu_lv3(self.linear_lvl3(x)))
        
        # Simple without relu
        #level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        #level_2 = self.softmax_reg2(self.relu_lv2(self.linear_lvl2(x)))
        #level_3 = self.softmax_reg3(self.relu_lv3(self.linear_lvl3(x)))
                
        return level_1, level_2, level_3
    
        """
        level_1 = self.softmax_reg1(self.linear_lvl1(x))
        
        if self.simple:
            level_2 = self.softmax_reg2(self.linear_lvl2(x))
        else:
            level_2 = self.softmax_reg2(torch.cat((level_1, self.linear_lvl2(x)), dim=1))

        return level_1, level_2
        """