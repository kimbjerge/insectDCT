'''Hierarchical Loss Network with two or three layers
'''

import torch
import torch.nn as nn

class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, hierarchyL1, hierarchyL2, labelsL1, labelsL2, labelsL3, lossFnLx, 
                 device='cpu', total_level=3, alpha=0.5, p_loss=2.718281828459, simple=False): #p_loss = 3 values of alpha and beta KBE???
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = 1-alpha
        self.p_loss = p_loss
        print("Hierachical loss levels=%d, alpha=%.2f, beta = %.2f, ploss = %.4f" % (self.total_level, self.alpha, self.beta, self.p_loss))
        self.device = device
        self.hierarchical_labelsL1 = hierarchyL1
        self.hierarchical_labelsL2 = hierarchyL2
        self.level_one_labels = labelsL1 
        self.level_two_labels = labelsL2
        self.level_three_labels = labelsL3
        self.numeric_hierarchyL1, self.numeric_hierarchyL2 = self.words_to_indices()
        self.simple = simple
        self.lossFnLx = lossFnLx
        if self.simple:
           print("Loss simple")
        else:
           print("Loss orginal")


    def words_to_indices(self):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchyL1 = {}
        for k, v in self.hierarchical_labelsL1.items():
            numeric_hierarchyL1[self.level_one_labels.index(k)] = [self.level_two_labels.index(i) for i in v]

        numeric_hierarchyL2 = {}
        for k, v in self.hierarchical_labelsL2.items():
            numeric_hierarchyL2[self.level_two_labels.index(k)] = [self.level_three_labels.index(i) for i in v]

        return numeric_hierarchyL1, numeric_hierarchyL2 


    def check_hierarchy_list(self, level, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        if level == 1:
            bool_tensor = [current_level[i] in self.numeric_hierarchyL1[previous_level[i]] for i in range(len(previous_level))]
        if level == 2:
            bool_tensor = [current_level[i] in self.numeric_hierarchyL2[previous_level[i]] for i in range(len(previous_level))]

        return bool_tensor


    def check_hierarchy(self, level, current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''
        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        if level == 1:
            bool_tensor = [not current_level[i] in self.numeric_hierarchyL1[previous_level[i].item()] for i in range(previous_level.size()[0])]
        if level == 2:
            bool_tensor = [not current_level[i] in self.numeric_hierarchyL2[previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''
        
        lloss = 0
        for l in range(self.total_level):
            #lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])
            lloss += self.lossFnLx[l](predictions[l], true_labels[l])

        return self.alpha * lloss


    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)

            D_l = self.check_hierarchy(l, current_lvl_pred, prev_lvl_pred)

            if self.simple:
                dloss += torch.sum(torch.pow(self.p_loss, D_l) - 1) # Simplified version
            else:
                l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
                l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
                dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
