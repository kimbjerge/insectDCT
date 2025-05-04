import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        self.cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        self.reduction = reduction

    def forward(self, logits, target):
        cls_num_list = self.cls_num_list.to(logits.device)
        cls_weights = cls_num_list / cls_num_list.sum()

        balanced_logits = logits + torch.log(cls_weights + 1e-6)  

        return F.cross_entropy(balanced_logits, target, reduction=self.reduction)
