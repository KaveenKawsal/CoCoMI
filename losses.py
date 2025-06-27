# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiceLoss(nn.Module):
    """Standard Dice Loss."""
    def __init__(self, smooth=1e-6):
        super(SimpleDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class SimpleFocalLoss(nn.Module):
    """Standard Focal Loss without class weighting."""
    def __init__(self, alpha=1, gamma=2):
        super(SimpleFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CrossEntropyLoss(nn.Module):
    """Standard Cross Entropy Loss."""
    def __init__(self, **kwargs): # Accept kwargs to be compatible with weighted losses
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def forward(self, inputs, targets):
        return self.ce(inputs, targets.long())

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        dice = torch.zeros(num_classes, device=pred.device)
        for c in range(num_classes):
            pred_c = pred[:, c]
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice[c] = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class AdvancedFocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5, gamma=2, smooth=1e-6, alpha=None):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice