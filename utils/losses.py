import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class FocalLoss(nn.Module):
    """Focal Loss实现
    
    参数:
        gamma (float): 聚焦参数，降低易分类样本的权重
        alpha (tensor): 类别权重
    """
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha).to(Config.DEVICE)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, smoothing=Config.LABEL_SMOOTHING):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class CombinedLoss(nn.Module):
    """组合损失函数：Focal Loss + Label Smoothing"""
    def __init__(self, alpha=Config.CLASS_WEIGHTS, gamma=2, smoothing=Config.LABEL_SMOOTHING):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.smoothing = LabelSmoothingLoss(smoothing=smoothing)
        
    def forward(self, inputs, targets):
        return self.focal(inputs, targets) + self.smoothing(inputs, targets) 