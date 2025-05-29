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
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha).to(Config.DEVICE)
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
    def __init__(self, alpha=None, gamma=1.0, smoothing=0.05, reduction='mean'):
        """
        组合损失函数：Focal Loss + 标签平滑
        
        Args:
            alpha: 类别权重，默认为 [0.5, 1.5, 3.0]
            gamma: Focal Loss的聚焦参数，降低到1.0
            smoothing: 标签平滑参数，降低到0.05
            reduction: 损失聚合方式
        """
        super().__init__()
        if alpha is None:
            alpha = [0.5, 1.5, 3.0]  # 更温和的权重
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.label_smoothing = LabelSmoothingLoss(smoothing=smoothing)
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets) + self.label_smoothing(inputs, targets) 