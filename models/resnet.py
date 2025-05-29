import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ResNetTransfer(nn.Module):
    """基于ResNet-18的迁移学习模型"""
    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()
        
        # 加载预训练ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 冻结主干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 替换最后的分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # 降低dropout
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # 降低dropout
            nn.Linear(128, num_classes)
        )
        
        # 初始化新的分类层
        for layer in self.backbone.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=0):
        """解冻指定数量的层"""
        if num_layers == 0:
            return
        
        # 解冻最后几个层
        trainable_layers = [
            self.backbone.layer4,
            self.backbone.layer3,
            self.backbone.layer2,
            self.backbone.layer1
        ]
        
        for layer in trainable_layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True
    
    def get_trainable_params(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_params(self):
        """打印可训练参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"冻结参数数量: {total_params - trainable_params:,}")
        
        # 打印每层的参数状态
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel():,} 参数, 可训练: {param.requires_grad}")

def create_resnet_model(unfreeze_layers=0):
    """创建ResNet迁移学习模型"""
    model = ResNetTransfer(freeze_backbone=True)
    if unfreeze_layers > 0:
        model.unfreeze_layers(unfreeze_layers)
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 