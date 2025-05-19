import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ResNetTransfer(nn.Module):
    """基于ResNet-18的迁移学习模型"""
    def __init__(self, freeze_layers=True):
        super().__init__()
        # 加载预训练的ResNet-18
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        if freeze_layers:
            # 冻结所有卷积层
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 修改最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, Config.NUM_CLASSES)
        )
        
        # 初始化新添加的层
        self._initialize_weights()
    
    def forward(self, x):
        return self.resnet(x)
    
    def _initialize_weights(self):
        """初始化新添加层的权重"""
        for m in self.resnet.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def unfreeze_layers(self, num_layers=0):
        """解冻指定数量的层"""
        if num_layers == 0:
            return
        
        # 解冻最后几个层
        trainable_layers = [
            self.resnet.layer4,
            self.resnet.layer3,
            self.resnet.layer2,
            self.resnet.layer1
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
    model = ResNetTransfer(freeze_layers=True)
    if unfreeze_layers > 0:
        model.unfreeze_layers(unfreeze_layers)
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 