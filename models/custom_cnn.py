import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ConvBlock(nn.Module):
    """
    Convolutional block: Conv -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

class CustomCNN(nn.Module):
    """
    Optimized CNN model
    Improvements:
    1. Add global average pooling layer to replace the first fully connected layer
    2. Simplify fully connected layer structure
    3. Reduce parameter count
    4. Maintain model expressiveness
    """
    def __init__(self):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = ConvBlock(3, 32)     
        self.conv2 = ConvBlock(32, 64)    
        self.conv3 = ConvBlock(64, 128)   
        self.conv4 = ConvBlock(128, 256)  
        self.conv5 = ConvBlock(256, 512)  
        
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simplified fully connected layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, Config.NUM_CLASSES)
        
        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        
        # Weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Use He initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 