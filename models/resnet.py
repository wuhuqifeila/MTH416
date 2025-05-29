import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class ResNetTransfer(nn.Module):
    """ResNet-18 based transfer learning model"""
    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze backbone network
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # Reduce dropout
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduce dropout
            nn.Linear(128, num_classes)
        )
        
        # Initialize new classification layer
        for layer in self.backbone.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=0):
        """Unfreeze specified number of layers"""
        if num_layers == 0:
            return
        
        # Unfreeze last few layers
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
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def print_trainable_params(self):
        """Print number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Print parameter status for each layer
        for name, param in self.named_parameters():
            print(f"{name}: {param.numel():,} parameters, trainable: {param.requires_grad}")

def create_resnet_model(unfreeze_layers=0):
    """Create ResNet transfer learning model"""
    model = ResNetTransfer(freeze_backbone=True)
    if unfreeze_layers > 0:
        model.unfreeze_layers(unfreeze_layers)
    return model

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 