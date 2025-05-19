import torch
import torchvision.transforms as T
import random
import numpy as np
from config import Config

class MixUpAugmentation:
    """MixUp数据增强"""
    def __init__(self, alpha=Config.AUGMENTATION['mixup_alpha']):
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        对一个批次的数据进行mixup
        x: 图像数据 [B, C, H, W]
        y: 标签 [B]
        """
        if self.alpha > 0 and random.random() < 0.5:  # 50%的概率应用mixup
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CustomDataAugmentation:
    """自定义数据增强类"""
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        self.mixup = MixUpAugmentation() if is_training else None
    
    def _get_train_transforms(self):
        """获取训练数据增强"""
        return T.Compose([
            # 调整大小
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            
            # 基础变换（在PIL图像上进行）
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(
                degrees=Config.AUGMENTATION['random_rotate_degrees'],
                fill=0
            ),
            
            # 转换为张量
            T.ToTensor(),
            
            # 在张量上进行的变换
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # 随机擦除（在标准化后进行）
            T.RandomErasing(p=Config.AUGMENTATION['random_erase_prob'])
        ])
    
    def _get_val_transforms(self):
        """获取验证数据增强"""
        return T.Compose([
            T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, img):
        """
        应用数据增强
        img: PIL图像
        """
        if self.is_training:
            return self.train_transform(img)
        return self.val_transform(img)
    
    def apply_mixup(self, x, y):
        """
        应用MixUp增强
        x: 图像张量 [B, C, H, W]
        y: 标签 [B]
        """
        if self.mixup and self.is_training:
            return self.mixup(x, y)
        return x, y, None, None 