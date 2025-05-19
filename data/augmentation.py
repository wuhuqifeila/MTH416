import torch
import torchvision.transforms as T
import random
import numpy as np
from config import Config
from PIL import Image
import torch.nn.functional as F

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

class GaussianNoise:
    """添加高斯噪声"""
    def __init__(self, std=Config.AUGMENTATION['gaussian_noise']):
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)

class CutMix:
    """CutMix数据增强"""
    def __init__(self, prob=Config.AUGMENTATION['cutmix_prob']):
        self.prob = prob
    
    def __call__(self, x, y):
        if random.random() > self.prob:
            return x, y, None, None
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # 生成随机框
        lam = np.random.beta(1, 1)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        
        # 混合图像
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整标签权重
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        return x, y, y[index], lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class CustomDataAugmentation:
    """自定义数据增强类"""
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        self.mixup = MixUpAugmentation() if is_training else None
        self.cutmix = CutMix() if is_training else None
        self.gaussian_noise = GaussianNoise() if is_training else None
    
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
            T.RandomResizedCrop(
                size=Config.IMAGE_SIZE,
                scale=Config.AUGMENTATION['random_crop_scale'],
                ratio=Config.AUGMENTATION['random_crop_ratio']
            ),
            T.RandomPerspective(
                distortion_scale=Config.AUGMENTATION['random_perspective'],
                p=0.5
            ),
            
            # 色彩变换
            T.ColorJitter(
                brightness=Config.AUGMENTATION['brightness_jitter'],
                contrast=Config.AUGMENTATION['contrast_jitter'],
                saturation=Config.AUGMENTATION['saturation_jitter'],
                hue=Config.AUGMENTATION['hue_jitter']
            ),
            
            # 转换为张量
            T.ToTensor(),
            
            # 在张量上进行的变换
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # 随机擦除
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
            img = self.train_transform(img)
            if self.gaussian_noise is not None:
                img = self.gaussian_noise(img)
            return img
        return self.val_transform(img)
    
    def apply_mixup(self, x, y):
        """
        应用MixUp增强
        x: 图像张量 [B, C, H, W]
        y: 标签 [B]
        """
        if not self.is_training:
            return x, y, None, None
            
        if random.random() < 0.5:  # 50%概率使用MixUp
            return self.mixup(x, y)
        else:  # 50%概率使用CutMix
            return self.cutmix(x, y) 