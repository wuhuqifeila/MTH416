import os
import torch

class Config:
    # Dataset paths
    DATA_ROOT = "dataset"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # Model parameters
    NUM_CLASSES = 3
    IMAGE_SIZE = 224  # ResNet默认输入大小
    BATCH_SIZE = 32
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0
    
    # Training parameters
    NUM_EPOCHS = 50  # 增加训练轮数
    LEARNING_RATE = 0.0005  # 降低学习率
    WEIGHT_DECAY = 5e-4  # 增加权重衰减
    
    # 早停设置
    EARLY_STOPPING_PATIENCE = 15  # 增加早停耐心值
    
    # 学习率调度器设置
    SCHEDULER_PATIENCE = 7  # 增加调度器耐心值
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    # 标签平滑
    LABEL_SMOOTHING = 0.1
    
    # Model architecture parameters
    DROPOUT_RATE = 0.5
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Performance optimization
    PIN_MEMORY = True if torch.cuda.is_available() else False
    PREFETCH_FACTOR = 2 if torch.cuda.is_available() else None
    PERSISTENT_WORKERS = True if torch.cuda.is_available() else False
    
    # 类别权重（基于数据集统计）
    CLASS_WEIGHTS = [0.1, 2.0, 8.0]  # 显著增加cancer类的权重
    
    # 数据增强参数
    AUGMENTATION = {
        'random_rotate_degrees': 45,  # 增加旋转角度
        'random_crop_scale': (0.8, 1.0),  # 减小裁剪范围
        'random_crop_ratio': (0.8, 1.2),  # 调整裁剪比例
        'brightness_jitter': 0.4,  # 增加亮度变化
        'contrast_jitter': 0.4,  # 增加对比度变化
        'saturation_jitter': 0.4,  # 添加饱和度变化
        'hue_jitter': 0.1,  # 添加色调变化
        'random_erase_prob': 0.5,  # 增加随机擦除概率
        'mixup_alpha': 0.4,  # 增加mixup强度
        'cutmix_prob': 0.5,  # 添加CutMix概率
        'gaussian_noise': 0.01,  # 添加高斯噪声
        'random_perspective': 0.3  # 添加透视变换
    }
    
    # 评估指标设置
    METRICS = {
        'accuracy': True,
        'precision': True,
        'recall': True,
        'f1': True,
        'auc': True,
        'confusion_matrix': True
    }
    
    # 迁移学习设置
    TRANSFER = {
        'unfreeze_layers': 3,  # 增加解冻层数
        'feature_extract': True,
        'learning_rate_fc': 0.001,
        'learning_rate_backbone': 0.00005,  # 降低主干网络学习率
        'progressive_unfreeze': True,
        'unfreeze_epoch': 8  # 增加解冻间隔
    }
    
    # 模型保存设置
    SAVE_DIR = 'results'
    CUSTOM_MODEL_PATH = os.path.join(SAVE_DIR, 'custom_model.pth')
    RESNET_MODEL_PATH = os.path.join(SAVE_DIR, 'resnet_model.pth')
    
    # 损失函数设置
    LOSS = {
        'focal_gamma': 2,  # Focal Loss的gamma参数
        'focal_alpha': CLASS_WEIGHTS,  # 使用类别权重作为alpha
        'label_smoothing': 0.1,  # 标签平滑系数
        'use_focal': True,  # 是否使用Focal Loss
        'use_label_smoothing': True  # 是否使用标签平滑
    }
    
    # Intel optimization
    USE_INTEL_OPTIMIZATION = False
    
    # ResNet parameters
    RESNET_PRETRAINED = True
    RESNET_FREEZE_LAYERS = True 