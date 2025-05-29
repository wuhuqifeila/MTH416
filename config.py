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
    NUM_EPOCHS = 10  # 增加训练轮数
    LEARNING_RATE = 0.0001  # 大幅降低学习率
    WEIGHT_DECAY = 1e-5  # 降低权重衰减
    
    # 早停设置
    EARLY_STOPPING_PATIENCE = 3  # 减少早停耐心值
    
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
    
    # 类别权重（调整为更温和的权重）
    CLASS_WEIGHTS = [0.5, 1.5, 3.0]  # 降低权重差异
    
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
        'unfreeze_layers': 2,  # 减少解冻层数
        'feature_extract': True,
        'learning_rate_fc': 0.001,  # 保持分类器学习率
        'learning_rate_backbone': 0.00001,  # 进一步降低主干网络学习率
        'progressive_unfreeze': True,
        'unfreeze_epoch': 5  # 更早开始解冻
    }
    
    # 模型保存设置
    SAVE_DIR = 'results'
    CUSTOM_MODEL_PATH = os.path.join(SAVE_DIR, 'custom_model.pth')
    RESNET_MODEL_PATH = os.path.join(SAVE_DIR, 'resnet_model.pth')
    
    # 损失函数设置
    LOSS = {
        'focal_gamma': 1,  # 降低Focal Loss强度
        'focal_alpha': [0.5, 1.5, 3.0],  # 使用更温和的权重
        'label_smoothing': 0.05,  # 降低标签平滑强度
        'use_focal': True,
        'use_label_smoothing': True
    }
    
    # Intel optimization
    USE_INTEL_OPTIMIZATION = False
    
    # ResNet parameters
    RESNET_PRETRAINED = True
    RESNET_FREEZE_LAYERS = True 