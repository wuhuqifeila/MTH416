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
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 2e-4
    
    # 早停设置
    EARLY_STOPPING_PATIENCE = 10
    
    # 学习率调度器设置
    SCHEDULER_PATIENCE = 5
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
    CLASS_WEIGHTS = [0.1, 3.0, 4.0]  # 更新的类别权重
    
    # 数据增强参数
    AUGMENTATION = {
        'random_rotate_degrees': 30,
        'random_crop_scale': (0.7, 1.0),
        'random_crop_ratio': (0.7, 1.3),
        'brightness_jitter': 0.3,
        'contrast_jitter': 0.3,
        'random_erase_prob': 0.3,
        'mixup_alpha': 0.3
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
        'unfreeze_layers': 2,  # 解冻最后两个块
        'feature_extract': True,  # 是否只训练分类器
        'learning_rate_fc': 0.001,  # 分类器学习率
        'learning_rate_backbone': 0.0001,  # 主干网络学习率
        'progressive_unfreeze': True,  # 是否使用渐进式解冻
        'unfreeze_epoch': 5  # 每隔多少epoch解冻一层
    }
    
    # 模型保存设置
    SAVE_DIR = 'results'
    CUSTOM_MODEL_PATH = os.path.join(SAVE_DIR, 'custom_model.pth')
    RESNET_MODEL_PATH = os.path.join(SAVE_DIR, 'resnet_model.pth')
    
    # Intel optimization
    USE_INTEL_OPTIMIZATION = False
    
    # ResNet parameters
    RESNET_PRETRAINED = True
    RESNET_FREEZE_LAYERS = True 