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
    IMAGE_SIZE = 224  # Default input size for ResNet
    BATCH_SIZE = 32
    NUM_WORKERS = 2 if torch.cuda.is_available() else 0
    
    # Training parameters
    NUM_EPOCHS = 10  # Increase training epochs
    LEARNING_RATE = 0.0001  # Significantly reduce learning rate
    WEIGHT_DECAY = 1e-5  # Reduce weight decay
    
    # Early stopping settings
    EARLY_STOPPING_PATIENCE = 3  # Reduce early stopping patience
    
    # Learning rate scheduler settings
    SCHEDULER_PATIENCE = 7  # Increase scheduler patience
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    # Label smoothing
    LABEL_SMOOTHING = 0.1
    
    # Model architecture parameters
    DROPOUT_RATE = 0.5
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Performance optimization
    PIN_MEMORY = True if torch.cuda.is_available() else False
    PREFETCH_FACTOR = 2 if torch.cuda.is_available() else None
    PERSISTENT_WORKERS = True if torch.cuda.is_available() else False
    
    # Class weights (adjusted to more moderate weights)
    CLASS_WEIGHTS = [0.5, 1.5, 3.0]  # Reduce weight differences
    
    # Data augmentation parameters
    AUGMENTATION = {
        'random_rotate_degrees': 45,  # Increase rotation angle
        'random_crop_scale': (0.8, 1.0),  # Reduce crop range
        'random_crop_ratio': (0.8, 1.2),  # Adjust crop ratio
        'brightness_jitter': 0.4,  # Increase brightness variation
        'contrast_jitter': 0.4,  # Increase contrast variation
        'saturation_jitter': 0.4,  # Add saturation variation
        'hue_jitter': 0.1,  # Add hue variation
        'random_erase_prob': 0.5,  # Increase random erase probability
        'mixup_alpha': 0.4,  # Increase mixup strength
        'cutmix_prob': 0.5,  # Add CutMix probability
        'gaussian_noise': 0.01,  # Add Gaussian noise
        'random_perspective': 0.3  # Add perspective transformation
    }
    
    # Evaluation metrics settings
    METRICS = {
        'accuracy': True,
        'precision': True,
        'recall': True,
        'f1': True,
        'auc': True,
        'confusion_matrix': True
    }
    
    # Transfer learning settings
    TRANSFER = {
        'unfreeze_layers': 2,  # Reduce number of unfrozen layers
        'feature_extract': True,
        'learning_rate_fc': 0.001,  # Maintain classifier learning rate
        'learning_rate_backbone': 0.00001,  # Further reduce backbone learning rate
        'progressive_unfreeze': True,
        'unfreeze_epoch': 5  # Start unfreezing earlier
    }
    
    # Model saving settings
    SAVE_DIR = 'results'
    CUSTOM_MODEL_PATH = os.path.join(SAVE_DIR, 'custom_model.pth')
    RESNET_MODEL_PATH = os.path.join(SAVE_DIR, 'resnet_model.pth')
    
    # Loss function settings
    LOSS = {
        'focal_gamma': 1,  # Reduce Focal Loss strength
        'focal_alpha': [0.5, 1.5, 3.0],  # Use more moderate weights
        'label_smoothing': 0.05,  # Reduce label smoothing strength
        'use_focal': True,
        'use_label_smoothing': True
    }
    
    # Intel optimization
    USE_INTEL_OPTIMIZATION = False
    
    # ResNet parameters
    RESNET_PRETRAINED = True
    RESNET_FREEZE_LAYERS = True 