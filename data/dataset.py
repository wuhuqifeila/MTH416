import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from config import Config
from data.augmentation import CustomDataAugmentation

class BreastCancerDataset(Dataset):
    """Breast cancer dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'benign', 'malignant']  # Test set uses malignant
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load data samples"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                # In test set 'cancer' is replaced with 'malignant'
                if class_name == 'malignant':
                    class_dir = os.path.join(self.root_dir, 'cancer')
                if not os.path.exists(class_dir):
                    continue
            
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders():
    """Get data loaders"""
    # Create data augmentation
    train_transform = CustomDataAugmentation(is_training=True)
    val_transform = CustomDataAugmentation(is_training=False)
    
    # Create datasets
    train_dataset = BreastCancerDataset(
        root_dir=Config.TRAIN_DIR,
        transform=train_transform
    )
    
    val_dataset = BreastCancerDataset(
        root_dir=Config.VAL_DIR,
        transform=val_transform
    )
    
    test_dataset = BreastCancerDataset(
        root_dir=Config.TEST_DIR,
        transform=val_transform
    )
    
    # Calculate sampling weights
    class_counts = np.zeros(Config.NUM_CLASSES)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    weights = 1.0 / class_counts
    sample_weights = [weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        prefetch_factor=Config.PREFETCH_FACTOR,
        persistent_workers=Config.PERSISTENT_WORKERS
    )
    
    print("\nDataset Statistics:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {train_dataset.classes[i]}: {count} samples ({100*count/sum(class_counts):.2f}%)")
    
    print("\nSampling Weights:")
    for i, w in enumerate(weights):
        print(f"Class {train_dataset.classes[i]}: {w:.4f}")
    
    return train_loader, val_loader, test_loader 