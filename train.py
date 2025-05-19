import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from data.dataset import get_data_loaders
from models.custom_cnn import CustomCNN
from models.resnet import ResNetTransfer as ResNetModel
from utils.metrics import MetricsCalculator, print_metrics_summary
from utils.early_stopping import EarlyStopping

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

def train_epoch(model, train_loader, criterion, optimizer, device, metrics_calculator):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        scores = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.detach().cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算训练指标
    metrics = metrics_calculator.calculate_all_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    metrics['loss'] = running_loss / len(train_loader)
    
    return metrics

def validate(model, val_loader, criterion, device, metrics_calculator):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            scores = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # 计算验证指标
    metrics = metrics_calculator.calculate_all_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_scores)
    )
    metrics['loss'] = running_loss / len(val_loader)
    
    return metrics

def train_model(model, train_loader, val_loader, num_epochs, device):
    """训练模型"""
    # 设置损失函数（带标签平滑）
    criterion = LabelSmoothingLoss(smoothing=Config.LABEL_SMOOTHING)
    
    # 设置优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 设置学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=Config.MIN_LR
    )
    
    # 设置早停
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        mode='max'  # 监控验证准确率
    )
    
    # 创建指标计算器
    metrics_calculator = MetricsCalculator()
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 训练阶段
        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, device, metrics_calculator
        )
        
        # 验证阶段
        val_metrics = validate(
            model, val_loader, criterion,
            device, metrics_calculator
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印指标
        print('\nTraining Metrics:')
        print_metrics_summary(train_metrics)
        print('\nValidation Metrics:')
        print_metrics_summary(val_metrics)
        
        # 保存混淆矩阵和PR曲线
        metrics_calculator.plot_confusion_matrix(
            val_metrics['confusion_matrix'],
            save_path=os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        )
        
        # 检查早停
        early_stopping(val_metrics['accuracy'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_metrics, val_metrics

def main():
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 创建模型
    model = CustomCNN().to(device)
    
    # 训练模型
    train_metrics, val_metrics = train_model(
        model, train_loader, val_loader,
        Config.NUM_EPOCHS, device
    )
    
    # 保存最终结果
    print("\nTraining completed!")
    print("\nFinal Training Metrics:")
    print_metrics_summary(train_metrics)
    print("\nFinal Validation Metrics:")
    print_metrics_summary(val_metrics)

if __name__ == '__main__':
    main() 