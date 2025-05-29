import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from data.dataset import get_data_loaders
from models.custom_cnn import CustomCNN
from models.resnet import ResNetTransfer
from utils.metrics import MetricsCalculator

class FocalLoss(nn.Module):
    """改进的Focal Loss"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def create_balanced_loss():
    """创建平衡的损失函数"""
    # 更强的类别权重来处理不平衡
    class_weights = torch.tensor([0.3, 2.0, 5.0], dtype=torch.float32)
    
    # 使用Focal Loss
    return FocalLoss(alpha=class_weights, gamma=2.0)

def train_epoch_balanced(model, train_loader, criterion, optimizer, device):
    """平衡训练函数"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 统计各类别准确率
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1
        
        # 计算类别平衡准确率
        class_accs = [class_correct[i]/max(class_total[i], 1) for i in range(3)]
        balanced_acc = sum(class_accs) / 3
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{100.0 * correct / total:.1f}%',
            'bal_acc': f'{balanced_acc*100:.1f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    balanced_acc = sum([class_correct[i]/max(class_total[i], 1) for i in range(3)]) / 3
    
    return epoch_loss, epoch_acc, balanced_acc

def validate_balanced(model, val_loader, criterion, device):
    """平衡验证函数"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计各类别准确率
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            balanced_acc = sum([class_correct[i]/max(class_total[i], 1) for i in range(3)]) / 3
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{100.0 * correct / total:.1f}%',
                'bal_acc': f'{balanced_acc*100:.1f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    balanced_acc = sum([class_correct[i]/max(class_total[i], 1) for i in range(3)]) / 3
    
    return epoch_loss, epoch_acc, balanced_acc, np.array(all_labels), np.array(all_preds)

def train_model_improved(model, train_loader, val_loader, test_loader, model_name, device):
    """改进的训练函数"""
    print(f"\n🚀 开始训练 {model_name}...")
    
    # 使用平衡的损失函数
    criterion = create_balanced_loss()
    
    # 优化器设置
    if 'ResNet' in model_name:
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 学习率调度器 - 监控平衡准确率
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.7, min_lr=1e-7)
    
    best_balanced_acc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 7
    
    print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(20):  # 增加训练轮数
        print(f'\nEpoch {epoch+1}/20')
        
        # 训练
        train_loss, train_acc, train_bal_acc = train_epoch_balanced(
            model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_bal_acc, val_labels, val_preds = validate_balanced(
            model, val_loader, criterion, device)
        
        # 更新学习率 - 基于平衡准确率
        scheduler.step(val_bal_acc)
        
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Balanced Acc: {train_bal_acc:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Balanced Acc: {val_bal_acc:.4f}')
        
        # 基于平衡准确率保存最佳模型
        if val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"📈 最佳平衡准确率更新: {best_balanced_acc:.4f}")
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= max_patience:
            print("⏰ 早停触发")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试集评估
    print(f"\n🔍 {model_name} 测试集评估...")
    test_loss, test_acc, test_bal_acc, test_labels, test_preds = validate_balanced(
        model, test_loader, criterion, device)
    
    print(f"测试结果 - 准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"测试结果 - 平衡准确率: {test_bal_acc:.4f} ({test_bal_acc*100:.2f}%)")
    
    # 详细分类报告
    class_names = ['Normal', 'Benign', 'Cancer']
    print(f"\n详细分类报告:")
    print(classification_report(test_labels, test_preds, 
                              target_names=class_names, digits=4))
    
    return {
        'test_acc': test_acc,
        'test_bal_acc': test_bal_acc,
        'test_labels': test_labels,
        'test_preds': test_preds
    }

def plot_final_confusion_matrix(labels, preds, model_name, save_path):
    """绘制最终混淆矩阵"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Benign', 'Cancer'],
                yticklabels=['Normal', 'Benign', 'Cancer'])
    plt.title(f'{model_name} - 测试集混淆矩阵', fontsize=14, fontweight='bold')
    plt.ylabel('真实标签', fontsize=12)
    plt.xlabel('预测标签', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主训练函数"""
    print("="*60)
    print("          MTH416 改进版训练脚本 - 解决类别不平衡")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"数据集大小: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'results/improved_training_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # 训练CNN
    print("\n" + "="*40)
    print("          Q1: 自定义CNN训练")
    print("="*40)
    
    cnn_model = CustomCNN().to(device)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"CNN参数量: {cnn_params:,}")
    
    cnn_results = train_model_improved(cnn_model, train_loader, val_loader, test_loader, "CNN", device)
    results['cnn'] = cnn_results
    
    # 保存CNN模型和混淆矩阵
    torch.save(cnn_model.state_dict(), os.path.join(save_dir, 'cnn_model_improved.pth'))
    plot_final_confusion_matrix(cnn_results['test_labels'], cnn_results['test_preds'], 
                               'Q1 (自定义CNN)', os.path.join(save_dir, 'cnn_confusion_matrix.png'))
    
    # 训练ResNet
    print("\n" + "="*40)
    print("          Q2: ResNet迁移学习训练")
    print("="*40)
    
    resnet_model = ResNetTransfer().to(device)
    total_params = sum(p.numel() for p in resnet_model.parameters())
    trainable_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"ResNet总参数: {total_params:,}")
    print(f"ResNet可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    resnet_results = train_model_improved(resnet_model, train_loader, val_loader, test_loader, "ResNet", device)
    results['resnet'] = resnet_results
    
    # 保存ResNet模型和混淆矩阵
    torch.save(resnet_model.state_dict(), os.path.join(save_dir, 'resnet_model_improved.pth'))
    plot_final_confusion_matrix(resnet_results['test_labels'], resnet_results['test_preds'], 
                               'Q2 (ResNet迁移学习)', os.path.join(save_dir, 'resnet_confusion_matrix.png'))
    
    # 最终结果对比
    print("\n" + "="*60)
    print("                 最终结果对比")
    print("="*60)
    
    print(f"CNN:")
    print(f"  - 整体准确率: {cnn_results['test_acc']:.4f} ({cnn_results['test_acc']*100:.2f}%)")
    print(f"  - 平衡准确率: {cnn_results['test_bal_acc']:.4f} ({cnn_results['test_bal_acc']*100:.2f}%)")
    
    print(f"\nResNet:")
    print(f"  - 整体准确率: {resnet_results['test_acc']:.4f} ({resnet_results['test_acc']*100:.2f}%)")
    print(f"  - 平衡准确率: {resnet_results['test_bal_acc']:.4f} ({resnet_results['test_bal_acc']*100:.2f}%)")
    
    improvement = resnet_results['test_acc'] - cnn_results['test_acc']
    bal_improvement = resnet_results['test_bal_acc'] - cnn_results['test_bal_acc']
    
    print(f"\n性能提升:")
    print(f"  - 整体准确率提升: {improvement:+.4f} ({improvement*100:+.2f}个百分点)")
    print(f"  - 平衡准确率提升: {bal_improvement:+.4f} ({bal_improvement*100:+.2f}个百分点)")
    
    # 保存结果
    torch.save(results, os.path.join(save_dir, 'training_results_improved.pth'))
    
    print(f"\n💾 所有结果已保存到: {save_dir}")
    print(f"📊 混淆矩阵已生成: cnn_confusion_matrix.png, resnet_confusion_matrix.png")
    
    # 成功标准
    print(f"\n📊 结果评估:")
    if cnn_results['test_bal_acc'] > 0.4:
        print("✅ CNN平衡性能合格 (>40%)")
    else:
        print("⚠️  CNN平衡性能需要改进")
        
    if resnet_results['test_bal_acc'] > cnn_results['test_bal_acc']:
        print("✅ ResNet迁移学习在平衡准确率上优于CNN")
    
    if resnet_results['test_bal_acc'] > 0.5:
        print("✅ ResNet达到期望的平衡性能 (>50%)")
    else:
        print("⚠️  ResNet平衡性能仍需优化")

if __name__ == '__main__':
    main() 