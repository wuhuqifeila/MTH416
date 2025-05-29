import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

from config import Config
from data.dataset import get_data_loaders
from models.custom_cnn import CustomCNN
from models.resnet import ResNetTransfer
from utils.metrics import MetricsCalculator, print_metrics_summary
from utils.early_stopping import EarlyStopping

def simple_cross_entropy_loss():
    """使用简单的交叉熵损失"""
    # 计算类别权重（更温和）
    class_weights = torch.tensor([0.8, 1.2, 2.0], dtype=torch.float32)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()
    
    return nn.CrossEntropyLoss(weight=class_weights)

def train_epoch_simple(model, train_loader, criterion, optimizer, device):
    """简化的训练函数"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        acc = 100.0 * correct / total
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_simple(model, val_loader, criterion, device):
    """简化的验证函数"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
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
            
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)

def train_model_fixed(model, train_loader, val_loader, test_loader, model_name, device):
    """修复后的训练函数"""
    print(f"\n🚀 开始训练 {model_name}...")
    
    # 使用简单的交叉熵损失
    criterion = simple_cross_entropy_loss()
    
    # 保守的优化器设置
    if 'ResNet' in model_name:
        # ResNet使用更小的学习率
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    else:
        # CNN使用稍大的学习率
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # 早停
    early_stopping = EarlyStopping(patience=5, mode='max')
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(15):  # 最多15轮
        print(f'\nEpoch {epoch+1}/15')
        
        # 训练
        train_loss, train_acc = train_epoch_simple(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_labels, val_preds = validate_simple(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"📈 最佳验证准确率更新: {best_val_acc:.4f}")
        
        # 检查早停
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("⏰ 早停触发")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试集评估
    print(f"\n🔍 {model_name} 测试集评估...")
    test_loss, test_acc, test_labels, test_preds = validate_simple(model, test_loader, criterion, device)
    
    print(f"最终测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return {
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_labels': test_labels,
        'test_preds': test_preds
    }

def main():
    """主训练函数"""
    print("="*60)
    print("          MTH416 修复版重新训练脚本")
    print("="*60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"数据集大小: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'results/fixed_training_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # 训练CNN
    print("\n" + "="*40)
    print("          Q1: 自定义CNN训练")
    print("="*40)
    
    cnn_model = CustomCNN().to(device)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"CNN参数量: {cnn_params:,}")
    
    cnn_results = train_model_fixed(cnn_model, train_loader, val_loader, test_loader, "CNN", device)
    results['cnn'] = cnn_results
    
    # 保存CNN模型
    torch.save(cnn_model.state_dict(), os.path.join(save_dir, 'cnn_model_fixed.pth'))
    
    # 训练ResNet
    print("\n" + "="*40)
    print("          Q2: ResNet迁移学习训练")
    print("="*40)
    
    resnet_model = ResNetTransfer().to(device)
    total_params = sum(p.numel() for p in resnet_model.parameters())
    trainable_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"ResNet总参数: {total_params:,}")
    print(f"ResNet可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    resnet_results = train_model_fixed(resnet_model, train_loader, val_loader, test_loader, "ResNet", device)
    results['resnet'] = resnet_results
    
    # 保存ResNet模型
    torch.save(resnet_model.state_dict(), os.path.join(save_dir, 'resnet_model_fixed.pth'))
    
    # 结果对比
    print("\n" + "="*60)
    print("                 最终结果对比")
    print("="*60)
    
    print(f"CNN  - 测试准确率: {cnn_results['test_acc']:.4f} ({cnn_results['test_acc']*100:.2f}%)")
    print(f"ResNet - 测试准确率: {resnet_results['test_acc']:.4f} ({resnet_results['test_acc']*100:.2f}%)")
    
    improvement = resnet_results['test_acc'] - cnn_results['test_acc']
    print(f"性能提升: {improvement:+.4f} ({improvement*100:+.2f}个百分点)")
    
    # 保存结果
    torch.save(results, os.path.join(save_dir, 'training_results_fixed.pth'))
    
    print(f"\n💾 所有结果已保存到: {save_dir}")
    
    # 预期结果分析
    print(f"\n📊 结果分析:")
    if cnn_results['test_acc'] > 0.6:
        print("✅ CNN性能正常 (>60%)")
    else:
        print("⚠️  CNN性能仍需改进")
        
    if resnet_results['test_acc'] > cnn_results['test_acc']:
        print("✅ ResNet迁移学习表现优于CNN")
    else:
        print("⚠️  迁移学习优势不明显")
        
    if resnet_results['test_acc'] > 0.7:
        print("✅ ResNet达到期望性能 (>70%)")
    else:
        print("⚠️  ResNet性能仍需优化")

if __name__ == '__main__':
    main() 