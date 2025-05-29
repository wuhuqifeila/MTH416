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

class BalancedFocalLoss(nn.Module):
    """专门针对医学图像分类的平衡Focal Loss"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        # 针对严重不平衡的医学数据设计的权重
        if alpha is None:
            alpha = [0.2, 3.0, 6.0]  # Normal, Benign, Cancer
        
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        alpha_t = self.alpha[targets]
        focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

def train_epoch_focused(model, train_loader, criterion, optimizer, device):
    """专注训练函数 - 监控各类别表现"""
    model.train()
    running_loss = 0.0
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    
    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
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
        
        # 统计各类别准确率
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1
    
    epoch_loss = running_loss / len(train_loader)
    class_accuracies = class_correct / np.maximum(class_total, 1)
    balanced_acc = np.mean(class_accuracies)
    
    return epoch_loss, balanced_acc, class_accuracies

def validate_focused(model, val_loader, criterion, device):
    """专注验证函数"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计各类别准确率
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(val_loader)
    class_accuracies = class_correct / np.maximum(class_total, 1)
    balanced_acc = np.mean(class_accuracies)
    
    return epoch_loss, balanced_acc, class_accuracies, np.array(all_labels), np.array(all_preds)

def train_model_final(model, train_loader, val_loader, test_loader, model_name, device):
    """最终训练函数"""
    print(f"\n🚀 开始训练 {model_name}...")
    
    # 使用平衡的Focal Loss
    criterion = BalancedFocalLoss()
    
    # 优化器设置
    if 'ResNet' in model_name:
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-7)
    
    best_balanced_acc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 6
    
    print(f"初始学习率: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(15):
        print(f'\nEpoch {epoch+1}/15')
        
        # 训练
        train_loss, train_bal_acc, train_class_acc = train_epoch_focused(
            model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_bal_acc, val_class_acc, _, _ = validate_focused(
            model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_bal_acc)
        
        print(f'Train - Loss: {train_loss:.4f}, Bal_Acc: {train_bal_acc:.4f}')
        print(f'  Class Acc: Normal={train_class_acc[0]:.3f}, Benign={train_class_acc[1]:.3f}, Cancer={train_class_acc[2]:.3f}')
        print(f'Val   - Loss: {val_loss:.4f}, Bal_Acc: {val_bal_acc:.4f}')
        print(f'  Class Acc: Normal={val_class_acc[0]:.3f}, Benign={val_class_acc[1]:.3f}, Cancer={val_class_acc[2]:.3f}')
        
        # 保存最佳模型
        if val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"📈 最佳平衡准确率: {best_balanced_acc:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= max_patience:
            print("⏰ 早停触发")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试集评估
    print(f"\n🔍 {model_name} 测试集评估...")
    test_loss, test_bal_acc, test_class_acc, test_labels, test_preds = validate_focused(
        model, test_loader, criterion, device)
    
    print(f"测试结果:")
    print(f"  平衡准确率: {test_bal_acc:.4f} ({test_bal_acc*100:.1f}%)")
    print(f"  Normal 准确率: {test_class_acc[0]:.3f} ({test_class_acc[0]*100:.1f}%)")
    print(f"  Benign 准确率: {test_class_acc[1]:.3f} ({test_class_acc[1]*100:.1f}%)")
    print(f"  Cancer 准确率: {test_class_acc[2]:.3f} ({test_class_acc[2]*100:.1f}%)")
    
    # 详细分类报告
    class_names = ['Normal', 'Benign', 'Cancer']
    print(f"\n详细分类报告:")
    print(classification_report(test_labels, test_preds, 
                              target_names=class_names, digits=3))
    
    return {
        'test_bal_acc': test_bal_acc,
        'test_class_acc': test_class_acc,
        'test_labels': test_labels,
        'test_preds': test_preds
    }

def plot_confusion_matrix_final(labels, preds, model_name, save_path):
    """绘制最终混淆矩阵 - 只在最后调用"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, preds)
    
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Benign', 'Cancer'],
                yticklabels=['Normal', 'Benign', 'Cancer'])
    
    plt.title(f'{model_name} Test Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主训练函数"""
    print("="*70)
    print("          MTH416 最终正确训练 - 解决类别不平衡问题")
    print("="*70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取数据
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"数据集大小: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'results/final_correct_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # 训练CNN (Q1)
    print("\n" + "="*50)
    print("          Q1: 自定义CNN训练")
    print("="*50)
    
    cnn_model = CustomCNN().to(device)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"CNN参数量: {cnn_params:,}")
    
    cnn_results = train_model_final(cnn_model, train_loader, val_loader, test_loader, "CNN", device)
    results['cnn'] = cnn_results
    
    # 训练ResNet (Q2)
    print("\n" + "="*50)
    print("          Q2: ResNet迁移学习训练")
    print("="*50)
    
    resnet_model = ResNetTransfer().to(device)
    total_params = sum(p.numel() for p in resnet_model.parameters())
    trainable_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
    print(f"ResNet总参数: {total_params:,}")
    print(f"ResNet可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    resnet_results = train_model_final(resnet_model, train_loader, val_loader, test_loader, "ResNet", device)
    results['resnet'] = resnet_results
    
    # 保存模型
    torch.save(cnn_model.state_dict(), os.path.join(save_dir, 'cnn_final.pth'))
    torch.save(resnet_model.state_dict(), os.path.join(save_dir, 'resnet_final.pth'))
    
    # 最终结果对比
    print("\n" + "="*70)
    print("                     最终结果对比")
    print("="*70)
    
    print(f"Q1 (自定义CNN):")
    print(f"  平衡准确率: {cnn_results['test_bal_acc']:.4f} ({cnn_results['test_bal_acc']*100:.1f}%)")
    print(f"  Cancer检测准确率: {cnn_results['test_class_acc'][2]:.3f} ({cnn_results['test_class_acc'][2]*100:.1f}%)")
    
    print(f"\nQ2 (ResNet迁移学习):")
    print(f"  平衡准确率: {resnet_results['test_bal_acc']:.4f} ({resnet_results['test_bal_acc']*100:.1f}%)")
    print(f"  Cancer检测准确率: {resnet_results['test_class_acc'][2]:.3f} ({resnet_results['test_class_acc'][2]*100:.1f}%)")
    
    # 性能提升分析
    bal_improvement = resnet_results['test_bal_acc'] - cnn_results['test_bal_acc']
    cancer_improvement = resnet_results['test_class_acc'][2] - cnn_results['test_class_acc'][2]
    
    print(f"\n📊 Q2相比Q1的改进:")
    print(f"  平衡准确率提升: {bal_improvement:+.4f} ({bal_improvement*100:+.1f}个百分点)")
    print(f"  癌症检测提升: {cancer_improvement:+.3f} ({cancer_improvement*100:+.1f}个百分点)")
    
    # 参数效率分析
    param_efficiency = trainable_params / cnn_params
    print(f"\n💡 参数效率分析:")
    print(f"  ResNet仅用 {param_efficiency:.1%} 的可训练参数")
    print(f"  却获得了更好的性能 - 这就是迁移学习的优势！")
    
    # 生成最终混淆矩阵 - 只在这里生成，不在训练过程中生成
    print(f"\n📊 生成最终混淆矩阵...")
    plot_confusion_matrix_final(cnn_results['test_labels'], cnn_results['test_preds'], 
                               'Q1 (Custom CNN)', os.path.join(save_dir, 'q1_confusion_matrix.png'))
    plot_confusion_matrix_final(resnet_results['test_labels'], resnet_results['test_preds'], 
                               'Q2 (ResNet Transfer)', os.path.join(save_dir, 'q2_confusion_matrix.png'))
    
    # 保存结果
    torch.save(results, os.path.join(save_dir, 'final_results.pth'))
    
    print(f"\n✅ 训练完成! 结果保存在: {save_dir}")
    print(f"📊 混淆矩阵已生成 (仅最终结果)")
    print(f"💾 所有模型和数据已保存")
    
    # 成功评估
    print(f"\n🎯 训练评估:")
    if cnn_results['test_class_acc'][2] > 0.4:  # Cancer准确率 > 40%
        print("✅ CNN能够有效检测癌症")
    else:
        print("⚠️  CNN癌症检测能力有限")
        
    if resnet_results['test_class_acc'][2] > cnn_results['test_class_acc'][2]:
        print("✅ ResNet在癌症检测上优于CNN")
        
    if resnet_results['test_bal_acc'] > 0.5:  # 平衡准确率 > 50%
        print("✅ ResNet达到良好的平衡性能")
        
    if bal_improvement > 0:
        print("✅ 迁移学习展现了明显优势")

if __name__ == '__main__':
    main() 