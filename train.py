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
from utils.losses import CombinedLoss

# 创建结果保存目录
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join('results', timestamp)
os.makedirs(save_dir, exist_ok=True)

# 保存训练配置
with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            f.write(f'{key}: {value}\n')

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
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # 计算训练指标
    metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    metrics['loss'] = running_loss / len(train_loader)
    
    # 添加原始预测数据
    metrics['y_true'] = all_labels
    metrics['y_pred'] = all_preds
    metrics['y_score'] = all_scores
    
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
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # 计算验证指标
    metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    metrics['loss'] = running_loss / len(val_loader)
    
    # 添加原始预测数据
    metrics['y_true'] = all_labels
    metrics['y_pred'] = all_preds
    metrics['y_score'] = all_scores
    
    return metrics

def train_model(model, train_loader, val_loader, num_epochs, device, model_save_path):
    """训练模型"""
    # 设置损失函数
    criterion = CombinedLoss(
        alpha=Config.LOSS['focal_alpha'],
        gamma=Config.LOSS['focal_gamma'],
        smoothing=Config.LOSS['label_smoothing']
    )
    
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
    
    best_val_acc = 0.0
    best_model_state = None
    
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
        
        # 保存混淆矩阵
        metrics_calculator.plot_confusion_matrix(
            val_metrics['confusion_matrix'],
            save_path=os.path.join(save_dir, f'confusion_matrix_epoch_{epoch+1}.png')
        )
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, model_save_path)
            print(f"\nBest model saved with validation accuracy: {best_val_acc:.4f}")
        
        # 检查早停
        early_stopping(val_metrics['accuracy'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # 恢复最佳模型状态
            model.load_state_dict(best_model_state)
            break
    
    return train_metrics, val_metrics

def evaluate_on_test(model, test_loader, device, metrics_calculator, model_name):
    """在测试集上评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    print(f"\n评估 {model_name} 在测试集上的性能...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Testing {model_name}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            scores = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # 计算测试指标
    test_metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    
    # 添加原始预测数据
    test_metrics['y_true'] = all_labels
    test_metrics['y_pred'] = all_preds
    test_metrics['y_score'] = all_scores
    
    return test_metrics

def compare_models(cnn_metrics, resnet_metrics, cnn_params, resnet_params, save_dir):
    """对比两个模型的性能"""
    print("\n" + "="*80)
    print("                    模型性能对比分析 (Q1 vs Q2)")
    print("="*80)
    
    # 参数量对比
    print(f"\n📊 模型参数量对比:")
    print(f"├─ Q1 (自定义CNN):     {cnn_params['total']:,} 参数 (全部可训练)")
    print(f"├─ Q2 (ResNet迁移学习): {resnet_params['total']:,} 参数")
    print(f"│  ├─ 可训练参数:      {resnet_params['trainable']:,} 参数 ({resnet_params['trainable']/resnet_params['total']*100:.2f}%)")
    print(f"│  └─ 冻结参数:        {resnet_params['frozen']:,} 参数 ({resnet_params['frozen']/resnet_params['total']*100:.2f}%)")
    
    # 性能对比表格
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'ap_class_0', 'ap_class_1', 'ap_class_2']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'Normal AP', 'Benign AP', 'Cancer AP']
    
    print(f"\n📈 测试集性能对比:")
    print(f"{'指标':<15} {'Q1 (CNN)':<12} {'Q2 (ResNet)':<14} {'差异':<10} {'优势'}")
    print("-" * 70)
    
    for metric, name in zip(metrics_to_compare, metric_names):
        cnn_val = cnn_metrics.get(metric, 0)
        resnet_val = resnet_metrics.get(metric, 0)
        diff = resnet_val - cnn_val
        winner = "ResNet" if diff > 0 else "CNN" if diff < 0 else "平局"
        
        print(f"{name:<15} {cnn_val:<12.4f} {resnet_val:<14.4f} {diff:+.4f}   {winner}")
    
    # 类别级别分析
    print(f"\n🔍 各类别性能详细分析:")
    class_names = ['Normal', 'Benign', 'Cancer']
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name} 类别:")
        cnn_f1 = cnn_metrics.get(f'f1_class_{i}', 0)
        resnet_f1 = resnet_metrics.get(f'f1_class_{i}', 0)
        cnn_auc = cnn_metrics.get(f'auc_class_{i}', 0)
        resnet_auc = resnet_metrics.get(f'auc_class_{i}', 0)
        
        print(f"  F1分数:  CNN={cnn_f1:.4f}, ResNet={resnet_f1:.4f} (差异: {resnet_f1-cnn_f1:+.4f})")
        print(f"  AUC:     CNN={cnn_auc:.4f}, ResNet={resnet_auc:.4f} (差异: {resnet_auc-cnn_auc:+.4f})")
    
    # 迁移学习优劣分析
    print(f"\n💡 迁移学习分析:")
    
    # 参数效率
    param_efficiency = (resnet_metrics['accuracy'] / resnet_params['trainable']) / (cnn_metrics['accuracy'] / cnn_params['total'])
    print(f"├─ 参数效率: ResNet的参数效率{'高于' if param_efficiency > 1 else '低于'}自定义CNN {param_efficiency:.2f}倍")
    
    # 整体性能
    overall_better = resnet_metrics['accuracy'] > cnn_metrics['accuracy']
    acc_diff = resnet_metrics['accuracy'] - cnn_metrics['accuracy']
    print(f"├─ 整体性能: ResNet准确率{'高于' if overall_better else '低于'}CNN {abs(acc_diff)*100:.2f}个百分点")
    
    # 类别偏好分析
    cancer_performance = resnet_metrics.get('f1_class_2', 0) - cnn_metrics.get('f1_class_2', 0)
    print(f"├─ 癌症检测: ResNet在癌症类别F1分数{'优于' if cancer_performance > 0 else '劣于'}CNN {abs(cancer_performance):.4f}")
    
    # 保存对比结果
    comparison_results = {
        'parameter_comparison': {
            'cnn': cnn_params,
            'resnet': resnet_params
        },
        'performance_comparison': {
            'cnn': {k: v for k, v in cnn_metrics.items() if k in metrics_to_compare},
            'resnet': {k: v for k, v in resnet_metrics.items() if k in metrics_to_compare}
        },
        'analysis': {
            'parameter_efficiency': param_efficiency,
            'accuracy_difference': acc_diff,
            'cancer_detection_improvement': cancer_performance
        }
    }
    
    # 保存对比结果到文件
    torch.save(comparison_results, os.path.join(save_dir, 'model_comparison.pth'))
    
    # 生成对比报告文本
    with open(os.path.join(save_dir, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write("MTH416 深度学习项目 - 模型对比分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 模型参数量对比\n")
        f.write(f"   Q1 (自定义CNN): {cnn_params['total']:,} 参数\n")
        f.write(f"   Q2 (ResNet迁移学习): {resnet_params['trainable']:,} 可训练参数 / {resnet_params['total']:,} 总参数\n\n")
        
        f.write("2. 测试集性能对比\n")
        for metric, name in zip(metrics_to_compare, metric_names):
            cnn_val = cnn_metrics.get(metric, 0)
            resnet_val = resnet_metrics.get(metric, 0)
            f.write(f"   {name}: CNN={cnn_val:.4f}, ResNet={resnet_val:.4f}\n")
        
        f.write(f"\n3. 迁移学习优势分析\n")
        f.write(f"   - 参数效率: {param_efficiency:.2f}\n")
        f.write(f"   - 准确率提升: {acc_diff*100:+.2f}%\n")
        f.write(f"   - 癌症检测改进: {cancer_performance:+.4f}\n")
    
    return comparison_results

def count_model_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params, 
        'frozen': frozen_params
    }

def main():
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 创建指标计算器
    metrics_calculator = MetricsCalculator()
    
    # ===============================
    # Q1: 训练自定义CNN模型
    # ===============================
    print("\n🚀 Q1: 训练自定义CNN模型...")
    cnn_model = CustomCNN().to(device)
    
    # 统计CNN参数量
    cnn_params = count_model_parameters(cnn_model)
    print(f"\n📊 自定义CNN参数统计:")
    print(f"├─ 总参数数量: {cnn_params['total']:,}")
    print(f"├─ 可训练参数: {cnn_params['trainable']:,}")
    print(f"└─ 冻结参数: {cnn_params['frozen']:,}")
    
    cnn_save_path = os.path.join(save_dir, 'cnn_model.pth')
    cnn_train_metrics, cnn_val_metrics = train_model(
        cnn_model, train_loader, val_loader,
        Config.NUM_EPOCHS, device, cnn_save_path
    )
    
    # ===============================
    # Q2: 训练ResNet迁移学习模型
    # ===============================
    print("\n🚀 Q2: 训练ResNet迁移学习模型...")
    resnet_model = ResNetModel().to(device)
    
    # 统计ResNet参数量
    resnet_params = count_model_parameters(resnet_model)
    print(f"\n📊 ResNet迁移学习参数统计:")
    print(f"├─ 总参数数量: {resnet_params['total']:,}")
    print(f"├─ 可训练参数: {resnet_params['trainable']:,} ({resnet_params['trainable']/resnet_params['total']*100:.2f}%)")
    print(f"└─ 冻结参数: {resnet_params['frozen']:,} ({resnet_params['frozen']/resnet_params['total']*100:.2f}%)")
    
    resnet_save_path = os.path.join(save_dir, 'resnet_model.pth')
    resnet_train_metrics, resnet_val_metrics = train_model(
        resnet_model, train_loader, val_loader,
        Config.NUM_EPOCHS, device, resnet_save_path
    )
    
    # ===============================
    # 测试集评估
    # ===============================
    print("\n🔍 测试集最终评估...")
    
    # 加载最佳模型进行测试
    cnn_model.load_state_dict(torch.load(cnn_save_path, map_location=device))
    resnet_model.load_state_dict(torch.load(resnet_save_path, map_location=device))
    
    # 在测试集上评估
    cnn_test_metrics = evaluate_on_test(cnn_model, test_loader, device, metrics_calculator, "自定义CNN")
    resnet_test_metrics = evaluate_on_test(resnet_model, test_loader, device, metrics_calculator, "ResNet迁移学习")
    
    # ===============================
    # Q3: 生成混淆矩阵和PR曲线
    # ===============================
    print("\n📊 Q3: 生成评估图表...")
    
    # CNN模型图表
    metrics_calculator.plot_confusion_matrix(
        cnn_test_metrics['confusion_matrix'],
        save_path=os.path.join(save_dir, 'q1_cnn_confusion_matrix_test.png')
    )
    metrics_calculator.plot_pr_curves(
        cnn_test_metrics['y_true'],
        cnn_test_metrics['y_score'],
        save_path=os.path.join(save_dir, 'q1_cnn_precision_recall_test.png')
    )
    
    # ResNet模型图表  
    metrics_calculator.plot_confusion_matrix(
        resnet_test_metrics['confusion_matrix'],
        save_path=os.path.join(save_dir, 'q2_resnet_confusion_matrix_test.png')
    )
    metrics_calculator.plot_pr_curves(
        resnet_test_metrics['y_true'],
        resnet_test_metrics['y_score'],
        save_path=os.path.join(save_dir, 'q2_resnet_precision_recall_test.png')
    )
    
    # ===============================
    # Q4: 模型对比分析
    # ===============================
    print("\n⚖️ Q4: 模型对比分析...")
    comparison_results = compare_models(
        cnn_test_metrics, resnet_test_metrics, 
        cnn_params, resnet_params, save_dir
    )
    
    # ===============================
    # 保存所有结果
    # ===============================
    final_results = {
        'models': {
            'cnn': {
                'parameters': cnn_params,
                'train_metrics': cnn_train_metrics,
                'val_metrics': cnn_val_metrics,
                'test_metrics': cnn_test_metrics
            },
            'resnet': {
                'parameters': resnet_params, 
                'train_metrics': resnet_train_metrics,
                'val_metrics': resnet_val_metrics,
                'test_metrics': resnet_test_metrics
            }
        },
        'comparison': comparison_results
    }
    
    torch.save(final_results, os.path.join(save_dir, 'final_results.pth'))
    
    # ===============================
    # 最终报告输出
    # ===============================
    print("\n" + "="*80)
    print("                           最终实验结果总结")
    print("="*80)
    
    print(f"\n📈 Q1 (自定义CNN) 测试集性能:")
    print_metrics_summary(cnn_test_metrics)
    
    print(f"\n📈 Q2 (ResNet迁移学习) 测试集性能:")
    print_metrics_summary(resnet_test_metrics)
    
    print(f"\n💾 所有结果已保存到: {save_dir}")
    print(f"├─ 模型权重: cnn_model.pth, resnet_model.pth")
    print(f"├─ 混淆矩阵: *_confusion_matrix_test.png") 
    print(f"├─ PR曲线: *_precision_recall_test.png")
    print(f"├─ 对比报告: comparison_report.txt")
    print(f"└─ 完整结果: final_results.pth")

if __name__ == '__main__':
    main() 