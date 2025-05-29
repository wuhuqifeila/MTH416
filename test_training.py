import torch
import torch.nn as nn
from datetime import datetime
import os
from tqdm import tqdm

from config import Config
from data.dataset import get_data_loaders
from models.custom_cnn import CustomCNN, count_parameters
from models.resnet import ResNetTransfer, count_parameters as count_resnet_parameters
from utils.metrics import MetricsCalculator, print_metrics_summary

def quick_test():
    """快速测试训练流程"""
    print("🚀 开始快速测试...")
    
    # 设置设备
    device = torch.device(Config.DEVICE)
    print(f"使用设备: {device}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 创建指标计算器
    metrics_calculator = MetricsCalculator()
    
    # 测试CNN模型
    print("\n📊 测试自定义CNN模型...")
    cnn_model = CustomCNN().to(device)
    cnn_params = count_parameters(cnn_model)
    print(f"CNN总参数量: {cnn_params:,}")
    
    # 测试ResNet模型
    print("\n📊 测试ResNet迁移学习模型...")
    resnet_model = ResNetTransfer().to(device)
    resnet_params = count_resnet_parameters(resnet_model)
    print(f"ResNet可训练参数量: {resnet_params:,}")
    
    # 快速测试一个批次的前向传播
    print("\n🔄 测试前向传播...")
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # CNN前向传播
        cnn_outputs = cnn_model(inputs)
        print(f"CNN输出形状: {cnn_outputs.shape}")
        
        # ResNet前向传播
        resnet_outputs = resnet_model(inputs)
        print(f"ResNet输出形状: {resnet_outputs.shape}")
        
        break  # 只测试一个批次
    
    print("\n✅ 快速测试完成！模型结构正常。")
    
    # 生成参数对比报告
    print("\n" + "="*60)
    print("                  模型参数对比")
    print("="*60)
    print(f"Q1 (自定义CNN)     - 总参数: {cnn_params:,}")
    print(f"Q2 (ResNet迁移学习) - 可训练参数: {resnet_params:,}")
    print(f"参数效率提升: {cnn_params / resnet_params:.1f}倍")

def simple_evaluation():
    """简单评估现有模型"""
    device = torch.device(Config.DEVICE)
    
    # 检查是否有已训练的模型
    model_path = "best_model.pth"
    if os.path.exists(model_path):
        print(f"📂 找到已训练模型: {model_path}")
        
        # 加载数据
        _, _, test_loader = get_data_loaders()
        metrics_calculator = MetricsCalculator()
        
        # 创建模型
        model = CustomCNN().to(device)
        
        try:
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            print("📊 在测试集上评估模型...")
            
            all_preds = []
            all_labels = []
            all_scores = []
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc='评估中'):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    scores = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())
                    all_scores.extend(scores.cpu().numpy())
            
            import numpy as np
            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_scores = np.array(all_scores)
            
            # 计算指标
            test_metrics = metrics_calculator.calculate_all_metrics(
                all_labels, all_preds, all_scores
            )
            
            print("\n📈 测试集性能:")
            print_metrics_summary(test_metrics)
            
            # 生成并保存图表
            save_dir = f"results/quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(save_dir, exist_ok=True)
            
            metrics_calculator.plot_confusion_matrix(
                test_metrics['confusion_matrix'],
                save_path=os.path.join(save_dir, 'confusion_matrix.png')
            )
            
            metrics_calculator.plot_pr_curves(
                all_labels, all_scores,
                save_path=os.path.join(save_dir, 'precision_recall.png')
            )
            
            print(f"\n💾 结果已保存到: {save_dir}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    else:
        print("❌ 未找到已训练的模型文件")

if __name__ == "__main__":
    print("MTH416 深度学习项目 - 快速测试工具")
    print("="*50)
    
    # 运行快速测试
    quick_test()
    
    # 尝试评估现有模型
    simple_evaluation() 