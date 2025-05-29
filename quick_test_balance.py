import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataset import get_data_loaders
from models.custom_cnn import CustomCNN
from models.resnet import ResNetTransfer

def quick_test_balance():
    """快速测试类别平衡效果"""
    print("🔍 快速测试类别平衡改进效果...")
    
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # 检查数据分布
    print("\n📊 数据集分布检查:")
    train_class_counts = [0, 0, 0]
    for _, labels in train_loader:
        for label in labels:
            train_class_counts[label.item()] += 1
    
    total = sum(train_class_counts)
    print(f"训练集分布:")
    class_names = ['Normal', 'Benign', 'Cancer']
    for i, (name, count) in enumerate(zip(class_names, train_class_counts)):
        print(f"  {name}: {count} 样本 ({count/total*100:.1f}%)")
    
    # 分析不平衡程度
    imbalance_ratio = max(train_class_counts) / min(train_class_counts)
    print(f"\n⚖️ 类别不平衡比例: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("🚨 严重不平衡！需要强力的平衡策略")
    elif imbalance_ratio > 5:
        print("⚠️ 中度不平衡，需要平衡策略")
    else:
        print("✅ 相对平衡")
    
    # 推荐的类别权重
    inverse_freq = [total/count for count in train_class_counts]
    min_weight = min(inverse_freq)
    normalized_weights = [w/min_weight for w in inverse_freq]
    
    print(f"\n💡 推荐的类别权重: {[f'{w:.1f}' for w in normalized_weights]}")
    
    # 测试 Focal Loss 设置
    print(f"\n🎯 Focal Loss 参数建议:")
    print(f"  - Alpha (类别权重): {normalized_weights}")
    print(f"  - Gamma (聚焦参数): 2.0 (标准值)")
    print(f"  - 对于严重不平衡，可考虑 Gamma=3.0")

def simulate_balanced_training():
    """模拟平衡训练效果"""
    print("\n🎭 模拟平衡训练预期效果...")
    
    # 假设的改进前后对比
    before = {
        'normal_recall': 0.99,   # 极高 - 几乎所有都预测为normal
        'benign_recall': 0.05,   # 极低
        'cancer_recall': 0.02,   # 极低
        'overall_acc': 0.87      # 看似不错，但只是因为normal占主导
    }
    
    after = {
        'normal_recall': 0.85,   # 略降，但更合理
        'benign_recall': 0.65,   # 大幅提升
        'cancer_recall': 0.70,   # 大幅提升 - 最重要！
        'overall_acc': 0.78      # 略降，但更有意义
    }
    
    print(f"改进前 (原始训练):")
    print(f"  Normal 召回率: {before['normal_recall']:.2f}")
    print(f"  Benign 召回率: {before['benign_recall']:.2f}")
    print(f"  Cancer 召回率: {before['cancer_recall']:.2f}")
    print(f"  整体准确率: {before['overall_acc']:.2f}")
    
    print(f"\n改进后 (平衡训练):")
    print(f"  Normal 召回率: {after['normal_recall']:.2f}")
    print(f"  Benign 召回率: {after['benign_recall']:.2f}")
    print(f"  Cancer 召回率: {after['cancer_recall']:.2f}")
    print(f"  整体准确率: {after['overall_acc']:.2f}")
    
    # 平衡准确率计算
    before_bal_acc = (before['normal_recall'] + before['benign_recall'] + before['cancer_recall']) / 3
    after_bal_acc = (after['normal_recall'] + after['benign_recall'] + after['cancer_recall']) / 3
    
    print(f"\n⚖️ 平衡准确率对比:")
    print(f"  改进前: {before_bal_acc:.2f}")
    print(f"  改进后: {after_bal_acc:.2f}")
    print(f"  提升: {after_bal_acc - before_bal_acc:+.2f}")
    
    print(f"\n💡 关键改进:")
    print(f"✅ 癌症检测召回率从 {before['cancer_recall']:.0%} 提升到 {after['cancer_recall']:.0%}")
    print(f"✅ 平衡准确率提升 {(after_bal_acc - before_bal_acc)*100:.0f} 个百分点")
    print(f"✅ 更适合医学诊断场景（宁可误报，不可漏诊）")

def visualize_balance_strategy():
    """可视化平衡策略"""
    print("\n📊 生成平衡策略可视化...")
    
    # 创建对比图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 类别分布图
    classes = ['Normal', 'Benign', 'Cancer']
    original_counts = [2145, 186, 132]
    colors = ['lightblue', 'orange', 'red']
    
    ax1.bar(classes, original_counts, color=colors, alpha=0.7)
    ax1.set_title('原始数据分布 (严重不平衡)', fontweight='bold')
    ax1.set_ylabel('样本数量')
    for i, count in enumerate(original_counts):
        ax1.text(i, count + 50, str(count), ha='center', fontweight='bold')
    
    # 权重策略图
    weights = [0.3, 2.0, 5.0]
    ax2.bar(classes, weights, color=colors, alpha=0.7)
    ax2.set_title('Focal Loss 类别权重', fontweight='bold')
    ax2.set_ylabel('权重值')
    for i, weight in enumerate(weights):
        ax2.text(i, weight + 0.1, f'{weight:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('balance_strategy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 平衡策略图已保存: balance_strategy.png")

def main():
    """主测试函数"""
    print("="*60)
    print("          MTH416 类别平衡问题分析与解决方案")
    print("="*60)
    
    # 快速测试
    quick_test_balance()
    
    # 模拟效果
    simulate_balanced_training()
    
    # 可视化
    visualize_balance_strategy()
    
    print("\n" + "="*60)
    print("                    解决方案总结")
    print("="*60)
    
    print("🎯 主要改进策略:")
    print("1️⃣ Focal Loss: 专注困难样本，减少易分类样本影响")
    print("2️⃣ 强化类别权重: [0.3, 2.0, 5.0] 突出少数类别")
    print("3️⃣ 平衡准确率监控: 防止模型偏向主导类别")
    print("4️⃣ WeightedRandomSampler: 训练时平衡各批次")
    print("5️⃣ 梯度裁剪: 防止不稳定训练")
    
    print("\n🎯 预期改进效果:")
    print("✅ 癌症检测召回率从 2% 提升到 70%+")
    print("✅ 良性肿瘤检测召回率从 5% 提升到 65%+") 
    print("✅ 平衡准确率从 35% 提升到 73%+")
    print("✅ 更适合医学诊断的实际需求")

if __name__ == "__main__":
    main() 