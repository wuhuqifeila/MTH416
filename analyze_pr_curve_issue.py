import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

def simulate_problematic_predictions():
    """模拟有问题的预测来解释PR曲线异常"""
    print("🔍 分析PR曲线异常现象")
    print("="*50)
    
    # 模拟测试集分布 (基于您的数据)
    n_normal = int(305 * 0.87)    # ~265 样本
    n_benign = int(305 * 0.075)   # ~23 样本  
    n_cancer = int(305 * 0.054)   # ~17 样本
    
    print(f"模拟测试集分布:")
    print(f"Normal: {n_normal} 样本")
    print(f"Benign: {n_benign} 样本") 
    print(f"Cancer: {n_cancer} 样本")
    
    # 创建真实标签
    y_true = np.concatenate([
        np.zeros(n_normal),      # Normal = 0
        np.ones(n_benign),       # Benign = 1  
        np.full(n_cancer, 2)     # Cancer = 2
    ])
    
    # 模拟CNN的糟糕预测 (几乎全部预测为Benign)
    print(f"\n🤖 模拟CNN模型的预测模式:")
    
    # CNN: 90%预测为Benign，其他随机
    cnn_preds = np.random.choice([0, 1, 2], size=len(y_true), p=[0.05, 0.90, 0.05])
    
    # 模拟置信度分数 (非常糟糕的分布)
    cnn_scores = np.random.rand(len(y_true), 3)
    # 让Benign置信度异常高
    cnn_scores[:, 1] = np.random.uniform(0.7, 0.99, len(y_true))  
    # Normal和Cancer置信度很低
    cnn_scores[:, 0] = np.random.uniform(0.01, 0.3, len(y_true))
    cnn_scores[:, 2] = np.random.uniform(0.01, 0.1, len(y_true))
    
    # 归一化
    cnn_scores = cnn_scores / cnn_scores.sum(axis=1, keepdims=True)
    
    print(f"CNN预测分布: {np.bincount(cnn_preds, minlength=3)}")
    
    # 计算PR曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PR曲线异常现象分析', fontsize=16, fontweight='bold')
    
    class_names = ['Normal', 'Benign', 'Cancer']
    colors = ['blue', 'orange', 'green']
    
    # 绘制CNN的PR曲线
    ax = axes[0, 0]
    for i in range(3):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = cnn_scores[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
        ap = average_precision_score(y_true_binary, y_score_binary)
        
        ax.plot(recall, precision, label=f'{class_names[i]} (AP = {ap:.2f})', 
                color=colors[i], linewidth=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision') 
    ax.set_title('模拟CNN PR曲线 (类似您的结果)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 分析每个类别的问题
    axes[0, 1].text(0.1, 0.9, "Normal类别问题分析:", fontsize=12, fontweight='bold', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.8, "• 样本数量多，但模型几乎不预测", fontsize=10, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.7, "• 置信度分数普遍很低", fontsize=10, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.6, "• 导致PR曲线平滑下降", fontsize=10, transform=axes[0, 1].transAxes)
    
    axes[0, 1].text(0.1, 0.5, "Benign类别问题分析:", fontsize=12, fontweight='bold', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.4, "• 模型过度预测该类别", fontsize=10, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.3, "• 大量误报导致锯齿状曲线", fontsize=10, transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.2, "• AP值降低", fontsize=10, transform=axes[0, 1].transAxes)
    
    axes[0, 1].text(0.1, 0.1, "Cancer类别问题分析:", fontsize=12, fontweight='bold', transform=axes[0, 1].transAxes)
    axes[0, 1].text(0.1, 0.0, "• 样本极少且几乎不被预测", fontsize=10, transform=axes[0, 1].transAxes)
    axes[0, 1].axis('off')
    
    # 显示实际预测混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, cnn_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[1, 0].set_title('模拟CNN混淆矩阵')
    axes[1, 0].set_ylabel('真实标签')
    axes[1, 0].set_xlabel('预测标签')
    
    # 解决方案说明
    axes[1, 1].text(0.1, 0.9, "PR曲线异常的解决方案:", fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.8, "1️⃣ 强化Focal Loss权重", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, "2️⃣ 增加少数类别的数据增强", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, "3️⃣ 调整决策阈值", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, "4️⃣ 使用更平衡的采样策略", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, "5️⃣ 监控平衡准确率而非总准确率", fontsize=10, transform=axes[1, 1].transAxes)
    
    axes[1, 1].text(0.1, 0.2, "关键洞察:", fontsize=12, fontweight='bold', color='red', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.1, "• 奇怪的PR曲线反映了严重的类别不平衡", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.0, "• 这是医学AI中的典型挑战！", fontsize=10, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pr_curve_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 PR曲线异常的原因分析:")
    print(f"1. Normal类别 (AP=0.87):")
    print(f"   • 样本多但预测置信度低")
    print(f"   • 导致平滑但递减的PR曲线")
    
    print(f"\n2. Benign类别 (AP=0.18):")
    print(f"   • 模型过度预测该类别")
    print(f"   • 大量假阳性导致锯齿状曲线")
    
    print(f"\n3. Cancer类别 (AP=0.06):")
    print(f"   • 极少被正确识别")
    print(f"   • 几乎为水平线")
    
    print(f"\n💡 这种PR曲线形状在严重不平衡的医学数据中很常见！")
    print(f"✅ 分析图已保存: pr_curve_analysis.png")

def explain_weird_curves():
    """详细解释奇怪曲线的数学原因"""
    print(f"\n🔬 奇怪PR曲线的数学解释:")
    print(f"="*50)
    
    print(f"PR曲线异常的数学原因:")
    print(f"1. Precision = TP / (TP + FP)")
    print(f"2. Recall = TP / (TP + FN)")
    print(f"3. 当模型预测严重偏向某个类别时：")
    print(f"   • 主导类别: 高FP，低Precision")  
    print(f"   • 少数类别: 低TP，低Recall")
    print(f"   • 导致不规则的曲线形状")
    
    print(f"\n🎯 您的曲线特征符合典型的:")
    print(f"✅ 严重类别不平衡问题")
    print(f"✅ 模型预测偏差问题") 
    print(f"✅ 医学AI的常见挑战")

if __name__ == "__main__":
    simulate_problematic_predictions()
    explain_weird_curves() 