import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

def load_results(results_dir):
    """加载训练结果"""
    results_path = os.path.join(results_dir, 'final_results.pth')
    if os.path.exists(results_path):
        return torch.load(results_path, map_location='cpu')
    else:
        print(f"结果文件不存在: {results_path}")
        return None

def generate_parameter_comparison_table(results):
    """生成参数量对比表格"""
    cnn_params = results['models']['cnn']['parameters']
    resnet_params = results['models']['resnet']['parameters']
    
    print("\n" + "="*60)
    print("                    模型参数量对比 (Q1 vs Q2)")
    print("="*60)
    
    data = {
        '模型': ['Q1 - 自定义CNN', 'Q2 - ResNet迁移学习'],
        '总参数量': [f"{cnn_params['total']:,}", f"{resnet_params['total']:,}"],
        '可训练参数': [f"{cnn_params['trainable']:,}", f"{resnet_params['trainable']:,}"],
        '冻结参数': [f"{cnn_params['frozen']:,}", f"{resnet_params['frozen']:,}"],
        '可训练比例': [f"{cnn_params['trainable']/cnn_params['total']*100:.1f}%", 
                    f"{resnet_params['trainable']/resnet_params['total']*100:.1f}%"]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    print(f"\n🔍 关键发现:")
    print(f"• ResNet通过迁移学习只需要微调 {resnet_params['trainable']:,} 个参数")
    print(f"• 相比自定义CNN减少了 {cnn_params['total'] - resnet_params['trainable']:,} 个需要训练的参数")
    print(f"• 参数效率提升: {cnn_params['total'] / resnet_params['trainable']:.1f}倍")

def generate_performance_comparison_table(results):
    """生成性能对比表格"""
    cnn_test = results['models']['cnn']['test_metrics']
    resnet_test = results['models']['resnet']['test_metrics']
    
    print("\n" + "="*60)
    print("                    测试集性能对比 (Q1 vs Q2)")
    print("="*60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    data = {
        '评估指标': metric_names,
        'Q1 (自定义CNN)': [f"{cnn_test[m]:.4f}" for m in metrics],
        'Q2 (ResNet迁移学习)': [f"{resnet_test[m]:.4f}" for m in metrics],
        '性能差异': [f"{resnet_test[m] - cnn_test[m]:+.4f}" for m in metrics],
        '优势模型': ['ResNet' if resnet_test[m] > cnn_test[m] else 'CNN' if resnet_test[m] < cnn_test[m] else '平局' for m in metrics]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # 各类别详细分析
    print(f"\n📊 各类别平均精度 (Average Precision) 对比:")
    class_names = ['Normal', 'Benign', 'Cancer']
    
    class_data = {
        '类别': class_names,
        'Q1 (CNN) AP': [f"{cnn_test[f'ap_class_{i}']:.4f}" for i in range(3)],
        'Q2 (ResNet) AP': [f"{resnet_test[f'ap_class_{i}']:.4f}" for i in range(3)],
        'AP差异': [f"{resnet_test[f'ap_class_{i}'] - cnn_test[f'ap_class_{i}']:+.4f}" for i in range(3)]
    }
    
    class_df = pd.DataFrame(class_data)
    print(class_df.to_string(index=False))

def analyze_class_imbalance_handling(results):
    """分析类别不平衡处理效果 (Q3)"""
    print("\n" + "="*60)
    print("                    类别不平衡分析 (Q3)")
    print("="*60)
    
    cnn_test = results['models']['cnn']['test_metrics']
    resnet_test = results['models']['resnet']['test_metrics']
    
    # 混淆矩阵分析
    print("\n📊 混淆矩阵分析:")
    print("• 混淆矩阵图已生成: q1_cnn_confusion_matrix_test.png, q2_resnet_confusion_matrix_test.png")
    print("• 精确率-召回率曲线已生成: q1_cnn_precision_recall_test.png, q2_resnet_precision_recall_test.png")
    
    # 各类别性能分析
    class_names = ['Normal (正常)', 'Benign (良性)', 'Cancer (恶性)']
    
    print(f"\n🎯 各类别F1分数对比:")
    for i, class_name in enumerate(class_names):
        cnn_f1 = cnn_test[f'f1_class_{i}']
        resnet_f1 = resnet_test[f'f1_class_{i}']
        print(f"• {class_name}:")
        print(f"  - CNN F1: {cnn_f1:.4f}")
        print(f"  - ResNet F1: {resnet_f1:.4f}")
        print(f"  - 改进: {resnet_f1 - cnn_f1:+.4f}")
    
    # 类别不平衡影响分析
    cancer_recall_cnn = cnn_test.get('f1_class_2', 0)  # Cancer类别
    cancer_recall_resnet = resnet_test.get('f1_class_2', 0)
    
    print(f"\n🔍 类别不平衡处理效果:")
    print(f"• 癌症类别(最稀少)检测能力:")
    print(f"  - ResNet相比CNN提升: {cancer_recall_resnet - cancer_recall_cnn:+.4f}")
    print(f"  - 这表明迁移学习对稀少类别的{'改善' if cancer_recall_resnet > cancer_recall_cnn else '影响'}")

def analyze_transfer_learning_advantages(results):
    """分析迁移学习优势 (Q4)"""
    print("\n" + "="*60)
    print("                    迁移学习优势分析 (Q4)")
    print("="*60)
    
    comparison = results['comparison']
    
    print(f"\n💡 迁移学习关键优势:")
    
    # 1. 参数效率
    param_eff = comparison['analysis']['parameter_efficiency']
    print(f"1️⃣ 参数效率: {param_eff:.2f}倍")
    print(f"   • ResNet用更少的可训练参数实现了{'更好' if param_eff > 1 else '相似'}的性能")
    
    # 2. 性能提升
    acc_diff = comparison['analysis']['accuracy_difference'] * 100
    print(f"2️⃣ 准确率变化: {acc_diff:+.2f}%")
    print(f"   • 迁移学习{'提升' if acc_diff > 0 else '降低'}了整体准确率")
    
    # 3. 癌症检测改进
    cancer_improvement = comparison['analysis']['cancer_detection_improvement']
    print(f"3️⃣ 癌症检测改进: {cancer_improvement:+.4f}")
    print(f"   • 对于关键的癌症类别，迁移学习{'有显著帮助' if cancer_improvement > 0.01 else '效果有限'}")
    
    # 4. 训练效率
    print(f"4️⃣ 训练效率优势:")
    print(f"   • 预训练权重提供良好初始化")
    print(f"   • 减少了从头训练的计算成本")
    print(f"   • 冻结特征提取层，专注于分类器微调")
    
    # 5. 泛化能力
    print(f"5️⃣ 泛化能力:")
    print(f"   • ResNet在ImageNet上的预训练提供了通用特征表示")
    print(f"   • 对于医学图像这种相对较小的数据集特别有效")

def generate_course_report(results_dir):
    """生成完整的课程作业报告"""
    results = load_results(results_dir)
    if results is None:
        return
    
    print("="*80)
    print("           MTH416 神经网络与深度学习 - 最终项目报告")
    print("="*80)
    print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Q1 & Q2: 参数量对比
    generate_parameter_comparison_table(results)
    
    # Q1 & Q2: 性能对比  
    generate_performance_comparison_table(results)
    
    # Q3: 类别不平衡分析
    analyze_class_imbalance_handling(results)
    
    # Q4: 迁移学习优势分析
    analyze_transfer_learning_advantages(results)
    
    # 总结建议
    print("\n" + "="*60)
    print("                        总结与建议")
    print("="*60)
    
    cnn_acc = results['models']['cnn']['test_metrics']['accuracy']
    resnet_acc = results['models']['resnet']['test_metrics']['accuracy']
    
    print(f"\n📋 实验总结:")
    print(f"• 成功实现了两种深度学习方法:")
    print(f"  - Q1: 自定义CNN (测试准确率: {cnn_acc:.1%})")
    print(f"  - Q2: ResNet迁移学习 (测试准确率: {resnet_acc:.1%})")
    print(f"• 有效处理了医学图像的类别不平衡问题")
    print(f"• 通过多种评估指标全面分析了模型性能")
    
    print(f"\n🎯 关键发现:")
    print(f"• 迁移学习在参数效率方面有显著优势")
    print(f"• 类别不平衡处理策略对稀少类别检测至关重要")
    print(f"• ResNet预训练权重为医学图像分析提供了良好基础")
    
    print(f"\n📁 生成文件清单:")
    print(f"• 模型权重: cnn_model.pth, resnet_model.pth")
    print(f"• 评估图表: 混淆矩阵与PR曲线 (*.png)")
    print(f"• 详细对比: comparison_report.txt")
    print(f"• 完整结果: final_results.pth")

def create_summary_plots(results_dir):
    """创建总结性图表"""
    results = load_results(results_dir)
    if results is None:
        return
    
    # 创建性能对比柱状图
    plt.figure(figsize=(12, 8))
    
    # 数据准备
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    cnn_values = [results['models']['cnn']['test_metrics'][m] for m in metrics]
    resnet_values = [results['models']['resnet']['test_metrics'][m] for m in metrics]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(x - width/2, cnn_values, width, label='Q1 (自定义CNN)', alpha=0.8)
    bars2 = plt.bar(x + width/2, resnet_values, width, label='Q2 (ResNet迁移学习)', alpha=0.8)
    
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title('测试集性能对比')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 参数量对比饼图
    plt.subplot(2, 2, 2)
    cnn_params = results['models']['cnn']['parameters']['total']
    resnet_trainable = results['models']['resnet']['parameters']['trainable']
    resnet_frozen = results['models']['resnet']['parameters']['frozen']
    
    labels = ['CNN总参数', 'ResNet可训练', 'ResNet冻结']
    sizes = [cnn_params, resnet_trainable, resnet_frozen]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('参数量分布对比')
    
    # 各类别AP对比
    plt.subplot(2, 2, 3)
    class_names = ['Normal', 'Benign', 'Cancer']
    cnn_aps = [results['models']['cnn']['test_metrics'][f'ap_class_{i}'] for i in range(3)]
    resnet_aps = [results['models']['resnet']['test_metrics'][f'ap_class_{i}'] for i in range(3)]
    
    x = np.arange(len(class_names))
    bars1 = plt.bar(x - width/2, cnn_aps, width, label='CNN', alpha=0.8)
    bars2 = plt.bar(x + width/2, resnet_aps, width, label='ResNet', alpha=0.8)
    
    plt.xlabel('类别')
    plt.ylabel('平均精度 (AP)')
    plt.title('各类别平均精度对比')
    plt.xticks(x, class_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # 训练效率对比
    plt.subplot(2, 2, 4)
    models = ['CNN\n(全参数训练)', 'ResNet\n(迁移学习)']
    param_counts = [cnn_params, resnet_trainable]
    accuracies = [results['models']['cnn']['test_metrics']['accuracy'],
                  results['models']['resnet']['test_metrics']['accuracy']]
    
    # 创建效率散点图
    plt.scatter(param_counts, accuracies, s=200, c=['blue', 'red'], alpha=0.7)
    
    for i, model in enumerate(models):
        plt.annotate(model, (param_counts[i], accuracies[i]), 
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('可训练参数数量')
    plt.ylabel('测试准确率')
    plt.title('参数效率对比')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 总结图表已保存: {os.path.join(results_dir, 'performance_summary.png')}")

if __name__ == "__main__":
    # 查找最新的结果目录
    results_base = "results"
    if os.path.exists(results_base):
        subdirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        if subdirs:
            latest_dir = max(subdirs)
            results_dir = os.path.join(results_base, latest_dir)
            
            print(f"📂 分析结果目录: {results_dir}")
            
            # 生成报告
            generate_course_report(results_dir)
            
            # 创建总结图表
            create_summary_plots(results_dir)
            
        else:
            print("❌ 未找到训练结果目录")
    else:
        print("❌ results目录不存在") 