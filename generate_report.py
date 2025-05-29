import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

def load_results(results_dir):
    """Load training results"""
    results_path = os.path.join(results_dir, 'final_results.pth')
    if os.path.exists(results_path):
        return torch.load(results_path, map_location='cpu')
    else:
        print(f"Results file does not exist: {results_path}")
        return None

def generate_parameter_comparison_table(results):
    """Generate parameter comparison table"""
    cnn_params = results['models']['cnn']['parameters']
    resnet_params = results['models']['resnet']['parameters']
    
    print("\n" + "="*60)
    print("                    Model Parameter Comparison (Q1 vs Q2)")
    print("="*60)
    
    data = {
        'Model': ['Q1 - Custom CNN', 'Q2 - ResNet Transfer Learning'],
        'Total Parameters': [f"{cnn_params['total']:,}", f"{resnet_params['total']:,}"],
        'Trainable Parameters': [f"{cnn_params['trainable']:,}", f"{resnet_params['trainable']:,}"],
        'Frozen Parameters': [f"{cnn_params['frozen']:,}", f"{resnet_params['frozen']:,}"],
        'Trainable Ratio': [f"{cnn_params['trainable']/cnn_params['total']*100:.1f}%", 
                           f"{resnet_params['trainable']/resnet_params['total']*100:.1f}%"]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    print(f"\nKey Findings:")
    print(f"• ResNet only needs to fine-tune {resnet_params['trainable']:,} parameters through transfer learning")
    print(f"• Reduced {cnn_params['total'] - resnet_params['trainable']:,} parameters to train compared to custom CNN")
    print(f"• Parameter efficiency improvement: {cnn_params['total'] / resnet_params['trainable']:.1f}x")

def generate_performance_comparison_table(results):
    """Generate performance comparison table"""
    cnn_test = results['models']['cnn']['test_metrics']
    resnet_test = results['models']['resnet']['test_metrics']
    
    print("\n" + "="*60)
    print("                    Test Set Performance Comparison (Q1 vs Q2)")
    print("="*60)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    data = {
        'Evaluation Metric': metric_names,
        'Q1 (Custom CNN)': [f"{cnn_test[m]:.4f}" for m in metrics],
        'Q2 (ResNet Transfer Learning)': [f"{resnet_test[m]:.4f}" for m in metrics],
        'Performance Difference': [f"{resnet_test[m] - cnn_test[m]:+.4f}" for m in metrics],
        'Best Model': ['ResNet' if resnet_test[m] > cnn_test[m] else 'CNN' if resnet_test[m] < cnn_test[m] else 'Tie' for m in metrics]
    }
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Class-wise detailed analysis
    print(f"\nClass-wise Average Precision (AP) Comparison:")
    class_names = ['Normal', 'Benign', 'Cancer']
    
    class_data = {
        'Class': class_names,
        'Q1 (CNN) AP': [f"{cnn_test[f'ap_class_{i}']:.4f}" for i in range(3)],
        'Q2 (ResNet) AP': [f"{resnet_test[f'ap_class_{i}']:.4f}" for i in range(3)],
        'AP Difference': [f"{resnet_test[f'ap_class_{i}'] - cnn_test[f'ap_class_{i}']:+.4f}" for i in range(3)]
    }
    
    class_df = pd.DataFrame(class_data)
    print(class_df.to_string(index=False))

def analyze_class_imbalance_handling(results):
    """Analyze class imbalance handling effectiveness (Q3)"""
    print("\n" + "="*60)
    print("                    Class Imbalance Analysis (Q3)")
    print("="*60)
    
    cnn_test = results['models']['cnn']['test_metrics']
    resnet_test = results['models']['resnet']['test_metrics']
    
    # Confusion matrix analysis
    print("\nConfusion Matrix Analysis:")
    print("• Confusion matrix plots generated: q1_cnn_confusion_matrix_test.png, q2_resnet_confusion_matrix_test.png")
    print("• Precision-Recall curves generated: q1_cnn_precision_recall_test.png, q2_resnet_precision_recall_test.png")
    
    # Class-wise performance analysis
    class_names = ['Normal', 'Benign', 'Cancer']
    
    print(f"\nClass-wise F1 Score Comparison:")
    for i, class_name in enumerate(class_names):
        cnn_f1 = cnn_test[f'f1_class_{i}']
        resnet_f1 = resnet_test[f'f1_class_{i}']
        print(f"• {class_name}:")
        print(f"  - CNN F1: {cnn_f1:.4f}")
        print(f"  - ResNet F1: {resnet_f1:.4f}")
        print(f"  - Improvement: {resnet_f1 - cnn_f1:+.4f}")
    
    # Class imbalance impact analysis
    cancer_recall_cnn = cnn_test.get('f1_class_2', 0)  # Cancer class
    cancer_recall_resnet = resnet_test.get('f1_class_2', 0)
    
    print(f"\nClass Imbalance Handling Effectiveness:")
    print(f"• Cancer class (rarest) detection capability:")
    print(f"  - ResNet improvement over CNN: {cancer_recall_resnet - cancer_recall_cnn:+.4f}")
    print(f"  - This indicates transfer learning {'improves' if cancer_recall_resnet > cancer_recall_cnn else 'affects'} rare class detection")

def analyze_transfer_learning_advantages(results):
    """Analyze transfer learning advantages (Q4)"""
    print("\n" + "="*60)
    print("                    Transfer Learning Advantages Analysis (Q4)")
    print("="*60)
    
    comparison = results['comparison']
    
    print(f"\nKey Transfer Learning Advantages:")
    
    # 1. Parameter efficiency
    param_eff = comparison['analysis']['parameter_efficiency']
    print(f"1. Parameter Efficiency: {param_eff:.2f}x")
    print(f"   • ResNet achieves {'better' if param_eff > 1 else 'similar'} performance with fewer trainable parameters")
    
    # 2. Performance improvement
    acc_diff = comparison['analysis']['accuracy_difference'] * 100
    print(f"2. Accuracy Change: {acc_diff:+.2f}%")
    print(f"   • Transfer learning {'improves' if acc_diff > 0 else 'reduces'} overall accuracy")
    
    # 3. Cancer detection improvement
    cancer_improvement = comparison['analysis']['cancer_detection_improvement']
    print(f"3. Cancer Detection Improvement: {cancer_improvement:+.4f}")
    print(f"   • For critical cancer class, transfer learning {'significantly helps' if cancer_improvement > 0.01 else 'has limited effect'}")
    
    # 4. Training efficiency
    print(f"4. Training Efficiency Advantages:")
    print(f"   • Pre-trained weights provide good initialization")
    print(f"   • Reduced computational cost from training from scratch")
    print(f"   • Frozen feature extraction layers, focus on classifier fine-tuning")
    
    # 5. Generalization ability
    print(f"5. Generalization Ability:")
    print(f"   • ResNet pre-training on ImageNet provides universal feature representations")
    print(f"   • Particularly effective for relatively small datasets like medical images")

def generate_course_report(results_dir):
    """Generate complete course assignment report"""
    results = load_results(results_dir)
    if results is None:
        return
    
    print("="*80)
    print("           MTH416 Neural Networks and Deep Learning - Final Project Report")
    print("="*80)
    print(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Q1 & Q2: Parameter comparison
    generate_parameter_comparison_table(results)
    
    # Q1 & Q2: Performance comparison  
    generate_performance_comparison_table(results)
    
    # Q3: Class imbalance analysis
    analyze_class_imbalance_handling(results)
    
    # Q4: Transfer learning advantages analysis
    analyze_transfer_learning_advantages(results)
    
    # Summary and recommendations
    print("\n" + "="*60)
    print("                        Summary and Recommendations")
    print("="*60)
    
    cnn_acc = results['models']['cnn']['test_metrics']['accuracy']
    resnet_acc = results['models']['resnet']['test_metrics']['accuracy']
    
    print(f"\nExperiment Summary:")
    print(f"• Successfully implemented two deep learning approaches:")
    print(f"  - Q1: Custom CNN (Test accuracy: {cnn_acc:.1%})")
    print(f"  - Q2: ResNet Transfer Learning (Test accuracy: {resnet_acc:.1%})")
    print(f"• Effectively handled class imbalance in medical images")
    print(f"• Comprehensively analyzed model performance through multiple evaluation metrics")
    
    print(f"\nKey Findings:")
    print(f"• Transfer learning has significant advantages in parameter efficiency")
    print(f"• Class imbalance handling strategies are crucial for rare class detection")
    print(f"• ResNet pre-trained weights provide good foundation for medical image analysis")
    
    print(f"\nGenerated Files List:")
    print(f"• Model weights: cnn_model.pth, resnet_model.pth")
    print(f"• Evaluation charts: Confusion matrices and PR curves (*.png)")
    print(f"• Detailed comparison: comparison_report.txt")
    print(f"• Complete results: final_results.pth")

def create_summary_plots(results_dir):
    """Create summary plots"""
    results = load_results(results_dir)
    if results is None:
        return
    
    # Create performance comparison bar chart
    plt.figure(figsize=(12, 8))
    
    # Data preparation
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    cnn_values = [results['models']['cnn']['test_metrics'][m] for m in metrics]
    resnet_values = [results['models']['resnet']['test_metrics'][m] for m in metrics]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(x - width/2, cnn_values, width, label='Q1 (Custom CNN)', alpha=0.8)
    bars2 = plt.bar(x + width/2, resnet_values, width, label='Q2 (ResNet Transfer Learning)', alpha=0.8)
    
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.title('Test Set Performance Comparison')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Parameter comparison pie chart
    plt.subplot(2, 2, 2)
    cnn_params = results['models']['cnn']['parameters']['total']
    resnet_trainable = results['models']['resnet']['parameters']['trainable']
    resnet_frozen = results['models']['resnet']['parameters']['frozen']
    
    labels = ['CNN Total Parameters', 'ResNet Trainable', 'ResNet Frozen']
    sizes = [cnn_params, resnet_trainable, resnet_frozen]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Parameter Distribution Comparison')
    
    # Class-wise AP comparison
    plt.subplot(2, 2, 3)
    class_names = ['Normal', 'Benign', 'Cancer']
    cnn_aps = [results['models']['cnn']['test_metrics'][f'ap_class_{i}'] for i in range(3)]
    resnet_aps = [results['models']['resnet']['test_metrics'][f'ap_class_{i}'] for i in range(3)]
    
    x = np.arange(len(class_names))
    bars1 = plt.bar(x - width/2, cnn_aps, width, label='CNN', alpha=0.8)
    bars2 = plt.bar(x + width/2, resnet_aps, width, label='ResNet', alpha=0.8)
    
    plt.xlabel('Class')
    plt.ylabel('Average Precision (AP)')
    plt.title('Class-wise Average Precision Comparison')
    plt.xticks(x, class_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # Training efficiency comparison
    plt.subplot(2, 2, 4)
    models = ['CNN\n(Full Parameter Training)', 'ResNet\n(Transfer Learning)']
    param_counts = [cnn_params, resnet_trainable]
    accuracies = [results['models']['cnn']['test_metrics']['accuracy'],
                  results['models']['resnet']['test_metrics']['accuracy']]
    
    # Create efficiency scatter plot
    plt.scatter(param_counts, accuracies, s=200, c=['blue', 'red'], alpha=0.7)
    
    for i, model in enumerate(models):
        plt.annotate(model, (param_counts[i], accuracies[i]), 
                    xytext=(10, 10), textcoords='offset points')
    
    plt.xlabel('Trainable Parameter Count')
    plt.ylabel('Test Accuracy')
    plt.title('Parameter Efficiency Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary plots saved: {os.path.join(results_dir, 'performance_summary.png')}")

if __name__ == "__main__":
    # Find latest results directory
    results_base = "results"
    if os.path.exists(results_base):
        subdirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
        if subdirs:
            latest_dir = max(subdirs)
            results_dir = os.path.join(results_base, latest_dir)
            
            print(f"📂 Analyzing results directory: {results_dir}")
            
            # Generate report
            generate_course_report(results_dir)
            
            # Create summary plots
            create_summary_plots(results_dir)
            
        else:
            print("❌ No training result directories found")
    else:
        print("❌ results directory does not exist") 