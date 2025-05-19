import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, average_precision_score,
    roc_curve, auc, f1_score, precision_score, recall_score
)
import seaborn as sns
from config import Config

def count_parameters(model):
    """
    统计模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(labels, preds, scores):
    """
    计算评估指标
    """
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    # 计算每个类别的精确率和召回率
    precision = dict()
    recall = dict()
    ap = dict()
    
    for i in range(scores.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(
            (labels == i).astype(int),
            scores[:, i]
        )
        ap[i] = average_precision_score(
            (labels == i).astype(int),
            scores[:, i]
        )
    
    return cm, precision, recall, ap

def plot_confusion_matrix(cm, class_names):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return plt.gcf()

def plot_precision_recall_curve(precision, recall, ap, class_names):
    """
    绘制精确率-召回率曲线
    """
    plt.figure(figsize=(8, 6))
    
    for i in range(len(class_names)):
        plt.plot(recall[i], precision[i],
                label=f'{class_names[i]} (AP = {ap[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    return plt.gcf()

def print_model_summary(model, input_size=(1, 3, 128, 128)):
    """
    打印模型摘要
    """
    print("\n模型架构:")
    print(model)
    
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 计算模型大小
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"模型大小: {size_all_mb:.2f} MB")
    
    # 计算每层的参数数量
    print("\n各层参数统计:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} 参数")

class MetricsCalculator:
    """评估指标计算器"""
    def __init__(self, num_classes=Config.NUM_CLASSES):
        self.num_classes = num_classes
        self.class_names = ['normal', 'benign', 'cancer']
    
    def calculate_all_metrics(self, y_true, y_pred, y_score):
        """计算所有评估指标"""
        metrics = {}
        
        # 基础指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # 每个类别的指标
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score_binary = y_score[:, i]
            
            # PR曲线
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
            metrics[f'ap_class_{i}'] = average_precision_score(y_true_binary, y_score_binary)
            
            # ROC曲线
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            metrics[f'auc_class_{i}'] = auc(fpr, tpr)
            
            # 类别特定F1
            metrics[f'f1_class_{i}'] = f1_score(y_true_binary, (y_pred == i).astype(int))
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_pr_curves(self, y_true, y_score, save_path=None):
        """绘制PR曲线"""
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score_binary = y_score[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
            ap = average_precision_score(y_true_binary, y_score_binary)
            
            plt.plot(recall, precision, label=f'{self.class_names[i]} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curves(self, y_true, y_score, save_path=None):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))
        
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score_binary = y_score[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def print_metrics_summary(metrics):
    """打印评估指标摘要"""
    print("\n=== Metrics Summary ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['precision']:.4f}")
    print(f"Weighted Recall: {metrics['recall']:.4f}")
    print(f"Weighted F1: {metrics['f1']:.4f}")
    
    print("\nPer-class metrics:")
    for i in range(Config.NUM_CLASSES):
        print(f"\nClass {i}:")
        print(f"  F1: {metrics[f'f1_class_{i}']:.4f}")
        print(f"  AP: {metrics[f'ap_class_{i}']:.4f}")
        print(f"  AUC: {metrics[f'auc_class_{i}']:.4f}") 