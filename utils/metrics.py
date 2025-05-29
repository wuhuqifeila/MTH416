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
    Count model parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(labels, preds, scores):
    """
    Calculate evaluation metrics
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Calculate precision and recall for each class
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
    Plot confusion matrix
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add numerical labels
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
    Plot precision-recall curve
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
    Print model summary
    """
    print("\nModel Architecture:")
    print(model)
    
    print("\nModel Parameter Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")
    
    # Calculate parameters per layer
    print("\nPer-layer Parameter Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")

class MetricsCalculator:
    """Evaluation metrics calculator"""
    def __init__(self, num_classes=Config.NUM_CLASSES):
        self.num_classes = num_classes
        self.class_names = ['normal', 'benign', 'cancer']
    
    def calculate_all_metrics(self, y_true, y_pred, y_score):
        """Calculate all evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score_binary = y_score[:, i]
            
            # PR curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
            metrics[f'ap_class_{i}'] = average_precision_score(y_true_binary, y_score_binary)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            metrics[f'auc_class_{i}'] = auc(fpr, tpr)
            
            # Class-specific F1
            metrics[f'f1_class_{i}'] = f1_score(y_true_binary, (y_pred == i).astype(int))
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
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
        """Plot PR curves"""
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
        """Plot ROC curves"""
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
    """Print evaluation metrics summary"""
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
    
    print("\nPer-class metrics:")
    class_names = ['normal', 'benign', 'cancer']
    for i, name in enumerate(class_names):
        print(f"{name.capitalize()}:")
        if f'ap_class_{i}' in metrics:
            print(f"  Average Precision: {metrics[f'ap_class_{i}']:.4f}")
        if f'auc_class_{i}' in metrics:
            print(f"  AUC: {metrics[f'auc_class_{i}']:.4f}")
        if f'f1_class_{i}' in metrics:
            print(f"  F1 Score: {metrics[f'f1_class_{i}']:.4f}") 