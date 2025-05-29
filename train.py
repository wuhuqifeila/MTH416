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

# Create results save directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join('results', timestamp)
os.makedirs(save_dir, exist_ok=True)

# Save training configuration
with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            f.write(f'{key}: {value}\n')

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss function"""
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
    """Train for one epoch"""
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
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # Calculate training metrics
    metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    metrics['loss'] = running_loss / len(train_loader)
    
    # Add raw prediction data
    metrics['y_true'] = all_labels
    metrics['y_pred'] = all_preds
    metrics['y_score'] = all_scores
    
    return metrics

def validate(model, val_loader, criterion, device, metrics_calculator):
    """Validate model"""
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
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # Calculate validation metrics
    metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    metrics['loss'] = running_loss / len(val_loader)
    
    # Add raw prediction data
    metrics['y_true'] = all_labels
    metrics['y_pred'] = all_preds
    metrics['y_score'] = all_scores
    
    return metrics

def train_model(model, train_loader, val_loader, num_epochs, device, model_save_path):
    """Train model"""
    # Set loss function
    criterion = CombinedLoss(
        alpha=Config.LOSS['focal_alpha'],
        gamma=Config.LOSS['focal_gamma'],
        smoothing=Config.LOSS['label_smoothing']
    )
    
    # Set optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Set learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=Config.MIN_LR
    )
    
    # Set early stopping
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        mode='max'  # Monitor validation accuracy
    )
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, device, metrics_calculator
        )
        
        # Validation phase
        val_metrics = validate(
            model, val_loader, criterion,
            device, metrics_calculator
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print('\nTraining Metrics:')
        print_metrics_summary(train_metrics)
        print('\nValidation Metrics:')
        print_metrics_summary(val_metrics)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, model_save_path)
            print(f"\nBest model saved with validation accuracy: {best_val_acc:.4f}")
        
        # Check early stopping
        early_stopping(val_metrics['accuracy'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            # Restore best model state
            model.load_state_dict(best_model_state)
            break
    
    return train_metrics, val_metrics

def evaluate_on_test(model, test_loader, device, metrics_calculator, model_name):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    print(f"\nEvaluate {model_name} performance on test set...")
    
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
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    
    # Calculate test metrics
    test_metrics = metrics_calculator.calculate_all_metrics(
        all_labels,
        all_preds,
        all_scores
    )
    
    # Add raw prediction data
    test_metrics['y_true'] = all_labels
    test_metrics['y_pred'] = all_preds
    test_metrics['y_score'] = all_scores
    
    return test_metrics

def compare_models(cnn_metrics, resnet_metrics, cnn_params, resnet_params, save_dir):
    """Compare two models' performance"""
    print("\n" + "="*80)
    print("                     Model Performance Comparison (Q1 vs Q2)")
    print("="*80)
    
    # Parameter comparison
    print(f"\nModel Parameter Comparison:")
    print(f"├─ Q1 (Custom CNN):     {cnn_params['total']:,} parameters (all trainable)")
    print(f"├─ Q2 (ResNet Transfer Learning): {resnet_params['total']:,} parameters")
    print(f"│  ├─ Trainable Parameters:      {resnet_params['trainable']:,} parameters ({resnet_params['trainable']/resnet_params['total']*100:.2f}%)")
    print(f"│  └─ Frozen Parameters:        {resnet_params['frozen']:,} parameters ({resnet_params['frozen']/resnet_params['total']*100:.2f}%)")
    
    # Performance comparison table
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'ap_class_0', 'ap_class_1', 'ap_class_2']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Normal AP', 'Benign AP', 'Cancer AP']
    
    print(f"\n📈 Test Set Performance Comparison:")
    print(f"{'Metric':<15} {'Q1 (CNN)':<12} {'Q2 (ResNet)':<14} {'Difference':<10} {'Winner'}")
    print("-" * 70)
    
    for metric, name in zip(metrics_to_compare, metric_names):
        cnn_val = cnn_metrics.get(metric, 0)
        resnet_val = resnet_metrics.get(metric, 0)
        diff = resnet_val - cnn_val
        winner = "ResNet" if diff > 0 else "CNN" if diff < 0 else "Tie"
        
        print(f"{name:<15} {cnn_val:<12.4f} {resnet_val:<14.4f} {diff:+.4f}   {winner}")
    
    # Class-level analysis
    print(f"\n🔍 Class-level Performance Details Analysis:")
    class_names = ['Normal', 'Benign', 'Cancer']
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name} Class:")
        cnn_f1 = cnn_metrics.get(f'f1_class_{i}', 0)
        resnet_f1 = resnet_metrics.get(f'f1_class_{i}', 0)
        cnn_auc = cnn_metrics.get(f'auc_class_{i}', 0)
        resnet_auc = resnet_metrics.get(f'auc_class_{i}', 0)
        
        print(f"  F1 Score:  CNN={cnn_f1:.4f}, ResNet={resnet_f1:.4f} (Difference: {resnet_f1-cnn_f1:+.4f})")
        print(f"  AUC:     CNN={cnn_auc:.4f}, ResNet={resnet_auc:.4f} (Difference: {resnet_auc-cnn_auc:+.4f})")
    
    # Transfer learning advantage analysis
    print(f"\n💡 Transfer Learning Analysis:")
    
    # Parameter efficiency
    param_efficiency = (resnet_metrics['accuracy'] / resnet_params['trainable']) / (cnn_metrics['accuracy'] / cnn_params['total'])
    print(f"├─ Parameter Efficiency: ResNet parameter efficiency {'higher' if param_efficiency > 1 else 'lower'} than Custom CNN {param_efficiency:.2f}x")
    
    # Overall performance
    overall_better = resnet_metrics['accuracy'] > cnn_metrics['accuracy']
    acc_diff = resnet_metrics['accuracy'] - cnn_metrics['accuracy']
    print(f"├─ Overall Performance: ResNet accuracy {'higher' if overall_better else 'lower'} than CNN {abs(acc_diff)*100:.2f} percentage points")
    
    # Class preference analysis
    cancer_performance = resnet_metrics.get('f1_class_2', 0) - cnn_metrics.get('f1_class_2', 0)
    print(f"├─ Cancer Detection: ResNet cancer F1 score {'better' if cancer_performance > 0 else 'worse'} than CNN {abs(cancer_performance):.4f}")
    
    # Save comparison results
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
    
    # Save comparison results to file
    torch.save(comparison_results, os.path.join(save_dir, 'model_comparison.pth'))
    
    # Generate comparison report text
    with open(os.path.join(save_dir, 'comparison_report.txt'), 'w', encoding='utf-8') as f:
        f.write("MTH416 Deep Learning Project - Model Comparison Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. Model Parameter Comparison\n")
        f.write(f"   Q1 (Custom CNN): {cnn_params['total']:,} parameters\n")
        f.write(f"   Q2 (ResNet Transfer Learning): {resnet_params['trainable']:,} trainable parameters / {resnet_params['total']:,} total parameters\n\n")
        
        f.write("2. Test Set Performance Comparison\n")
        for metric, name in zip(metrics_to_compare, metric_names):
            cnn_val = cnn_metrics.get(metric, 0)
            resnet_val = resnet_metrics.get(metric, 0)
            f.write(f"   {name}: CNN={cnn_val:.4f}, ResNet={resnet_val:.4f}\n")
        
        f.write(f"\n3. Transfer Learning Advantage Analysis\n")
        f.write(f"   - Parameter Efficiency: {param_efficiency:.2f}\n")
        f.write(f"   - Accuracy Improvement: {acc_diff*100:+.2f}%\n")
        f.write(f"   - Cancer Detection Improvement: {cancer_performance:+.4f}\n")
    
    return comparison_results

def count_model_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params, 
        'frozen': frozen_params
    }

def main():
    # Set device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create metrics calculator
    metrics_calculator = MetricsCalculator()

    # Q1: Train Custom CNN model
    print("\nQ1: Train Custom CNN model...")
    cnn_model = CustomCNN().to(device)
    
    # Count CNN parameters
    cnn_params = count_model_parameters(cnn_model)
    print(f"\nCustom CNN Parameter Statistics:")
    print(f"├─ Total Parameters: {cnn_params['total']:,}")
    print(f"├─ Trainable Parameters: {cnn_params['trainable']:,}")
    print(f"└─ Frozen Parameters: {cnn_params['frozen']:,}")
    
    cnn_save_path = os.path.join(save_dir, 'cnn_model.pth')
    cnn_train_metrics, cnn_val_metrics = train_model(
        cnn_model, train_loader, val_loader,
        Config.NUM_EPOCHS, device, cnn_save_path
    )
    
    # Q2: Train ResNet Transfer Learning model
    print("\nQ2: Train ResNet Transfer Learning model...")
    resnet_model = ResNetModel().to(device)
    
    # Count ResNet parameters
    resnet_params = count_model_parameters(resnet_model)
    print(f"\nResNet Transfer Learning Parameter Statistics:")
    print(f"├─ Total Parameters: {resnet_params['total']:,}")
    print(f"├─ Trainable Parameters: {resnet_params['trainable']:,} ({resnet_params['trainable']/resnet_params['total']*100:.2f}%)")
    print(f"└─ Frozen Parameters: {resnet_params['frozen']:,} ({resnet_params['frozen']/resnet_params['total']*100:.2f}%)")
    
    resnet_save_path = os.path.join(save_dir, 'resnet_model.pth')
    resnet_train_metrics, resnet_val_metrics = train_model(
        resnet_model, train_loader, val_loader,
        Config.NUM_EPOCHS, device, resnet_save_path
    )
    
    # Test set evaluation
    print("\nFinal Test Set Evaluation...")
    
    # Load best model for testing
    cnn_model.load_state_dict(torch.load(cnn_save_path, map_location=device))
    resnet_model.load_state_dict(torch.load(resnet_save_path, map_location=device))
    
    # Evaluate on test set
    cnn_test_metrics = evaluate_on_test(cnn_model, test_loader, device, metrics_calculator, "Custom CNN")
    resnet_test_metrics = evaluate_on_test(resnet_model, test_loader, device, metrics_calculator, "ResNet Transfer Learning")
    
    # Q3: Generate confusion matrix and PR curves
    print("\nQ3: Generate Evaluation Charts...")
    
    # CNN model charts
    metrics_calculator.plot_confusion_matrix(
        cnn_test_metrics['confusion_matrix'],
        save_path=os.path.join(save_dir, 'q1_cnn_confusion_matrix_test.png')
    )
    metrics_calculator.plot_pr_curves(
        cnn_test_metrics['y_true'],
        cnn_test_metrics['y_score'],
        save_path=os.path.join(save_dir, 'q1_cnn_precision_recall_test.png')
    )
    
    # ResNet model charts  
    metrics_calculator.plot_confusion_matrix(
        resnet_test_metrics['confusion_matrix'],
        save_path=os.path.join(save_dir, 'q2_resnet_confusion_matrix_test.png')
    )
    metrics_calculator.plot_pr_curves(
        resnet_test_metrics['y_true'],
        resnet_test_metrics['y_score'],
        save_path=os.path.join(save_dir, 'q2_resnet_precision_recall_test.png')
    )
    
    # Q4: Model Comparison Analysis
    print("\nQ4: Model Comparison Analysis...")
    comparison_results = compare_models(
        cnn_test_metrics, resnet_test_metrics, 
        cnn_params, resnet_params, save_dir
    )
    
    # Save all results
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
    
    # Final Report Output
    print("\n" + "="*80)
    print("                            Final Experiment Results Summary")
    print("="*80)
    
    print(f"\nQ1 (Custom CNN) Test Set Performance:")
    print_metrics_summary(cnn_test_metrics)
    
    print(f"\nQ2 (ResNet Transfer Learning) Test Set Performance:")
    print_metrics_summary(resnet_test_metrics)
    
    print(f"\nAll results saved to: {save_dir}")
    print(f"├─ Model Weights: cnn_model.pth, resnet_model.pth")
    print(f"├─ Confusion Matrices: *_confusion_matrix_test.png") 
    print(f"├─ PR Curves: *_precision_recall_test.png")
    print(f"├─ Comparison Report: comparison_report.txt")
    print(f"└─ Complete Results: final_results.pth")

if __name__ == '__main__':
    main() 