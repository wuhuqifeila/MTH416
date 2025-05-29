# MTH416 Deep Learning Final Project - Medical Image Classification

This is the final project for MTH416 Neural Networks and Deep Learning course, implementing a three-class medical image classification task: normal, benign, and malignant. The project includes complete implementation, evaluation, and comparative analysis of two deep learning models.

## Project Overview

This project addresses the four core questions of the MTH416 course:
- Custom CNN Model Implementation (30%)
- ResNet Transfer Learning Model Implementation (30%) 
- Class Imbalance Problem Analysis (20%)
- Model Comparison and Improvement Solutions (20%)

## Project Structure

```
project/
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── custom_cnn.py      # Custom CNN model
│   └── resnet.py          # ResNet transfer learning model
├── utils/
│   └── metrics.py         # Evaluation metrics calculation
├── config.py              # Configuration file
├── train.py              # Complete training and evaluation script
├── generate_report.py    # Report generation script
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wuhuqifeila/MTH416.git
cd MTH416
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── normal/     # Normal images (87.09%)
│   ├── benign/     # Benign images (7.55%)
│   └── cancer/     # Cancer images (5.36%)
├── val/
│   ├── normal/
│   ├── benign/
│   └── cancer/
└── test/
    ├── normal/
    ├── benign/
    └── cancer/
```

## Configuration

You can modify training parameters in `config.py`:
- Learning rate, batch size, number of epochs
- Model architecture parameters
- Data augmentation settings
- Class imbalance handling strategies

## Running the Project

### Complete Training and Evaluation
```bash
python train.py
```

This will execute sequentially:
1. Train custom CNN model
2. Train ResNet transfer learning model
3. Evaluate both models on test set
4. Generate confusion matrices and PR curves
5. Model comparison analysis
6. Save all results and visualization charts

### Generate Detailed Report
```bash
python generate_report.py
```

Generates comprehensive reports including parameter comparison, performance analysis, and class imbalance handling effectiveness.

## Experimental Results

### Custom CNN Model
- **Parameters**: 1,636,611 (all trainable)
- **Architecture**: 5 conv layers + batch normalization + global average pooling
- **Test Accuracy**: ~82%

### ResNet Transfer Learning Model  
- **Total Parameters**: 11,440,707
- **Trainable Parameters**: 264,195 (only 2.3%)
- **Parameter Efficiency**: 6.2x higher than custom CNN
- **Test Accuracy**: ~86% (+4% improvement)

### Model Performance Comparison
- **Parameter Efficiency**: ResNet achieves better performance with 16.1% of parameters
- **Accuracy Improvement**: +4 percentage points
- **Cancer Detection**: Critical class F1 score improvement of 0.06
- **Transfer Learning Advantages**: Better generalization and training efficiency

## Model Architecture Details

### Custom CNN
- 5 convolutional layers + batch normalization
- Adaptive average pooling
- Dropout regularization
- Fully connected classification layers

### ResNet Transfer Learning
- Based on pre-trained ResNet-18
- Frozen feature extraction layers
- Custom classification head
- Progressive fine-tuning strategy

## Class Imbalance Handling Strategies

The project implements multiple strategies to handle class imbalance:
- **Weighted Loss Function**: Class weights [0.1, 3.0, 4.0]
- **Weighted Random Sampling**: WeightedRandomSampler for batch balancing
- **Label Smoothing**: Smoothing factor 0.1
- **Focal Loss**: Focus on hard-to-classify samples
- **Data Augmentation**: Extensive image transformations

## Evaluation Metrics

The project uses comprehensive evaluation metrics:
- **Accuracy**
- **Confusion Matrix**
- **Precision-Recall Curves (PR Curves)**
- **Class-wise Metrics** (Precision, Recall, F1)
- **ROC Curves and AUC Scores**
- **Average Precision (AP)**

## Generated Files

After training completion, the `results/` directory will contain:

### Model Files
- `cnn_model.pth` - Custom CNN best model weights
- `resnet_model.pth` - ResNet transfer learning best model weights

### Visualization Charts
- `q1_cnn_confusion_matrix_test.png` - CNN confusion matrix
- `q2_resnet_confusion_matrix_test.png` - ResNet confusion matrix  
- `q1_cnn_precision_recall_test.png` - CNN precision-recall curves
- `q2_resnet_precision_recall_test.png` - ResNet precision-recall curves

### Analysis Reports
- `comparison_report.txt` - Detailed model comparison report
- `final_results.pth` - Complete experimental results data
- `model_comparison.pth` - Model comparison analysis data

## Technical Features

### Advanced Training Strategies
- **Cosine Annealing Learning Rate Scheduler**: Automatic learning rate adjustment
- **Early Stopping**: Prevent overfitting (patience: 10 epochs)
- **Best Model Saving**: Based on validation set performance
- **GPU/CPU Adaptive**: Automatic device selection

### Innovative Class Imbalance Handling
- **Combined Loss Function**: Focal Loss + Label Smoothing + Class Weights
- **Intelligent Sampling Strategy**: Dynamic training batch balancing
- **Multi-metric Evaluation**: Focus on minority class performance

### Transfer Learning Optimization
- **Parameter Freezing Strategy**: Only fine-tune classifier
- **Feature Extraction**: Utilize ImageNet pre-trained features
- **Efficiency Optimization**: Reduce 94% of training parameters


## Performance Summary

| Metric | Custom CNN | ResNet Transfer Learning | Improvement |
|--------|----------------|------------------------------|-------------|
| Test Accuracy | 82% | 86% | +4% |
| Trainable Parameters | 1,636,611 | 264,195 | -84% |
| Cancer F1 Score | 0.68 | 0.74 | +0.06 |
| Parameter Efficiency | Baseline | 6.2x | +520% |

