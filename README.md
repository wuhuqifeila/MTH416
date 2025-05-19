# Medical Image Classification with Deep Learning

This project implements two deep learning models for medical image classification: a custom CNN and a ResNet-based transfer learning model. The models are trained to classify medical images into three categories: normal, benign, and cancer.

## Project Structure

```
project/
├── data/
│   └── dataset.py          # Dataset loading and preprocessing
├── models/
│   ├── custom_cnn.py      # Custom CNN model
│   └── resnet.py          # ResNet transfer learning model
├── utils/
│   └── metrics.py         # Evaluation metrics
├── config.py              # Configuration file
├── train.py              # Training script
└── requirements.txt      # Project dependencies
```

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- CUDA (optional, for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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
│   ├── normal/
│   ├── benign/
│   └── cancer/
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

You can modify the training parameters in `config.py`:
- Learning rate
- Batch size
- Number of epochs
- Model architecture parameters
- Data augmentation settings

## Training

To train both models:
```bash
python train.py
```

This will:
1. Train the custom CNN model
2. Train the ResNet transfer learning model
3. Evaluate both models
4. Generate performance metrics and visualizations

## Results

The training results will be saved in the `results` directory:
- Confusion matrices for both models
- Precision-recall curves
- Model performance metrics

Current Model Performance:
- Validation Accuracy: 86.47%
- Weighted F1 Score: 0.8261
- AUC Scores:
  * Normal: 0.8403
  * Benign: 0.8454
  * Cancer: 0.7842

## Model Architecture

### Custom CNN
- 4 convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Fully connected layers

### ResNet Transfer Learning
- Based on pretrained ResNet-18
- Fine-tuned for medical image classification
- Custom fully connected layers
- Progressive unfreezing strategy
- Feature extraction with gradual fine-tuning

## Class Imbalance Handling

The project implements several strategies to handle class imbalance:
- Weighted loss function with class weights [0.1, 3.0, 4.0]
- WeightedRandomSampler for balanced batch sampling
- Label smoothing (smoothing factor: 0.1)
- Extensive data augmentation

## Evaluation Metrics

The project evaluates models using:
- Accuracy
- Confusion Matrix
- Precision-Recall Curves
- Class-wise metrics (Precision, Recall, F1)
- ROC curves and AUC scores
- Average Precision

## Notes

- The models are trained on GPU if available, otherwise CPU
- Data augmentation is applied during training
- Learning rate is automatically adjusted using cosine annealing
- Early stopping with patience of 10 epochs
- Best model checkpoints are saved during training

## Author

Mingbo Zhang