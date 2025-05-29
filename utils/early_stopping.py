import numpy as np
import torch
from config import Config

class EarlyStopping:
    """Early stopping mechanism
    
    Stop training when validation performance stops improving
    Args:
        patience (int): Number of epochs to wait for improvement
        min_delta (float): Minimum improvement amount
        mode (str): 'min' for monitoring minimization metrics (like loss), 'max' for maximization metrics (like accuracy)
    """
    def __init__(self, patience=Config.EARLY_STOPPING_PATIENCE, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, score, model, path='best_model.pth'):
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, path):
        """Save model"""
        if self.mode == 'min':
            val_loss = -val_loss
        if val_loss < self.val_loss_min:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss 