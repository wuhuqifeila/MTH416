import numpy as np
import torch
from config import Config

class EarlyStopping:
    """早停机制
    
    当验证集性能不再提升时停止训练
    参数:
        patience (int): 等待改善的轮数
        min_delta (float): 最小改善量
        mode (str): 'min' 用于监控最小化指标(如损失), 'max' 用于监控最大化指标(如准确率)
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
        """保存模型"""
        if self.mode == 'min':
            val_loss = -val_loss
        if val_loss < self.val_loss_min:
            print(f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
            torch.save(model.state_dict(), path)
            self.val_loss_min = val_loss 