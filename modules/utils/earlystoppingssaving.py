"""Class to handle early stopping and model saving"""
import os
import torch

class EarlyStoppingAndSaving:
    def __init__(self, hparams):
        self.early_stopping = hparams.early_stopping
        self.early_stopping_mode = hparams.early_stopping_mode
        self.early_stopping_metric = hparams.early_stopping_metric        
        self.patience = hparams.early_stopping_patience
        self.early_stopping_delta = hparams.early_stopping_delta
        self.save_best_model = hparams.save_best_model
        self.save_last_model = hparams.save_last_model
        self.save_path = hparams.output_path
        self.early_stopping_counter = 0        
        self.best_value = None
    
    def __call__(self, metric_dict, model):
        
        metric_val = metric_dict[self.early_stopping_metric]
        if self.best_value is None:
            self.best_value = metric_val
            self.save_model(model)
        
        if self.early_stopping_mode == 'max':
            if metric_val > (self.best_value + self.early_stopping_delta):
                self.early_stopping_counter = 0
                self.best_value = metric_val
                self.save_model(model)
            else:
                self.early_stopping_counter +=1        
        elif self.early_stopping_mode == 'min':
            if metric_val < (self.best_value + self.early_stopping_delta):
                self.early_stopping_counter = 0
                self.best_value = metric_val
                self.save_model(model)
            else:
                self.early_stopping_counter +=1
        if self.early_stopping:
            if self.early_stopping_counter >= self.patience:
                return True
        return False
        
    
    def save_model(self, model):
        models_path = os.path.join(self.save_path, 'models')
        os.makedirs(models_path, exist_ok=True)
        if self.save_best_model:
            torch.save(model.state_dict(), os.path.join(models_path, "best_model.pt"))
        if self.save_last_model:
            torch.save(model.state_dict(), os.path.join(models_path, "last_model.pt"))






