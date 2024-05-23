"""Class to handle early stopping and model saving"""
import os
import torch

class ModelSaving:
    def __init__(self, hparams):
        self.save_model_metric = hparams.save_model_metric
        self.save_model_mode = hparams.save_model_mode
        self.save_best_model = hparams.save_best_model
        self.save_path = hparams.output_path
        self.best_value = None
    
    def __call__(self, metric_dict, model):
        
        metric_val = metric_dict[self.save_model_metric]
        if self.best_value is None:
            self.best_value = metric_val
            self.save_model(model)
        
        if self.save_model_mode == 'max':
            if metric_val > self.best_value:
                self.best_value = metric_val
                self.save_model(model)     
        elif self.save_model_mode == 'min':
            if metric_val < self.best_value:
                self.best_value = metric_val
                self.save_model(model)
        return
        
    
    def save_model(self, model):
        models_path = os.path.join(self.save_path, 'models')
        os.makedirs(models_path, exist_ok=True)
        if self.save_best_model:
            torch.save(model.state_dict(), os.path.join(models_path, "best_model.pt"))


