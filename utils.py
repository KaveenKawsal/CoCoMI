# utils.py

import torch
import numpy as np
import random
import logging
import os

from models import VanillaUNet, AttentionUNet
from losses import CrossEntropyLoss, SimpleDiceLoss, SimpleFocalLoss, AdvancedFocalDiceLoss

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging():
    """Configures the logging format."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    """Gets the available device (CUDA or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device

def get_optimizer(optimizer_name, model_params, learning_rate, weight_decay=1e-4, momentum=0.9):
    """Factory function to create an optimizer."""
    opt_name = optimizer_name.lower()
    if opt_name == 'sgd':
        return torch.optim.SGD(model_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(model_params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return torch.optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return torch.optim.RMSprop(model_params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif opt_name == 'adadelta':
        return torch.optim.Adadelta(model_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(scheduler_name, optimizer, **kwargs):
    """Factory function to create a learning rate scheduler."""
    sched_name = scheduler_name.lower()
    if sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 20), gamma=kwargs.get('gamma', 0.1))
    elif sched_name == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=kwargs.get('factor', 0.5), patience=kwargs.get('patience', 5))
    elif sched_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50), eta_min=kwargs.get('min_lr', 1e-6))
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def calculate_class_weights(dataloader, num_classes, device):
    """Calculates class weights using Median Frequency Balancing."""
    class_counts = torch.zeros(num_classes, device=device)
    total_pixels = 0
    
    for _, masks in dataloader:
        masks = masks.to(device)
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum()
        total_pixels += masks.numel()
    
    # Avoid division by zero for classes that are not present
    freq = class_counts / total_pixels
    median_freq = torch.median(freq[freq > 0])
    
    class_weights = median_freq / (freq + 1e-6) # Add epsilon to avoid div by zero
    class_weights = class_weights / class_weights.mean() # Normalize
    
    logging.info(f"Calculated class weights: {class_weights.cpu().numpy()}")
    return class_weights

def setup_model_and_loss(
    model_type, loss_type, n_classes, device, class_weights=None
):
    """Factory function to create the model and loss function."""
    # Model Selection
    if model_type == 'vanilla_unet':
        model = VanillaUNet(n_channels=3, n_classes=n_classes)
    elif model_type == 'attention_unet':
        model = AttentionUNet(n_channels=3, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(device)
    logging.info(f"Using {model_type} model.")
    
    # Loss Function Selection
    if loss_type == 'cross_entropy':
        criterion = CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'dice':
        criterion = SimpleDiceLoss()
    elif loss_type == 'focal':
        criterion = SimpleFocalLoss()
    elif loss_type == 'advanced_focal_dice':
        if class_weights is None:
            logging.warning("AdvancedFocalDiceLoss used without pre-calculated class weights.")
        criterion = AdvancedFocalDiceLoss(alpha=class_weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    logging.info(f"Using {loss_type} loss function.")
    return model, criterion

def save_model_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Saves a model checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)
    logging.info(f"Model saved to {save_path}")