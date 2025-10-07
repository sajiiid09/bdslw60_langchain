import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config.get("optimizer", "adam").lower()
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    
    # Get model parameters
    if hasattr(model, 'parameters'):
        parameters = model.parameters()
    else:
        parameters = model
    
    # Create optimizer
    if optimizer_name == "adam":
        optimizer = Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get("betas", [0.9, 0.999]),
            eps=float(config.get("eps", 1e-8))
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=config.get("betas", [0.9, 0.999]),
            eps=float(config.get("eps", 1e-8))
        )
    elif optimizer_name == "sgd":
        optimizer = SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", False)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Scheduler configuration
        
    Returns:
        Scheduler instance or None
    """
    scheduler_name = config.get("scheduler", "").lower()
    
    if not scheduler_name:
        return None
    
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("num_epochs", 100)),
            eta_min=float(config.get("min_lr", 1e-6))
        )
    elif scheduler_name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=int(config.get("step_size", 30)),
            gamma=float(config.get("gamma", 0.1))
        )
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(
            optimizer,
            gamma=float(config.get("gamma", 0.95))
        )
    elif scheduler_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.get("mode", "min"),
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 10)),
            min_lr=float(config.get("min_lr", 1e-6))
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, 
                       warmup_epochs: int, 
                       warmup_factor: float = 0.1) -> torch.optim.lr_scheduler._LRScheduler:
    """Create warmup learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        warmup_factor: Warmup factor (final_lr = base_lr * warmup_factor)
        
    Returns:
        Warmup scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_optimizer_and_scheduler(model: nn.Module, 
                                  training_config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Create both optimizer and scheduler from configuration.
    
    Args:
        model: PyTorch model
        training_config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config)
    
    return optimizer, scheduler

