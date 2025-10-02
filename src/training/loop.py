import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np

from ..utils.metrics import compute_accuracy, compute_top_k_accuracy
from ..utils.logging import log_training_progress


def train_epoch(model: nn.Module, 
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device,
                epoch: int,
                logger: Optional[Any] = None) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        if isinstance(batch, dict):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
        else:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * total_correct / total_samples:.2f}%'
        })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int,
                   logger: Optional[Any] = None) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, dict):
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            # Store predictions and targets for additional metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * total_correct / total_samples:.2f}%'
            })
    
    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    
    # Compute top-k accuracy using probabilities
    all_targets_array = np.array(all_targets)
    all_probabilities_array = np.array(all_probabilities)
    top3_accuracy = compute_top_k_accuracy(all_targets_array, all_probabilities_array, k=3)
    top5_accuracy = compute_top_k_accuracy(all_targets_array, all_probabilities_array, k=5)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "top5_accuracy": top5_accuracy
    }


def train_model(model: nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                criterion: nn.Module,
                device: torch.device,
                num_epochs: int,
                logger: Optional[Any] = None,
                save_dir: Optional[str] = None) -> Dict[str, Any]:
    """Train model for multiple epochs."""
    
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer, criterion, device, epoch, logger
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_dataloader, criterion, device, epoch, logger
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Log progress
        if logger:
            log_training_progress(
                logger, epoch, train_metrics["loss"], val_metrics["loss"],
                val_metrics["accuracy"], optimizer.param_groups[0]["lr"]
            )
        
        # Store metrics
        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_accuracies.append(train_metrics["accuracy"])
        val_accuracies.append(val_metrics["accuracy"])
        
        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            if save_dir:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_accuracy': val_metrics["accuracy"],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }, f"{save_dir}/best_model.pth")
    
    return {
        "best_val_accuracy": best_val_accuracy,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies
    }
