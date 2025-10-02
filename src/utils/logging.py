import logging
import os
from typing import Optional
from datetime import datetime


def setup_logging(log_dir: str, log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Setup logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("bdsl")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"bdsl_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_model_info(logger: logging.Logger, model, config: dict) -> None:
    """Log model information."""
    logger.info(f"Model: {config.get('name', 'Unknown')}")
    logger.info(f"Type: {config.get('type', 'Unknown')}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")


def log_training_progress(logger: logging.Logger, epoch: int, train_loss: float, 
                         val_loss: float, val_accuracy: float, lr: float) -> None:
    """Log training progress."""
    logger.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {lr:.2e}")


def log_evaluation_results(logger: logging.Logger, results: dict) -> None:
    """Log evaluation results."""
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

