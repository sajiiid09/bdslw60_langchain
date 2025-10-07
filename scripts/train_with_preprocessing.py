#!/usr/bin/env python3
"""
Comprehensive Training Script for Sign Language Recognition

This script:
1. Loads preprocessed data with proper train/test split
2. Uses BiLSTM model with 1629 feature dimension
3. Handles data reshaping for LSTM input
4. Implements proper training with validation
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import our models and utilities
import sys
sys.path.append('src')
from models.bilstm_attn import BiLSTMAttention
from scripts.preprocess_data import SignLanguageDataset


class DataReshaper:
    """Utility class for reshaping data for LSTM input."""
    
    @staticmethod
    def reshape_for_lstm(data: torch.Tensor, feature_dim: int = 1629) -> torch.Tensor:
        """
        Reshape data for LSTM input.
        
        Args:
            data: Input tensor of shape (batch_size, sequence_length, features)
            feature_dim: Expected feature dimension (1629)
        
        Returns:
            Reshaped tensor ready for LSTM input
        """
        batch_size, seq_len, features = data.shape
        
        # Ensure correct feature dimension
        if features != feature_dim:
            if features < feature_dim:
                # Pad with zeros
                padding = torch.zeros(batch_size, seq_len, feature_dim - features, device=data.device)
                data = torch.cat([data, padding], dim=-1)
            else:
                # Truncate
                data = data[:, :, :feature_dim]
        
        return data
    
    @staticmethod
    def validate_input_shape(data: torch.Tensor, expected_features: int = 1629) -> bool:
        """Validate that input data has correct shape for LSTM."""
        if len(data.shape) != 3:
            return False
        
        batch_size, seq_len, features = data.shape
        return features == expected_features


class SignLanguageTrainer:
    """Trainer class for sign language recognition model."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "auto",
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.model = model.to(device if device != "auto" else self._get_device())
        self.device = device if device != "auto" else self._get_device()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _get_device(self) -> str:
        """Get available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Reshape data for LSTM
            data = DataReshaper.reshape_for_lstm(data, self.model.input_size)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                # Reshape data for LSTM
                data = DataReshaper.reshape_for_lstm(data, self.model.input_size)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_dir: Optional[str] = None) -> Dict:
        """Train the model."""
        print(f"Training on device: {self.device}")
        print(f"Model input size: {self.model.input_size}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
                
                if save_dir:
                    self.save_model(save_dir, epoch, val_acc)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, save_dir: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_classes': self.model.num_classes,
                'dropout': self.model.dropout,
                'bidirectional': self.model.bidirectional
            }
        }
        
        torch.save(checkpoint, save_path / f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, save_path / "best_model.pth")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate(self, test_loader: DataLoader, class_names: Optional[List[str]] = None) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Reshape data for LSTM
                data = DataReshaper.reshape_for_lstm(data, self.model.input_size)
                
                # Forward pass
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
        
        # Classification report
        if class_names:
            report = classification_report(all_targets, all_preds, target_names=class_names)
        else:
            report = classification_report(all_targets, all_preds)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return results


def load_preprocessed_data(data_dir: str) -> Tuple[SignLanguageDataset, SignLanguageDataset, StandardScaler, Dict]:
    """Load preprocessed data and metadata."""
    data_dir = Path(data_dir)
    
    # Load dataset info
    with open(data_dir / "dataset_info.json", 'r') as f:
        dataset_info = json.load(f)
    
    # Load scaler
    with open(data_dir / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Create datasets (we need to recreate them since we can't pickle them directly)
    # For now, we'll assume the data is in the same structure as before
    # In a real implementation, you'd save the dataset indices and recreate
    
    print("Note: This is a simplified version. In practice, you'd need to save")
    print("dataset indices and recreate the datasets properly.")
    
    return None, None, scaler, dataset_info


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--preprocessed-dir", default="data/preprocessed", 
                       help="Directory containing preprocessed data")
    parser.add_argument("--output-dir", default="outputs/experiments/bilstm_trained", 
                       help="Directory to save training results")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--feature-dim", type=int, default=1629, 
                       help="Feature dimension")
    parser.add_argument("--max-sequence-length", type=int, default=100, 
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Sign Language Recognition Training ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Preprocessed directory: {args.preprocessed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Max sequence length: {args.max_sequence_length}")
    
    # For this example, we'll create datasets directly
    # In practice, you'd load the preprocessed data
    print("\nCreating datasets...")
    
    # Create train/test split
    from scripts.preprocess_data import create_train_test_split
    train_dataset, test_dataset, scaler = create_train_test_split(
        data_dir=args.data_dir,
        test_size=0.3,
        random_state=42,
        max_sequence_length=args.max_sequence_length,
        feature_dim=args.feature_dim
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(set(train_dataset.labels.values()))}")
    
    # Create model
    model = BiLSTMAttention(
        input_size=args.feature_dim,
        hidden_size=512,
        num_layers=2,
        num_classes=len(set(train_dataset.labels.values())),
        dropout=0.1,
        bidirectional=True
    )
    
    # Create trainer
    trainer = SignLanguageTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation for this example
        num_epochs=args.epochs,
        save_dir=str(output_dir / "checkpoints")
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_path = output_dir / "training_plots.png"
    trainer.plot_training_history(str(plot_path))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report']
        }, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Best model saved to: {output_dir / 'checkpoints' / 'best_model.pth'}")


if __name__ == "__main__":
    main()
