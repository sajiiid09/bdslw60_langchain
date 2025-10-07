#!/usr/bin/env python3
"""
Simplified Training Script for Sign Language Recognition

This script trains the BiLSTM model without matplotlib dependencies.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Import our models
import sys
sys.path.append('src')
from models.bilstm_attn import BiLSTMAttention


class SimpleDataset:
    """Simple dataset class for training."""
    
    def __init__(self, data_dir: str, max_length: int = 100, feature_dim: int = 1629):
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        self.feature_dim = feature_dim
        
        # Find all data files
        self.files = list(self.data_dir.rglob("*.npz"))
        print(f"Found {len(self.files)} data files")
        
        # Create simple labels (for demo purposes)
        self.labels = {f.stem: hash(f.stem) % 6 for f in self.files}  # 6 classes
        self.num_classes = len(set(self.labels.values()))
        print(f"Number of classes: {self.num_classes}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load data
        data = np.load(file_path, allow_pickle=True)
        codes = data['codes']  # Shape: (frames, features)
        
        # Get label
        label = self.labels[file_path.stem]
        
        # Ensure correct feature dimension
        if codes.shape[1] != self.feature_dim:
            if codes.shape[1] < self.feature_dim:
                padding = np.zeros((codes.shape[0], self.feature_dim - codes.shape[1]))
                codes = np.hstack([codes, padding])
            else:
                codes = codes[:, :self.feature_dim]
        
        # Pad or truncate sequence
        if codes.shape[0] > self.max_length:
            codes = codes[:self.max_length]
        elif codes.shape[0] < self.max_length:
            padding = np.zeros((self.max_length - codes.shape[0], self.feature_dim))
            codes = np.vstack([codes, padding])
        
        # Convert to tensor
        codes_tensor = torch.from_numpy(codes).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return codes_tensor, label_tensor


class SimpleTrainer:
    """Simple trainer class."""
    
    def __init__(self, model, device="auto", learning_rate=1e-3):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in tqdm(train_loader, desc="Training"):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Train the model."""
        print(f"Training on device: {self.device}")
        print(f"Model input size: {self.model.input_size}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        return self.history


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--output-dir", default="outputs/experiments/bilstm_trained", 
                       help="Directory to save training results")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, 
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
    print(f"Output directory: {args.output_dir}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Max sequence length: {args.max_sequence_length}")
    
    # Create dataset
    dataset = SimpleDataset(
        data_dir=args.data_dir,
        max_length=args.max_sequence_length,
        feature_dim=args.feature_dim
    )
    
    if len(dataset) == 0:
        print("No data found for training!")
        return
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = BiLSTMAttention(
        input_size=args.feature_dim,
        hidden_size=512,
        num_layers=2,
        num_classes=dataset.num_classes,
        dropout=0.1,
        bidirectional=True
    )
    
    # Create trainer
    trainer = SimpleTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    # Save model
    model_path = output_dir / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': args.feature_dim,
            'hidden_size': 512,
            'num_layers': 2,
            'num_classes': dataset.num_classes,
            'dropout': 0.1
        },
        'history': history
    }, model_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
