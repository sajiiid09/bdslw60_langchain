#!/usr/bin/env python3
"""
VideoMAE Training Script for Sign Language Recognition

This script trains VideoMAE models on sequence data with proper data handling.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml

# Import our models
import sys
sys.path.append('src')
from models.videomae import VideoMAE
from models.videomae_adapter import VideoMAEFromCodes


class VideoMAEDataset:
    """Dataset class for VideoMAE training on sequence data."""
    
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


class VideoMAETrainer:
    """Trainer class for VideoMAE model."""
    
    def __init__(self, model, device="auto", learning_rate=1e-4, weight_decay=0.05):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
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
    
    def train(self, train_loader, val_loader, num_epochs=50, warmup_epochs=5):
        """Train the model."""
        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"New best validation accuracy: {best_val_acc:.2f}%")
        
        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
        return self.history


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict, num_classes: int) -> nn.Module:
    """Create VideoMAE model from configuration."""
    model_config = config['model']
    
    # Check if we need the adapter for sequence data
    if model_config.get('name') == 'videomae_from_codes':
        # Use VideoMAE with adapter for sequence data
        videomae_kwargs = {
            'patch_size': model_config.get('patch_size', 16),
            'num_frames': model_config.get('num_frames', 16),
            'tubelet_size': model_config.get('tubelet_size', 2),
            'hidden_size': model_config.get('hidden_size', 768),
            'num_hidden_layers': model_config.get('num_hidden_layers', 12),
            'num_attention_heads': model_config.get('num_attention_heads', 12),
            'intermediate_size': model_config.get('intermediate_size', 3072),
            'hidden_act': model_config.get('hidden_act', 'gelu'),
            'hidden_dropout_prob': model_config.get('hidden_dropout_prob', 0.0),
            'attention_probs_dropout_prob': model_config.get('attention_probs_dropout_prob', 0.0),
            'initializer_range': model_config.get('initializer_range', 0.02),
            'layer_norm_eps': model_config.get('layer_norm_eps', 1e-12),
            'image_size': model_config.get('image_size', 224),
            'num_classes': num_classes,
            'classifier_dropout': model_config.get('classifier_dropout', 0.1)
        }
        
        model = VideoMAEFromCodes(
            videomae_kwargs=videomae_kwargs,
            input_feature_dim=1629,  # Your feature dimension
            adapter_channels=model_config.get('adapter_channels', 3),
            adapter_small_h=model_config.get('adapter_small_h', 8),
            adapter_small_w=model_config.get('adapter_small_w', 8)
        )
    else:
        # Standard VideoMAE for video data
        model = VideoMAE(
            patch_size=model_config.get('patch_size', 16),
            num_frames=model_config.get('num_frames', 16),
            tubelet_size=model_config.get('tubelet_size', 2),
            hidden_size=model_config.get('hidden_size', 768),
            num_hidden_layers=model_config.get('num_hidden_layers', 12),
            num_attention_heads=model_config.get('num_attention_heads', 12),
            intermediate_size=model_config.get('intermediate_size', 3072),
            hidden_act=model_config.get('hidden_act', 'gelu'),
            hidden_dropout_prob=model_config.get('hidden_dropout_prob', 0.0),
            attention_probs_dropout_prob=model_config.get('attention_probs_dropout_prob', 0.0),
            initializer_range=model_config.get('initializer_range', 0.02),
            layer_norm_eps=model_config.get('layer_norm_eps', 1e-12),
            input_channels=model_config.get('input_channels', 3),
            image_size=model_config.get('image_size', 224),
            num_classes=num_classes,
            classifier_dropout=model_config.get('classifier_dropout', 0.1)
        )
    
    return model


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VideoMAE model for sign language recognition")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--config", default="configs/model_videomae_from_codes.yaml", 
                       help="Path to model configuration file")
    parser.add_argument("--output-dir", default="outputs/experiments/videomae_trained", 
                       help="Directory to save training results")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, 
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
    
    print("=== VideoMAE Sign Language Recognition Training ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Config file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Max sequence length: {args.max_sequence_length}")
    
    # Load configuration
    config = load_config(args.config)
    print(f"Model configuration: {config['model']['name']}")
    
    # Create dataset
    dataset = VideoMAEDataset(
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
    model = create_model_from_config(config, dataset.num_classes)
    
    # Create trainer
    trainer = VideoMAETrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=config['training'].get('weight_decay', 0.05)
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        warmup_epochs=config['training'].get('warmup_epochs', 5)
    )
    
    # Save model
    model_path = output_dir / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': config['model'],
        'history': history
    }, model_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Configuration saved to: {config_path}")


if __name__ == "__main__":
    main()
