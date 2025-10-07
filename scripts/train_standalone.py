#!/usr/bin/env python3
"""
Standalone Training Script

This script trains a sign language recognition model using the processed RQE data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from tqdm import tqdm
import json
import random


def read_yaml(file_path: str) -> Dict:
    """Read YAML configuration file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RQEDataset(Dataset):
    """Dataset for RQE-encoded sequences."""
    
    def __init__(self, processed_dir: str, labels: Optional[Dict[str, int]] = None, max_length: int = 2000):
        self.processed_dir = Path(processed_dir)
        self.labels = labels or {}
        self.max_length = max_length
        
        # Find all processed files
        self.files = list(self.processed_dir.glob("*.npz"))
        
        # Create labels if not provided (for demonstration)
        if not self.labels:
            self.labels = self._create_demo_labels()
        
        # Filter files that have labels
        self.files = [f for f in self.files if f.stem in self.labels]
        
        print(f"Dataset initialized with {len(self.files)} files")
    
    def _create_demo_labels(self) -> Dict[str, int]:
        """Create demonstration labels based on file patterns."""
        labels = {}
        for file_path in self.files:
            # Create labels based on file name patterns for demonstration
            if "U1" in file_path.stem:
                labels[file_path.stem] = 0  # Class 0
            elif "U2" in file_path.stem:
                labels[file_path.stem] = 1  # Class 1
            elif "U3" in file_path.stem:
                labels[file_path.stem] = 2  # Class 2
            elif "U4" in file_path.stem:
                labels[file_path.stem] = 3  # Class 3
            elif "U5" in file_path.stem:
                labels[file_path.stem] = 4  # Class 4
            elif "U10" in file_path.stem:
                labels[file_path.stem] = 5  # Class 5
            else:
                labels[file_path.stem] = 0  # Default class
        
        return labels
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        # Load RQE data
        data = np.load(file_path, allow_pickle=True)
        codes = data['codes']  # Shape: (frames, features)
        meta = data['meta'].item()
        
        # Get label
        label = self.labels[file_path.stem]
        
        # Pad or truncate sequence to max_length
        if codes.shape[0] > self.max_length:
            codes = codes[:self.max_length]
        elif codes.shape[0] < self.max_length:
            # Pad with zeros
            padding = np.zeros((self.max_length - codes.shape[0], codes.shape[1]))
            codes = np.vstack([codes, padding])
        
        # Convert to tensor
        codes_tensor = torch.from_numpy(codes).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return codes_tensor, label_tensor


class BiLSTMAttentionModel(nn.Module):
    """BiLSTM with Attention model for sign language recognition."""
    
    def __init__(self, input_size: int = 1629, hidden_size: int = 512, 
                 num_layers: int = 2, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        
        # Apply attention
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)
        # attended_output shape: (batch_size, hidden_size * 2)
        
        # Classification
        logits = self.classifier(attended_output)
        # logits shape: (batch_size, num_classes)
        
        return logits


def train_model(model, train_loader, val_loader, num_epochs: int = 50, 
                learning_rate: float = 1e-3, device: str = "auto") -> Dict:
    """Train the model."""
    
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100. * val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_val_acc


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Directory containing processed RQE data")
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
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Sign Language Recognition Model Training ===")
    print(f"Processed data directory: {args.processed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Create dataset
    dataset = RQEDataset(args.processed_dir)
    
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
    model = BiLSTMAttentionModel(
        input_size=1629,
        hidden_size=512,
        num_layers=2,
        num_classes=6,  # Based on our demo labels
        dropout=0.1
    )
    
    # Train model
    print("\nStarting training...")
    history, best_val_acc = train_model(
        model, train_loader, val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save model
    model_path = output_dir / "best_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': 1629,
            'hidden_size': 512,
            'num_layers': 2,
            'num_classes': 6,
            'dropout': 0.1
        },
        'best_val_acc': best_val_acc,
        'history': history
    }, model_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
