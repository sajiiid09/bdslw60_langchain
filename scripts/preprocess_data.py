#!/usr/bin/env python3
"""
Data Preprocessing Script for Sign Language Recognition

This script handles:
1. Loading and preprocessing data with StandardScaler normalization
2. Proper train/test split (70%/30%)
3. Data reshaping for LSTM input (feature dimension 1629)
4. Saving preprocessed data for training
"""

import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from tqdm import tqdm
import random


class SignLanguageDataset(Dataset):
    """Dataset for sign language sequences with proper preprocessing."""
    
    def __init__(self, 
                 data_dir: str,
                 max_sequence_length: int = 100,
                 feature_dim: int = 1629,
                 normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the processed data files
            max_sequence_length: Maximum sequence length for padding/truncation
            feature_dim: Feature dimension (1629 for your case)
            normalize: Whether to apply StandardScaler normalization
            scaler: Pre-fitted scaler for normalization
        """
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.scaler = scaler
        
        # Find all data files
        self.files = list(self.data_dir.rglob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        # Create labels mapping
        self.labels = self._create_labels()
        
        print(f"Found {len(self.files)} data files")
        print(f"Feature dimension: {feature_dim}")
        print(f"Max sequence length: {max_sequence_length}")
        print(f"Number of classes: {len(set(self.labels.values()))}")
    
    def _create_labels(self) -> Dict[str, int]:
        """Create labels mapping from file names."""
        labels = {}
        class_names = set()
        
        for file_path in self.files:
            # Extract class name from file path
            # Assuming structure like: data/processed/class_name/video_name.npz
            parts = file_path.parts
            if len(parts) >= 3:
                class_name = parts[-2]  # Parent directory name
                class_names.add(class_name)
            else:
                # Fallback: use filename without extension
                class_name = file_path.stem
                class_names.add(class_name)
            
            labels[file_path.stem] = class_name
        
        # Create numerical labels
        class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
        self.class_to_idx = class_to_idx
        
        # Convert string labels to numerical
        numerical_labels = {}
        for file_stem, class_name in labels.items():
            numerical_labels[file_stem] = class_to_idx[class_name]
        
        return numerical_labels
    
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
                # Pad with zeros if features are fewer
                padding = np.zeros((codes.shape[0], self.feature_dim - codes.shape[1]))
                codes = np.hstack([codes, padding])
            else:
                # Truncate if features are more
                codes = codes[:, :self.feature_dim]
        
        # Pad or truncate sequence to max_length
        if codes.shape[0] > self.max_sequence_length:
            codes = codes[:self.max_sequence_length]
        elif codes.shape[0] < self.max_sequence_length:
            # Pad with zeros
            padding = np.zeros((self.max_sequence_length - codes.shape[0], self.feature_dim))
            codes = np.vstack([codes, padding])
        
        # Apply normalization if requested
        if self.normalize and self.scaler is not None:
            # Reshape for scaler: (samples, features) -> (samples * features, 1)
            original_shape = codes.shape
            codes_flat = codes.flatten().reshape(-1, 1)
            codes_normalized = self.scaler.transform(codes_flat)
            codes = codes_normalized.reshape(original_shape)
        
        # Convert to tensor
        codes_tensor = torch.from_numpy(codes).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return codes_tensor, label_tensor


def fit_scaler_on_data(dataset: SignLanguageDataset) -> StandardScaler:
    """Fit StandardScaler on the entire dataset."""
    print("Fitting StandardScaler on dataset...")
    
    all_data = []
    for i in tqdm(range(len(dataset)), desc="Collecting data for scaler"):
        file_path = dataset.files[i]
        data = np.load(file_path, allow_pickle=True)
        codes = data['codes']
        
        # Ensure correct feature dimension
        if codes.shape[1] != dataset.feature_dim:
            if codes.shape[1] < dataset.feature_dim:
                padding = np.zeros((codes.shape[0], dataset.feature_dim - codes.shape[1]))
                codes = np.hstack([codes, padding])
            else:
                codes = codes[:, :dataset.feature_dim]
        
        all_data.append(codes)
    
    # Concatenate all data
    all_data = np.vstack(all_data)
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(all_data)
    
    print(f"Scaler fitted on {all_data.shape[0]} samples with {all_data.shape[1]} features")
    return scaler


def create_train_test_split(data_dir: str,
                          test_size: float = 0.3,
                          random_state: int = 42,
                          max_sequence_length: int = 100,
                          feature_dim: int = 1629) -> Tuple[SignLanguageDataset, SignLanguageDataset, StandardScaler]:
    """
    Create train/test split with proper preprocessing.
    
    Args:
        data_dir: Directory containing processed data
        test_size: Fraction of data to use for testing (default: 0.3)
        random_state: Random seed for reproducibility
        max_sequence_length: Maximum sequence length
        feature_dim: Feature dimension (1629)
    
    Returns:
        train_dataset, test_dataset, fitted_scaler
    """
    print("Creating train/test split...")
    
    # Create initial dataset without normalization
    full_dataset = SignLanguageDataset(
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
        feature_dim=feature_dim,
        normalize=False
    )
    
    # Get file indices and labels for stratified split
    file_indices = list(range(len(full_dataset)))
    labels = [full_dataset.labels[full_dataset.files[i].stem] for i in file_indices]
    
    # Create stratified train/test split
    train_indices, test_indices = train_test_split(
        file_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"Train samples: {len(train_indices)}")
    print(f"Test samples: {len(test_indices)}")
    
    # Create train dataset for fitting scaler
    train_files = [full_dataset.files[i] for i in train_indices]
    train_labels = {full_dataset.files[i].stem: full_dataset.labels[full_dataset.files[i].stem] 
                   for i in train_indices}
    
    # Create temporary train dataset for scaler fitting
    temp_train_dataset = SignLanguageDataset(
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
        feature_dim=feature_dim,
        normalize=False
    )
    temp_train_dataset.files = train_files
    temp_train_dataset.labels = train_labels
    
    # Fit scaler on training data only
    scaler = fit_scaler_on_data(temp_train_dataset)
    
    # Create final datasets with normalization
    train_dataset = SignLanguageDataset(
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
        feature_dim=feature_dim,
        normalize=True,
        scaler=scaler
    )
    train_dataset.files = train_files
    train_dataset.labels = train_labels
    
    test_dataset = SignLanguageDataset(
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
        feature_dim=feature_dim,
        normalize=True,
        scaler=scaler
    )
    test_dataset.files = [full_dataset.files[i] for i in test_indices]
    test_dataset.labels = {full_dataset.files[i].stem: full_dataset.labels[full_dataset.files[i].stem] 
                          for i in test_indices}
    
    return train_dataset, test_dataset, scaler


def save_preprocessed_data(train_dataset: SignLanguageDataset,
                          test_dataset: SignLanguageDataset,
                          scaler: StandardScaler,
                          output_dir: str):
    """Save preprocessed data and scaler for later use."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save dataset info
    dataset_info = {
        'feature_dim': train_dataset.feature_dim,
        'max_sequence_length': train_dataset.max_sequence_length,
        'num_classes': len(set(train_dataset.labels.values())),
        'class_to_idx': train_dataset.class_to_idx,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset)
    }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Preprocessed data saved to: {output_dir}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Dataset info saved to: {info_path}")


def main():
    """Main preprocessing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess sign language data")
    parser.add_argument("--data-dir", default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--output-dir", default="data/preprocessed", 
                       help="Directory to save preprocessed data")
    parser.add_argument("--test-size", type=float, default=0.3, 
                       help="Fraction of data for testing")
    parser.add_argument("--max-sequence-length", type=int, default=100, 
                       help="Maximum sequence length")
    parser.add_argument("--feature-dim", type=int, default=1629, 
                       help="Feature dimension")
    parser.add_argument("--random-state", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    
    print("=== Sign Language Data Preprocessing ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Max sequence length: {args.max_sequence_length}")
    
    # Create train/test split with preprocessing
    train_dataset, test_dataset, scaler = create_train_test_split(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        max_sequence_length=args.max_sequence_length,
        feature_dim=args.feature_dim
    )
    
    # Save preprocessed data
    save_preprocessed_data(train_dataset, test_dataset, scaler, args.output_dir)
    
    print("\nPreprocessing completed successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(set(train_dataset.labels.values()))}")


if __name__ == "__main__":
    main()
