#!/usr/bin/env python3
"""
Example Usage Script for Sign Language Recognition

This script demonstrates how to:
1. Preprocess data with StandardScaler normalization
2. Create proper train/test split (70%/30%)
3. Train BiLSTM model with 1629 feature dimension
4. Handle data reshaping for LSTM input
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

def main():
    """Main example function."""
    print("=== Sign Language Recognition Example ===")
    print("This example demonstrates the complete pipeline:")
    print("1. Data preprocessing with StandardScaler")
    print("2. Train/test split (70%/30%)")
    print("3. BiLSTM model with 1629 feature dimension")
    print("4. Proper data reshaping for LSTM input")
    print()
    
    # Step 1: Preprocess data
    print("Step 1: Preprocessing data...")
    print("Run: python scripts/preprocess_data.py --data-dir data/processed --output-dir data/preprocessed")
    print("This will:")
    print("- Apply StandardScaler normalization")
    print("- Create 70%/30% train/test split")
    print("- Handle 1629 feature dimension")
    print("- Save preprocessed data and scaler")
    print()
    
    # Step 2: Train model
    print("Step 2: Training model...")
    print("Run: python scripts/train_with_preprocessing.py --data-dir data/processed --output-dir outputs/experiments/bilstm_trained")
    print("This will:")
    print("- Load preprocessed data")
    print("- Create BiLSTM model with input_size=1629")
    print("- Handle data reshaping automatically")
    print("- Train with proper validation")
    print("- Save model and results")
    print()
    
    # Step 3: Key components explanation
    print("=== Key Components ===")
    print()
    
    print("1. Data Preprocessing (scripts/preprocess_data.py):")
    print("   - SignLanguageDataset class handles 1629 feature dimension")
    print("   - StandardScaler normalization applied")
    print("   - Stratified train/test split (70%/30%)")
    print("   - Proper sequence padding/truncation")
    print()
    
    print("2. BiLSTM Model (src/models/bilstm_attn.py):")
    print("   - Input size: 1629 (feature dimension)")
    print("   - Bidirectional LSTM with attention")
    print("   - Handles variable sequence lengths")
    print("   - Proper weight initialization")
    print()
    
    print("3. Data Reshaping (scripts/train_with_preprocessing.py):")
    print("   - DataReshaper class ensures correct input shape")
    print("   - Handles padding/truncation to 1629 features")
    print("   - Validates input shape before LSTM")
    print()
    
    print("4. Training Pipeline:")
    print("   - SignLanguageTrainer class manages training")
    print("   - Automatic data reshaping in each batch")
    print("   - Proper validation and checkpointing")
    print("   - Training history and visualization")
    print()
    
    # Step 4: Usage commands
    print("=== Usage Commands ===")
    print()
    print("# 1. Preprocess your data")
    print("python scripts/preprocess_data.py \\")
    print("    --data-dir data/processed \\")
    print("    --output-dir data/preprocessed \\")
    print("    --feature-dim 1629 \\")
    print("    --max-sequence-length 100 \\")
    print("    --test-size 0.3")
    print()
    
    print("# 2. Train the model")
    print("python scripts/train_with_preprocessing.py \\")
    print("    --data-dir data/processed \\")
    print("    --output-dir outputs/experiments/bilstm_trained \\")
    print("    --epochs 100 \\")
    print("    --batch-size 32 \\")
    print("    --learning-rate 1e-3 \\")
    print("    --feature-dim 1629")
    print()
    
    print("# 3. Monitor training")
    print("# Check outputs/experiments/bilstm_trained/ for:")
    print("# - training_history.json")
    print("# - training_plots.png")
    print("# - checkpoints/best_model.pth")
    print("# - evaluation_results.json")
    print()
    
    # Step 5: Model configuration
    print("=== Model Configuration ===")
    print()
    print("BiLSTM Model Parameters:")
    print("- input_size: 1629 (your feature dimension)")
    print("- hidden_size: 512")
    print("- num_layers: 2")
    print("- bidirectional: True")
    print("- dropout: 0.1")
    print("- attention_type: 'dot'")
    print()
    
    print("Data Shape Handling:")
    print("- Input: (batch_size, sequence_length, 1629)")
    print("- LSTM processes each timestep with 1629 features")
    print("- Output: (batch_size, num_classes)")
    print()
    
    print("Training Configuration:")
    print("- Optimizer: Adam (lr=1e-3, weight_decay=1e-4)")
    print("- Scheduler: StepLR (step_size=30, gamma=0.1)")
    print("- Loss: CrossEntropyLoss")
    print("- Batch size: 32")
    print("- Epochs: 100")
    print()
    
    print("=== Summary ===")
    print("Your setup is now ready to handle:")
    print("✓ 1629 feature dimension input")
    print("✓ Proper train/test split (70%/30%)")
    print("✓ StandardScaler normalization")
    print("✓ BiLSTM model with correct input size")
    print("✓ Automatic data reshaping")
    print("✓ Comprehensive training pipeline")
    print()
    print("Run the preprocessing and training commands above to get started!")


if __name__ == "__main__":
    main()
