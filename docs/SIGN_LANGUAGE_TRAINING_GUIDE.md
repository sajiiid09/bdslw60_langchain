# Sign Language Recognition Training Guide

This guide explains how to properly handle a dataset with 1629 feature dimensions for BiLSTM training, including proper data preprocessing, train/test split, and model configuration.

## Overview

Your dataset has sequences with a feature dimension of 1629, and you need to:
1. Apply StandardScaler normalization
2. Split data 70% training / 30% testing
3. Configure BiLSTM model for 1629 input features
4. Handle proper data reshaping for LSTM input

## Solution Components

### 1. Data Preprocessing (`scripts/preprocess_data.py`)

**Key Features:**
- Handles 1629 feature dimension correctly
- Applies StandardScaler normalization
- Creates stratified 70%/30% train/test split
- Proper sequence padding/truncation
- Saves preprocessed data and scaler

**Usage:**
```bash
python scripts/preprocess_data.py \
    --data-dir data/processed \
    --output-dir data/preprocessed \
    --feature-dim 1629 \
    --max-sequence-length 100 \
    --test-size 0.3
```

### 2. BiLSTM Model (`src/models/bilstm_attn.py`)

**Configuration:**
- `input_size: 1629` - Matches your feature dimension
- Bidirectional LSTM with attention mechanism
- Proper weight initialization
- Handles variable sequence lengths

**Model Architecture:**
```python
BiLSTMAttention(
    input_size=1629,        # Your feature dimension
    hidden_size=512,
    num_layers=2,
    dropout=0.1,
    bidirectional=True,
    num_classes=60          # Adjust based on your classes
)
```

### 3. Training Script (`scripts/train_with_preprocessing.py`)

**Key Features:**
- Automatic data reshaping for LSTM input
- Comprehensive training pipeline
- Validation and checkpointing
- Training history visualization
- Model evaluation

**Usage:**
```bash
python scripts/train_with_preprocessing.py \
    --data-dir data/processed \
    --output-dir outputs/experiments/bilstm_trained \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --feature-dim 1629
```

### 4. Data Reshaping Utilities

**DataReshaper Class:**
- Ensures correct input shape for LSTM
- Handles padding/truncation to 1629 features
- Validates input dimensions
- Automatic reshaping in training loop

## Complete Workflow

### Step 1: Preprocess Your Data

```bash
# Create preprocessed data with proper normalization and split
python scripts/preprocess_data.py \
    --data-dir data/processed \
    --output-dir data/preprocessed \
    --feature-dim 1629 \
    --max-sequence-length 100 \
    --test-size 0.3
```

This will:
- Apply StandardScaler normalization to your 1629-dimensional features
- Create stratified 70%/30% train/test split
- Handle sequence padding/truncation
- Save scaler and dataset metadata

### Step 2: Train Your Model

```bash
# Train BiLSTM model with proper data handling
python scripts/train_with_preprocessing.py \
    --data-dir data/processed \
    --output-dir outputs/experiments/bilstm_trained \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --feature-dim 1629
```

This will:
- Load preprocessed data with proper train/test split
- Create BiLSTM model configured for 1629 input features
- Handle data reshaping automatically
- Train with validation and save checkpoints
- Generate training plots and evaluation results

### Step 3: Monitor Training

Check the output directory for:
- `training_history.json` - Training metrics
- `training_plots.png` - Loss and accuracy plots
- `checkpoints/best_model.pth` - Best model checkpoint
- `evaluation_results.json` - Test set evaluation

## Data Shape Handling

### Input Data Shape
```
Original: (batch_size, sequence_length, 1629)
LSTM Input: (batch_size, sequence_length, 1629) ✓
```

### Model Configuration
```python
# BiLSTM Layer
nn.LSTM(
    input_size=1629,        # Matches your feature dimension
    hidden_size=512,
    num_layers=2,
    bidirectional=True,
    batch_first=True
)
```

### Data Flow
1. **Input**: `(batch_size, seq_len, 1629)`
2. **LSTM**: Processes each timestep with 1629 features
3. **Attention**: Aggregates LSTM outputs
4. **Classifier**: Final prediction

## Key Features

### ✅ Proper Feature Dimension Handling
- Model configured for 1629 input features
- Automatic data reshaping utilities
- Input validation

### ✅ StandardScaler Normalization
- Applied to training data only
- Fitted scaler saved for inference
- Proper normalization pipeline

### ✅ Train/Test Split
- Stratified 70%/30% split
- Maintains class distribution
- Reproducible with random seed

### ✅ BiLSTM Configuration
- Bidirectional LSTM with attention
- Proper weight initialization
- Handles variable sequence lengths

### ✅ Training Pipeline
- Comprehensive trainer class
- Automatic data reshaping
- Validation and checkpointing
- Training visualization

## Example Usage

See `scripts/example_usage.py` for a complete example:

```bash
python scripts/example_usage.py
```

## Troubleshooting

### Common Issues

1. **Feature Dimension Mismatch**
   - Ensure your data has exactly 1629 features
   - Use `DataReshaper.reshape_for_lstm()` for automatic handling

2. **Memory Issues**
   - Reduce batch size if needed
   - Use gradient accumulation for large models

3. **Training Convergence**
   - Adjust learning rate
   - Try different optimizers
   - Check data normalization

### Data Validation

```python
# Validate your data shape
data = torch.randn(batch_size, seq_len, 1629)
reshaped = DataReshaper.reshape_for_lstm(data, feature_dim=1629)
print(f"Input shape: {data.shape}")
print(f"LSTM input shape: {reshaped.shape}")
```

## Results

After training, you'll have:
- Trained BiLSTM model with 1629 input features
- Proper train/test split with normalization
- Training history and evaluation metrics
- Model checkpoints for inference

The model is now properly configured to handle your 1629-dimensional feature sequences with appropriate preprocessing and training pipeline.
