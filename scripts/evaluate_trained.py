#!/usr/bin/env python3
"""
Trained Model Evaluation and Prediction Script

This script loads the trained model and generates predictions on the processed data.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm


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


class TrainedSignPredictor:
    """Sign language predictor using trained model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint['model_config']
        
        # Create model
        self.model = BiLSTMAttentionModel(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create class mapping
        self.class_to_word = self._create_class_mapping()
        
        print(f"Loaded trained model from {model_path}")
        print(f"Model config: {model_config}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    def _create_class_mapping(self) -> Dict[int, str]:
        """Create class to word mapping."""
        words = [
            "class_0",  # U1 videos
            "class_1",  # U2 videos  
            "class_2",  # U3 videos
            "class_3",  # U4 videos
            "class_4",  # U5 videos
            "class_5"   # U10 videos
        ]
        return {i: word for i, word in enumerate(words)}
    
    def predict_from_codes(self, codes: np.ndarray, max_length: int = 2000) -> Dict[str, any]:
        """Predict sign language class from RQE codes."""
        # Pad or truncate sequence
        if codes.shape[0] > max_length:
            codes = codes[:max_length]
        elif codes.shape[0] < max_length:
            padding = np.zeros((max_length - codes.shape[0], codes.shape[1]))
            codes = np.vstack([codes, padding])
        
        # Convert to tensor
        input_tensor = torch.from_numpy(codes).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1)[0].item()
        
        predicted_word = self.class_to_word.get(predicted_class, f"unknown_class_{predicted_class}")
        
        return {
            "predicted_class": predicted_class,
            "predicted_word": predicted_word,
            "confidence": confidence,
            "probabilities": probabilities.cpu().numpy()[0].tolist()
        }


def evaluate_trained_model(model_path: str, processed_dir: str = "data/processed", 
                         output_dir: str = "outputs/predictions_trained") -> None:
    """Evaluate trained model on processed data."""
    
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all processed files
    npz_files = list(processed_path.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {processed_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to evaluate")
    
    # Initialize predictor
    predictor = TrainedSignPredictor(model_path)
    
    # Process each file
    results = {}
    
    for npz_file in tqdm(npz_files, desc="Evaluating with trained model"):
        try:
            # Load RQE data
            data = np.load(npz_file, allow_pickle=True)
            codes = data['codes']  # Shape: (frames, features)
            meta = data['meta'].item()
            
            # Make prediction
            prediction = predictor.predict_from_codes(codes)
            
            # Save prediction to text file
            output_file = output_path / f"{npz_file.stem}_trained.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Video: {npz_file.name}\n")
                f.write(f"Predicted Word: {prediction['predicted_word']}\n")
                f.write(f"Confidence: {prediction['confidence']:.4f}\n")
                f.write(f"Predicted Class: {prediction['predicted_class']}\n")
                f.write(f"Source: {meta.get('source', 'unknown')}\n")
                f.write(f"Frames: {meta.get('frames', 'unknown')}\n")
                f.write(f"Model: Trained BiLSTM with Attention\n")
            
            results[npz_file.name] = prediction
            print(f"✓ Processed {npz_file.name} -> {prediction['predicted_word']} (confidence: {prediction['confidence']:.4f})")
            
        except Exception as e:
            print(f"✗ Error processing {npz_file.name}: {e}")
            results[npz_file.name] = {"error": str(e)}
    
    # Save summary results
    summary_file = output_path / "trained_predictions_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted evaluation with trained model")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_file}")


def compare_predictions(original_dir: str = "outputs/predictions", 
                       trained_dir: str = "outputs/predictions_trained") -> None:
    """Compare predictions from simple model vs trained model."""
    
    original_path = Path(original_dir)
    trained_path = Path(trained_dir)
    
    print("\n=== Prediction Comparison ===")
    print("File\t\t\tSimple Model\t\tTrained Model")
    print("-" * 60)
    
    for file in original_path.glob("*.txt"):
        if not file.name.endswith("_trained.txt"):
            # Read original prediction
            with open(file, 'r') as f:
                original_lines = f.readlines()
                original_word = original_lines[1].split(": ")[1].strip()
                original_conf = float(original_lines[2].split(": ")[1].strip())
            
            # Read trained prediction
            trained_file = trained_path / f"{file.stem}_trained.txt"
            if trained_file.exists():
                with open(trained_file, 'r') as f:
                    trained_lines = f.readlines()
                    trained_word = trained_lines[1].split(": ")[1].strip()
                    trained_conf = float(trained_lines[2].split(": ")[1].strip())
                
                print(f"{file.stem:<20}\t{original_word} ({original_conf:.3f})\t{trained_word} ({trained_conf:.3f})")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained sign language model")
    parser.add_argument("--model-path", default="outputs/experiments/bilstm_trained/best_model.pth", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Directory containing processed RQE data")
    parser.add_argument("--output-dir", default="outputs/predictions_trained", 
                       help="Directory to save evaluation results")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare with original simple model predictions")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # Evaluate trained model
    evaluate_trained_model(
        model_path=args.model_path,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir
    )
    
    # Compare predictions if requested
    if args.compare:
        compare_predictions()


if __name__ == "__main__":
    main()
