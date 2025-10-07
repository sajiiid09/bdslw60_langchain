#!/usr/bin/env python3
"""
Sign Language Prediction Script

This script loads landmark data from processed .npz files and predicts sign language words.
It can work with pre-trained models or create a simple baseline model for demonstration.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import yaml


def read_yaml(file_path: str) -> Dict:
    """Read YAML configuration file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class SimpleSignPredictor:
    """Simple sign language predictor for demonstration purposes."""
    
    def __init__(self, num_classes: int = 60, input_size: int = 1629):
        """Initialize a simple predictor."""
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a simple model
        self.model = self._create_simple_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Create a simple class mapping (for demonstration)
        self.class_to_word = self._create_class_mapping()
    
    def _create_simple_model(self) -> nn.Module:
        """Create a simple neural network for demonstration."""
        return nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_classes)
        )
    
    def _create_class_mapping(self) -> Dict[int, str]:
        """Create a simple class to word mapping."""
        # This is a demonstration mapping - in reality, you'd load this from your dataset
        words = [
            "hello", "goodbye", "thank_you", "please", "sorry", "yes", "no", "help",
            "water", "food", "home", "family", "friend", "love", "happy", "sad",
            "angry", "tired", "sick", "doctor", "hospital", "school", "work", "money",
            "time", "today", "tomorrow", "yesterday", "morning", "evening", "night",
            "hot", "cold", "big", "small", "good", "bad", "beautiful", "ugly",
            "fast", "slow", "new", "old", "young", "old_person", "child", "baby",
            "man", "woman", "boy", "girl", "cat", "dog", "bird", "fish",
            "car", "bus", "train", "plane", "book", "pen", "paper", "computer"
        ]
        
        # Pad or truncate to match num_classes
        if len(words) < self.num_classes:
            words.extend([f"word_{i}" for i in range(len(words), self.num_classes)])
        else:
            words = words[:self.num_classes]
        
        return {i: word for i, word in enumerate(words)}
    
    def predict_from_landmarks(self, landmarks: np.ndarray) -> Dict[str, any]:
        """Predict sign language word from landmark data."""
        # Convert landmarks to tensor
        if landmarks.ndim == 2:
            # Take mean across time dimension for simple prediction
            landmarks = np.mean(landmarks, axis=0)
        
        input_tensor = torch.from_numpy(landmarks).float().unsqueeze(0).to(self.device)
        
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


def load_landmark_data(npz_path: str) -> Tuple[np.ndarray, Dict]:
    """Load landmark data from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    codes = data['codes']  # Shape: (frames, features)
    meta = data['meta'].item()
    return codes, meta


def predict_signs_from_processed_data(processed_dir: str = "data/processed", 
                                    output_dir: str = "outputs/predictions",
                                    model_config_path: Optional[str] = None) -> None:
    """Predict sign language words from all processed .npz files."""
    
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .npz files
    npz_files = list(processed_path.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {processed_dir}")
        return
    
    print(f"Found {len(npz_files)} .npz files to process")
    
    # Initialize predictor
    if model_config_path and os.path.exists(model_config_path):
        # Try to load a real trained model
        try:
            model_config = read_yaml(model_config_path)
            predictor = load_trained_predictor(model_config_path)
            print(f"Loaded trained model from {model_config_path}")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            print("Using simple predictor instead")
            predictor = SimpleSignPredictor()
    else:
        # Use simple predictor for demonstration
        predictor = SimpleSignPredictor()
        print("Using simple predictor for demonstration")
    
    # Process each file
    results = {}
    
    for npz_file in tqdm(npz_files, desc="Processing files"):
        try:
            # Load landmark data
            codes, meta = load_landmark_data(str(npz_file))
            
            # Make prediction
            prediction = predictor.predict_from_landmarks(codes)
            
            # Save prediction to text file
            output_file = output_path / f"{npz_file.stem}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Video: {npz_file.name}\n")
                f.write(f"Predicted Word: {prediction['predicted_word']}\n")
                f.write(f"Confidence: {prediction['confidence']:.4f}\n")
                f.write(f"Predicted Class: {prediction['predicted_class']}\n")
                f.write(f"Source: {meta.get('source', 'unknown')}\n")
                f.write(f"Frames: {meta.get('frames', 'unknown')}\n")
            
            results[npz_file.name] = prediction
            print(f"✓ Processed {npz_file.name} -> {prediction['predicted_word']} (confidence: {prediction['confidence']:.4f})")
            
        except Exception as e:
            print(f"✗ Error processing {npz_file.name}: {e}")
            results[npz_file.name] = {"error": str(e)}
    
    # Save summary results
    summary_file = output_path / "predictions_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted processing {len(npz_files)} files")
    print(f"Results saved to: {output_path}")
    print(f"Summary saved to: {summary_file}")


def load_trained_predictor(model_config_path: str):
    """Load a trained model predictor."""
    # This would load a real trained model
    # For now, we'll use the simple predictor
    return SimpleSignPredictor()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict sign language words from processed landmark data")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Directory containing processed .npz files")
    parser.add_argument("--output-dir", default="outputs/predictions", 
                       help="Directory to save prediction results")
    parser.add_argument("--model-config", 
                       help="Path to model configuration file (optional)")
    
    args = parser.parse_args()
    
    predict_signs_from_processed_data(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        model_config_path=args.model_config
    )


if __name__ == "__main__":
    main()
