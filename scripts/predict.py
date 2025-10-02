import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm

from src.inference.predictor import SignLanguagePredictor, load_predictor_from_checkpoint
from src.utils.io import read_yaml, write_json


def main():
    parser = argparse.ArgumentParser(description="Generate predictions from trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-config", required=True, help="Path to model config")
    parser.add_argument("--input", required=True, help="Path to input video or directory of videos")
    parser.add_argument("--output", required=True, help="Path to output predictions file")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()
    
    # Load predictor
    predictor = load_predictor_from_checkpoint(
        args.checkpoint, 
        args.model_config, 
        args.device
    )
    
    print(f"Loaded model: {predictor.get_model_info()}")
    
    # Get input files
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = []
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    input_files.append(os.path.join(root, file))
        input_files.sort()
    else:
        raise ValueError(f"Input path does not exist: {args.input}")
    
    print(f"Found {len(input_files)} input files")
    
    # Generate predictions
    predictions = []
    
    for input_file in tqdm(input_files, desc="Generating predictions"):
        try:
            result = predictor.predict(input_file)
            predictions.append({
                "input_file": input_file,
                "predicted_class": int(result["predicted_class"]),
                "confidence": float(result["confidence"]),
                "probabilities": result["probabilities"].tolist(),
                "logits": result["logits"].tolist()
            })
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            predictions.append({
                "input_file": input_file,
                "error": str(e)
            })
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_json(args.output, {
        "model_info": predictor.get_model_info(),
        "predictions": predictions
    })
    
    print(f"Predictions saved to {args.output}")
    print(f"Processed {len(predictions)} files")


if __name__ == "__main__":
    main()
