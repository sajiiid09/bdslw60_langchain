import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference.predictor import SignLanguagePredictor, load_predictor_from_checkpoint
from src.data.bdslw60 import BdSLW60ProcessedDataset
from src.data.collate import pad_collate_torch
from src.utils.io import read_yaml, write_json
from src.utils.metrics import (
    compute_accuracy, compute_top_k_accuracy, compute_f1_score,
    compute_precision_recall
)
from src.utils.viz import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-config", required=True, help="Path to model config")
    parser.add_argument("--data-config", required=True, help="Path to data config")
    parser.add_argument("--config", required=True, help="Path to default config")
    parser.add_argument("--output", required=True, help="Path to output evaluation results")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    # Split not supported by BdSLW60ProcessedDataset; kept for compatibility but unused
    parser.add_argument("--split", default="all", help="Dataset split to evaluate (unused)")
    args = parser.parse_args()
    
    # Load configs
    config = read_yaml(args.config)
    data_config = read_yaml(args.data_config)
    eval_config = read_yaml("configs/eval.yaml")
    
    # Load predictor
    predictor = load_predictor_from_checkpoint(
        args.checkpoint, 
        args.model_config, 
        args.device
    )
    
    print(f"Loaded model: {predictor.get_model_info()}")
    
    # Create dataset (no split support; evaluate on full processed dataset)
    processed_root = config["paths"]["processed"]
    dataset = BdSLW60ProcessedDataset(processed_root)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_collate_torch
    )
    
    # Evaluate model
    all_predictions = []
    all_targets = []
    all_confidences = []
    all_probabilities = []
    
    print(f"Evaluating on {args.split} split...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = batch["input"]
        targets = batch["target"]
        
        # Generate predictions
        for i in range(inputs.size(0)):
            input_tensor = inputs[i:i+1]  # Keep batch dimension
            result = predictor.predict(input_tensor)
            
            all_predictions.append(result["predicted_class"])
            all_targets.append(targets[i].item())
            all_confidences.append(result["confidence"])
            if "probabilities" in result:
                all_probabilities.append(np.array(result["probabilities"]))
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    confidences = np.array(all_confidences)
    
    # Compute metrics
    results = {
        "model_info": predictor.get_model_info(),
        "dataset_split": args.split,
        "num_samples": len(predictions),
        "metrics": {}
    }
    
    # Classification metrics
    if eval_config["metrics"]["accuracy"]:
        results["metrics"]["accuracy"] = compute_accuracy(predictions, targets)
    
    if eval_config["metrics"]["top_k_accuracy"] and len(all_probabilities) == len(predictions) and len(all_probabilities) > 0:
        y_score = np.stack(all_probabilities, axis=0)
        for k in eval_config["metrics"]["top_k_accuracy"]:
            results["metrics"][f"top_{k}_accuracy"] = compute_top_k_accuracy(y_score, targets, k)
    
    if eval_config["metrics"]["f1_score"]:
        results["metrics"]["f1_score"] = compute_f1_score(predictions, targets)
    
    if eval_config["metrics"]["precision_recall"]:
        pr_results = compute_precision_recall(predictions, targets)
        results["metrics"].update(pr_results)
    
    # Additional metrics
    results["metrics"]["mean_confidence"] = float(np.mean(confidences))
    results["metrics"]["std_confidence"] = float(np.std(confidences))
    
    # Per-class accuracy
    if eval_config["metrics"]["per_class_accuracy"]:
        unique_classes = np.unique(targets)
        per_class_acc = {}
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() > 0:
                per_class_acc[int(cls)] = float((predictions[mask] == cls).mean())
        results["metrics"]["per_class_accuracy"] = per_class_acc
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_json(args.output, results)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Dataset split: {args.split}")
    print(f"Number of samples: {len(predictions)}")
    print("\nMetrics:")
    for metric, value in results["metrics"].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Plot confusion matrix if requested
    if eval_config["evaluation"]["plot_confusion_matrix"]:
        plot_path = os.path.join(os.path.dirname(args.output), "confusion_matrix.png")
        plot_confusion_matrix(
            targets, predictions, 
            save_path=plot_path,
            title=f"Confusion Matrix - {args.split} split"
        )
        print(f"Confusion matrix saved to {plot_path}")
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
