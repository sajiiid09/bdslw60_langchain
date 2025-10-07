#!/usr/bin/env python3
"""
Standalone RQE Processing Script

This script processes landmark data using Relative Quantization Encoding (RQE)
and saves the processed sequences to data/processed/ directory.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from tqdm import tqdm


def read_yaml(file_path: str) -> Dict:
    """Read YAML configuration file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class RelativeQuantizationEncoder:
    """Relative Quantization Encoder for landmark sequences."""
    
    def __init__(self, num_bins: int = 21, clip_sigma: float = 3.0):
        self.num_bins = num_bins
        self.clip_sigma = clip_sigma
    
    def encode(self, landmarks: np.ndarray, mean: Optional[np.ndarray] = None, 
               std: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode landmark sequence using RQE."""
        if landmarks.shape[0] < 2:
            # Single frame, return zeros
            return np.zeros((landmarks.shape[0], landmarks.shape[1]), dtype=np.int32)
        
        # Compute relative differences
        diffs = np.diff(landmarks, axis=0)
        
        # Normalize using provided statistics or compute from data
        if mean is not None and std is not None:
            normalized_diffs = (diffs - mean) / (std + 1e-8)
        else:
            # Compute statistics from current data
            mean = np.nanmean(diffs, axis=0)
            std = np.nanstd(diffs, axis=0)
            normalized_diffs = (diffs - mean) / (std + 1e-8)
        
        # Clip extreme values
        normalized_diffs = np.clip(normalized_diffs, -self.clip_sigma, self.clip_sigma)
        
        # Quantize to bins
        bin_width = 2 * self.clip_sigma / self.num_bins
        codes = np.floor((normalized_diffs + self.clip_sigma) / bin_width).astype(np.int32)
        codes = np.clip(codes, 0, self.num_bins - 1)
        
        return codes
    
    def decode(self, codes: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Decode RQE codes back to landmark differences."""
        bin_width = 2 * self.clip_sigma / self.num_bins
        normalized_diffs = codes * bin_width - self.clip_sigma
        diffs = normalized_diffs * std + mean
        return diffs


def process_landmarks_to_rqe(landmarks_dir: str = "data/landmarks",
                           processed_dir: str = "data/processed",
                           config: Optional[Dict] = None) -> None:
    """Process landmark data using RQE encoding."""
    
    landmarks_path = Path(landmarks_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Find all landmark files
    landmark_files = list(landmarks_path.glob("*.npz"))
    
    if not landmark_files:
        print(f"No landmark files found in {landmarks_dir}")
        return
    
    print(f"Found {len(landmark_files)} landmark files to process")
    
    # Initialize RQE encoder
    rqe_encoder = RelativeQuantizationEncoder()
    
    # Collect all differences for global statistics
    all_diffs = []
    landmark_data = {}
    
    print("Collecting landmark data...")
    for landmark_file in tqdm(landmark_files, desc="Loading landmarks"):
        try:
            data = np.load(landmark_file, allow_pickle=True)
            landmarks = data['landmarks']
            
            if landmarks.shape[0] >= 2:
                diffs = np.diff(landmarks, axis=0)
                all_diffs.append(diffs)
                landmark_data[landmark_file.stem] = landmarks
            else:
                print(f"⚠ Skipping {landmark_file.name} (too few frames)")
                
        except Exception as e:
            print(f"✗ Error loading {landmark_file.name}: {e}")
    
    if not all_diffs:
        print("No valid landmark data found")
        return
    
    # Compute global statistics
    print("Computing global statistics...")
    all_diffs = np.concatenate(all_diffs, axis=0)
    global_mean = np.nanmean(all_diffs, axis=0)
    global_std = np.nanstd(all_diffs, axis=0)
    
    print(f"Global mean shape: {global_mean.shape}")
    print(f"Global std shape: {global_std.shape}")
    
    # Process each landmark file
    print("Processing landmarks with RQE...")
    for landmark_file in tqdm(landmark_files, desc="Processing"):
        try:
            # Check if already processed
            processed_file = processed_path / f"{landmark_file.stem}.npz"
            if processed_file.exists():
                print(f"✓ Already processed {landmark_file.name}, skipping")
                continue
            
            if landmark_file.stem not in landmark_data:
                continue
            
            landmarks = landmark_data[landmark_file.stem]
            
            # Encode using RQE
            codes = rqe_encoder.encode(landmarks, global_mean, global_std)
            
            # Create metadata
            meta = {
                'num_bins': rqe_encoder.num_bins,
                'clip_sigma': rqe_encoder.clip_sigma,
                'dims': landmarks.shape[1],
                'frames': landmarks.shape[0],
                'source': f'src/data/videos/{landmark_file.stem}.mp4'
            }
            
            # Save processed data
            np.savez_compressed(
                processed_file,
                codes=codes,
                meta=meta
            )
            
            print(f"✓ Processed {landmark_file.name} -> {codes.shape[0]} frames, {codes.shape[1]} features")
            
        except Exception as e:
            print(f"✗ Error processing {landmark_file.name}: {e}")
    
    print(f"\nCompleted RQE processing")
    print(f"Processed data saved to: {processed_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process landmarks using RQE encoding")
    parser.add_argument("--landmarks-dir", default="data/landmarks", 
                       help="Directory containing landmark .npz files")
    parser.add_argument("--processed-dir", default="data/processed", 
                       help="Directory to save processed RQE data")
    parser.add_argument("--config", 
                       help="Path to configuration file (optional)")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        config = read_yaml(args.config)
    
    process_landmarks_to_rqe(
        landmarks_dir=args.landmarks_dir,
        processed_dir=args.processed_dir,
        config=config
    )


if __name__ == "__main__":
    main()

