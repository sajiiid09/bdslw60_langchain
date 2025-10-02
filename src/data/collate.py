from typing import List, Tuple
import numpy as np
import torch


def pad_sequences_numpy(sequences: List[np.ndarray], pad_value: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int32)
    max_len = int(lengths.max()) if lengths.size > 0 else 0
    if max_len == 0:
        return np.empty((0, 0, 0), dtype=np.int32), lengths
    feat_dim = sequences[0].shape[1]
    batch = np.full((len(sequences), max_len, feat_dim), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        L = seq.shape[0]
        batch[i, :L] = seq
    return batch, lengths


def pad_collate_torch(batch: List[dict], pad_value: int = -1) -> dict:
    """Collate function for PyTorch DataLoader that handles dictionary batches."""
    # Extract sequences and targets
    sequences = [item["input"] for item in batch]
    targets = [item["target"] for item in batch]
    
    # Pad sequences
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
    if max_len == 0:
        return {
            "input": torch.empty(0),
            "target": torch.empty(0, dtype=torch.long),
            "lengths": lengths
        }
    feat_dim = sequences[0].size(1)
    padded_sequences = torch.full((len(sequences), max_len, feat_dim), pad_value, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        L = seq.size(0)
        padded_sequences[i, :L] = seq
    
    return {
        "input": padded_sequences,
        "target": torch.stack(targets),
        "lengths": lengths
    }
