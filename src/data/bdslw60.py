from typing import List, Dict, Optional, Tuple
import os
import pandas as pd
import numpy as np


class BdSLW60LandmarkDataset:
    """Dataset over landmark npz files with labels from labels.csv.

    Expects labels_csv with columns: video_path, label, signer_id (configurable).
    This loader maps video paths to corresponding landmark npz in data/landmarks with same relative path.
    """

    def __init__(self, labels_csv: str, landmarks_root: str, video_column: str = "video_path", label_column: str = "label", signer_column: str = "signer_id"):
        self.df = pd.read_csv(labels_csv)
        self.video_column = video_column
        self.label_column = label_column
        self.signer_column = signer_column
        self.landmarks_root = landmarks_root

    def _npz_for_video(self, video_rel_path: str) -> str:
        rel = os.path.splitext(video_rel_path)[0] + ".npz"
        return os.path.normpath(os.path.join(self.landmarks_root, rel))

    def items(self) -> List[Dict]:
        out: List[Dict] = []
        for _, row in self.df.iterrows():
            video_rel = row[self.video_column]
            label = row[self.label_column]
            signer = row.get(self.signer_column, None)
            npz_path = self._npz_for_video(video_rel)
            out.append({"npz": npz_path, "label": label, "signer": signer, "video": video_rel})
        return out


class BdSLW60ProcessedDataset:
    """Dataset over processed (RQE) sequences .npz files with labels.

    Each npz contains fields: codes (T-1, D), meta dict (json-serializable), label
    """

    def __init__(self, processed_root: str, labels_csv: Optional[str] = None, video_column: str = "video_path", label_column: str = "label"):
        self.processed_root = processed_root
        self.labels: Optional[pd.DataFrame] = None
        self.video_column = video_column
        self.label_column = label_column
        if labels_csv is not None and os.path.exists(labels_csv):
            self.labels = pd.read_csv(labels_csv)

    def list_files(self) -> List[str]:
        files: List[str] = []
        for root, _, fnames in os.walk(self.processed_root):
            for f in fnames:
                if f.endswith('.npz'):
                    files.append(os.path.join(root, f))
        return sorted(files)

    def load_item(self, path: str) -> Dict:
        data = np.load(path, allow_pickle=True)
        item = {
            "codes": data["codes"],
            "meta": data["meta"].item() if "meta" in data else {},
            "label": data["label"] if "label" in data else None,
        }
        return item

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.list_files())

    def __getitem__(self, idx: int) -> Dict:
        """Get item by index for PyTorch DataLoader."""
        files = self.list_files()
        if idx >= len(files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(files)}")
        
        item = self.load_item(files[idx])
        
        # Convert to PyTorch tensors
        import torch
        codes = torch.from_numpy(item["codes"]).float()
        label = torch.tensor(item["label"] if item["label"] is not None else 0, dtype=torch.long)
        
        return {
            "input": codes,
            "target": label,
            "meta": item["meta"]
        }