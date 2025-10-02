"""Lightweight package init to avoid importing heavy deps (e.g., torch) at import time.

Modules should be imported explicitly by consumers, e.g.:
    from src.data.mediapipe_runner import HolisticExtractor
    from src.data.bdslw60 import BdSLW60LandmarkDataset
"""

__all__ = [
    "bdslw60",
    "mediapipe_runner",
    "collate",
]
