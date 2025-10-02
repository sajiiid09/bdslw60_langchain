from typing import Tuple, Optional
import numpy as np


class RelativeQuantizationEncoder:
    """Relative Quantization Encoding (RQE) for frame-wise landmarks.

    Given a T x D sequence, compute deltas along time and quantize into integer bins.
    NaNs are handled by propagating NaNs; optional imputation can be added later.
    """

    def __init__(self, num_bins: int = 21, clip_sigma: float = 3.0):
        self.num_bins = int(num_bins)
        self.clip_sigma = float(clip_sigma)

    def fit_params(self, sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate per-dimension mean and std from valid deltas.

        sequence: (T, D)
        returns: (mean[D], std[D]) used for clipping and scaling
        """
        deltas = np.diff(sequence, axis=0)  # (T-1, D)
        mask = ~np.isnan(deltas)
        mean = np.zeros(sequence.shape[1], dtype=np.float32)
        std = np.ones(sequence.shape[1], dtype=np.float32)
        for d in range(sequence.shape[1]):
            valid = deltas[:, d][mask[:, d]]
            if valid.size > 0:
                mean[d] = valid.mean()
                std[d] = valid.std() if valid.std() > 1e-6 else 1.0
            else:
                mean[d] = 0.0
                std[d] = 1.0
        return mean, std

    def encode(self, sequence: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
        """Encode to integer bins in [0, num_bins-1].

        sequence: (T, D) float with NaNs
        mean,std: per-dim params; if None, computed from sequence deltas
        """
        if sequence.ndim != 2:
            raise ValueError("sequence must be 2D (T, D)")
        if mean is None or std is None:
            mean, std = self.fit_params(sequence)
        deltas = np.diff(sequence, axis=0)
        # standardize
        z = (deltas - mean) / (std + 1e-8)
        # clip
        z = np.clip(z, -self.clip_sigma, self.clip_sigma)
        # map to [0, 1]
        z01 = (z + self.clip_sigma) / (2 * self.clip_sigma)
        # quantize
        bins = np.floor(z01 * self.num_bins).astype(np.int32)
        bins = np.clip(bins, 0, self.num_bins - 1)
        # keep NaNs where deltas invalid
        bins[np.isnan(deltas)] = -1  # -1 as missing token
        return bins  # shape (T-1, D)

    def decode(self, bins: np.ndarray, start: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Approximate inverse decoding (for analysis), reconstruct sequence from codes.

        bins: (T-1, D), integers in [0, num_bins-1] or -1 for missing
        start: (D,) starting frame to integrate from
        returns: (T, D) reconstructed
        """
        Tm1, D = bins.shape
        z01 = (bins.astype(np.float32) + 0.5) / self.num_bins
        z = z01 * (2 * self.clip_sigma) - self.clip_sigma
        deltas = z * std + mean
        deltas[bins < 0] = 0.0
        out = np.empty((Tm1 + 1, D), dtype=start.dtype)
        out[0] = start
        for t in range(Tm1):
            out[t + 1] = out[t] + deltas[t]
        return out
