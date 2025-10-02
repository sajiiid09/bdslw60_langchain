import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.signal import resample

from src.utils.io import read_yaml, ensure_dir
from src.utils.seed import set_seed
from src.utils.rqe import RelativeQuantizationEncoder


def resample_time(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """Resample along time dimension to target length using Fourier method.
    sequence: (T, D) -> (target_len, D)
    NaNs are linearly interpolated per-dim before resampling.
    """
    if sequence.shape[0] == 0 or target_len <= 0:
        return np.empty((0, sequence.shape[1]), dtype=sequence.dtype)
    seq = sequence.copy()
    # simple NaN interpolation
    for d in range(seq.shape[1]):
        s = seq[:, d]
        nans = np.isnan(s)
        if nans.any():
            idx = np.arange(len(s))
            valid = ~nans
            if valid.sum() == 0:
                s[:] = 0.0
            else:
                s[nans] = np.interp(idx[nans], idx[valid], s[valid])
        seq[:, d] = s
    if target_len == seq.shape[0]:
        return seq
    return resample(seq, num=target_len, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=21)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    dcfg = read_yaml(args.data_config)
    set_seed(cfg.get("seed", 42))

    lm_root = cfg["paths"]["landmarks"]
    out_root = cfg["paths"]["processed"]
    ensure_dir(out_root)

    target_fps = args.fps or cfg.get("fps", 30)

    files = []
    for root, _, names in os.walk(lm_root):
        for f in names:
            if f.endswith('.npz'):
                files.append(os.path.join(root, f))
    files = sorted(files)

    encoder = RelativeQuantizationEncoder(num_bins=args.num_bins, clip_sigma=args.clip_sigma)

    for path in tqdm(files, desc="Building sequences"):
        rel = os.path.relpath(path, lm_root)
        out_path = os.path.join(out_root, rel)
        out_dir = os.path.dirname(out_path)
        ensure_dir(out_dir)
        if os.path.exists(out_path):
            continue
        data = np.load(path, allow_pickle=True)
        landmarks = data["landmarks"].astype(np.float32)  # (T, D)
        # deduce original fps from metadata if available, else assume target
        # For simplicity here, just resample to fixed length proportional to target fps
        # Use duration estimate via number of frames and assumed src fps (target)
        T = landmarks.shape[0]
        if T == 0:
            np.savez_compressed(out_path, codes=np.empty((0, 0), dtype=np.int32), meta={"source": str(data.get("source", ""))})
            continue
        # Normalize to desired FPS by matching time axis length: T_target = T * (target_fps / src_fps)
        # If src_fps unknown, keep T as-is; here we keep as-is to avoid artifacts.
        # Optionally set a fixed max length or proportional scaling; here we keep T.
        seq = landmarks  # (T, D)
        # Encode via RQE
        mean, std = encoder.fit_params(seq)
        codes = encoder.encode(seq, mean, std)  # (T-1, D)
        meta = {
            "num_bins": encoder.num_bins,
            "clip_sigma": encoder.clip_sigma,
            "dims": seq.shape[1],
            "frames": seq.shape[0],
            "source": str(data.get("source", "")),
        }
        np.savez_compressed(out_path, codes=codes.astype(np.int16), meta=meta)

    print("Sequence building complete.")


if __name__ == "__main__":
    main()
