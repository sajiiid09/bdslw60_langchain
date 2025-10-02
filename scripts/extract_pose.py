import argparse
import os
import glob
import numpy as np
from tqdm import tqdm

from src.utils.io import read_yaml, ensure_dir
from src.utils.seed import set_seed
from src.data.mediapipe_runner import HolisticExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    dcfg = read_yaml(args.data_config)
    # set_seed(cfg.get("seed", 42))  # Skip due to numpy compatibility issue

    # Prefer explicit videos path from data config if provided; else use default raw path
    videos_path = dcfg.get("data", {}).get("videos_path")
    raw_root = videos_path if videos_path else cfg["paths"]["raw"]
    lm_root = cfg["paths"]["landmarks"]
    ensure_dir(lm_root)

    pattern = os.path.join(raw_root, cfg.get("video_glob", "**/*.mp4"))
    videos = sorted(glob.glob(pattern, recursive=True))
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    extractor = HolisticExtractor()

    for vp in tqdm(videos, desc="Extracting pose"):
        rel = os.path.relpath(vp, raw_root)
        out_path = os.path.join(lm_root, os.path.splitext(rel)[0] + ".npz")
        out_dir = os.path.dirname(out_path)
        ensure_dir(out_dir)
        if os.path.exists(out_path):
            continue
        data = extractor.extract_video(vp)
        landmarks = data["landmarks"].astype(cfg.get("landmark_dtype", "float32"))
        np.savez_compressed(out_path, landmarks=landmarks, source=vp)

    print("Pose extraction complete.")


if __name__ == "__main__":
    main()
