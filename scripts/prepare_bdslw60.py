import argparse
import os
from src.utils.io import read_yaml, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    dcfg = read_yaml(args.data_config)

    # ensure directory structure
    for key in ["raw", "interim", "landmarks", "processed", "metadata"]:
        ensure_dir(cfg["paths"][key])

    # ensure videos directory if specified in data config (fallback to default path)
    videos_path = dcfg.get("data", {}).get("videos_path", "src/data/videos/")
    ensure_dir(videos_path)

    labels_csv = dcfg.get("labels_csv")
    if not (labels_csv and os.path.exists(labels_csv)):
        print(f"[WARN] labels csv not found: {labels_csv}")
    else:
        print(f"[OK] labels csv found: {labels_csv}")

    print("Dataset directories prepared.")
    print(f"Videos directory: {videos_path}")


if __name__ == "__main__":
    main()
