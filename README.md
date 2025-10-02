# BdSLW60 Preprocessing (Pose -> RQE Sequences)

This project preprocesses BdSLW60 videos for Bangla Sign Language recognition by:
- Extracting MediaPipe Holistic pose landmarks (face, hands, body)
- Normalizing video frame rate
- Applying Relative Quantization Encoding (RQE)
- Emitting ready-to-train sequences

No LLM/GPT correction is included in this phase.

## Quickstart

1) Create environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate bdsl-pre

# or with pip
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Configure paths in `configs/data_bdslw60.yaml` and defaults in `configs/default.yaml`.

3) Verify dataset structure (optional):

```bash
python scripts/prepare_bdslw60.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
```

4) Extract pose landmarks to `data/landmarks/`:

```bash
python scripts/extract_pose.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
```

5) Build RQE-encoded sequences into `data/processed/`:

```bash
python scripts/build_sequences.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
```

6) Load processed data later for training via `src/data/bdslw60.py` and `src/data/collate.py`.

## How to run

- With Make:

```bash
make setup
make prepare
make extract
make build
make train MODEL=videomae
make evaluate CHECKPOINT=outputs/experiments/videomae_run/checkpoints/best_model.pth
make predict CHECKPOINT=outputs/experiments/videomae_run/checkpoints/best_model.pth INPUT=data/raw/path/to/video.mp4
make demo CHECKPOINT=outputs/experiments/videomae_run/checkpoints/best_model.pth
```

- Without Make (Windows-safe):

```bash
# Prepare
python scripts/prepare_bdslw60.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
# Extract pose
python scripts/extract_pose.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
# Build sequences (RQE)
python scripts/build_sequences.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml
# Train
python scripts/train.py --config configs/default.yaml --data-config configs/data_bdslw60.yaml --model-config configs/model_videomae.yaml --output-dir outputs/experiments/videomae_run
# Evaluate
python scripts/evaluate.py --checkpoint outputs/experiments/videomae_run/checkpoints/best_model.pth --model-config configs/model_videomae.yaml --data-config configs/data_bdslw60.yaml --config configs/default.yaml --output outputs/reports/eval_videomae.json
# Predict
python scripts/predict.py --checkpoint outputs/experiments/videomae_run/checkpoints/best_model.pth --model-config configs/model_videomae.yaml --input data/raw/path/to/video.mp4 --output outputs/predictions/predictions.json
# Demo server
python scripts/demo_server.py --checkpoint outputs/experiments/videomae_run/checkpoints/best_model.pth --model-config configs/model_videomae.yaml --port 5000
```

## Layout

Key directories:
- `data/raw/`: original BdSLW60 videos and labels CSV
- `data/landmarks/`: extracted holistic landmarks (`.npz` per video)
- `data/processed/`: RQE sequences (`.npz` per sample)
- `data/metadata/`: label maps, splits

## Notes
- MediaPipe Holistic returns up to 543 landmarks per frame (468 face, 33 pose, 21 LH, 21 RH). We export normalized x,y (0..1), z in image-depth units (as provided by MediaPipe), filling missing parts with NaN.
- FPS normalization uses linear time resampling to the target FPS.
- RQE discretizes relative frame-to-frame deltas into integer codes; metadata is stored alongside outputs.

## License
MIT
