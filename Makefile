# BdSLW60 Sign Language Recognition Makefile

.PHONY: help install setup prepare extract build train predict evaluate demo clean test

# Default target
help:
	@echo "BdSLW60 Sign Language Recognition Project"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  setup       - Setup project environment"
	@echo "  prepare     - Prepare dataset structure"
	@echo "  extract     - Extract pose landmarks from videos"
	@echo "  build       - Build RQE sequences"
	@echo "  train       - Train model"
	@echo "  predict     - Generate predictions"
	@echo "  evaluate    - Evaluate model"
	@echo "  demo        - Start demo server"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean temporary files"
	@echo ""
	@echo "Example usage:"
	@echo "  make install"
	@echo "  make setup"
	@echo "  make prepare"
	@echo "  make extract"
	@echo "  make build"
	@echo "  make train MODEL=videomae"
	@echo "  make predict CHECKPOINT=models/checkpoints/best_model.pth"
	@echo "  make evaluate CHECKPOINT=models/checkpoints/best_model.pth"

# Installation
install:
	pip install -r requirements.txt

setup:
	@echo "Setting up project directories..."
	mkdir -p data/raw data/interim data/landmarks data/processed data/metadata
	mkdir -p models/checkpoints models/exports
	mkdir -p outputs/logs outputs/predictions outputs/reports
	mkdir -p notebooks
	@echo "Project setup complete!"

# Dataset preparation
prepare:
	python scripts/prepare_bdslw60.py \
		--config configs/default.yaml \
		--data-config configs/data_bdslw60.yaml

# Pose extraction
extract:
	python scripts/extract_pose.py \
		--config configs/default.yaml \
		--data-config configs/data_bdslw60.yaml

# Sequence building
build:
	python scripts/build_sequences.py \
		--config configs/default.yaml \
		--data-config configs/data_bdslw60.yaml

# Training
MODEL ?= videomae
OUTPUT_DIR ?= outputs/experiments/$(MODEL)_$(shell date +%Y%m%d_%H%M%S)

train:
	@echo "Training $(MODEL) model..."
	@echo "Output directory: $(OUTPUT_DIR)"
	python scripts/train.py \
		--config configs/default.yaml \
		--data-config configs/data_bdslw60.yaml \
		--model-config configs/model_$(MODEL).yaml \
		--output-dir $(OUTPUT_DIR)

# Prediction
CHECKPOINT ?= models/checkpoints/best_model.pth
MODEL_CONFIG ?= configs/model_videomae.yaml
INPUT ?= data/raw/test_video.mp4
OUTPUT ?= outputs/predictions/predictions.json

predict:
	python scripts/predict.py \
		--checkpoint $(CHECKPOINT) \
		--model-config $(MODEL_CONFIG) \
		--input $(INPUT) \
		--output $(OUTPUT)

# Evaluation
EVAL_OUTPUT ?= outputs/reports/evaluation_results.json

evaluate:
	python scripts/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--model-config $(MODEL_CONFIG) \
		--data-config configs/data_bdslw60.yaml \
		--config configs/default.yaml \
		--output $(EVAL_OUTPUT)

# Demo server
DEMO_PORT ?= 5000

demo:
	python scripts/demo_server.py \
		--checkpoint $(CHECKPOINT) \
		--model-config $(MODEL_CONFIG) \
		--port $(DEMO_PORT)

# Testing
test:
	python -m pytest tests/ -v

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	@echo "Cleanup complete!"

# Development
dev-install:
	pip install -r requirements.txt
	pip install -e .

# Format code
format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

# Lint code
lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

# Type checking
type-check:
	mypy src/ scripts/

# Full pipeline
pipeline: setup prepare extract build
	@echo "Full preprocessing pipeline completed!"

# Quick start
quick-start: install setup
	@echo "Quick start completed! Next steps:"
	@echo "1. Add your BdSLW60 videos to data/raw/"
	@echo "2. Create labels.csv in data/raw/"
	@echo "3. Run: make prepare"
	@echo "4. Run: make extract"
	@echo "5. Run: make build"
	@echo "6. Run: make train MODEL=videomae"

