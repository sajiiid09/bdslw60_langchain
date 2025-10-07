import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.factory import create_model
from src.data.bdslw60 import BdSLW60ProcessedDataset
from src.data.collate import pad_collate_torch
from src.training.loop import train_model
from src.training.optimizer import create_optimizer_and_scheduler
from src.utils.io import read_yaml, ensure_dir
from src.utils.seed import set_seed
from src.utils.logging import setup_logging, log_model_info


def main():
    parser = argparse.ArgumentParser(description="Train sign language recognition model")
    parser.add_argument("--config", required=True, help="Path to default config")
    parser.add_argument("--data-config", required=True, help="Path to data config")
    parser.add_argument("--model-config", required=True, help="Path to model config")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    args = parser.parse_args()
    
    # Load configs
    config = read_yaml(args.config)
    data_config = read_yaml(args.data_config)
    model_config = read_yaml(args.model_config)
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Setup logging
    log_dir = os.path.join(args.output_dir, "logs")
    ensure_dir(log_dir)
    logger = setup_logging(log_dir)
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create datasets first (needed to infer feature dims for adapters)
    processed_root = config["paths"]["processed"]
    # For now, use all data as training set since we don't have splits
    train_dataset = BdSLW60ProcessedDataset(processed_root)
    val_dataset = BdSLW60ProcessedDataset(processed_root)  # Same as train for now

    # Infer input feature dimension (T, D) from first item for sequence-based adapters
    try:
        sample_item = train_dataset[0]
        feature_dim = int(sample_item["input"].shape[1])
    except Exception:
        feature_dim = None

    # Create model (handle special adapters that need feature_dim)
    model_cfg = model_config["model"]
    if model_cfg.get("name", "").lower() == "videomae_from_codes":
        if feature_dim is None:
            raise RuntimeError("Could not infer input feature dimension from dataset for videomae_from_codes.")
        # Build videomae backbone kwargs from model config
        videomae_keys = [
            "patch_size","num_frames","tubelet_size","hidden_size","num_hidden_layers",
            "num_attention_heads","intermediate_size","hidden_act","hidden_dropout_prob",
            "attention_probs_dropout_prob","initializer_range","layer_norm_eps","input_channels",
            "image_size","num_classes","classifier_dropout"
        ]
        videomae_kwargs = {k: v for k, v in model_cfg.items() if k in videomae_keys}
        # Adapter params (optional)
        adapter_params = {k: v for k, v in model_cfg.items() if k.startswith("adapter_")}
        # Assemble final params for factory
        model_params = {
            "input_feature_dim": feature_dim,
            "videomae_kwargs": videomae_kwargs,
            **adapter_params,
        }
        # Replace model section to pass only necessary keys
        model_config["model"] = {"name": "videomae_from_codes", **model_params}
    
    model = create_model(model_config["model"])
    model = model.to(device)
    
    # Log model info
    log_model_info(logger, model, model_config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["training"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_collate_torch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=pad_collate_torch
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, model_config["training"])
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup TensorBoard
    tb_writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    ensure_dir(checkpoint_dir)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Train model
    training_results = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=model_config["training"]["num_epochs"],
        logger=logger,
        save_dir=checkpoint_dir
    )
    
    # Log final results
    logger.info(f"Training completed. Best validation accuracy: {training_results['best_val_accuracy']:.4f}")
    
    # Close TensorBoard writer
    tb_writer.close()


if __name__ == "__main__":
    main()
