#!/usr/bin/env python3
"""
Main entry point for Twin DCAL training.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir
sys.path.insert(0, str(src_dir))

# Handle both relative and absolute imports
try:
    from utils.config import Config, load_config
    from utils.tracking import create_tracker
    from data.dataset import create_data_loaders
    from data.transforms import TwinFaceTransforms
    from models.backbone import deit_tiny_patch16_224
    from models.dcal_model import create_dcal_model
    from training.trainer import Trainer
except ImportError as e:
    print(f"Import error: {e}")
    # Try alternative import paths
    sys.path.insert(0, str(current_dir.parent))
    from src.utils.config import Config, load_config
    from src.utils.tracking import create_tracker
    from src.data.dataset import create_data_loaders
    from src.data.transforms import TwinFaceTransforms
    from src.models.backbone import deit_tiny_patch16_224
    from src.models.dcal_model import create_dcal_model
    from src.training.trainer import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Twin DCAL model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for checkpoints and logs'
    )
    
    parser.add_argument(
        '--config-overrides',
        nargs='*',
        help='Config overrides in format key=value'
    )
    
    # Environment specific overrides
    parser.add_argument('--local', action='store_true', help='Use local server config')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle config')
    parser.add_argument('--no-tracking', action='store_true', help='Disable experiment tracking')
    
    return parser.parse_args()


def setup_device(config: Config):
    """Setup device and distributed training."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def create_model(config: Config):
    """Create model from config."""
    model_name = config.get('model.backbone', 'deit_tiny_patch16_224')
    use_dcal = config.get('model.use_glca', False) or config.get('model.use_pwca', False)
    
    if use_dcal:
        # Use DCAL model with cross-attention mechanisms
        model = create_dcal_model(config.to_dict())
        print(f"Created DCAL model: {model_name}")
        print(f"- GLCA enabled: {config.get('model.use_glca', False)}")
        print(f"- PWCA enabled: {config.get('model.use_pwca', False)}")
    else:
        # Use standard ViT backbone
        num_classes = config.get('model.num_classes', 356)
        
        if model_name == 'deit_tiny_patch16_224':
            model = deit_tiny_patch16_224(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Created standard model: {model_name}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Apply environment-specific configs
    if args.local:
        local_config = load_config('configs/local_config.yaml')
        config.config.update(local_config.config)
    elif args.kaggle:
        kaggle_config = load_config('configs/kaggle_config.yaml')
        config.config.update(kaggle_config.config)
    
    # Apply command line overrides
    config.update_from_args(args)
    
    # Override tracking if specified
    if args.no_tracking:
        config.set('tracking.method', 'none')
    
    print("Configuration:")
    print(f"- Model: {config.get('model.backbone')}")
    print(f"- Image size: {config.get('model.image_size')}")
    print(f"- Batch size: {config.get('data.batch_size')}")
    print(f"- Learning rate: {config.get('training.learning_rate')}")
    print(f"- Epochs: {config.get('training.epochs')}")
    print(f"- Tracking: {config.get('tracking.method')}")
    
    # Setup device
    device = setup_device(config)
    
    # Create transforms
    train_transform = TwinFaceTransforms.get_train_transforms(config.to_dict())
    val_transform = TwinFaceTransforms.get_val_transforms(config.to_dict())
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config.to_dict(),
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Setup experiment tracking
    tracker = create_tracker(config.to_dict())
    tracker.init_experiment(
        project_name=config.get('tracking.project_name', 'twin_dcal'),
        experiment_name=config.get('tracking.experiment_name', 'baseline'),
        config=config.to_dict()
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    
    # Save config
    config.save(os.path.join(args.output_dir, 'config.yaml'))
    
    try:
        # Start training
        print("Starting training...")
        trainer.train(tracker, checkpoint_dir=checkpoint_dir)
        print("Training completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
        
    finally:
        # Finish tracking
        tracker.finish()


if __name__ == '__main__':
    main()
