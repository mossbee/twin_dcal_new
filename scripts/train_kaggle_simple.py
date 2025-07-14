#!/usr/bin/env python3
"""
Simplified training script for Kaggle environment.
"""

import sys
import os
from pathlib import Path
import json

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / 'src'

# Add src to Python path
sys.path.insert(0, str(src_dir))

# Set up minimal environment
os.chdir(str(project_root))

def main():
    """Run training with hardcoded Kaggle settings."""
    import torch
    from utils.config import load_config
    from utils.tracking import create_tracker
    from data.dataset import create_data_loaders
    from data.transforms import TwinFaceTransforms
    from models.dcal_model import create_dcal_model
    from training.trainer import Trainer
    
    print("🚀 Starting Twin DCAL training on Kaggle...")
    
    # Load base config
    config = load_config('configs/base_config.yaml')
    
    # Kaggle-specific overrides
    config.set('data.dataset_root', '/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224')
    config.set('data.batch_size', 16)
    config.set('data.num_workers', 2)
    config.set('training.epochs', 50)
    config.set('training.save_every', 1)
    config.set('tracking.method', 'wandb')
    config.set('tracking.project_name', 'twin_dcal_kaggle')
    config.set('tracking.entity', 'hunchoquavodb-hanoi-university-of-science-and-technology')
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create transforms
    transforms = TwinFaceTransforms(
        image_size=config.get('model.image_size', 224),
        augment=True,
        **config.get('data.augmentation', {})
    )
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        config.to_dict(),
        train_transform=transforms.train_transform,
        val_transform=transforms.val_transform
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("🏗️ Creating DCAL model...")
    model = create_dcal_model(config.to_dict())
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Create experiment tracker
    tracker = create_tracker(
        method=config.get('tracking.method', 'none'),
        project_name=config.get('tracking.project_name', 'twin_dcal'),
        experiment_name='kaggle_training'
    )
    
    # Initialize experiment
    tracker.init_experiment(
        project_name=config.get('tracking.project_name', 'twin_dcal'),
        experiment_name='kaggle_training',
        config=config.to_dict()
    )
    
    # Start training
    print("🎯 Starting training...")
    trainer.train(tracker, checkpoint_dir='outputs/checkpoints')
    
    print("✅ Training completed!")

if __name__ == '__main__':
    main()
