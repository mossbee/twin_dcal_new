#!/usr/bin/env python3
"""
Fixed Kaggle training script with robust error handling.
"""

import os
import sys
from pathlib import Path

# Set up working directory and paths
try:
    os.chdir('/kaggle/working/twin_dcal_new')
    project_root = Path('/kaggle/working/twin_dcal_new')
    src_dir = project_root / 'src'
    
    # Add to Python path
    for path in [str(src_dir), str(project_root)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print("✓ Paths configured successfully")
except Exception as e:
    print(f"❌ Error setting up paths: {e}")
    sys.exit(1)

import torch
print(f"Using PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

def main():
    print("🚀 Starting Twin DCAL training on Kaggle...")
    
    # Import modules
    try:
        from utils.config import load_config
        from utils.tracking import create_tracker
        from data.dataset import create_data_loaders
        from data.transforms import TwinFaceTransforms
        from models.dcal_model import create_dcal_model
        from training.trainer import Trainer
        print("✓ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load configuration
    try:
        config = load_config('configs/kaggle_config.yaml')
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Override with Kaggle-specific settings
    config.set('data.dataset_root', '/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224')
    config.set('data.batch_size', 16)
    config.set('data.num_workers', 2)
    config.set('training.epochs', 10)  # Start with fewer epochs for testing
    config.set('tracking.method', 'wandb')
    config.set('tracking.project_name', 'twin_dcal_kaggle')
    config.set('tracking.entity', 'hunchoquavodb-hanoi-university-of-science-and-technology')
    
    print("Configuration:")
    print(f"- Model: {config.get('model.name', 'Unknown')}")
    print(f"- Image size: {config.get('model.image_size', 'Unknown')}")
    print(f"- Batch size: {config.get('data.batch_size', 'Unknown')}")
    print(f"- Learning rate: {config.get('training.learning_rate', 'Unknown')}")
    print(f"- Epochs: {config.get('training.epochs', 'Unknown')}")
    print(f"- Tracking: {config.get('tracking.method', 'Unknown')}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Create transforms
    print("Creating data transforms...")
    try:
        train_transform = TwinFaceTransforms.get_train_transforms(config.to_dict())
        val_transform = TwinFaceTransforms.get_val_transforms(config.to_dict())
        print("✓ Transforms created successfully")
    except Exception as e:
        print(f"❌ Error creating transforms: {e}")
        return
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            config.to_dict(), train_transform, val_transform
        )
        print(f"✓ Data loaders created successfully")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        # Debug dataset path
        dataset_root = config.get('data.dataset_root')
        print(f"Dataset root: {dataset_root}")
        print(f"Dataset root exists: {os.path.exists(dataset_root)}")
        if os.path.exists(dataset_root):
            print(f"Contents: {os.listdir(dataset_root)}")
        
        # Check for JSON files
        for json_file in ['data/train_dataset_infor.json', 'data/train_twin_pairs.json']:
            full_path = os.path.join(project_root, json_file)
            print(f"{json_file} exists: {os.path.exists(full_path)}")
        
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    print("Creating DCAL model...")
    try:
        model = create_dcal_model(config.to_dict())
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created successfully")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - GLCA enabled: {config.get('model.use_glca', False)}")
        print(f"  - PWCA enabled: {config.get('model.use_pwca', False)}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create trainer
    print("Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        print("✓ Trainer created successfully")
    except Exception as e:
        print(f"❌ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create experiment tracker
    print("Setting up experiment tracking...")
    try:
        tracker = create_tracker(
            method=config.get('tracking.method', 'none'),
            project_name=config.get('tracking.project_name', 'twin_dcal'),
            experiment_name=config.get('tracking.experiment_name', 'kaggle_run')
        )
        
        # Initialize experiment
        tracker.init_experiment(
            project_name=config.get('tracking.project_name', 'twin_dcal'),
            experiment_name=config.get('tracking.experiment_name', 'kaggle_run'),
            config=config.to_dict()
        )
        print("✓ Experiment tracking initialized")
    except Exception as e:
        print(f"⚠️ Warning: Error setting up tracking: {e}")
        # Continue without tracking
        try:
            from utils.tracking import NoTracker
            tracker = NoTracker()
            print("Continuing without experiment tracking...")
        except:
            print("Creating dummy tracker...")
            class DummyTracker:
                def init_experiment(self, *args, **kwargs): pass
                def log_metrics(self, *args, **kwargs): pass
                def log_artifact(self, *args, **kwargs): pass
                def finish(self): pass
            tracker = DummyTracker()
    
    # Start training
    print("Starting training...")
    try:
        trainer.train(tracker, checkpoint_dir='checkpoints_kaggle')
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            tracker.finish()
        except:
            pass

if __name__ == '__main__':
    main()
