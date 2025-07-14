#!/usr/bin/env python3
"""
Enhanced Kaggle training script with additional debug output and error handling.
"""

import sys
import os
import yaml
import traceback
from pathlib import Path

def setup_environment():
    """Setup environment for Kaggle execution."""
    # Get the current working directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Add src to Python path for imports
    src_path = current_dir / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        print(f"Added to Python path: {src_path}")
    else:
        print(f"Warning: src directory not found at {src_path}")
    
    return current_dir

def load_and_validate_config(config_path):
    """Load config with validation and type conversion."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✓ Config loaded successfully")
        
        # Validate and convert critical numeric values
        if 'training' in config:
            training = config['training']
            
            # Convert learning rate to float if it's a string
            if 'learning_rate' in training:
                lr = training['learning_rate']
                print(f"Learning rate raw value: {lr} (type: {type(lr)})")
                training['learning_rate'] = float(lr)
                print(f"Learning rate after conversion: {training['learning_rate']} (type: {type(training['learning_rate'])})")
            
            # Convert weight decay to float if it's a string
            if 'weight_decay' in training:
                wd = training['weight_decay']
                print(f"Weight decay raw value: {wd} (type: {type(wd)})")
                training['weight_decay'] = float(wd)
                print(f"Weight decay after conversion: {training['weight_decay']} (type: {type(training['weight_decay'])})")
        
        return config
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        traceback.print_exc()
        raise

def main():
    """Main training function with enhanced error handling."""
    try:
        print("=== Enhanced Kaggle Training Script ===")
        
        # Setup environment
        current_dir = setup_environment()
        
        # Load and validate config
        config_path = current_dir / "configs" / "kaggle_config.yaml"
        print(f"Loading config from: {config_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = load_and_validate_config(config_path)
        
        # Now try to import and run training
        print("Importing training modules...")
        
        try:
            from main import main as train_main
            print("✓ Successfully imported main")
        except ImportError as e:
            print(f"✗ Failed to import main: {e}")
            traceback.print_exc()
            
            # Try alternative import
            try:
                sys.path.append(str(current_dir))
                from src.main import main as train_main
                print("✓ Successfully imported main (alternative path)")
            except ImportError as e2:
                print(f"✗ Failed alternative import: {e2}")
                raise
        
        # Run training
        print("Starting training...")
        train_main()
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
