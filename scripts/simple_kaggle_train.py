#!/usr/bin/env python3
"""
Simplified Kaggle training script that disables GLCA temporarily to test the pipeline.
"""

import sys
import os
import yaml
from pathlib import Path

def setup_environment():
    """Setup environment for Kaggle execution."""
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Add src to Python path for imports
    src_path = current_dir / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
        print(f"Added to Python path: {src_path}")
    
    return current_dir

def main():
    """Main training function with GLCA disabled for testing."""
    try:
        print("=== Simplified Kaggle Training Script (GLCA Disabled) ===")
        
        # Setup environment
        current_dir = setup_environment()
        
        # Load config and disable GLCA temporarily
        config_path = current_dir / "configs" / "kaggle_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Temporarily disable GLCA to test the pipeline
        config['model']['use_glca'] = False
        print("✓ GLCA disabled for testing")
        
        # Convert numeric values to ensure they're floats
        if 'training' in config:
            training = config['training']
            if 'learning_rate' in training:
                training['learning_rate'] = float(training['learning_rate'])
            if 'weight_decay' in training:
                training['weight_decay'] = float(training['weight_decay'])
        
        # Save temporary config
        temp_config_path = current_dir / "configs" / "temp_kaggle_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Set environment variable to use temp config
        os.environ['DCAL_CONFIG_PATH'] = str(temp_config_path)
        
        # Import and run training
        from main import main as train_main
        train_main()
        
    except Exception as e:
        print(f"✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
