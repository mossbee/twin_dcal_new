#!/usr/bin/env python3
"""
Debug script to test configuration loading and type conversion.
"""

import yaml
import sys
from pathlib import Path

def test_config():
    """Test configuration loading."""
    config_path = Path("configs/kaggle_config.yaml")
    
    print(f"Testing config at: {config_path}")
    print(f"Config exists: {config_path.exists()}")
    
    if not config_path.exists():
        print("Config file not found!")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n=== Config Structure ===")
    print(f"Top-level keys: {list(config.keys())}")
    
    if 'training' in config:
        training = config['training']
        print(f"\nTraining keys: {list(training.keys())}")
        
        # Check learning rate
        if 'learning_rate' in training:
            lr = training['learning_rate']
            print(f"Learning rate: {lr} (type: {type(lr)})")
            try:
                lr_float = float(lr)
                print(f"Converted to float: {lr_float}")
            except Exception as e:
                print(f"Conversion failed: {e}")
        
        # Check weight decay
        if 'weight_decay' in training:
            wd = training['weight_decay']
            print(f"Weight decay: {wd} (type: {type(wd)})")
            try:
                wd_float = float(wd)
                print(f"Converted to float: {wd_float}")
            except Exception as e:
                print(f"Conversion failed: {e}")
    
    print("\n=== Full Config ===")
    print(yaml.dump(config, default_flow_style=False))

if __name__ == "__main__":
    test_config()
