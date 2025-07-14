#!/usr/bin/env python3
"""
Training script for Kaggle environment.
"""

import sys
import os
from pathlib import Path

# Ensure we can import from src
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / 'src'

# Add paths to Python path
for path in [str(src_dir), str(project_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import after setting up paths
import torch

# Now import and run main
from main import main

if __name__ == '__main__':
    # Set up sys.argv for the main function
    sys.argv = ['main.py'] + [
        '--config', 'configs/kaggle_config.yaml',
        '--kaggle',
        '--config-overrides',
        'data.dataset_root=/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224',
        'data.batch_size=16',
        'data.num_workers=2',
        'training.epochs=50',
        'training.save_every=1',
        'tracking.method=wandb',
        'tracking.project_name=twin_dcal_kaggle',
        'tracking.entity=hunchoquavodb-hanoi-university-of-science-and-technology'
    ]
    
    main()
