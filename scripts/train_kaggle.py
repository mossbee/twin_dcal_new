#!/usr/bin/env python3
"""
Training script for Kaggle environment.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main
import sys

if __name__ == '__main__':
    # Set Kaggle config arguments
    sys.argv.extend(['--kaggle'])
    
    # Override config for Kaggle
    sys.argv.extend([
        '--config-overrides',
        'data.dataset_root=/kaggle/input/nd-twin/ND_TWIN_Dataset_224/ND_TWIN_Dataset_224',
        'data.batch_size=16',
        'data.num_workers=2',
        'training.epochs=50',  # Reduced for 12h limit
        'training.save_every=1',
        'tracking.method=wandb',
        'tracking.project_name=twin_dcal_kaggle',
        'tracking.entity=hunchoquavodb-hanoi-university-of-science-and-technology'
    ])
    
    main()
