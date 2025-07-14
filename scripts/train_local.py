#!/usr/bin/env python3
"""
Training script for local Ubuntu server.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import main
import sys

if __name__ == '__main__':
    # Set local config arguments
    sys.argv.extend(['--local'])
    
    # Override config for local server
    sys.argv.extend([
        '--config-overrides',
        'data.dataset_root=/path/to/local/dataset',  # Update this path
        'data.batch_size=32',
        'data.num_workers=8',
        'training.epochs=100',
        'tracking.method=mlflow',
        'tracking.mlflow_uri=http://localhost:5000'
    ])
    
    main()
