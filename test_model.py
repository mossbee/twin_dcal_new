#!/usr/bin/env python3
"""
Test script to verify model creation and basic forward pass.
"""

import sys
import torch
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_model():
    """Test model creation and forward pass."""
    try:
        print("Testing model creation...")
        
        # Import model
        from models.dcal_model import create_dcal_model
        from utils.config import Config
        
        # Load test config
        config = Config()
        config.load("configs/test_kaggle_config.yaml")
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = create_dcal_model(config)
        model = model.to(device)
        
        print(f"✓ Model created successfully")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        image_size = config.get('model.image_size', 224)
        
        # Create dummy input
        x = torch.randn(batch_size, 3, image_size, image_size).to(device)
        secondary_x = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        print(f"Testing forward pass with input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            # Test SA branch only
            output = model(x, return_features=True)
            print(f"✓ SA forward pass successful")
            print(f"SA features shape: {output['sa_features'].shape}")
            
            # Test with secondary input (PWCA)
            if config.get('model.use_pwca', False):
                output = model(x, secondary_x, return_features=True)
                print(f"✓ PWCA forward pass successful")
            
            # Test embedding extraction
            embedding = model.get_embedding(x)
            print(f"✓ Embedding extraction successful")
            print(f"Embedding shape: {embedding.shape}")
        
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_model()
