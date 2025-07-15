#!/usr/bin/env python3
"""
Test script to verify GLCA is working properly.
"""

import sys
import torch
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def test_glca():
    """Test GLCA functionality."""
    try:
        print("Testing GLCA functionality...")
        
        # Import required modules
        from models.dcal_model import create_dcal_model
        from utils.config import Config
        
        # Load config
        config = Config()
        config.load("configs/kaggle_config.yaml")
        
        # Ensure GLCA is enabled
        config.set('model.use_glca', True)
        config.set('model.glca_blocks', 1)  # Last layer only
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create model
        model = create_dcal_model(config)
        model = model.to(device)
        model.train()  # Set to training mode
        
        print(f"✓ Model created with GLCA enabled")
        print(f"GLCA layers: {model.glca_layers}")
        
        # Check which blocks have GLCA
        glca_blocks = []
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'use_glca') and block.use_glca:
                glca_blocks.append(i)
        print(f"Blocks with GLCA: {glca_blocks}")
        
        # Test forward pass
        batch_size = 2
        image_size = config.get('model.image_size', 224)
        
        # Create dummy input
        x = torch.randn(batch_size, 3, image_size, image_size).to(device)
        secondary_x = torch.randn(batch_size, 3, image_size, image_size).to(device)
        
        print(f"Testing forward pass with input shape: {x.shape}")
        
        # Forward pass with GLCA
        print("Testing SA + GLCA forward pass...")
        with torch.no_grad():
            output = model(x, return_features=True)
            
            print(f"✓ Forward pass successful")
            print(f"SA features shape: {output['sa_features'].shape}")
            
            if 'glca_features' in output and output['glca_features'] is not None:
                print(f"✓ GLCA features shape: {output['glca_features'].shape}")
                print("🟢 GLCA is working!")
                
                # Check if GLCA features are different from SA features
                sa_feat = output['sa_features']
                glca_feat = output['glca_features']
                
                diff = torch.norm(sa_feat - glca_feat)
                print(f"Feature difference (SA vs GLCA): {diff.item():.6f}")
                
                if diff.item() > 1e-6:
                    print("✓ GLCA produces different features than SA")
                else:
                    print("⚠️ GLCA and SA features are identical - check implementation")
                    
            else:
                print("❌ GLCA features not found in output")
                return False
        
        # Test with secondary input (PWCA)
        if config.get('model.use_pwca', False):
            print("Testing SA + GLCA + PWCA forward pass...")
            with torch.no_grad():
                output = model(x, secondary_x, return_features=True)
                print(f"✓ PWCA forward pass successful")
        
        print("✓ All GLCA tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ GLCA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_rollout():
    """Test attention rollout mechanism."""
    try:
        print("\nTesting attention rollout...")
        
        from models.utils import AttentionRollout
        
        # Create dummy attention matrices
        batch_size = 2
        num_heads = 3
        num_tokens = 197
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create realistic attention matrices
        attentions = []
        for layer in range(3):  # 3 layers
            # Create attention matrix [batch, heads, tokens, tokens]
            attn = torch.randn(batch_size, num_heads, num_tokens, num_tokens).to(device)
            attn = torch.softmax(attn, dim=-1)  # Normalize
            attentions.append(attn)
        
        # Test attention rollout
        rollout_module = AttentionRollout(local_query_ratio=0.3)
        selected_indices, rollout = rollout_module(attentions)
        
        print(f"✓ Attention rollout successful")
        print(f"Selected indices shape: {selected_indices.shape}")
        print(f"Rollout shape: {rollout.shape}")
        print(f"Number of selected patches: {selected_indices.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention rollout test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GLCA Functionality Test ===")
    
    glca_ok = test_glca()
    rollout_ok = test_attention_rollout()
    
    if glca_ok and rollout_ok:
        print("\n🟢 All tests passed! GLCA should be working in training.")
    else:
        print("\n🔴 Some tests failed. Check the implementation.")
    
    print("\nTo enable GLCA in training:")
    print("1. Make sure configs/kaggle_config.yaml has 'use_glca: true'")
    print("2. Use the updated model code")
    print("3. GLCA should now show non-zero loss in training")
