#!/usr/bin/env python3
"""
Quick evaluation script to test current model performance.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def evaluate_model(checkpoint_path):
    """Evaluate model performance on test set."""
    try:
        from models.dcal_model import create_dcal_model
        from data.dataset import create_data_loaders
        from utils.config import Config
        
        # Load config and model
        config = Config()
        config.load("configs/kaggle_config.yaml")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and load checkpoint
        model = create_dcal_model(config)
        
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print("⚠️ No checkpoint found, using randomly initialized model")
        
        model = model.to(device)
        model.eval()
        
        # Load data
        _, _, test_loader = create_data_loaders(config)
        
        # Extract embeddings
        embeddings = []
        labels = []
        
        print("Extracting embeddings...")
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                batch_labels = batch['label']
                
                # Get embeddings
                emb = model.get_embedding(images)
                embeddings.append(emb.cpu())
                labels.append(batch_labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print(f"✓ Extracted {len(embeddings)} embeddings")
        
        # Compute pairwise similarities for verification
        print("Computing verification metrics...")
        
        # Create genuine and impostor pairs
        genuine_scores = []
        impostor_scores = []
        
        # Simple approach: compare each embedding with others
        for i in range(0, min(1000, len(embeddings)), 10):  # Sample to avoid memory issues
            for j in range(i+1, min(i+100, len(embeddings))):
                # Cosine similarity
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0), 
                    embeddings[j].unsqueeze(0)
                ).item()
                
                if labels[i] == labels[j]:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        print(f"Genuine pairs: {len(genuine_scores)}")
        print(f"Impostor pairs: {len(impostor_scores)}")
        
        # Compute metrics
        print("\n=== Performance Metrics ===")
        print(f"Genuine score mean: {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
        print(f"Impostor score mean: {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")
        print(f"Score separation: {genuine_scores.mean() - impostor_scores.mean():.4f}")
        
        # Simple EER estimation
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        all_labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find EER
        eer_idx = np.argmin(np.abs(fpr + tpr - 1))
        eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
        
        print(f"AUC: {roc_auc:.4f}")
        print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
        
        # Assessment
        print("\n=== Assessment ===")
        if eer < 0.05:
            print("🟢 EXCELLENT performance (EER < 5%)")
        elif eer < 0.10:
            print("🟡 GOOD performance (EER < 10%)")
        elif eer < 0.20:
            print("🟠 FAIR performance (EER < 20%)")
        else:
            print("🔴 POOR performance (EER > 20%)")
        
        if genuine_scores.mean() - impostor_scores.mean() > 0.3:
            print("✓ Good score separation")
        else:
            print("⚠️ Poor score separation - consider more training")
        
        return eer, roc_auc
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Look for latest checkpoint
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Using checkpoint: {latest_checkpoint}")
            evaluate_model(latest_checkpoint)
        else:
            print("No checkpoints found, evaluating random model")
            evaluate_model(Path("nonexistent.pth"))
    else:
        print("No checkpoint directory found")
        evaluate_model(Path("nonexistent.pth"))
