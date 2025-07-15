#!/usr/bin/env python3
"""
Simple evaluation script for current model using basic PyTorch operations.
"""

import sys
import os
import torch
import json
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def simple_eer(genuine_scores, impostor_scores):
    """Simple EER computation using PyTorch."""
    # Combine scores and labels
    all_scores = torch.cat([genuine_scores, impostor_scores])
    all_labels = torch.cat([torch.ones(len(genuine_scores)), torch.zeros(len(impostor_scores))])
    
    # Sort by score
    sorted_indices = torch.argsort(all_scores, descending=True)
    sorted_labels = all_labels[sorted_indices]
    
    # Compute FPR and TPR for different thresholds
    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)
    
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)
    
    tpr = tp.float() / n_genuine
    fpr = fp.float() / n_impostor
    
    # Find EER (where FPR ≈ 1-TPR)
    diff = torch.abs(fpr - (1 - tpr))
    eer_idx = torch.argmin(diff)
    eer = (fpr[eer_idx] + (1 - tpr[eer_idx])) / 2
    
    return eer.item()

def evaluate_simple():
    """Simple evaluation with minimal dependencies."""
    try:
        print("=== Simple Model Evaluation ===")
        
        # Import modules
        from models.dcal_model import create_dcal_model
        from data.dataset import create_data_loaders
        from utils.config import Config
        
        # Load config
        config = Config()
        config.load("configs/kaggle_config.yaml")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Find checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
        
        if not checkpoints:
            print("❌ No checkpoints found!")
            return None
        
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Using: {latest_checkpoint.name}")
        
        # Load model
        model = create_dcal_model(config)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded from epoch {epoch}")
        except Exception as e:
            print(f"Warning: {e}")
            epoch = "unknown"
        
        model = model.to(device)
        model.eval()
        
        # Load test data
        print("Loading test data...")
        _, _, test_loader = create_data_loaders(config)
        
        # Extract embeddings
        print("Extracting embeddings...")
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(test_loader)}")
                
                images = batch['image'].to(device)
                batch_labels = batch['label']
                
                # Get embeddings
                emb = model.get_embedding(images)
                embeddings.append(emb.cpu())
                labels.append(batch_labels)
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print(f"Extracted {len(embeddings)} embeddings")
        print(f"Unique identities: {len(torch.unique(labels))}")
        
        # Create verification pairs
        print("Creating verification pairs...")
        genuine_scores = []
        impostor_scores = []
        
        unique_labels = torch.unique(labels)
        max_pairs_per_identity = 20
        max_total_pairs = 5000
        
        pair_count = 0
        
        # Genuine pairs
        for label in unique_labels:
            if pair_count >= max_total_pairs // 2:
                break
                
            indices = torch.where(labels == label)[0]
            if len(indices) < 2:
                continue
            
            # Sample pairs within this identity
            for i in range(min(len(indices), max_pairs_per_identity)):
                if pair_count >= max_total_pairs // 2:
                    break
                for j in range(i+1, min(i+5, len(indices))):
                    emb1 = embeddings[indices[i]]
                    emb2 = embeddings[indices[j]]
                    
                    # Cosine similarity
                    sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    genuine_scores.append(sim)
                    pair_count += 1
        
        print(f"Generated {len(genuine_scores)} genuine pairs")
        
        # Impostor pairs
        pair_count = 0
        for i in range(0, min(len(embeddings), 500), 5):  # Sample every 5th
            if pair_count >= max_total_pairs // 2:
                break
            for j in range(i+1, min(i+20, len(embeddings))):
                if labels[i] != labels[j] and pair_count < max_total_pairs // 2:
                    emb1 = embeddings[i]
                    emb2 = embeddings[j]
                    
                    sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
                    impostor_scores.append(sim)
                    pair_count += 1
        
        print(f"Generated {len(impostor_scores)} impostor pairs")
        
        # Convert to tensors
        genuine_scores = torch.stack(genuine_scores)
        impostor_scores = torch.stack(impostor_scores)
        
        # Compute metrics
        print("\n=== Results ===")
        print(f"Genuine scores: {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f}")
        print(f"Impostor scores: {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f}")
        print(f"Score separation: {genuine_scores.mean() - impostor_scores.mean():.4f}")
        
        # Compute EER
        eer = simple_eer(genuine_scores, impostor_scores)
        print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
        
        # Assessment
        print("\n=== Assessment ===")
        if eer < 0.05:
            assessment = "🟢 EXCELLENT (EER < 5%)"
        elif eer < 0.10:
            assessment = "🟡 GOOD (EER < 10%)"
        elif eer < 0.20:
            assessment = "🟠 FAIR (EER < 20%)"
        else:
            assessment = "🔴 POOR (EER > 20%)"
        
        print(f"Performance: {assessment}")
        
        if genuine_scores.mean() - impostor_scores.mean() > 0.3:
            print("✅ Good score separation")
        else:
            print("⚠️ Poor score separation")
        
        # Save results
        results = {
            'epoch': str(epoch),
            'eer': float(eer),
            'genuine_mean': float(genuine_scores.mean()),
            'genuine_std': float(genuine_scores.std()),
            'impostor_mean': float(impostor_scores.mean()),
            'impostor_std': float(impostor_scores.std()),
            'score_separation': float(genuine_scores.mean() - impostor_scores.mean()),
            'num_genuine_pairs': len(genuine_scores),
            'num_impostor_pairs': len(impostor_scores),
            'assessment': assessment
        }
        
        with open(f'simple_eval_epoch_{epoch}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to simple_eval_epoch_{epoch}.json")
        
        # Recommendations
        print("\n=== Recommendations ===")
        if eer < 0.1:
            print("✅ Current model performs well!")
            print("💡 You can:")
            print("   - Use this model as-is")
            print("   - Try enabling GLCA for potential improvement")
            print("   - Continue training with lower learning rate")
        else:
            print("📈 Model needs improvement:")
            print("   - Enable GLCA (main DCAL feature)")
            print("   - Continue training with current setup")
            print("   - Try different hyperparameters")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = evaluate_simple()
    
    if results:
        print(f"\n🎯 Final EER: {results['eer']:.4f} ({results['eer']*100:.2f}%)")
    else:
        print("\n❌ Evaluation failed")
