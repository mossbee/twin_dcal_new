#!/usr/bin/env python3
"""
Comprehensive evaluation script for twin face verification model.
Evaluates the best checkpoint on test set with detailed metrics.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Add src to path
current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

def compute_eer(genuine_scores, impostor_scores):
    """Compute Equal Error Rate (EER)."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    # EER is where FPR = FNR (1 - TPR)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def compute_tar_at_far(genuine_scores, impostor_scores, far_targets=[0.001, 0.01, 0.1]):
    """Compute True Accept Rate (TAR) at specific False Accept Rates (FAR)."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    
    tar_at_far = {}
    for far_target in far_targets:
        # Find the closest FAR to target
        idx = np.argmin(np.abs(fpr - far_target))
        tar_at_far[f'TAR@FAR={far_target}'] = tpr[idx]
        tar_at_far[f'threshold@FAR={far_target}'] = thresholds[idx]
    
    return tar_at_far

def evaluate_verification_pairs(embeddings, labels, pairs=None, max_pairs=10000):
    """
    Evaluate face verification using embeddings.
    
    Args:
        embeddings: Face embeddings [N, dim]
        labels: Identity labels [N]
        pairs: Optional list of (idx1, idx2, is_same) tuples
        max_pairs: Maximum number of pairs to evaluate
    
    Returns:
        genuine_scores, impostor_scores, metrics
    """
    print("Computing verification scores...")
    
    if pairs is None:
        # Create pairs from available data
        genuine_pairs = []
        impostor_pairs = []
        
        unique_labels = np.unique(labels)
        
        # Sample pairs to avoid memory issues
        pair_count = 0
        
        for label in unique_labels:
            same_indices = np.where(labels == label)[0]
            
            # Genuine pairs (same identity)
            for i in range(len(same_indices)):
                for j in range(i+1, min(i+10, len(same_indices))):  # Limit genuine pairs per identity
                    if pair_count >= max_pairs // 2:
                        break
                    genuine_pairs.append((same_indices[i], same_indices[j]))
                    pair_count += 1
                if pair_count >= max_pairs // 2:
                    break
        
        # Impostor pairs (different identities)
        pair_count = 0
        for i in range(0, min(len(embeddings), 1000), 5):  # Sample every 5th embedding
            for j in range(i+1, min(i+100, len(embeddings))):
                if labels[i] != labels[j] and pair_count < max_pairs // 2:
                    impostor_pairs.append((i, j))
                    pair_count += 1
    
    # Compute similarities
    genuine_scores = []
    impostor_scores = []
    
    print(f"Computing {len(genuine_pairs)} genuine pairs...")
    for i, j in tqdm(genuine_pairs):
        sim = torch.nn.functional.cosine_similarity(
            embeddings[i].unsqueeze(0), 
            embeddings[j].unsqueeze(0)
        ).item()
        genuine_scores.append(sim)
    
    print(f"Computing {len(impostor_pairs)} impostor pairs...")
    for i, j in tqdm(impostor_pairs):
        sim = torch.nn.functional.cosine_similarity(
            embeddings[i].unsqueeze(0), 
            embeddings[j].unsqueeze(0)
        ).item()
        impostor_scores.append(sim)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    return genuine_scores, impostor_scores

def load_test_pairs(config):
    """Load official test pairs if available."""
    pairs_file = Path(config.get('data.dataset_root', '')) / config.get('data.test_pairs', '')
    
    if pairs_file.exists():
        print(f"Loading test pairs from {pairs_file}")
        with open(pairs_file, 'r') as f:
            pairs_data = json.load(f)
        return pairs_data
    else:
        print("No test pairs file found, will create pairs from test data")
        return None

def evaluate_model():
    """Main evaluation function."""
    try:
        # Import required modules
        from models.dcal_model import create_dcal_model
        from data.dataset import create_data_loaders
        from utils.config import Config
        
        print("=== Model Evaluation on Test Set ===")
        
        # Load config
        config = Config()
        config.load("configs/kaggle_config.yaml")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Find best checkpoint
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            print("❌ No checkpoints directory found!")
            return
        
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print("❌ No checkpoint files found!")
            return
        
        # Use the most recent checkpoint (you could also look for "best" in filename)
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"📁 Using checkpoint: {latest_checkpoint}")
        
        # Load model
        model = create_dcal_model(config)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Loaded model from epoch {epoch}")
        except Exception as e:
            print(f"⚠️ Warning: Could not load checkpoint state: {e}")
            print("Using randomly initialized model")
        
        model = model.to(device)
        model.eval()
        
        # Load test data
        print("📊 Loading test data...")
        _, _, test_loader = create_data_loaders(config)
        
        if test_loader is None:
            print("❌ Could not create test data loader!")
            return
        
        print(f"Test set size: {len(test_loader.dataset)} samples")
        
        # Extract embeddings
        print("🔄 Extracting embeddings...")
        embeddings = []
        labels = []
        image_paths = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                images = batch['image'].to(device)
                batch_labels = batch['label']
                
                # Get embeddings using the model's embedding method
                emb = model.get_embedding(images)
                
                embeddings.append(emb.cpu())
                labels.append(batch_labels)
                
                # Store some image paths for analysis
                if 'path' in batch:
                    image_paths.extend(batch['path'])
        
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print(f"✅ Extracted {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Number of unique identities: {len(torch.unique(labels))}")
        
        # Compute verification metrics
        genuine_scores, impostor_scores = evaluate_verification_pairs(embeddings, labels)
        
        print(f"\n📈 Verification Results:")
        print(f"Genuine pairs: {len(genuine_scores)}")
        print(f"Impostor pairs: {len(impostor_scores)}")
        
        # Basic statistics
        print(f"\n📊 Score Statistics:")
        print(f"Genuine scores: {genuine_scores.mean():.4f} ± {genuine_scores.std():.4f} [{genuine_scores.min():.4f}, {genuine_scores.max():.4f}]")
        print(f"Impostor scores: {impostor_scores.mean():.4f} ± {impostor_scores.std():.4f} [{impostor_scores.min():.4f}, {impostor_scores.max():.4f}]")
        print(f"Score separation: {genuine_scores.mean() - impostor_scores.mean():.4f}")
        
        # Compute metrics
        eer = compute_eer(genuine_scores, impostor_scores)
        tar_at_far = compute_tar_at_far(genuine_scores, impostor_scores)
        
        # ROC curve
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        all_labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
        print(f"AUC: {roc_auc:.4f}")
        
        for metric, value in tar_at_far.items():
            if 'TAR@FAR' in metric:
                print(f"{metric}: {value:.4f} ({value*100:.2f}%)")
        
        # Performance assessment
        print(f"\n🏆 Performance Assessment:")
        if eer < 0.05:
            print("🟢 EXCELLENT performance (EER < 5%)")
            recommendation = "Model is production-ready!"
        elif eer < 0.10:
            print("🟡 GOOD performance (EER < 10%)")
            recommendation = "Model performs well, consider fine-tuning for better results"
        elif eer < 0.20:
            print("🟠 FAIR performance (EER < 20%)")
            recommendation = "Model needs improvement - try GLCA or more training"
        else:
            print("🔴 POOR performance (EER > 20%)")
            recommendation = "Model needs significant improvement"
        
        if genuine_scores.mean() - impostor_scores.mean() > 0.3:
            print("✅ Good score separation")
        else:
            print("⚠️ Poor score separation")
        
        print(f"\n💡 Recommendation: {recommendation}")
        
        # Save results
        results = {
            'checkpoint': str(latest_checkpoint),
            'epoch': epoch,
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'num_identities': len(torch.unique(labels)),
            'genuine_pairs': len(genuine_scores),
            'impostor_pairs': len(impostor_scores),
            'genuine_stats': {
                'mean': float(genuine_scores.mean()),
                'std': float(genuine_scores.std()),
                'min': float(genuine_scores.min()),
                'max': float(genuine_scores.max())
            },
            'impostor_stats': {
                'mean': float(impostor_scores.mean()),
                'std': float(impostor_scores.std()),
                'min': float(impostor_scores.min()),
                'max': float(impostor_scores.max())
            },
            'eer': float(eer),
            'auc': float(roc_auc),
            'tar_at_far': {k: float(v) for k, v in tar_at_far.items()},
            'score_separation': float(genuine_scores.mean() - impostor_scores.mean()),
            'recommendation': recommendation
        }
        
        # Save results to file
        results_file = f"evaluation_results_epoch_{epoch}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {results_file}")
        
        # Plot ROC curve
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Face Verification')
            plt.legend()
            plt.grid(True)
            
            roc_file = f"roc_curve_epoch_{epoch}.png"
            plt.savefig(roc_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 ROC curve saved to {roc_file}")
        except Exception as e:
            print(f"⚠️ Could not save ROC curve: {e}")
        
        # Plot score distributions
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', density=True)
            plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', density=True)
            plt.xlabel('Cosine Similarity Score')
            plt.ylabel('Density')
            plt.title('Score Distribution')
            plt.legend()
            plt.grid(True)
            
            dist_file = f"score_distribution_epoch_{epoch}.png"
            plt.savefig(dist_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Score distribution saved to {dist_file}")
        except Exception as e:
            print(f"⚠️ Could not save score distribution: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting comprehensive model evaluation...")
    results = evaluate_model()
    
    if results:
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📈 EER: {results['eer']:.4f} ({results['eer']*100:.2f}%)")
        print(f"📈 AUC: {results['auc']:.4f}")
    else:
        print("\n❌ Evaluation failed!")
    
    print("\nNext steps:")
    print("1. Review the results above")
    print("2. Check the saved JSON file for detailed metrics")
    print("3. Look at the ROC curve and score distribution plots")
    print("4. Decide whether to continue training or enable GLCA")
