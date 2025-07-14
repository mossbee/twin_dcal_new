#!/usr/bin/env python3
"""
Evaluation script for Twin DCAL model.
"""

import os
import sys
import argparse
import torch
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import Config, load_config
from data.dataset import TwinFaceDataset, create_data_loaders
from data.transforms import TwinFaceTransforms
from models.dcal_model import create_dcal_model
from models.backbone import deit_tiny_patch16_224
from training.metrics import VerificationEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Twin DCAL model')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    parser.add_argument(
        '--distance-metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='Distance metric for verification'
    )
    
    # Environment specific overrides
    parser.add_argument('--local', action='store_true', help='Use local server config')
    parser.add_argument('--kaggle', action='store_true', help='Use Kaggle config')
    
    return parser.parse_args()


def create_verification_pairs(dataset, twin_pairs):
    """
    Create verification pairs for evaluation.
    
    Args:
        dataset: TwinFaceDataset instance
        twin_pairs: List of twin pairs
    
    Returns:
        pairs: List of (img1_path, img2_path, is_same_person)
    """
    pairs = []
    
    # Create twin pairs mapping
    twin_map = {}
    for pair in twin_pairs:
        twin_map[pair[0]] = pair[1]
        twin_map[pair[1]] = pair[0]
    
    # Positive pairs (same person)
    for person_id, img_paths in dataset.dataset_info.items():
        if len(img_paths) >= 2:
            # Create pairs within same person
            for i in range(len(img_paths)):
                for j in range(i + 1, len(img_paths)):
                    pairs.append((img_paths[i], img_paths[j], 1))
    
    # Negative pairs (different persons, focus on twins)
    person_ids = list(dataset.dataset_info.keys())
    for person_id in person_ids:
        twin_id = twin_map.get(person_id)
        if twin_id and twin_id in dataset.dataset_info:
            # Create twin pairs (hardest negatives)
            person_imgs = dataset.dataset_info[person_id]
            twin_imgs = dataset.dataset_info[twin_id]
            
            # Sample a few pairs
            for i, img1 in enumerate(person_imgs[:3]):  # Limit to avoid too many pairs
                for j, img2 in enumerate(twin_imgs[:3]):
                    pairs.append((img1, img2, 0))
    
    return pairs


def evaluate_model(model, pairs, transform, device, batch_size=32):
    """
    Evaluate model on verification pairs.
    
    Args:
        model: Trained model
        pairs: List of verification pairs
        transform: Image transform
        device: Device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        embeddings1, embeddings2, labels: Evaluation data
    """
    from PIL import Image
    import torch.utils.data as data
    
    class VerificationDataset(data.Dataset):
        def __init__(self, pairs, transform):
            self.pairs = pairs
            self.transform = transform
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            img1_path, img2_path, label = self.pairs[idx]
            
            # Load images
            try:
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')
            except:
                # Fallback for path issues
                img1 = Image.new('RGB', (224, 224), color=(128, 128, 128))
                img2 = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            return {
                'image1': img1,
                'image2': img2,
                'label': label
            }
    
    # Create dataset and loader
    dataset = VerificationDataset(pairs, transform)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model.eval()
    all_embeddings1 = []
    all_embeddings2 = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images1 = batch['image1'].to(device)
            images2 = batch['image2'].to(device)
            labels = batch['label'].to(device)
            
            # Get embeddings
            if hasattr(model, 'get_embedding'):
                # DCAL model
                embeddings1 = model.get_embedding(images1)
                embeddings2 = model.get_embedding(images2)
            else:
                # Standard ViT
                embeddings1 = model.forward_features(images1)
                embeddings2 = model.forward_features(images2)
            
            all_embeddings1.append(embeddings1)
            all_embeddings2.append(embeddings2)
            all_labels.append(labels)
    
    # Concatenate results
    all_embeddings1 = torch.cat(all_embeddings1, dim=0)
    all_embeddings2 = torch.cat(all_embeddings2, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_embeddings1, all_embeddings2, all_labels


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply environment-specific configs
    if args.local:
        local_config = load_config('configs/local_config.yaml')
        config.config.update(local_config.config)
    elif args.kaggle:
        kaggle_config = load_config('configs/kaggle_config.yaml')
        config.config.update(kaggle_config.config)
    
    # Override batch size
    config.set('data.batch_size', args.batch_size)
    
    print(f"Evaluating model: {config.get('model.backbone')}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Distance metric: {args.distance_metric}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    use_dcal = config.get('model.use_glca', False) or config.get('model.use_pwca', False)
    
    if use_dcal:
        model = create_dcal_model(config.to_dict())
    else:
        model = deit_tiny_patch16_224(num_classes=config.get('model.num_classes', 356))
    
    model = model.to(device)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create transform
    transform = TwinFaceTransforms.get_val_transforms(config.to_dict())
    
    # Load test dataset
    test_dataset = TwinFaceDataset(
        dataset_info_path=config.get('data.test_info'),
        twin_pairs_path=config.get('data.test_pairs'),
        dataset_root=config.get('data.dataset_root'),
        transform=None,
        is_training=False
    )
    
    # Load twin pairs
    with open(config.get('data.test_pairs'), 'r') as f:
        twin_pairs = json.load(f)
    
    # Create verification pairs
    print("Creating verification pairs...")
    pairs = create_verification_pairs(test_dataset, twin_pairs)
    print(f"Created {len(pairs)} verification pairs")
    
    # Evaluate model
    print("Evaluating model...")
    embeddings1, embeddings2, labels = evaluate_model(
        model, pairs, transform, device, args.batch_size
    )
    
    # Compute metrics
    evaluator = VerificationEvaluator(
        distance_metric=args.distance_metric,
        far_targets=[0.001, 0.01, 0.1]
    )
    
    results = evaluator.evaluate_pairs(embeddings1, embeddings2, labels)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"EER: {results['EER']:.4f}")
    print(f"AUC: {results['AUC']:.4f}")
    print(f"Best Accuracy: {results['best_accuracy']:.4f}")
    
    for key, value in results.items():
        if key.startswith('TAR@FAR'):
            print(f"{key}: {value:.4f}")
    
    # Save results
    output_data = {
        'config': config.to_dict(),
        'checkpoint': args.checkpoint,
        'distance_metric': args.distance_metric,
        'num_pairs': len(pairs),
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()
