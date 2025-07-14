#!/usr/bin/env python3
"""
Phase 3 Advanced Features Demo Script for Twin DCAL.

This script demonstrates:
1. Model ensemble and test-time augmentation
2. Hyperparameter optimization
3. Extended evaluation metrics
4. Performance optimization
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config, load_config
from utils.tracking import create_tracker
from data.dataset import create_data_loaders
from models.dcal_model import create_dcal_model
from training.trainer import Trainer
from training.ensemble import (
    ModelEnsemble, 
    TestTimeAugmentation, 
    EnsembleTTAInference,
    create_ensemble_from_checkpoints,
    evaluate_ensemble
)
from training.hyperopt import run_hyperparameter_optimization
from training.metrics import ExtendedVerificationEvaluator, PerformanceProfiler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Twin DCAL Phase 3 Advanced Features')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to config file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['ensemble', 'hyperopt', 'evaluation', 'benchmark', 'all'],
        default='all',
        help='Which advanced feature to run'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory containing model checkpoints'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='phase3_results',
        help='Output directory for results'
    )
    
    # Hyperparameter optimization options
    parser.add_argument(
        '--hpo-trials',
        type=int,
        default=50,
        help='Number of hyperparameter optimization trials'
    )
    
    # Ensemble options
    parser.add_argument(
        '--n-models',
        type=int,
        default=3,
        help='Number of models for ensemble'
    )
    
    parser.add_argument(
        '--use-tta',
        action='store_true',
        help='Use test-time augmentation'
    )
    
    parser.add_argument(
        '--tta-augmentations',
        type=int,
        default=8,
        help='Number of TTA augmentations'
    )
    
    return parser.parse_args()


def run_model_ensemble(args, config, device):
    """Run model ensemble evaluation."""
    print("=== Running Model Ensemble Evaluation ===")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Check for available checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("*.pth"))
    
    if len(checkpoint_files) < args.n_models:
        print(f"Not enough checkpoints found ({len(checkpoint_files)} < {args.n_models})")
        print("Training multiple models for ensemble...")
        
        # Train multiple models with different seeds
        checkpoint_files = []
        for i in range(args.n_models):
            print(f"Training model {i+1}/{args.n_models}")
            
            # Set different random seed
            torch.manual_seed(42 + i)
            
            # Create model
            model = create_dcal_model(config.to_dict())
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device
            )
            
            # Create tracker
            tracker = create_tracker('none', 'ensemble_training', f'model_{i}')
            
            # Train for fewer epochs for demo
            config_copy = config.copy()
            config_copy.set('training.epochs', 10)  # Quick training
            
            # Train model
            trainer.train(tracker, checkpoint_dir=f"{args.output_dir}/ensemble_models")
            
            # Add checkpoint path
            checkpoint_path = f"{args.output_dir}/ensemble_models/final_model.pth"
            checkpoint_files.append(checkpoint_path)
    
    # Create ensemble
    ensemble = create_ensemble_from_checkpoints(
        checkpoint_paths=[str(f) for f in checkpoint_files[:args.n_models]],
        config=config.to_dict(),
        device=device
    )
    
    # Evaluate ensemble
    ensemble_metrics = evaluate_ensemble(
        ensemble=ensemble,
        data_loader=test_loader,
        device=device,
        use_tta=args.use_tta,
        n_augmentations=args.tta_augmentations
    )
    
    print("Ensemble Evaluation Results:")
    for metric, value in ensemble_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    import json
    results_path = f"{args.output_dir}/ensemble_results.json"
    with open(results_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    
    return ensemble_metrics


def run_hyperparameter_optimization(args, config):
    """Run hyperparameter optimization."""
    print("=== Running Hyperparameter Optimization ===")
    
    # Run HPO
    best_params = run_hyperparameter_optimization(
        config_path=args.config,
        output_dir=f"{args.output_dir}/hyperopt",
        n_trials=args.hpo_trials,
        study_name="dcal_phase3_hpo"
    )
    
    print("Best hyperparameters found:")
    import json
    print(json.dumps(best_params, indent=2))
    
    return best_params


def run_extended_evaluation(args, config, device):
    """Run extended evaluation with detailed metrics."""
    print("=== Running Extended Evaluation ===")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Load or create model
    checkpoint_files = list(Path(args.checkpoint_dir).glob("*.pth"))
    
    if checkpoint_files:
        # Load existing model
        print(f"Loading model from {checkpoint_files[0]}")
        model = create_dcal_model(config.to_dict())
        checkpoint = torch.load(checkpoint_files[0], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Create and quickly train a model
        print("No checkpoint found, training a demo model...")
        model = create_dcal_model(config.to_dict())
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Quick training
        config_copy = config.copy()
        config_copy.set('training.epochs', 5)
        tracker = create_tracker('none', 'demo_training', 'quick_model')
        trainer.train(tracker, checkpoint_dir=f"{args.output_dir}/demo_model")
    
    model.to(device)
    model.eval()
    
    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []
    
    print("Collecting embeddings for evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label']
            
            embeddings = model.get_embedding(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Extended evaluation
    evaluator = ExtendedVerificationEvaluator()
    
    evaluation_dir = f"{args.output_dir}/evaluation"
    detailed_metrics = evaluator.evaluate_with_analysis(
        embeddings=all_embeddings,
        labels=all_labels,
        save_plots=True,
        output_dir=evaluation_dir
    )
    
    print("Extended Evaluation Results:")
    for metric, value in detailed_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    import json
    results_path = f"{evaluation_dir}/detailed_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(detailed_metrics, f, indent=2)
    
    return detailed_metrics


def run_performance_benchmark(args, config, device):
    """Run performance benchmarking."""
    print("=== Running Performance Benchmark ===")
    
    profiler = PerformanceProfiler()
    
    # Create different model configurations for comparison
    models = {}
    
    # Standard ViT
    config_vit = config.copy()
    config_vit.set('model.use_glca', False)
    config_vit.set('model.use_pwca', False)
    models['ViT-Only'] = create_dcal_model(config_vit.to_dict())
    
    # DCAL with GLCA only
    config_glca = config.copy()
    config_glca.set('model.use_glca', True)
    config_glca.set('model.use_pwca', False)
    models['DCAL-GLCA'] = create_dcal_model(config_glca.to_dict())
    
    # Full DCAL
    config_full = config.copy()
    config_full.set('model.use_glca', True)
    config_full.set('model.use_pwca', True)
    models['DCAL-Full'] = create_dcal_model(config_full.to_dict())
    
    # Benchmark all models
    input_shape = (1, 3, 224, 224)
    benchmark_results = profiler.compare_models(models, input_shape, device)
    
    print("Performance Benchmark Results:")
    print("-" * 80)
    print(f"{'Model':<15} {'Params (M)':<12} {'FPS':<10} {'Memory (MB)':<12} {'Latency (ms)':<12}")
    print("-" * 80)
    
    for name, results in benchmark_results.items():
        params_m = results['total_parameters'] / 1e6
        fps = results['fps']
        memory = results['memory_allocated_mb']
        latency = results['avg_inference_time_ms']
        
        print(f"{name:<15} {params_m:<12.2f} {fps:<10.1f} {memory:<12.1f} {latency:<12.2f}")
    
    # Save benchmark results
    import json
    results_path = f"{args.output_dir}/benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    return benchmark_results


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run selected features
    results = {}
    
    if args.mode in ['ensemble', 'all']:
        try:
            results['ensemble'] = run_model_ensemble(args, config, device)
        except Exception as e:
            print(f"Ensemble evaluation failed: {e}")
            results['ensemble'] = None
    
    if args.mode in ['hyperopt', 'all']:
        try:
            results['hyperopt'] = run_hyperparameter_optimization(args, config)
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            results['hyperopt'] = None
    
    if args.mode in ['evaluation', 'all']:
        try:
            results['evaluation'] = run_extended_evaluation(args, config, device)
        except Exception as e:
            print(f"Extended evaluation failed: {e}")
            results['evaluation'] = None
    
    if args.mode in ['benchmark', 'all']:
        try:
            results['benchmark'] = run_performance_benchmark(args, config, device)
        except Exception as e:
            print(f"Performance benchmark failed: {e}")
            results['benchmark'] = None
    
    # Save combined results
    import json
    summary_path = f"{args.output_dir}/phase3_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nPhase 3 Advanced Features Demo completed!")
    print(f"Results saved to: {args.output_dir}")
    print("\nSummary:")
    for feature, result in results.items():
        status = "✓ Success" if result is not None else "✗ Failed"
        print(f"  {feature.capitalize()}: {status}")


if __name__ == "__main__":
    main()
