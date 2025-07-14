"""
Hyperparameter optimization using Optuna for DCAL model.
"""

import os
import json
from typing import Dict, Any, Optional, List
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader

from ..utils.config import Config
from ..models.dcal_model import create_dcal_model
from ..training.trainer import Trainer
from ..training.metrics import VerificationEvaluator
from ..utils.tracking import create_tracker


class HyperparameterOptimizer:
    """Hyperparameter optimization for DCAL model using Optuna."""
    
    def __init__(
        self,
        base_config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        study_name: str = "dcal_optimization",
        storage: Optional[str] = None,
        n_trials: int = 100
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            base_config: Base configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            study_name: Name of the optimization study
            storage: Optional storage URL for distributed optimization
            n_trials: Number of optimization trials
        """
        self.base_config = base_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.study_name = study_name
        self.n_trials = n_trials
        
        # Create Optuna study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',  # Maximize validation metric
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=10
            ),
            load_if_exists=True
        )
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            hyperparams: Dictionary of suggested hyperparameters
        """
        # Model architecture hyperparameters
        backbone = trial.suggest_categorical(
            'backbone', 
            ['deit_tiny_patch16_224', 'deit_small_patch16_224']
        )
        
        # DCAL specific parameters
        use_glca = trial.suggest_categorical('use_glca', [True, False])
        use_pwca = trial.suggest_categorical('use_pwca', [True, False])
        
        local_query_ratio = trial.suggest_float('local_query_ratio', 0.1, 0.5, step=0.1)
        
        # Training hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.01, 0.2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        
        # Loss hyperparameters
        triplet_margin = trial.suggest_float('triplet_margin', 0.1, 0.5, step=0.1)
        use_dynamic_weights = trial.suggest_categorical('use_dynamic_weights', [True, False])
        
        # Data augmentation
        horizontal_flip = trial.suggest_float('horizontal_flip', 0.0, 0.7, step=0.1)
        rotation = trial.suggest_int('rotation', 5, 20, step=5)
        color_jitter_strength = trial.suggest_float('color_jitter_strength', 0.1, 0.3, step=0.05)
        random_erasing = trial.suggest_float('random_erasing', 0.0, 0.4, step=0.1)
        
        # Scheduler parameters
        warmup_epochs = trial.suggest_int('warmup_epochs', 2, 10)
        
        return {
            'model': {
                'backbone': backbone,
                'use_glca': use_glca,
                'use_pwca': use_pwca,
                'local_query_ratio': local_query_ratio,
            },
            'training': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'warmup_epochs': warmup_epochs,
                'triplet_margin': triplet_margin,
                'use_dynamic_weights': use_dynamic_weights,
            },
            'data': {
                'batch_size': batch_size,
                'augmentation': {
                    'horizontal_flip': horizontal_flip,
                    'rotation': rotation,
                    'color_jitter': {
                        'brightness': color_jitter_strength,
                        'contrast': color_jitter_strength,
                        'saturation': color_jitter_strength,
                        'hue': color_jitter_strength / 2,
                    },
                    'random_erasing': random_erasing,
                }
            }
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            objective_value: Value to optimize (higher is better)
        """
        # Get suggested hyperparameters
        hyperparams = self.suggest_hyperparameters(trial)
        
        # Create modified config
        config = self.base_config.copy()
        config.update(hyperparams)
        
        # Reduce training epochs for faster optimization
        config.set('training.epochs', 20)  # Short training for HPO
        
        try:
            # Create model
            model = create_dcal_model(config.to_dict())
            
            # Create trainer
            trainer = Trainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=config,
                device=self.device
            )
            
            # Create no-tracking experiment tracker
            tracker = create_tracker('none', 'hpo_trial', f'trial_{trial.number}')
            
            # Train for a few epochs
            best_metric = 0.0
            for epoch in range(config.get('training.epochs', 20)):
                # Train epoch
                train_metrics = trainer.train_epoch()
                
                # Validate
                val_metrics = trainer.validate()
                
                # Calculate objective metric (you can customize this)
                # For twin verification, we want high embedding quality
                embedding_norm = val_metrics.get('val_embedding_norm', 1.0)
                train_loss = train_metrics.get('train_loss', float('inf'))
                
                # Simple objective: minimize loss while maintaining reasonable embedding norm
                objective_value = 1.0 / (train_loss + 1e-6) * min(embedding_norm, 2.0)
                
                if objective_value > best_metric:
                    best_metric = objective_value
                
                # Report intermediate value for pruning
                trial.report(objective_value, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return best_metric
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            best_params: Best hyperparameters found
        """
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        
        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"Optimization completed!")
        print(f"Best value: {best_value:.4f}")
        print(f"Best parameters: {json.dumps(best_params, indent=2)}")
        
        return best_params
    
    def save_study(self, output_path: str):
        """Save optimization study results."""
        # Save study as dataframe
        df = self.study.trials_dataframe()
        df.to_csv(f"{output_path}_trials.csv", index=False)
        
        # Save best parameters
        with open(f"{output_path}_best_params.json", 'w') as f:
            json.dump(self.study.best_params, f, indent=2)
        
        # Save study statistics
        stats = {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_trial_number': self.study.best_trial.number,
            'study_name': self.study_name
        }
        
        with open(f"{output_path}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def plot_optimization_history(self, output_path: str):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.savefig(f"{output_path}_history.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.savefig(f"{output_path}_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot parallel coordinate
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)
            plt.savefig(f"{output_path}_parallel.png", dpi=150, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("matplotlib not available, skipping plots")


def run_hyperparameter_optimization(
    config_path: str,
    output_dir: str = "hpo_results",
    n_trials: int = 100,
    study_name: str = "dcal_hpo"
) -> Dict[str, Any]:
    """
    Run hyperparameter optimization from config file.
    
    Args:
        config_path: Path to base configuration file
        output_dir: Directory to save results
        n_trials: Number of optimization trials
        study_name: Name of the optimization study
    
    Returns:
        best_params: Best hyperparameters found
    """
    from ..utils.config import load_config
    from ..data.dataset import create_data_loaders
    
    # Load base configuration
    config = load_config(config_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders (with smaller batch size for HPO)
    train_loader, val_loader, test_loader = create_data_loaders(
        config, batch_size=16  # Fixed smaller batch size for HPO
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        base_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        study_name=study_name,
        n_trials=n_trials
    )
    
    # Run optimization
    best_params = optimizer.optimize()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, study_name)
    
    optimizer.save_study(output_path)
    optimizer.plot_optimization_history(output_path)
    
    # Create optimized config
    optimized_config = config.copy()
    optimized_config.update(best_params)
    
    # Save optimized config
    optimized_config.save(os.path.join(output_dir, f"{study_name}_optimized_config.yaml"))
    
    return best_params
