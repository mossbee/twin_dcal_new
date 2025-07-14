import os
import sys
import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Handle both relative and absolute imports
try:
    from ..models.backbone import VisionTransformer
    from ..models.dcal_model import DCALModel
    from ..training.losses import CombinedLoss
    from ..training.metrics import VerificationEvaluator
    from ..utils.tracking import ExperimentTracker
    from ..utils.config import Config
except ImportError:
    # Fallback for when running as script
    from models.backbone import VisionTransformer
    from models.dcal_model import DCALModel
    from training.losses import CombinedLoss
    from training.metrics import VerificationEvaluator
    from utils.tracking import ExperimentTracker
    from utils.config import Config


class Trainer:
    """Trainer for Twin DCAL model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = CombinedLoss(
            triplet_margin=config.get('training.triplet_margin', 0.3),
            use_dynamic_weights=config.get('training.use_dynamic_weights', True)
        ).to(device)
        
        # Setup evaluator
        self.evaluator = VerificationEvaluator(
            distance_metric='cosine',
            far_targets=config.get('evaluation.far_thresholds', [0.001, 0.01, 0.1])
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_name = self.config.get('training.optimizer', 'adam').lower()
        lr = self.config.get('training.learning_rate', 5e-4)
        weight_decay = self.config.get('training.weight_decay', 0.05)
        
        if optimizer_name == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        scheduler_name = self.config.get('training.scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            total_steps = len(self.train_loader) * self.config.get('training.epochs', 100)
            warmup_steps = len(self.train_loader) * self.config.get('training.warmup_epochs', 5)
            
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_sa_loss = 0.0
        total_glca_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get triplet data
            anchor_data = batch['anchor']
            positive_data = batch['positive']
            negative_data = batch['negative']
            
            # Move to device
            anchor_imgs = anchor_data['image'].to(self.device)
            positive_imgs = positive_data['image'].to(self.device)
            negative_imgs = negative_data['image'].to(self.device)
            
            # For PWCA, we need secondary images (random pairing)
            if hasattr(self.model, 'use_pwca') and self.model.use_pwca:
                # Create random secondary images by shuffling
                batch_size = anchor_imgs.size(0)
                shuffle_idx = torch.randperm(batch_size)
                anchor_secondary = positive_imgs[shuffle_idx]  # Random secondary for anchor
                positive_secondary = negative_imgs[shuffle_idx]  # Random secondary for positive
                negative_secondary = anchor_imgs[shuffle_idx]  # Random secondary for negative
            else:
                anchor_secondary = positive_secondary = negative_secondary = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get embeddings from model
            if isinstance(self.model, DCALModel):
                # DCAL model with SA and GLCA branches
                anchor_outputs = self.model(anchor_imgs, anchor_secondary, return_features=True)
                positive_outputs = self.model(positive_imgs, positive_secondary, return_features=True)
                negative_outputs = self.model(negative_imgs, negative_secondary, return_features=True)
                
                # Extract SA embeddings
                anchor_sa = anchor_outputs['sa_features']
                positive_sa = positive_outputs['sa_features']
                negative_sa = negative_outputs['sa_features']
                
                # Extract GLCA embeddings (if available)
                anchor_glca = anchor_outputs.get('glca_features')
                positive_glca = positive_outputs.get('glca_features')
                negative_glca = negative_outputs.get('glca_features')
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    anchor_sa, positive_sa, negative_sa,
                    anchor_glca, positive_glca, negative_glca
                )
            else:
                # Standard ViT model (SA only)
                anchor_embeddings = self.model.forward_features(anchor_imgs)
                positive_embeddings = self.model.forward_features(positive_imgs)
                negative_embeddings = self.model.forward_features(negative_imgs)
                
                # Compute loss (SA only)
                loss, loss_dict = self.criterion(
                    anchor_embeddings, positive_embeddings, negative_embeddings
                )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('training.gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('training.gradient_clip')
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            total_sa_loss += loss_dict['sa_loss']
            if 'glca_loss' in loss_dict:
                total_glca_loss += loss_dict.get('glca_loss', 0)
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss']:.4f}",
                'SA': f"{loss_dict['sa_loss']:.4f}",
                'GLCA': f"{loss_dict.get('glca_loss', 0):.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            self.global_step += 1
        
        # Calculate average losses
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'train_sa_loss': total_sa_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        if total_glca_loss > 0:
            avg_metrics['train_glca_loss'] = total_glca_loss / num_batches
        
        return avg_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get embeddings
                if isinstance(self.model, DCALModel):
                    # Use the combined embedding method for DCAL
                    embeddings = self.model.get_embedding(images)
                else:
                    # Standard ViT model
                    embeddings = self.model.forward_features(images)
                
                all_embeddings.append(embeddings)
                all_labels.append(labels)
        
        # Concatenate all embeddings and labels
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # For validation, we need to create pairs
        # Simple approach: create all possible pairs within batch
        # In practice, you'd want to use the twin pairs from the dataset
        
        # For now, just return dummy metrics
        val_metrics = {
            'val_embedding_norm': torch.norm(all_embeddings, dim=1).mean().item(),
            'val_samples': len(all_embeddings)
        }
        
        return val_metrics
    
    def train(self, tracker: ExperimentTracker, checkpoint_dir: str = 'checkpoints'):
        """Main training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        total_epochs = self.config.get('training.epochs', 100)
        save_every = self.config.get('training.save_every', 1)
        
        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            epoch_time = time.time() - start_time
            all_metrics['epoch_time'] = epoch_time
            
            # Log metrics
            tracker.log_metrics(all_metrics, step=epoch)
            
            # Print progress
            print(f"Epoch {epoch}: Loss={train_metrics['train_loss']:.4f}, "
                  f"Time={epoch_time:.2f}s")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                self.save_checkpoint(checkpoint_path)
                tracker.log_artifact(checkpoint_path, f'checkpoint_epoch_{epoch+1}')
        
        # Save final model
        final_path = os.path.join(checkpoint_dir, 'final_model.pth')
        self.save_checkpoint(final_path)
        tracker.log_artifact(final_path, 'final_model')
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'criterion_state_dict': self.criterion.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
