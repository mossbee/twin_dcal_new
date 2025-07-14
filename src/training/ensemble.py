"""
Model ensemble and test-time augmentation for improved performance.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.dcal_model import DCALModel
from ..data.transforms import TwinFaceTransforms


class ModelEnsemble:
    """Ensemble of multiple DCAL models for improved performance."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize model ensemble.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        
        assert len(self.weights) == len(self.models), "Number of weights must match number of models"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble embedding by averaging individual model embeddings.
        
        Args:
            x: Input images [batch, channels, height, width]
        
        Returns:
            ensemble_embedding: Weighted average of model embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                if isinstance(model, DCALModel):
                    embedding = model.get_embedding(x)
                else:
                    embedding = model.forward_features(x)
                
                embeddings.append(embedding * weight)
        
        # Sum weighted embeddings
        ensemble_embedding = torch.stack(embeddings, dim=0).sum(dim=0)
        
        # Normalize the final embedding
        ensemble_embedding = nn.functional.normalize(ensemble_embedding, p=2, dim=1)
        
        return ensemble_embedding
    
    def to(self, device):
        """Move all models to device."""
        for model in self.models:
            model.to(device)
        return self


class TestTimeAugmentation:
    """Test-time augmentation for improved robustness."""
    
    def __init__(self, n_augmentations: int = 8, use_horizontal_flip: bool = True):
        """
        Initialize TTA.
        
        Args:
            n_augmentations: Number of augmentations to apply
            use_horizontal_flip: Whether to use horizontal flipping
        """
        self.n_augmentations = n_augmentations
        self.use_horizontal_flip = use_horizontal_flip
        
        # Create augmentation transforms
        self.transforms = TwinFaceTransforms(
            image_size=224,
            augment=True,
            horizontal_flip=0.5 if use_horizontal_flip else 0.0,
            rotation=10,
            color_jitter={'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05}
        )
    
    def augment_image(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate augmented versions of an image.
        
        Args:
            image: Input image [channels, height, width]
        
        Returns:
            augmented_images: List of augmented images
        """
        augmented = [image]  # Include original
        
        # Generate additional augmentations
        for _ in range(self.n_augmentations - 1):
            # Convert to PIL for transforms, then back to tensor
            pil_image = self.transforms.to_pil(image)
            aug_image = self.transforms.augment_transform(pil_image)
            augmented.append(aug_image)
        
        return augmented
    
    def get_embedding_with_tta(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        Get embedding with test-time augmentation.
        
        Args:
            model: Model to use for inference
            image: Input image [batch, channels, height, width]
        
        Returns:
            tta_embedding: Average embedding across augmentations
        """
        batch_size = image.size(0)
        embeddings = []
        
        with torch.no_grad():
            for i in range(batch_size):
                single_image = image[i]
                
                # Generate augmented versions
                augmented_images = self.augment_image(single_image)
                
                # Get embeddings for all augmentations
                aug_embeddings = []
                for aug_img in augmented_images:
                    aug_batch = aug_img.unsqueeze(0)  # Add batch dimension
                    
                    if isinstance(model, DCALModel):
                        embedding = model.get_embedding(aug_batch)
                    else:
                        embedding = model.forward_features(aug_batch)
                    
                    aug_embeddings.append(embedding)
                
                # Average embeddings
                avg_embedding = torch.stack(aug_embeddings, dim=0).mean(dim=0)
                embeddings.append(avg_embedding)
            
            # Stack all batch embeddings
            tta_embedding = torch.cat(embeddings, dim=0)
            
            # Normalize
            tta_embedding = nn.functional.normalize(tta_embedding, p=2, dim=1)
        
        return tta_embedding


class EnsembleTTAInference:
    """Combined ensemble and TTA inference for maximum performance."""
    
    def __init__(
        self, 
        models: List[nn.Module], 
        model_weights: Optional[List[float]] = None,
        use_tta: bool = True,
        n_augmentations: int = 8
    ):
        """
        Initialize ensemble + TTA inference.
        
        Args:
            models: List of trained models
            model_weights: Optional weights for model ensemble
            use_tta: Whether to use test-time augmentation
            n_augmentations: Number of augmentations for TTA
        """
        self.ensemble = ModelEnsemble(models, model_weights)
        self.use_tta = use_tta
        
        if use_tta:
            self.tta = TestTimeAugmentation(n_augmentations=n_augmentations)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding using ensemble and TTA.
        
        Args:
            x: Input images [batch, channels, height, width]
        
        Returns:
            final_embedding: Ensemble + TTA embedding
        """
        if self.use_tta:
            # Apply TTA to each model and then ensemble
            embeddings = []
            
            for model, weight in zip(self.ensemble.models, self.ensemble.weights):
                tta_embedding = self.tta.get_embedding_with_tta(model, x)
                embeddings.append(tta_embedding * weight)
            
            final_embedding = torch.stack(embeddings, dim=0).sum(dim=0)
            final_embedding = nn.functional.normalize(final_embedding, p=2, dim=1)
        else:
            # Use ensemble without TTA
            final_embedding = self.ensemble.get_embedding(x)
        
        return final_embedding
    
    def to(self, device):
        """Move to device."""
        self.ensemble.to(device)
        return self


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str], 
    config: Dict,
    device: torch.device
) -> ModelEnsemble:
    """
    Create model ensemble from checkpoint files.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        config: Model configuration
        device: Device to load models on
    
    Returns:
        ensemble: Model ensemble
    """
    from ..models.dcal_model import create_dcal_model
    
    models = []
    
    for checkpoint_path in checkpoint_paths:
        # Create model
        model = create_dcal_model(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device and set to eval
        model.to(device)
        model.eval()
        
        models.append(model)
    
    return ModelEnsemble(models)


def evaluate_ensemble(
    ensemble: ModelEnsemble,
    data_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    n_augmentations: int = 8
) -> Dict[str, float]:
    """
    Evaluate ensemble model performance.
    
    Args:
        ensemble: Model ensemble
        data_loader: Data loader for evaluation
        device: Device for evaluation
        use_tta: Whether to use test-time augmentation
        n_augmentations: Number of augmentations for TTA
    
    Returns:
        metrics: Evaluation metrics
    """
    from ..training.metrics import VerificationEvaluator
    
    # Create inference engine
    if use_tta:
        inference_engine = EnsembleTTAInference(
            ensemble.models, 
            ensemble.weights,
            use_tta=True,
            n_augmentations=n_augmentations
        ).to(device)
    else:
        inference_engine = ensemble
    
    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Ensemble Evaluation'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Get embeddings
            embeddings = inference_engine.get_embedding(images)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate results
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Evaluate using verification metrics
    evaluator = VerificationEvaluator()
    metrics = evaluator.evaluate_verification(all_embeddings, all_labels)
    
    return metrics
