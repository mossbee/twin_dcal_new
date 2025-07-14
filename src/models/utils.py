import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


def attention_rollout(attentions: List[torch.Tensor], discard_ratio: float = 0.9) -> torch.Tensor:
    """
    Compute attention rollout following Abnar & Zuidema.
    
    Args:
        attentions: List of attention matrices from each layer [batch, heads, tokens, tokens]
        discard_ratio: Ratio of attention to keep (higher = more focused)
    
    Returns:
        rollout: Accumulated attention scores [batch, tokens, tokens]
    """
    if not attentions:
        raise ValueError("attentions list is empty. Cannot compute attention rollout.")
    
    batch_size = attentions[0].size(0)
    num_tokens = attentions[0].size(-1)
    
    # Average attention across heads for each layer
    averaged_attentions = []
    for attention in attentions:
        # attention: [batch, heads, tokens, tokens]
        avg_attention = attention.mean(dim=1)  # [batch, tokens, tokens]
        averaged_attentions.append(avg_attention)
    
    Args:
        attentions: List of attention matrices from each layer [batch, heads, tokens, tokens]
        discard_ratio: Ratio of attention to keep (higher = more focused)
    
    Returns:
        rollout: Accumulated attention scores [batch, tokens, tokens]
    """
    batch_size = attentions[0].size(0)
    num_tokens = attentions[0].size(-1)
    
    # Average attention across heads for each layer
    averaged_attentions = []
    for attention in attentions:
        # attention: [batch, heads, tokens, tokens]
        avg_attention = attention.mean(dim=1)  # [batch, tokens, tokens]
        averaged_attentions.append(avg_attention)
    
    # Add identity matrix to account for residual connections
    # Following the paper: S_bar = 0.5 * S + 0.5 * I
    eye = torch.eye(num_tokens, device=attentions[0].device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Initialize with identity matrix
    rollout = eye
    
    # Progressively multiply attention matrices
    for attention in averaged_attentions:
        # Add residual connection
        attention_with_residual = 0.5 * attention + 0.5 * eye
        
        # Matrix multiplication for rollout
        rollout = torch.matmul(attention_with_residual, rollout)
    
    return rollout


def get_high_response_regions(
    rollout: torch.Tensor, 
    ratio: float = 0.3,
    exclude_cls: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-R high-response regions from attention rollout.
    
    Args:
        rollout: Attention rollout matrix [batch, tokens, tokens]
        ratio: Ratio of tokens to select (R)
        exclude_cls: Whether to exclude CLS token from selection
    
    Returns:
        selected_indices: Indices of selected high-response regions [batch, num_selected]
        selected_scores: Attention scores of selected regions [batch, num_selected]
    """
    batch_size, num_tokens, _ = rollout.shape
    
    # Get attention scores from CLS token (first row)
    cls_attention = rollout[:, 0, :]  # [batch, tokens]
    
    if exclude_cls:
        # Exclude CLS token itself from selection
        patch_attention = cls_attention[:, 1:]  # [batch, num_patches]
        start_idx = 1
    else:
        patch_attention = cls_attention
        start_idx = 0
    
    # Calculate number of regions to select
    num_patches = patch_attention.size(1)
    num_selected = max(1, int(num_patches * ratio))
    
    # Get top-k patches
    selected_scores, relative_indices = torch.topk(patch_attention, k=num_selected, dim=1)
    
    # Convert relative indices to absolute indices
    selected_indices = relative_indices + start_idx
    
    return selected_indices, selected_scores


class AttentionRollout(nn.Module):
    """Attention rollout module for extracting high-response regions."""
    
    def __init__(self, local_query_ratio: float = 0.3, exclude_cls: bool = True):
        """
        Args:
            local_query_ratio: Ratio of patches to select as local queries (R)
            exclude_cls: Whether to exclude CLS token from selection
        """
        super().__init__()
        self.local_query_ratio = local_query_ratio
        self.exclude_cls = exclude_cls
        
    def forward(self, attentions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get high-response regions.
        
        Args:
            attentions: List of attention matrices from transformer blocks
        
        Returns:
            selected_indices: Indices of high-response regions [batch, num_selected]
            rollout: Full attention rollout matrix [batch, tokens, tokens]
        """
        # Compute attention rollout
        rollout = attention_rollout(attentions)
        
        # Get high-response regions
        selected_indices, selected_scores = get_high_response_regions(
            rollout, self.local_query_ratio, self.exclude_cls
        )
        
        return selected_indices, rollout


def extract_patch_embeddings(embeddings: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Extract patch embeddings based on selected indices.
    
    Args:
        embeddings: Input embeddings [batch, tokens, dim]
        indices: Selected patch indices [batch, num_selected]
    
    Returns:
        selected_embeddings: Selected patch embeddings [batch, num_selected, dim]
    """
    batch_size, num_selected = indices.shape
    batch_indices = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand(-1, num_selected)
    
    # Use advanced indexing to select embeddings
    selected_embeddings = embeddings[batch_indices, indices]
    
    return selected_embeddings


def visualize_attention_rollout(
    rollout: torch.Tensor, 
    image_size: Tuple[int, int] = (224, 224),
    patch_size: int = 16,
    selected_indices: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Visualize attention rollout as heatmap.
    
    Args:
        rollout: Attention rollout matrix [1, tokens, tokens] (single sample)
        image_size: Original image size (H, W)
        patch_size: Patch size
        selected_indices: Optional selected patch indices for highlighting
    
    Returns:
        attention_map: Attention heatmap as numpy array [H, W]
    """
    if rollout.dim() == 3 and rollout.size(0) == 1:
        rollout = rollout.squeeze(0)  # Remove batch dimension
    
    # Get CLS attention to patches
    cls_attention = rollout[0, 1:]  # Exclude CLS token
    
    # Reshape to spatial dimensions
    h_patches = image_size[0] // patch_size
    w_patches = image_size[1] // patch_size
    
    attention_map = cls_attention.reshape(h_patches, w_patches)
    attention_map = attention_map.cpu().numpy()
    
    # Resize to original image size using nearest neighbor
    from scipy.ndimage import zoom
    zoom_factor = (image_size[0] / h_patches, image_size[1] / w_patches)
    attention_map = zoom(attention_map, zoom_factor, order=1)
    
    # Normalize to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    return attention_map


class AttentionVisualization:
    """Utility class for attention visualization."""
    
    @staticmethod
    def save_attention_heatmap(
        attention_map: np.ndarray,
        output_path: str,
        original_image: Optional[np.ndarray] = None,
        alpha: float = 0.4
    ):
        """
        Save attention heatmap visualization.
        
        Args:
            attention_map: Attention heatmap [H, W]
            output_path: Path to save the visualization
            original_image: Optional original image to overlay [H, W, 3]
            alpha: Transparency for overlay
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        if original_image is not None:
            ax.imshow(original_image)
            ax.imshow(attention_map, cmap='jet', alpha=alpha)
        else:
            ax.imshow(attention_map, cmap='jet')
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    @staticmethod
    def create_attention_grid(
        attention_maps: List[np.ndarray],
        titles: Optional[List[str]] = None,
        output_path: str = None
    ):
        """
        Create a grid of attention visualizations.
        
        Args:
            attention_maps: List of attention heatmaps
            titles: Optional titles for each heatmap
            output_path: Optional path to save the grid
        """
        import matplotlib.pyplot as plt
        
        n_maps = len(attention_maps)
        cols = min(4, n_maps)
        rows = (n_maps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, attention_map in enumerate(attention_maps):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            im = ax.imshow(attention_map, cmap='jet')
            ax.axis('off')
            
            if titles and i < len(titles):
                ax.set_title(titles[i])
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(n_maps, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
