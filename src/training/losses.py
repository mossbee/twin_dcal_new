import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """Triplet loss implementation with hard mining."""
    
    def __init__(self, margin: float = 0.3, hard_mining: bool = True):
        """
        Args:
            margin: Margin for triplet loss
            hard_mining: Whether to use hard negative mining
        """
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]
        
        Returns:
            triplet_loss: Scalar loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        if self.hard_mining:
            # Only consider hard triplets (positive cases where loss > 0)
            hard_triplets = loss > 0
            if hard_triplets.sum() > 0:
                loss = loss[hard_triplets].mean()
            else:
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        else:
            loss = loss.mean()
        
        return loss


class BatchHardTripletLoss(nn.Module):
    """Batch hard triplet loss implementation."""
    
    def __init__(self, margin: float = 0.3):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Feature embeddings [batch_size, embed_dim]
            labels: Ground truth labels [batch_size]
        
        Returns:
            triplet_loss: Scalar loss value
        """
        # Compute pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get hardest positive and negative for each sample
        hardest_positive_dist, hardest_negative_dist = self._get_hardest_pairs(
            pairwise_dist, labels
        )
        
        # Compute triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between embeddings."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute cosine similarity and convert to distance
        dot_product = torch.matmul(embeddings, embeddings.t())
        distances = 1.0 - dot_product
        
        return distances
    
    def _get_hardest_pairs(self, pairwise_dist: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hardest positive and negative pairs for each sample."""
        batch_size = pairwise_dist.size(0)
        
        # Create mask for same/different labels
        labels = labels.unsqueeze(1)
        same_label_mask = labels == labels.t()
        diff_label_mask = ~same_label_mask
        
        # Set diagonal to infinity for positive pairs (exclude self)
        same_label_mask = same_label_mask.float()
        same_label_mask = same_label_mask + torch.diag(torch.ones(batch_size) * float('inf')).to(pairwise_dist.device)
        
        # Get hardest positive (maximum distance among same labels)
        hardest_positive_dist, _ = torch.max(pairwise_dist * same_label_mask, dim=1)
        
        # Get hardest negative (minimum distance among different labels)
        # Set same label distances to infinity
        masked_dist = pairwise_dist + same_label_mask * float('inf')
        hardest_negative_dist, _ = torch.min(masked_dist, dim=1)
        
        return hardest_positive_dist, hardest_negative_dist


class CombinedLoss(nn.Module):
    """Combined loss for SA and GLCA branches."""
    
    def __init__(
        self,
        triplet_margin: float = 0.3,
        use_dynamic_weights: bool = True,
        alpha: float = 0.5
    ):
        """
        Args:
            triplet_margin: Margin for triplet loss
            use_dynamic_weights: Whether to use dynamic loss weights
            alpha: Weight for SA branch when not using dynamic weights
        """
        super().__init__()
        self.triplet_loss = TripletLoss(margin=triplet_margin, hard_mining=True)
        self.use_dynamic_weights = use_dynamic_weights
        self.alpha = alpha
        
        if use_dynamic_weights:
            # Learnable weights for dynamic balancing
            self.log_vars = nn.Parameter(torch.zeros(2))  # For SA and GLCA
    
    def forward(
        self,
        sa_anchor: torch.Tensor,
        sa_positive: torch.Tensor,
        sa_negative: torch.Tensor,
        glca_anchor: Optional[torch.Tensor] = None,
        glca_positive: Optional[torch.Tensor] = None,
        glca_negative: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            sa_anchor, sa_positive, sa_negative: SA branch embeddings
            glca_anchor, glca_positive, glca_negative: GLCA branch embeddings (optional)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Compute SA loss
        sa_loss = self.triplet_loss(sa_anchor, sa_positive, sa_negative)
        
        loss_dict = {'sa_loss': sa_loss.item()}
        
        if glca_anchor is not None:
            # Compute GLCA loss
            glca_loss = self.triplet_loss(glca_anchor, glca_positive, glca_negative)
            loss_dict['glca_loss'] = glca_loss.item()
            
            if self.use_dynamic_weights:
                # Dynamic weight balancing (uncertainty weighting)
                precision1 = torch.exp(-self.log_vars[0])
                precision2 = torch.exp(-self.log_vars[1])
                
                total_loss = (
                    precision1 * sa_loss + self.log_vars[0] +
                    precision2 * glca_loss + self.log_vars[1]
                )
                
                loss_dict['sa_weight'] = precision1.item()
                loss_dict['glca_weight'] = precision2.item()
            else:
                # Fixed weight balancing
                total_loss = self.alpha * sa_loss + (1 - self.alpha) * glca_loss
                loss_dict['sa_weight'] = self.alpha
                loss_dict['glca_weight'] = 1 - self.alpha
        else:
            # Only SA loss
            total_loss = sa_loss
            loss_dict['sa_weight'] = 1.0
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class ContrastiveLoss(nn.Module):
    """Contrastive loss for face verification."""
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output1: First embedding [batch_size, embed_dim]
            output2: Second embedding [batch_size, embed_dim]
            label: Binary labels (1 if same person, 0 if different) [batch_size]
        
        Returns:
            contrastive_loss: Scalar loss value
        """
        # Compute Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2, p=2)
        
        # Contrastive loss
        loss_contrastive = torch.mean(
            label.float() * torch.pow(euclidean_distance, 2) +
            (1 - label.float()) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive
