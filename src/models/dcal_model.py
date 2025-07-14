import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from .backbone import VisionTransformer, PatchEmbedding
from .attention import DCALBlock


class DCALModel(nn.Module):
    """
    DCAL (Dual Cross-Attention Learning) model for twin face verification.
    
    Integrates Self-Attention (SA), Global-Local Cross-Attention (GLCA),
    and Pair-wise Cross-Attention (PWCA) mechanisms.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        # DCAL specific parameters
        use_glca: bool = True,
        use_pwca: bool = True,
        glca_layers: List[int] = None,  # Which layers to apply GLCA
        pwca_layers: List[int] = None,  # Which layers to apply PWCA
        local_query_ratio: float = 0.3,
    ):
        """
        Args:
            img_size: Input image size
            patch_size: Patch size for patch embedding
            in_chans: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in QKV projections
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            use_glca: Whether to use GLCA
            use_pwca: Whether to use PWCA
            glca_layers: List of layer indices to apply GLCA (default: last layer)
            pwca_layers: List of layer indices to apply PWCA (default: all layers)
            local_query_ratio: Ratio of patches to select for local queries
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_glca = use_glca
        self.use_pwca = use_pwca
        self.local_query_ratio = local_query_ratio
        
        # Default layer configurations
        if glca_layers is None:
            glca_layers = [depth - 1]  # Only last layer by default (M=1)
        if pwca_layers is None:
            pwca_layers = list(range(depth))  # All layers by default (T=depth)
        
        self.glca_layers = set(glca_layers)
        self.pwca_layers = set(pwca_layers)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # DCAL transformer blocks
        self.blocks = nn.ModuleList([
            DCALBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                use_glca=(i in self.glca_layers) and use_glca,
                use_pwca=(i in self.pwca_layers) and use_pwca,
                local_query_ratio=local_query_ratio
            )
            for i in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification heads
        self.sa_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if use_glca:
            self.glca_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Initialize position embeddings and cls token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize weights for a module."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(
        self, 
        x: torch.Tensor, 
        secondary_x: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through feature extraction layers.
        
        Args:
            x: Primary input images [batch, channels, height, width]
            secondary_x: Secondary input images for PWCA [batch, channels, height, width]
            return_all_tokens: Whether to return all tokens or just CLS token
            return_attention: Whether to return attention weights
        
        Returns:
            sa_features: Features from SA branch
            glca_features: Features from GLCA branch (if enabled)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process secondary input if provided
        secondary_tokens = None
        if secondary_x is not None and self.use_pwca:
            secondary_x = self.patch_embed(secondary_x)
            secondary_cls_tokens = self.cls_token.expand(B, -1, -1)
            secondary_x = torch.cat((secondary_cls_tokens, secondary_x), dim=1)
            secondary_x = secondary_x + self.pos_embed
            secondary_x = self.pos_drop(secondary_x)
            secondary_tokens = secondary_x
        
        # Forward through DCAL blocks
        sa_attentions = []
        sa_output = x
        glca_output = None
        
        for i, block in enumerate(self.blocks):
            sa_out, glca_out, pwca_out = block(
                sa_output,
                secondary_x=secondary_tokens,
                sa_attentions=sa_attentions if i in self.glca_layers else None,
                return_attention=return_attention
            )
            
            # Update outputs
            sa_output = sa_out
            if glca_out is not None:
                glca_output = glca_out
            
            # For PWCA, we use the PWCA output as input to next layer during training
            if pwca_out is not None and self.training:
                sa_output = pwca_out
            
            # Store attention for GLCA (we need accumulated attention)
            if return_attention and hasattr(block.self_attn, 'last_attention'):
                sa_attentions.append(block.self_attn.last_attention)
        
        # Final layer norm
        sa_output = self.norm(sa_output)
        if glca_output is not None:
            glca_output = self.norm(glca_output)
        
        # Return appropriate tokens
        if return_all_tokens:
            sa_features = sa_output
            glca_features = glca_output
        else:
            # Return CLS token only
            sa_features = sa_output[:, 0]
            glca_features = glca_output[:, 0] if glca_output is not None else None
        
        return sa_features, glca_features
    
    def forward(
        self, 
        x: torch.Tensor, 
        secondary_x: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with classification outputs.
        
        Args:
            x: Primary input images [batch, channels, height, width]
            secondary_x: Secondary input images for PWCA [batch, channels, height, width]
            return_features: Whether to return features instead of logits
        
        Returns:
            outputs: Dictionary containing SA and GLCA outputs
        """
        # Get features
        sa_features, glca_features = self.forward_features(x, secondary_x)
        
        outputs = {}
        
        if return_features:
            # Return raw features for metric learning
            outputs['sa_features'] = sa_features
            if glca_features is not None:
                outputs['glca_features'] = glca_features
        else:
            # Return classification logits
            outputs['sa_logits'] = self.sa_head(sa_features)
            if glca_features is not None and hasattr(self, 'glca_head'):
                outputs['glca_logits'] = self.glca_head(glca_features)
        
        return outputs
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding for face verification.
        
        Args:
            x: Input images [batch, channels, height, width]
        
        Returns:
            embedding: Combined embedding from SA and GLCA branches
        """
        sa_features, glca_features = self.forward_features(x)
        
        if glca_features is not None:
            # Combine SA and GLCA features
            embedding = (sa_features + glca_features) / 2
        else:
            embedding = sa_features
        
        return embedding


def create_dcal_model(config: Dict) -> DCALModel:
    """
    Create DCAL model from configuration.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        model: DCAL model instance
    """
    model_config = config.get('model', {})
    
    # Extract architecture parameters
    backbone = model_config.get('backbone', 'deit_tiny_patch16_224')
    
    # Architecture mapping
    arch_configs = {
        'deit_tiny_patch16_224': {
            'embed_dim': 192, 'depth': 12, 'num_heads': 3
        },
        'deit_small_patch16_224': {
            'embed_dim': 384, 'depth': 12, 'num_heads': 6
        },
        'deit_base_patch16_224': {
            'embed_dim': 768, 'depth': 12, 'num_heads': 12
        }
    }
    
    if backbone not in arch_configs:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    arch_config = arch_configs[backbone]
    
    # Create model
    model = DCALModel(
        img_size=model_config.get('image_size', 224),
        patch_size=model_config.get('patch_size', 16),
        in_chans=3,
        num_classes=model_config.get('num_classes', 1000),
        embed_dim=arch_config['embed_dim'],
        depth=arch_config['depth'],
        num_heads=arch_config['num_heads'],
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        qkv_bias=True,
        drop_rate=model_config.get('dropout', 0.0),
        attn_drop_rate=model_config.get('attention_dropout', 0.0),
        use_glca=model_config.get('use_glca', True),
        use_pwca=model_config.get('use_pwca', True),
        glca_layers=[arch_config['depth'] - 1],  # Last layer only
        pwca_layers=list(range(arch_config['depth'])),  # All layers
        local_query_ratio=model_config.get('local_query_ratio', 0.3)
    )
    
    return model
