import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from einops import rearrange

# Handle both relative and absolute imports
try:
    from .utils import AttentionRollout, extract_patch_embeddings
except ImportError:
    from utils import AttentionRollout, extract_patch_embeddings


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention mechanism."""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, 
                 attn_drop: float = 0., proj_drop: float = 0.):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass for cross attention.
        
        Args:
            query: Query tensor [batch, num_queries, dim]
            key: Key tensor [batch, num_keys, dim]
            value: Value tensor [batch, num_values, dim]
            return_attention: Whether to return attention weights
        
        Returns:
            output: Cross attention output [batch, num_queries, dim]
            attention: Optional attention weights [batch, num_heads, num_queries, num_keys]
        """
        B, N_q, C = query.shape
        B, N_k, C = key.shape
        B, N_v, C = value.shape
        
        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_v, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N_q, N_k]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)  # [B, N_q, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class GlobalLocalCrossAttention(nn.Module):
    """Global-Local Cross Attention (GLCA) module."""
    
    def __init__(self, dim: int, num_heads: int = 8, local_query_ratio: float = 0.3,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            local_query_ratio: Ratio of patches to select as local queries (R)
            qkv_bias: Whether to use bias in projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.dim = dim
        self.local_query_ratio = local_query_ratio
        
        # Attention rollout for selecting high-response regions
        self.attention_rollout = AttentionRollout(
            local_query_ratio=local_query_ratio,
            exclude_cls=True
        )
        
        # Cross attention mechanism
        self.cross_attention = MultiHeadCrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, attentions: List[torch.Tensor], 
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass for GLCA.
        
        Args:
            x: Input embeddings [batch, tokens, dim] (includes CLS token)
            attentions: List of attention matrices from previous layers
            return_attention: Whether to return attention weights
        
        Returns:
            output: GLCA output [batch, tokens, dim]
            cross_attn: Optional cross attention weights
        """
        # Get high-response regions using attention rollout
        selected_indices, rollout = self.attention_rollout(attentions)
        
        # Extract local queries (high-response patch embeddings)
        local_queries = extract_patch_embeddings(x, selected_indices)  # [B, num_selected, dim]
        
        # Global key-value pairs (all tokens including CLS)
        global_keys = x  # [B, tokens, dim]
        global_values = x  # [B, tokens, dim]
        
        # Apply layer norm
        local_queries = self.norm(local_queries)
        global_keys = self.norm(global_keys)
        global_values = self.norm(global_values)
        
        # Compute cross attention between local queries and global key-values
        if return_attention:
            cross_output, cross_attn = self.cross_attention(
                local_queries, global_keys, global_values, return_attention=True
            )
        else:
            cross_output = self.cross_attention(local_queries, global_keys, global_values)
            cross_attn = None
        
        # Scatter cross attention output back to original positions
        output = self._scatter_local_features(x, cross_output, selected_indices)
        
        if return_attention:
            return output, cross_attn
        return output
    
    def _scatter_local_features(self, x: torch.Tensor, local_features: torch.Tensor, 
                               indices: torch.Tensor) -> torch.Tensor:
        """
        Scatter local features back to their original positions.
        
        Args:
            x: Original embeddings [batch, tokens, dim]
            local_features: Local feature updates [batch, num_selected, dim]
            indices: Selected patch indices [batch, num_selected]
        
        Returns:
            updated_x: Updated embeddings [batch, tokens, dim]
        """
        batch_size, num_selected, dim = local_features.shape
        output = x.clone()
        
        # Create batch indices for advanced indexing
        batch_indices = torch.arange(batch_size, device=indices.device).unsqueeze(1).expand(-1, num_selected)
        
        # Update selected positions with local features
        output[batch_indices, indices] = local_features
        
        return output


class PairwiseCrossAttention(nn.Module):
    """Pair-wise Cross Attention (PWCA) module for training regularization."""
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0., proj_drop: float = 0.):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in projections
            attn_drop: Attention dropout rate
            proj_drop: Projection dropout rate
        """
        super().__init__()
        self.dim = dim
        
        # Cross attention mechanism
        self.cross_attention = MultiHeadCrossAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, primary: torch.Tensor, secondary: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass for PWCA.
        
        Args:
            primary: Primary image embeddings [batch, tokens, dim]
            secondary: Secondary image embeddings [batch, tokens, dim]
            return_attention: Whether to return attention weights
        
        Returns:
            output: PWCA output [batch, tokens, dim]
            cross_attn: Optional cross attention weights
        """
        # Query from primary image
        queries = self.norm(primary)  # [B, tokens, dim]
        
        # Concatenate key-value pairs from both images
        combined_keys = torch.cat([primary, secondary], dim=1)  # [B, 2*tokens, dim]
        combined_values = torch.cat([primary, secondary], dim=1)  # [B, 2*tokens, dim]
        
        combined_keys = self.norm(combined_keys)
        combined_values = self.norm(combined_values)
        
        # Compute cross attention
        if return_attention:
            output, cross_attn = self.cross_attention(
                queries, combined_keys, combined_values, return_attention=True
            )
        else:
            output = self.cross_attention(queries, combined_keys, combined_values)
            cross_attn = None
        
        if return_attention:
            return output, cross_attn
        return output


class DCALBlock(nn.Module):
    """DCAL Transformer block with SA, GLCA, and PWCA."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.,
                 use_glca: bool = False, use_pwca: bool = False,
                 local_query_ratio: float = 0.3):
        """
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            qkv_bias: Whether to use bias in projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            use_glca: Whether to use GLCA
            use_pwca: Whether to use PWCA (training only)
            local_query_ratio: Ratio for local query selection
        """
        super().__init__()
        self.use_glca = use_glca
        self.use_pwca = use_pwca
        
        # Self attention (always present)
        try:
            from .backbone import MultiHeadAttention, MLP
        except ImportError:
            from backbone import MultiHeadAttention, MLP
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # GLCA (optional)
        if use_glca:
            self.norm_glca = nn.LayerNorm(dim)
            self.glca = GlobalLocalCrossAttention(
                dim=dim, num_heads=num_heads, local_query_ratio=local_query_ratio,
                qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
            )
        
        # PWCA (optional, training only)
        if use_pwca:
            self.norm_pwca = nn.LayerNorm(dim)
            self.pwca = PairwiseCrossAttention(
                dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop
            )
    
    def forward(self, x: torch.Tensor, secondary_x: Optional[torch.Tensor] = None,
                sa_attentions: Optional[List[torch.Tensor]] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for DCAL block.
        
        Args:
            x: Input embeddings [batch, tokens, dim]
            secondary_x: Secondary image embeddings for PWCA [batch, tokens, dim]
            sa_attentions: Previous self-attention matrices for GLCA
            return_attention: Whether to return attention weights
        
        Returns:
            sa_output: Self-attention output
            glca_output: GLCA output (if enabled)
            pwca_output: PWCA output (if enabled)
        """
        # Self-attention branch
        if return_attention:
            sa_residual, sa_attn = self.self_attn(self.norm1(x), return_attention=True)
        else:
            sa_residual = self.self_attn(self.norm1(x))
            sa_attn = None
        
        sa_output = x + sa_residual
        sa_output = sa_output + self.mlp(self.norm2(sa_output))
        
        # GLCA branch (if enabled)
        glca_output = None
        if self.use_glca and sa_attentions is not None:
            # Use attention from SA branch for rollout
            current_attentions = sa_attentions + [sa_attn] if sa_attn is not None else sa_attentions
            glca_residual = self.glca(self.norm_glca(x), current_attentions)
            glca_output = x + glca_residual
            glca_output = glca_output + self.mlp(self.norm2(glca_output))
        
        # PWCA branch (if enabled and secondary input provided)
        pwca_output = None
        if self.use_pwca and secondary_x is not None and self.training:
            pwca_residual = self.pwca(self.norm_pwca(x), secondary_x)
            pwca_output = x + pwca_residual
            pwca_output = pwca_output + self.mlp(self.norm2(pwca_output))
        
        return sa_output, glca_output, pwca_output
