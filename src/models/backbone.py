import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self attention."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class MLP(nn.Module):
    """MLP with GELU activation."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
    
    def forward(self, x, return_attention=False):
        if return_attention:
            y, attn = self.attn(self.norm1(x), return_attention=True)
            x = x + y
            x = x + self.mlp(self.norm2(x))
            return x, attn
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x, return_all_tokens=False, return_attention=False):
        """Forward pass through feature extraction layers."""
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        attentions = []
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attentions.append(attn)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        if return_all_tokens:
            if return_attention:
                return x, attentions
            return x
        else:
            if return_attention:
                return x[:, 0], attentions  # return cls token
            return x[:, 0]  # return cls token
    
    def forward(self, x):
        """Forward pass."""
        x = self.forward_features(x)
        x = self.pre_logits(x)
        x = self.head(x)
        return x


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    """DeiT Tiny model."""
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, **kwargs
    )
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    """DeiT Small model."""
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, **kwargs
    )
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    """DeiT Base model."""
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, **kwargs
    )
    return model
