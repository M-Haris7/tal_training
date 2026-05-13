"""
Hybrid BiMamba-Transformer Temporal Feature Encoder.

This module implements a NOVEL hybrid architecture that combines bidirectional
Mamba scanning with Transformer self-attention in a parallel-gated design.

Key differences from architectures tested in the paper:
- MambaFormer (paper): Sequential Mamba -> Attn -> Mamba (replaces positional
  encoding and MLP in Transformer with Mamba blocks)
- CausalTAD (paper): Parallel causal attention + DBM with shared weights
  
Our BiMambaFormer: Parallel BiMamba scanner + Windowed Transformer with 
learned gated fusion. Both branches process the input independently, and a
gating mechanism dynamically weights each branch's contribution per timestep.

This design allows:
1. BiMamba captures bidirectional local-to-global temporal patterns efficiently
2. Transformer captures pairwise relationships within attention windows
3. Gated fusion learns when to rely on each branch per-timestep

The paper notes (Section 4.9) that they "focused on a limited set of hybrid
models" and suggests exploring alternative hybrid designs as future work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bimamba import BiMambaBlock, MambaOriginalBlock
from .transformer import TransformerBlock, MaskedMHA


class GatedFusion(nn.Module):
    """
    Learnable gated fusion for combining two feature streams.
    Produces a per-channel, per-timestep gate to weight contributions
    from the BiMamba branch and the Transformer branch.
    
    gate = sigmoid(W_g * [x_mamba || x_transformer] + b_g)
    output = gate * x_mamba + (1 - gate) * x_transformer
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x_mamba, x_transformer):
        """
        Args:
            x_mamba: (B, C, T) - BiMamba branch output
            x_transformer: (B, C, T) - Transformer branch output
        Returns:
            fused: (B, C, T)
        """
        # Transpose to (B, T, C) for linear layers
        xm = x_mamba.permute(0, 2, 1)
        xt = x_transformer.permute(0, 2, 1)
        
        # Concatenate and compute gate
        combined = torch.cat([xm, xt], dim=-1)  # (B, T, 2C)
        gate = self.gate_proj(combined)  # (B, T, C), values in [0, 1]
        
        # Gated combination
        fused = gate * xm + (1.0 - gate) * xt  # (B, T, C)
        fused = self.out_proj(fused)  # (B, T, C)
        
        # Back to (B, C, T)
        return fused.permute(0, 2, 1)


class AdditivesFusion(nn.Module):
    """Simple additive fusion with learned scaling."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x_mamba, x_transformer):
        alpha = torch.sigmoid(self.alpha)
        xm = x_mamba.permute(0, 2, 1)
        xt = x_transformer.permute(0, 2, 1)
        fused = alpha * xm + (1.0 - alpha) * xt
        fused = self.out_proj(fused)
        return fused.permute(0, 2, 1)


class ConcatFusion(nn.Module):
    """Concatenation-based fusion with projection."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x_mamba, x_transformer):
        xm = x_mamba.permute(0, 2, 1)
        xt = x_transformer.permute(0, 2, 1)
        fused = torch.cat([xm, xt], dim=-1)
        fused = self.proj(fused)
        return fused.permute(0, 2, 1)


class BiMambaTransformerBlock(nn.Module):
    """
    A single block of the hybrid BiMamba-Transformer encoder.
    
    Architecture:
        Input -> [BiMamba Branch]  -> |
                                      | -> Gated Fusion -> Output
        Input -> [Transformer Branch] -> |
    
    Both branches receive the same input and process it independently.
    The gated fusion dynamically combines their outputs.
    
    Args:
        d_model: Model dimension
        n_head: Number of attention heads for Transformer branch
        d_state: SSM state dimension for Mamba
        d_conv: Local conv width for Mamba
        expand: Mamba expansion factor
        dropout: Dropout rate
        attn_dropout: Attention dropout
        use_window: Use sliding window attention
        window_size: Sliding window size
        fusion_type: Type of fusion ("gated", "add", "concat")
    """
    
    def __init__(
        self,
        d_model: int,
        n_head: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_window: bool = True,
        window_size: int = 19,
        fusion_type: str = "gated",
    ):
        super().__init__()
        
        # BiMamba branch (bidirectional scanning)
        self.bimamba = BiMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # Transformer branch (windowed self-attention + MLP)
        self.transformer = TransformerBlock(
            n_embd=d_model,
            n_head=n_head,
            dropout=dropout,
            attn_dropout=attn_dropout,
            use_window=use_window,
            window_size=window_size,
        )
        
        # Fusion mechanism
        if fusion_type == "gated":
            self.fusion = GatedFusion(d_model)
        elif fusion_type == "add":
            self.fusion = AdditivesFusion(d_model)
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(d_model)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Post-fusion layer norm
        self.post_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T) - input features
            mask: (B, 1, T) - validity mask
        Returns:
            out: (B, C, T)
        """
        # Process through both branches independently
        x_bimamba = self.bimamba(x, mask)
        x_transformer = self.transformer(x, mask)
        
        # Fuse outputs
        fused = self.fusion(x_bimamba, x_transformer)
        
        # Post-norm (applied in channel-last format)
        fused = self.post_norm(fused.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Residual from input
        out = x + fused
        
        if mask is not None:
            out = out * mask
        
        return out


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale temporal feature encoder using hybrid BiMamba-Transformer blocks.
    
    This follows the multi-scale design from ActionFormer (Zhang et al., 2022),
    where the temporal resolution is downsampled at each level to enable
    detection of actions at different scales.
    
    Architecture:
        Level 0: Full resolution -> BiMambaTransformerBlock -> Downsample
        Level 1: 1/2 resolution -> BiMambaTransformerBlock -> Downsample
        ...
        Level L-1: 1/2^(L-1) resolution -> BiMambaTransformerBlock
    
    Args:
        d_model: Model dimension
        num_levels: Number of multi-scale levels
        scale_factor: Downsample factor between levels
        n_head: Attention heads
        d_state: Mamba SSM state dim
        d_conv: Mamba conv width
        expand: Mamba expansion
        dropout: Dropout
        attn_dropout: Attention dropout
        use_window: Sliding window attention
        window_size: Window size
        fusion_type: Fusion type for hybrid blocks
        n_blocks_per_level: Number of encoder blocks per level
    """
    
    def __init__(
        self,
        d_model: int,
        num_levels: int = 6,
        scale_factor: int = 2,
        n_head: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_window: bool = True,
        window_size: int = 19,
        fusion_type: str = "gated",
        n_blocks_per_level: int = 1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        
        # Encoder blocks for each level
        self.encoder_blocks = nn.ModuleList()
        for _ in range(num_levels):
            level_blocks = nn.ModuleList()
            for _ in range(n_blocks_per_level):
                level_blocks.append(
                    BiMambaTransformerBlock(
                        d_model=d_model,
                        n_head=n_head,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        use_window=use_window,
                        window_size=window_size,
                        fusion_type=fusion_type,
                    )
                )
            self.encoder_blocks.append(level_blocks)
        
        # Downsampling layers between levels (strided convolution)
        self.downsample = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.downsample.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=scale_factor,
                              stride=scale_factor, padding=0),
                    nn.LayerNorm(d_model),  # Applied after permute
                )
            )
    
    def forward(self, x, mask):
        """
        Args:
            x: (B, C, T) - embedded features
            mask: (B, 1, T) - validity mask
        Returns:
            feat_list: List of (B, C, T_l) for each level l
            mask_list: List of (B, 1, T_l) for each level l
        """
        feat_list = []
        mask_list = []
        
        current_feat = x
        current_mask = mask
        
        for level in range(self.num_levels):
            # Apply encoder blocks at this level
            for block in self.encoder_blocks[level]:
                current_feat = block(current_feat, current_mask)
            
            # Store output for this level
            feat_list.append(current_feat)
            mask_list.append(current_mask)
            
            # Downsample for next level (except last)
            if level < self.num_levels - 1:
                ds = self.downsample[level]
                # Apply conv1d
                current_feat = ds[0](current_feat)
                # Apply layer norm (needs B, T, C format)
                B, C, T_new = current_feat.shape
                current_feat = ds[1](current_feat.permute(0, 2, 1)).permute(0, 2, 1)
                
                # Downsample mask using max pooling
                current_mask = F.max_pool1d(
                    current_mask.float(),
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                    padding=0,
                ).bool().float()
                
                # Ensure mask shape matches
                if current_mask.shape[-1] != current_feat.shape[-1]:
                    current_mask = current_mask[:, :, :current_feat.shape[-1]]
        
        return feat_list, mask_list
