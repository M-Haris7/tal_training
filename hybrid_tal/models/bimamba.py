"""
Bidirectional Mamba (BiMamba) Scanner Block for Temporal Action Localization.

This implements a bidirectional Mamba block inspired by Vision Mamba (ViM)
(Zhu et al., 2024), adapted for 1D temporal sequences. The block processes
input in both forward and backward directions to capture full temporal context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    raise ImportError(
        "mamba-ssm is required. Install with: pip install mamba-ssm==1.1.1"
    )


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block that processes temporal sequences in both
    forward and backward directions, similar to ViM (Zhu et al., 2024).
    
    Unlike the paper's ViM which shares input projections, this implementation
    uses separate Mamba instances for forward and backward to allow each
    direction to learn independent representations.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Block expansion factor (default: 2)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Layer norm before Mamba (pre-norm architecture)
        self.norm = nn.LayerNorm(d_model)
        
        # Forward Mamba
        self.mamba_forward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Backward Mamba (separate instance for independent learning)
        self.mamba_backward = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Fusion projection: combine forward + backward
        self.out_proj = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T) - batch, channels, time
            mask: (B, 1, T) - validity mask (1 = valid, 0 = padded)
        Returns:
            out: (B, C, T)
        """
        B, C, T = x.shape
        
        # Transpose to (B, T, C) for Mamba
        x_t = x.permute(0, 2, 1).contiguous()
        
        # Apply mask if provided (zero out padded positions)
        if mask is not None:
            mask_t = mask.squeeze(1).unsqueeze(-1)  # (B, T, 1)
            x_t = x_t * mask_t
        
        # Pre-norm
        x_normed = self.norm(x_t)
        
        # Forward pass
        y_forward = self.mamba_forward(x_normed)
        
        # Backward pass: flip, process, flip back
        x_flipped = torch.flip(x_normed, dims=[1])
        y_backward = self.mamba_backward(x_flipped)
        y_backward = torch.flip(y_backward, dims=[1])
        
        # Concatenate and project
        y_combined = torch.cat([y_forward, y_backward], dim=-1)  # (B, T, 2C)
        y_out = self.out_proj(y_combined)  # (B, T, C)
        
        # Residual connection + dropout
        out = x_t + self.dropout(y_out)
        
        # Back to (B, C, T)
        out = out.permute(0, 2, 1).contiguous()
        
        # Re-apply mask
        if mask is not None:
            out = out * mask
        
        return out


class MambaOriginalBlock(nn.Module):
    """
    Original (unidirectional) Mamba block as described in Gu & Dao (2023).
    Kept for ablation studies and comparison.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T)
            mask: (B, 1, T)
        Returns:
            out: (B, C, T)
        """
        x_t = x.permute(0, 2, 1).contiguous()  # (B, T, C)
        
        if mask is not None:
            mask_t = mask.squeeze(1).unsqueeze(-1)
            x_t = x_t * mask_t
        
        x_normed = self.norm(x_t)
        y = self.mamba(x_normed)
        out = x_t + self.dropout(y)
        out = out.permute(0, 2, 1).contiguous()
        
        if mask is not None:
            out = out * mask
        
        return out
