"""
Transformer block with (sliding window) multi-head self-attention
for Temporal Action Localization.

Implements the standard Transformer encoder block as used in
ActionFormer (Zhang et al., 2022), with optional sliding window
attention to reduce quadratic complexity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMHA(nn.Module):
    """
    Multi-Head Attention with optional sliding window and masking support.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        attn_dropout: Attention dropout rate
        proj_dropout: Output projection dropout rate
        use_window: Whether to use sliding window attention
        window_size: Size of the sliding window (must be odd for symmetric)
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_window: bool = False,
        window_size: int = 19,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = math.sqrt(self.head_dim)
        
        self.use_window = use_window
        self.window_size = window_size
        
        # QKV projection
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.out_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    def _build_window_mask(self, seq_len, device):
        """Build a sliding window attention mask."""
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # Distance matrix
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Window mask: True where within window
        half_window = self.window_size // 2
        window_mask = dist <= half_window
        return window_mask  # (T, T)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T)
            mask: (B, 1, T) - validity mask
        Returns:
            out: (B, C, T)
        """
        B, C, T = x.shape
        
        # Transpose to (B, T, C)
        x_t = x.permute(0, 2, 1)
        
        # Compute Q, K, V
        qkv = self.qkv(x_t)  # (B, T, 3C)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Apply sliding window mask
        if self.use_window:
            window_mask = self._build_window_mask(T, x.device)  # (T, T)
            attn = attn.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask
        if mask is not None:
            # mask: (B, 1, T) -> (B, 1, 1, T) for key masking
            key_mask = mask.unsqueeze(2)  # (B, 1, 1, T)
            attn = attn.masked_fill(key_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Handle NaN from softmax when entire row is -inf
        attn = attn.nan_to_num(0.0)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        
        # Output projection
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        # Back to (B, C, T)
        out = out.permute(0, 2, 1)
        
        return out


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block: LayerNorm -> MHA -> LayerNorm -> MLP.
    Uses pre-norm architecture with residual connections.
    
    Args:
        n_embd: Embedding dimension
        n_head: Number of attention heads
        dropout: Dropout rate
        attn_dropout: Attention-specific dropout
        use_window: Use sliding window attention
        window_size: Sliding window size
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_window: bool = True,
        window_size: int = 19,
    ):
        super().__init__()
        
        # Pre-norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # Multi-head attention
        self.attn = MaskedMHA(
            n_embd=n_embd,
            n_head=n_head,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_window=use_window,
            window_size=window_size,
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, C, T)
            mask: (B, 1, T)
        Returns:
            out: (B, C, T)
        """
        B, C, T = x.shape
        
        # Self-attention with residual
        x_ln = self._apply_ln(self.ln1, x)
        x = x + self.attn(x_ln, mask)
        
        # MLP with residual
        x_ln = self._apply_ln(self.ln2, x)
        x_t = x_ln.permute(0, 2, 1)  # (B, T, C) for MLP
        mlp_out = self.mlp(x_t).permute(0, 2, 1)  # back to (B, C, T)
        x = x + mlp_out
        
        # Apply mask
        if mask is not None:
            x = x * mask
        
        return x
    
    def _apply_ln(self, ln, x):
        """Apply LayerNorm to (B, C, T) tensor."""
        return ln(x.permute(0, 2, 1)).permute(0, 2, 1)
