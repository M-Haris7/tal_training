"""
Hybrid Blocks for Temporal Action Localization.

Implements two hybrid architectures combining BiMamba with Transformer:

1. ParallelHybridBlock: BiMamba and Attention run in parallel on the same input,
   outputs are fused with learnable weights. Novel combination not explored in
   the base paper (Zhang et al., 2025 VISIGRAPP).

2. SequentialHybridBlock: MambaFormer-style (Park et al., 2024) adapted for TAL.
   BiMamba replaces positional encoding + MLP in the Transformer block.
   
Both blocks support masking and downsampling for the multi-scale pyramid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    MaskedConv1D, MaskedMHCA, MaskedMHA, LocalMaskedMHCA,
    LayerNorm, AffineDropPath
)
from .mamba_simple import BiMambaBlock, MambaBlock


class ParallelHybridBlock(nn.Module):
    """
    Parallel BiMamba + Sliding Window Attention Block.
    
    Architecture:
        Input x ──┬──► BiMamba Branch ──► y_mamba
                  │
                  └──► Attention Branch ──► y_attn
                  
        Fusion: α * y_mamba + (1-α) * y_attn  (learnable per-channel weight)
        └──► LayerNorm ──► MLP ──► Output
    
    This combination (ViM-BiMamba ∥ Sliding Window Attention) is the key novel
    contribution not tested in the base paper.
    """
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        use_mamba_cuda=True,
    ):
        super().__init__()

        # ---- Attention Branch ----
        self.ln_attn = LayerNorm(n_embd)
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd, n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHCA(
                n_embd, n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        # ---- BiMamba Branch ----
        self.bimamba = BiMambaBlock(
            d_model=n_embd,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            use_cuda=use_mamba_cuda,
        )

        # ---- Fusion: learnable per-channel weight ----
        # alpha controls how much Mamba vs Attention contributes
        # Initialized to 0.5 (equal contribution)
        self.fusion_alpha = nn.Parameter(torch.ones(1, n_embd, 1) * 0.5)

        # ---- Skip connection handling for downsampling ----
        if n_ds_strides[0] > 1:
            kernel_size = n_ds_strides[0] + 1
            stride = n_ds_strides[0]
            padding = (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
            # Downsample for mamba branch output
            self.pool_mamba = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()
            self.pool_mamba = nn.Identity()

        # ---- MLP (FFN) ----
        if n_out is None:
            n_out = n_embd
        if n_hidden is None:
            n_hidden = 4 * n_embd

        self.ln_mlp = LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # ---- Drop path ----
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        """
        Args:
            x: (B, C, T) input features
            mask: (B, 1, T) boolean mask
            pos_embd: optional position embedding (not used by default)
        Returns:
            out: (B, C, T') output features (T' may differ if downsampled)
            out_mask: (B, 1, T') output mask
        """
        # ---- Attention branch ----
        attn_out, out_mask = self.attn(self.ln_attn(x), mask)
        out_mask_float = out_mask.to(attn_out.dtype)

        # ---- BiMamba branch ----
        mamba_out, _ = self.bimamba(x, mask)  # (B, C, T)
        # Downsample mamba output if needed to match attention output
        if isinstance(self.pool_mamba, nn.MaxPool1d):
            mamba_out = self.pool_mamba(mamba_out)

        # ---- Learnable fusion ----
        alpha = torch.sigmoid(self.fusion_alpha)  # (0, 1)
        fused = alpha * mamba_out * out_mask_float + (1 - alpha) * attn_out

        # ---- Residual + drop path ----
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(fused)

        # ---- MLP ----
        out = out + self.drop_path_mlp(self.mlp(self.ln_mlp(out)) * out_mask_float)

        if pos_embd is not None:
            out += pos_embd * out_mask_float

        return out, out_mask


class SequentialHybridBlock(nn.Module):
    """
    Sequential BiMamba + Attention Block (MambaFormer-style).
    
    Architecture (adapted from Park et al., 2024):
        Input x ──► BiMamba₁ ──► (+residual) ──► Attention ──► (+residual) ──► BiMamba₂ ──► (+residual) ──► Output
    
    The BiMamba blocks replace the positional encoding and MLP in the
    standard Transformer block, following the MambaFormer design.
    """
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        use_mamba_cuda=True,
    ):
        super().__init__()

        if n_out is None:
            n_out = n_embd

        # ---- BiMamba₁ (replaces positional encoding) ----
        self.bimamba1 = BiMambaBlock(
            d_model=n_embd,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            use_cuda=use_mamba_cuda,
        )

        # ---- Attention layer ----
        self.ln_attn = LayerNorm(n_embd)
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd, n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHCA(
                n_embd, n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

        # ---- BiMamba₂ (replaces MLP) ----
        self.bimamba2 = BiMambaBlock(
            d_model=n_embd,
            d_state=d_state,
            d_conv=d_conv,
            expand=mamba_expand,
            use_cuda=use_mamba_cuda,
        )

        # ---- Skip connection handling ----
        if n_ds_strides[0] > 1:
            kernel_size = n_ds_strides[0] + 1
            stride = n_ds_strides[0]
            padding = (n_ds_strides[0] + 1) // 2
            self.pool_skip = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # ---- Drop path ----
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mamba = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mamba = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        """
        Args:
            x: (B, C, T) input features
            mask: (B, 1, T) boolean mask
        Returns:
            out: (B, C, T') output features
            out_mask: (B, 1, T') output mask
        """
        # ---- Step 1: BiMamba₁ (position-aware processing) ----
        x_mamba, _ = self.bimamba1(x, mask)  # residual inside BiMambaBlock

        # ---- Step 2: Attention ----
        attn_out, out_mask = self.attn(self.ln_attn(x_mamba), mask)
        out_mask_float = out_mask.to(attn_out.dtype)

        out = self.pool_skip(x_mamba) * out_mask_float + self.drop_path_attn(attn_out)

        # ---- Step 3: BiMamba₂ (replaces MLP) ----
        # Need to handle potential dimension change from downsampling
        mamba2_out, out_mask = self.bimamba2(out, out_mask)
        out = out + self.drop_path_mamba(mamba2_out - out)  # extract residual from bimamba2

        if pos_embd is not None:
            out += pos_embd * out_mask_float

        return out, out_mask
