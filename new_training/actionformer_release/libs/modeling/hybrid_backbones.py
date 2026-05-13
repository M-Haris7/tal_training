"""
Hybrid Backbones for Temporal Action Localization.

Registers two new backbone types with ActionFormer:
  - 'biMambaParallel': Parallel BiMamba + Sliding Window Attention
  - 'biMambaSequential': Sequential BiMamba → Attention → BiMamba (MambaFormer-style)

Both follow the same multi-scale pyramid design as the original
ConvTransformerBackbone, with identical embedding and output structure.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone
from .blocks import (
    get_sinusoid_encoding, MaskedConv1D, LayerNorm, TransformerBlock
)
from .hybrid_blocks import ParallelHybridBlock, SequentialHybridBlock


@register_backbone("biMambaParallel")
class BiMambaParallelBackbone(nn.Module):
    """
    Multi-scale backbone with Parallel BiMamba + Attention blocks.
    
    Novel architecture: At each pyramid level, features are processed
    simultaneously by a BiMamba branch (for causal/sequential modeling)
    and a sliding window attention branch (for context aggregation),
    with learnable per-level fusion weights.
    """
    def __init__(
        self,
        n_in,
        n_embd,
        n_head,
        n_embd_ks,
        max_len,
        arch=(2, 2, 5),
        mha_win_size=[-1]*6,
        scale_factor=2,
        with_ln=False,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_rel_pe=False,
        # BiMamba-specific params
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        use_mamba_cuda=True,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # Feature projection (same as ActionFormer)
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # Embedding network (same as ActionFormer)
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in_cur = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in_cur, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # Position embedding
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # Stem: use standard Transformer blocks (no downsampling)
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                )
            )

        # Branch: Parallel Hybrid blocks with downsampling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                ParallelHybridBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    use_mamba_cuda=use_mamba_cuda,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        B, C, T = x.size()

        # Feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                 for proj, s in zip(self.proj, x.split(self.n_in, dim=1))],
                dim=1
            )

        # Embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # Position embedding
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # Stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # Outputs
        out_feats = (x,)
        out_masks = (mask,)

        # Branch with hybrid blocks + downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


@register_backbone("biMambaSequential")
class BiMambaSequentialBackbone(nn.Module):
    """
    Multi-scale backbone with Sequential BiMamba + Attention blocks.
    
    MambaFormer-style: BiMamba → Attention → BiMamba at each level.
    The first BiMamba captures sequential/causal patterns (replaces 
    positional encoding), attention captures global context, and the
    second BiMamba replaces the MLP for non-linear transformation.
    """
    def __init__(
        self,
        n_in,
        n_embd,
        n_head,
        n_embd_ks,
        max_len,
        arch=(2, 2, 5),
        mha_win_size=[-1]*6,
        scale_factor=2,
        with_ln=False,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        use_abs_pe=False,
        use_rel_pe=False,
        # BiMamba-specific params
        d_state=16,
        d_conv=4,
        mamba_expand=2,
        use_mamba_cuda=True,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # Feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # Embedding network
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in_cur = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in_cur, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # Position embedding
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # Stem: standard Transformer blocks
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                )
            )

        # Branch: Sequential Hybrid blocks with downsampling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                SequentialHybridBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    d_state=d_state,
                    d_conv=d_conv,
                    mamba_expand=mamba_expand,
                    use_mamba_cuda=use_mamba_cuda,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        B, C, T = x.size()

        # Feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0]
                 for proj, s in zip(self.proj, x.split(self.n_in, dim=1))],
                dim=1
            )

        # Embedding
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # Position embedding
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # Stem
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # Outputs
        out_feats = (x,)
        out_masks = (mask,)

        # Branch
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
