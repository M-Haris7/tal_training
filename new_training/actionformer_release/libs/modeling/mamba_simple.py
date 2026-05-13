"""
BiMamba Block Implementation for Temporal Action Localization.

Implements:
  - SelectiveSSM: Core selective state space model operation
  - MambaBlock: Original unidirectional Mamba block (Gu & Dao, 2023)
  - BiMambaBlock: ViM-style bidirectional Mamba block (Zhu et al., 2024)

Uses mamba-ssm CUDA kernels if available, falls back to pure PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import optimized CUDA kernels
try:
    from mamba_ssm import Mamba as MambaCUDA
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False
    print("[INFO] mamba-ssm not found, using pure PyTorch SSM implementation.")


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - pure PyTorch implementation.
    Implements: h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C * h_t + D * x_t
    where A, B, C are input-dependent (selective).
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection: x -> (x_proj, z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Conv1D for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters projection
        # Projects x to dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A as a diagonal matrix (log-space for stability)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) input tensor
        Returns:
            y: (B, T, D) output tensor
        """
        B, T, D = x.shape

        # Input projection
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Conv1D (causal)
        x_conv = x_proj.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # truncate to original length
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, T, d_inner)

        # SSM parameter projection
        x_ssm = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        dt, B_param, C_param = torch.split(
            x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt projection + softplus
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)

        # Discretize A
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Run selective scan
        y = self._selective_scan(x_conv, dt, A, B_param, C_param, self.D)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)
        return out

    def _selective_scan(self, x, dt, A, B, C, D):
        """
        Selective scan implementation.
        Args:
            x: (B, T, d_inner) - input after conv
            dt: (B, T, d_inner) - time step
            A: (d_inner, d_state) - state transition (negative)
            B: (B, T, d_state) - input matrix
            C: (B, T, d_state) - output matrix
            D: (d_inner,) - skip connection
        Returns:
            y: (B, T, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B using dt
        # dA = exp(dt * A) -> (B, T, d_inner, d_state)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, T, d_inner, d_state)
        dA = torch.exp(dt_A)

        # dB = dt * B -> (B, T, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)

        # Add skip connection
        y = y + x * D.unsqueeze(0).unsqueeze(0)

        return y


class MambaBlock(nn.Module):
    """
    Original Mamba block for 1D temporal sequences.
    Input/Output format: (B, C, T) to match ActionFormer conventions.
    Supports masking for variable-length sequences.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_cuda=True):
        super().__init__()
        self.d_model = d_model
        self.use_cuda_impl = use_cuda and HAS_MAMBA_CUDA

        if self.use_cuda_impl:
            self.mamba = MambaCUDA(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, T) - feature tensor
            mask: (B, 1, T) - boolean mask
        Returns:
            out: (B, C, T)
            mask: (B, 1, T)
        """
        B, C, T = x.shape

        # Convert to (B, T, C) for Mamba processing
        x_t = x.permute(0, 2, 1)  # (B, T, C)

        # Layer norm
        x_normed = self.norm(x_t)

        # Apply mask before Mamba
        mask_float = mask.permute(0, 2, 1).to(x_normed.dtype)  # (B, T, 1)
        x_normed = x_normed * mask_float

        # Mamba forward
        if self.use_cuda_impl:
            out_t = self.mamba(x_normed)
        else:
            out_t = self.ssm(x_normed)

        # Apply mask to output
        out_t = out_t * mask_float

        # Residual connection
        out_t = out_t + x_t

        # Convert back to (B, C, T)
        out = out_t.permute(0, 2, 1)
        return out, mask


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba (ViM-style) block for TAL.
    Processes the input in both forward and backward directions.
    
    Architecture (ViM style):
        x -> Proj_x -> Conv1D_fwd -> SSM_fwd -> x'_fwd
                    -> Conv1D_bwd -> SSM_bwd -> x'_bwd
        x -> Proj_z -> SiLU(z)
        out = Proj_out(SiLU(z) * (x'_fwd + x'_bwd))
    
    Input/Output format: (B, C, T) to match ActionFormer conventions.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, use_cuda=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.use_cuda_impl = use_cuda and HAS_MAMBA_CUDA
        self.norm = nn.LayerNorm(d_model)

        if self.use_cuda_impl:
            # Use two separate CUDA Mamba instances
            self.mamba_fwd = MambaCUDA(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.mamba_bwd = MambaCUDA(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Pure PyTorch: shared projections, separate SSM directions
            # Input projections (shared between forward and backward)
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

            # Forward branch
            self.conv1d_fwd = nn.Conv1d(
                self.d_inner, self.d_inner, d_conv,
                padding=d_conv - 1, groups=self.d_inner, bias=True,
            )
            self.x_proj_fwd = nn.Linear(
                self.d_inner, math.ceil(d_model / 16) + d_state * 2, bias=False
            )
            self.dt_proj_fwd = nn.Linear(
                math.ceil(d_model / 16), self.d_inner, bias=True
            )

            # Backward branch
            self.conv1d_bwd = nn.Conv1d(
                self.d_inner, self.d_inner, d_conv,
                padding=d_conv - 1, groups=self.d_inner, bias=True,
            )
            self.x_proj_bwd = nn.Linear(
                self.d_inner, math.ceil(d_model / 16) + d_state * 2, bias=False
            )
            self.dt_proj_bwd = nn.Linear(
                math.ceil(d_model / 16), self.d_inner, bias=True
            )

            # Shared A and D
            A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

            # Output projection
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_branch(self, x_conv, x_proj_layer, dt_proj_layer, reverse=False):
        """Run one direction of the SSM."""
        B, T, d_inner = x_conv.shape
        dt_rank = math.ceil(self.d_model / 16)

        if reverse:
            x_conv = x_conv.flip(1)

        x_ssm = x_proj_layer(x_conv)
        dt, B_param, C_param = torch.split(
            x_ssm, [dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(dt_proj_layer(dt))
        A = -torch.exp(self.A_log)

        # Selective scan
        batch_size, seq_len, _ = x_conv.shape
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        dA = torch.exp(dt_A)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)

        h = torch.zeros(batch_size, d_inner, self.d_state,
                        device=x_conv.device, dtype=x_conv.dtype)
        ys = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)

        if reverse:
            y = y.flip(1)

        return y

    def forward(self, x, mask):
        """
        Args:
            x: (B, C, T) - feature tensor
            mask: (B, 1, T) - boolean mask
        Returns:
            out: (B, C, T)
            mask: (B, 1, T)
        """
        B, C, T = x.shape

        # Convert to (B, T, C)
        x_t = x.permute(0, 2, 1)
        x_normed = self.norm(x_t)

        mask_float = mask.permute(0, 2, 1).to(x_normed.dtype)  # (B, T, 1)
        x_normed = x_normed * mask_float

        if self.use_cuda_impl:
            # CUDA path: use two Mamba instances
            y_fwd = self.mamba_fwd(x_normed)
            y_bwd = self.mamba_bwd(x_normed.flip(1)).flip(1)
            out_t = (y_fwd + y_bwd) * mask_float
        else:
            # Pure PyTorch path: ViM-style
            xz = self.in_proj(x_normed)
            x_proj, z = xz.chunk(2, dim=-1)

            # Forward branch
            x_conv_fwd = self.conv1d_fwd(x_proj.transpose(1, 2))[:, :, :T]
            x_conv_fwd = F.silu(x_conv_fwd).transpose(1, 2)
            y_fwd = self._ssm_branch(x_conv_fwd, self.x_proj_fwd, self.dt_proj_fwd, reverse=False)

            # Backward branch
            x_conv_bwd = self.conv1d_bwd(x_proj.transpose(1, 2))[:, :, :T]
            x_conv_bwd = F.silu(x_conv_bwd).transpose(1, 2)
            y_bwd = self._ssm_branch(x_conv_bwd, self.x_proj_bwd, self.dt_proj_bwd, reverse=True)

            # Gate and project
            out_t = self.out_proj(F.silu(z) * (y_fwd + y_bwd)) * mask_float

        # Residual
        out_t = out_t + x_t

        # Back to (B, C, T)
        out = out_t.permute(0, 2, 1)
        return out, mask
