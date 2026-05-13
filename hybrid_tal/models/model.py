"""
Complete Temporal Action Localization Model.

Assembles the full pipeline:
1. Feature Embedding: Projects input features to model dimension
2. Multi-Scale Temporal Encoder: BiMamba-Transformer hybrid
3. Action Localization Head: Classification + regression branches

Based on the ActionFormer (Zhang et al., 2022) framework.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_encoder import MultiScaleEncoder


class FeatureEmbedding(nn.Module):
    """
    Feature embedding layer that projects pre-extracted video features
    (e.g., I3D with 2048-dim) to the model's working dimension.
    
    Uses 1D convolution for temporal-aware projection.
    """
    
    def __init__(
        self,
        input_dim: int,
        embd_dim: int,
        kernel_size: int = 3,
        with_ln: bool = True,
        max_seq_len: int = 2304,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embd_dim = embd_dim
        self.max_seq_len = max_seq_len
        
        # Projection
        self.proj = nn.Conv1d(
            input_dim, embd_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(embd_dim) if with_ln else nn.Identity()
        
        # Learnable positional encoding
        self.pos_embd = nn.Parameter(
            torch.zeros(1, embd_dim, max_seq_len)
        )
        nn.init.trunc_normal_(self.pos_embd, std=0.02)
    
    def forward(self, x, mask):
        """
        Args:
            x: (B, C_in, T) - raw features
            mask: (B, 1, T) - validity mask
        Returns:
            x: (B, C_embd, T) - embedded features
            mask: (B, 1, T) - same mask
        """
        B, C, T = x.shape
        
        # Project
        x = self.proj(x)
        
        # Add positional encoding (truncated to actual length)
        x = x + self.pos_embd[:, :, :T]
        
        # Layer norm (in B, T, C format)
        if isinstance(self.norm, nn.LayerNorm):
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Apply mask
        x = x * mask
        
        return x, mask


class ActionLocalizationHead(nn.Module):
    """
    Two-branch head for action classification and boundary regression.
    Shared across all multi-scale levels.
    
    - Classification branch: Predicts action label probabilities
    - Regression branch: Predicts relative start/end distances
    """
    
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_classes: int,
        kernel_size: int = 3,
        num_layers: int = 3,
        with_ln: bool = True,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared stem layers
        self.stem = nn.ModuleList()
        for i in range(num_layers):
            in_ch = d_model if i == 0 else head_dim
            self.stem.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, head_dim, kernel_size=kernel_size,
                              padding=kernel_size // 2),
                    nn.GroupNorm(16, head_dim) if with_ln else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
            )
        
        # Classification head
        self.cls_head = nn.Conv1d(head_dim, num_classes, kernel_size=kernel_size,
                                  padding=kernel_size // 2)
        
        # Regression head (predicts left and right distances)
        self.reg_head = nn.Conv1d(head_dim, 2, kernel_size=kernel_size,
                                  padding=kernel_size // 2)
        
        # Initialize classification bias with prior probability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)
        
        # Initialize regression head
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.constant_(self.reg_head.bias, 0)
    
    def forward(self, feat_list, mask_list):
        """
        Args:
            feat_list: List of (B, C, T_l) for each scale level
            mask_list: List of (B, 1, T_l) for each scale level
        Returns:
            cls_logits: List of (B, num_classes, T_l)
            reg_preds: List of (B, 2, T_l) - left/right distances
        """
        cls_logits = []
        reg_preds = []
        
        for feat, mask in zip(feat_list, mask_list):
            # Shared stem
            x = feat
            for layer in self.stem:
                x = layer(x)
                x = x * mask
            
            # Classification
            cls = self.cls_head(x) * mask
            cls_logits.append(cls)
            
            # Regression (ReLU to ensure positive distances)
            reg = F.relu(self.reg_head(x)) * mask
            reg_preds.append(reg)
        
        return cls_logits, reg_preds


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in action classification.
    From Lin et al. (2017) / Tian et al. (2019).
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, C, T) - raw logits
            target: (B, C, T) - one-hot targets
            mask: (B, 1, T)
        Returns:
            loss: scalar
        """
        pred_sigmoid = pred.sigmoid()
        
        # Focal weight
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(
            target == 1,
            self.alpha * torch.ones_like(pred),
            (1 - self.alpha) * torch.ones_like(pred),
        )
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Combined
        loss = focal_weight * alpha_weight * bce
        
        # Mask and average
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        
        return loss


class DIoULoss(nn.Module):
    """
    Distance IoU Loss for boundary regression.
    From Zheng et al. (2020), as used in the paper.
    """
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, 2, T) - predicted left/right distances
            target: (B, 2, T) - ground truth left/right distances
            mask: (B, 1, T) - valid positions with actions
        Returns:
            loss: scalar
        """
        # pred and target are (left_dist, right_dist) from each point
        pred_left = pred[:, 0:1, :]  # (B, 1, T)
        pred_right = pred[:, 1:2, :]
        target_left = target[:, 0:1, :]
        target_right = target[:, 1:2, :]
        
        # Compute IoU
        inter_left = torch.max(pred_left, target_left)
        inter_right = torch.max(pred_right, target_right)
        
        pred_area = pred_left + pred_right
        target_area = target_left + target_right
        inter = (pred_left + pred_right - inter_left - inter_right).clamp(min=0)
        
        union = pred_area + target_area - inter
        iou = inter / union.clamp(min=1e-6)
        
        # Enclosing box
        enclose_left = torch.max(pred_left, target_left)
        enclose_right = torch.max(pred_right, target_right)
        enclose = enclose_left + enclose_right
        
        # Center distance
        pred_center = (pred_right - pred_left) / 2.0
        target_center = (target_right - target_left) / 2.0
        center_dist = (pred_center - target_center) ** 2
        enclose_diag = enclose ** 2
        
        # DIoU
        diou = iou - center_dist / enclose_diag.clamp(min=1e-6)
        loss = 1.0 - diou
        
        # Mask: only compute for positive locations
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        
        return loss


class HybridTALModel(nn.Module):
    """
    Complete Hybrid BiMamba-Transformer model for Temporal Action Localization.
    
    Pipeline:
        Video features -> Feature Embedding -> Multi-Scale BiMamba-Transformer
        Encoder -> Action Localization Head -> (class_logits, boundary_regression)
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Feature embedding
        self.embedding = FeatureEmbedding(
            input_dim=cfg['dataset']['input_dim'],
            embd_dim=cfg['model']['embd_dim'],
            kernel_size=cfg['model']['embd_kernel_size'],
            with_ln=cfg['model']['embd_with_ln'],
            max_seq_len=cfg['dataset']['max_seq_len'],
        )
        
        # Multi-scale encoder
        self.encoder = MultiScaleEncoder(
            d_model=cfg['model']['embd_dim'],
            num_levels=cfg['model']['num_levels'],
            scale_factor=cfg['model']['scale_factor'],
            n_head=cfg['model']['n_head'],
            d_state=cfg['model']['mamba_d_state'],
            d_conv=cfg['model']['mamba_d_conv'],
            expand=cfg['model']['mamba_expand'],
            dropout=cfg['model']['dropout'],
            attn_dropout=cfg['model']['attn_dropout'],
            use_window=cfg['model']['use_window_attn'],
            window_size=cfg['model']['window_size'],
            fusion_type=cfg['model']['fusion_type'],
            n_blocks_per_level=cfg['model']['n_mamba_blocks'],
        )
        
        # Action localization head
        self.head = ActionLocalizationHead(
            d_model=cfg['model']['embd_dim'],
            head_dim=cfg['model']['head_dim'],
            num_classes=cfg['dataset']['num_classes'],
            kernel_size=cfg['model']['head_kernel_size'],
            num_layers=cfg['model']['head_num_layers'],
            with_ln=cfg['model']['head_with_ln'],
            prior_prob=cfg['model']['prior_prob'],
        )
        
        # Losses
        self.cls_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.reg_loss_fn = DIoULoss()
        
        # Loss weights
        self.cls_weight = cfg['training']['cls_loss_weight']
        self.reg_weight = cfg['training']['reg_loss_weight']
        
        # Store regression ranges
        self.regression_range = cfg['model']['regression_range']
        self.num_levels = cfg['model']['num_levels']
        self.scale_factor = cfg['model']['scale_factor']
        self.feat_stride = cfg['dataset']['feat_stride']
        self.num_frames = cfg['dataset']['num_frames']
    
    def forward(self, features, mask, targets=None):
        """
        Args:
            features: (B, C_in, T) - pre-extracted video features
            mask: (B, 1, T) - validity mask
            targets: dict with 'cls_targets', 'reg_targets', 'pos_mask'
                     for each scale level (only during training)
        Returns:
            During training: dict with 'cls_loss', 'reg_loss', 'total_loss'
            During inference: dict with 'cls_logits', 'reg_preds', 'masks'
        """
        # Feature embedding
        x, mask = self.embedding(features, mask)
        
        # Multi-scale encoding
        feat_list, mask_list = self.encoder(x, mask)
        
        # Action localization head
        cls_logits, reg_preds = self.head(feat_list, mask_list)
        
        if targets is not None and self.training:
            # Compute losses
            total_cls_loss = 0
            total_reg_loss = 0
            
            for level in range(len(cls_logits)):
                cls_target = targets['cls_targets'][level]
                reg_target = targets['reg_targets'][level]
                pos_mask = targets['pos_mask'][level]
                level_mask = mask_list[level]
                
                # Classification loss (all valid positions)
                cls_loss = self.cls_loss_fn(
                    cls_logits[level], cls_target, level_mask
                )
                total_cls_loss += cls_loss
                
                # Regression loss (only positive positions)
                if pos_mask.sum() > 0:
                    reg_loss = self.reg_loss_fn(
                        reg_preds[level], reg_target, pos_mask
                    )
                    total_reg_loss += reg_loss
            
            total_cls_loss /= len(cls_logits)
            total_reg_loss /= len(cls_logits)
            
            total_loss = (self.cls_weight * total_cls_loss +
                         self.reg_weight * total_reg_loss)
            
            return {
                'cls_loss': total_cls_loss,
                'reg_loss': total_reg_loss,
                'total_loss': total_loss,
            }
        else:
            return {
                'cls_logits': cls_logits,
                'reg_preds': reg_preds,
                'masks': mask_list,
            }
    
    def get_point_coordinates(self, feat_list):
        """
        Generate temporal point coordinates for each level.
        Used during inference to convert regression outputs to timestamps.
        
        Returns:
            points_list: List of (T_l,) tensors with temporal coordinates
        """
        points_list = []
        for level, feat in enumerate(feat_list):
            T_l = feat.shape[-1]
            stride = self.feat_stride * self.num_frames * (self.scale_factor ** level)
            # Center of each temporal bin
            points = torch.arange(T_l, device=feat.device).float() * stride + stride / 2.0
            points_list.append(points)
        return points_list
