"""
THUMOS14 Dataset Loader for Temporal Action Localization.

Loads pre-extracted I3D features and annotations, following the
same preprocessing as ActionFormer (Zhang et al., 2022).
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class THUMOS14Dataset(Dataset):
    """
    THUMOS14 dataset with I3D features.
    
    Features: I3D features pre-extracted on Kinetics, stored as .npy files.
    Annotations: JSON file with video-level action annotations.
    
    Args:
        feat_folder: Path to I3D feature folder
        json_file: Path to annotation JSON
        split: 'training' or 'testing'
        max_seq_len: Maximum sequence length (pad/crop)
        feat_stride: Temporal stride of features
        num_frames: Number of frames per clip
        num_classes: Number of action classes
        is_training: Whether in training mode
        regression_range: List of [min, max] for each level
        num_levels: Number of multi-scale levels
        scale_factor: Scale factor between levels
        downsample_rate: Feature downsample rate
    """
    
    def __init__(
        self,
        feat_folder: str,
        json_file: str,
        split: str = 'training',
        max_seq_len: int = 2304,
        feat_stride: int = 4,
        num_frames: int = 16,
        num_classes: int = 20,
        is_training: bool = True,
        regression_range: list = None,
        num_levels: int = 6,
        scale_factor: int = 2,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.feat_folder = feat_folder
        self.max_seq_len = max_seq_len
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.is_training = is_training
        self.regression_range = regression_range or [
            [0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]
        ]
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.downsample_rate = downsample_rate
        
        # Load annotations
        with open(json_file, 'r') as f:
            anno_data = json.load(f)
        
        self.data_list = []
        self.label_dict = anno_data.get('label_dict', None)
        
        # Parse annotations
        for video_id, video_info in anno_data['database'].items():
            if video_info['subset'] != split:
                continue
            
            # Check if feature file exists
            feat_file = os.path.join(feat_folder, f"{video_id}.npy")
            if not os.path.exists(feat_file):
                continue
            
            annotations = []
            if 'annotations' in video_info:
                for ann in video_info['annotations']:
                    annotations.append({
                        'segment': ann['segment'],
                        'label': ann['label'],
                        'label_id': ann['label_id'] if 'label_id' in ann else ann['label'],
                    })
            
            self.data_list.append({
                'id': video_id,
                'feat_file': feat_file,
                'duration': video_info.get('duration', 0),
                'fps': video_info.get('fps', 30),
                'annotations': annotations,
            })
        
        print(f"Loaded {len(self.data_list)} videos for {split}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        video_info = self.data_list[idx]
        
        # Load features
        features = np.load(video_info['feat_file']).astype(np.float32)
        
        # features shape: (T, C) -> (C, T)
        if features.ndim == 2:
            features = features.T
        
        # Downsample if needed
        if self.downsample_rate > 1:
            features = features[:, ::self.downsample_rate]
        
        C, T = features.shape
        
        # Handle variable-length: pad or crop to max_seq_len
        if T >= self.max_seq_len:
            # Crop
            if self.is_training:
                # Random crop during training
                start = np.random.randint(0, T - self.max_seq_len + 1)
            else:
                start = 0
            features = features[:, start:start + self.max_seq_len]
            mask = np.ones((1, self.max_seq_len), dtype=np.float32)
            offset = start
            T = self.max_seq_len
        else:
            # Pad
            pad_len = self.max_seq_len - T
            features = np.pad(features, ((0, 0), (0, pad_len)), mode='constant')
            mask = np.zeros((1, self.max_seq_len), dtype=np.float32)
            mask[:, :T] = 1.0
            offset = 0
        
        # Convert to torch tensors
        features = torch.from_numpy(features)
        mask = torch.from_numpy(mask)
        
        # Generate targets
        if self.is_training:
            targets = self._generate_targets(
                video_info, T, offset, features.device
            )
            return features, mask, targets, video_info['id']
        else:
            return features, mask, video_info, video_info['id']
    
    def _generate_targets(self, video_info, seq_len, offset, device):
        """
        Generate multi-scale classification and regression targets.
        
        For each scale level:
        - cls_target: (num_classes, T_l) one-hot classification target
        - reg_target: (2, T_l) left/right distance regression target
        - pos_mask: (1, T_l) positive position mask
        """
        fps = video_info.get('fps', 30)
        duration = video_info.get('duration', 0)
        annotations = video_info['annotations']
        
        # Temporal stride per feature step (in seconds)
        feat_step = self.feat_stride * self.num_frames / fps
        
        targets = {
            'cls_targets': [],
            'reg_targets': [],
            'pos_mask': [],
        }
        
        for level in range(self.num_levels):
            level_stride = self.scale_factor ** level
            T_l = self.max_seq_len // level_stride
            
            cls_target = torch.zeros(self.num_classes, T_l)
            reg_target = torch.zeros(2, T_l)
            pos_mask = torch.zeros(1, T_l)
            
            # Point coordinates (in feature steps)
            points = torch.arange(T_l).float() * level_stride + level_stride / 2.0
            
            # Regression range for this level
            reg_min, reg_max = self.regression_range[level]
            
            for ann in annotations:
                seg_start, seg_end = ann['segment']
                label_id = ann['label_id']
                
                if isinstance(label_id, str):
                    # If label is a string, try to convert
                    if self.label_dict and label_id in self.label_dict:
                        label_id = self.label_dict[label_id]
                    else:
                        continue
                
                # Convert timestamps to feature indices
                start_idx = (seg_start / feat_step - offset / level_stride)
                end_idx = (seg_end / feat_step - offset / level_stride)
                
                # Map to current level's resolution
                start_idx_level = start_idx / level_stride
                end_idx_level = end_idx / level_stride
                
                # For each point in this level
                for t in range(T_l):
                    # Left and right distances from this point to segment boundaries
                    left_dist = (points[t].item() / level_stride) - start_idx_level
                    right_dist = end_idx_level - (points[t].item() / level_stride)
                    
                    # Check if point is inside the segment
                    if left_dist >= 0 and right_dist >= 0:
                        # Check if within regression range
                        max_dist = max(left_dist, right_dist)
                        if max_dist >= reg_min and max_dist < reg_max:
                            cls_target[label_id, t] = 1.0
                            reg_target[0, t] = left_dist
                            reg_target[1, t] = right_dist
                            pos_mask[0, t] = 1.0
            
            targets['cls_targets'].append(cls_target)
            targets['reg_targets'].append(reg_target)
            targets['pos_mask'].append(pos_mask)
        
        return targets


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    features = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    video_ids = [b[3] for b in batch]
    
    if isinstance(batch[0][2], dict):
        # Training mode: collate targets
        targets = {
            'cls_targets': [],
            'reg_targets': [],
            'pos_mask': [],
        }
        num_levels = len(batch[0][2]['cls_targets'])
        for level in range(num_levels):
            targets['cls_targets'].append(
                torch.stack([b[2]['cls_targets'][level] for b in batch])
            )
            targets['reg_targets'].append(
                torch.stack([b[2]['reg_targets'][level] for b in batch])
            )
            targets['pos_mask'].append(
                torch.stack([b[2]['pos_mask'][level] for b in batch])
            )
        return features, masks, targets, video_ids
    else:
        # Inference mode
        video_infos = [b[2] for b in batch]
        return features, masks, video_infos, video_ids


def build_dataloader(cfg, split='training', is_training=True):
    """Build dataloader from config."""
    dataset = THUMOS14Dataset(
        feat_folder=cfg['dataset']['feat_folder'],
        json_file=cfg['dataset']['json_file'],
        split=split,
        max_seq_len=cfg['dataset']['max_seq_len'],
        feat_stride=cfg['dataset']['feat_stride'],
        num_frames=cfg['dataset']['num_frames'],
        num_classes=cfg['dataset']['num_classes'],
        is_training=is_training,
        regression_range=cfg['model']['regression_range'],
        num_levels=cfg['model']['num_levels'],
        scale_factor=cfg['model']['scale_factor'],
        downsample_rate=cfg['dataset'].get('downsample_rate', 1),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'] if is_training else 1,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=is_training,
    )
    
    return dataloader
