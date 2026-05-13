"""
Training script for Hybrid BiMamba-Transformer TAL model.

Usage:
    python train.py configs/thumos14_bimamba_transformer.yaml

Follows the same training procedure as described in the paper:
- Adam optimizer with warmup
- Focal loss for classification
- Distance IoU loss for regression
- Evaluation using mAP at tIoU thresholds [0.3:0.1:0.7]
"""

import os
import sys
import yaml
import time
import random
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import HybridTALModel
from datasets import build_dataloader
from utils import decode_predictions, evaluate_mAP


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg):
    """Build Adam optimizer with weight decay."""
    # Separate parameters: weight decay for weights, no decay for biases/norms
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': cfg['training']['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    optimizer = optim.Adam(
        param_groups,
        lr=cfg['training']['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    return optimizer


def build_scheduler(optimizer, cfg, num_training_steps):
    """Build cosine annealing scheduler with linear warmup."""
    warmup_epochs = cfg['training']['warmup_epochs']
    total_epochs = cfg['training']['epochs']
    
    def lr_lambda(current_step):
        # Linear warmup
        warmup_steps = warmup_epochs * (num_training_steps // total_epochs)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, num_training_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_reg_loss = 0
    num_batches = 0
    
    for batch_idx, (features, masks, targets, video_ids) in enumerate(dataloader):
        # Move to device
        features = features.to(device)
        masks = masks.to(device)
        
        targets_device = {
            'cls_targets': [t.to(device) for t in targets['cls_targets']],
            'reg_targets': [t.to(device) for t in targets['reg_targets']],
            'pos_mask': [t.to(device) for t in targets['pos_mask']],
        }
        
        # Forward pass
        loss_dict = model(features, masks, targets_device)
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_reg_loss += loss_dict['reg_loss'].item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} "
                f"(cls: {loss_dict['cls_loss'].item():.4f}, "
                f"reg: {loss_dict['reg_loss'].item():.4f}) "
                f"LR: {current_lr:.6f}"
            )
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_cls = total_cls_loss / max(num_batches, 1)
    avg_reg = total_reg_loss / max(num_batches, 1)
    
    # Log to tensorboard
    step = epoch
    writer.add_scalar('train/total_loss', avg_loss, step)
    writer.add_scalar('train/cls_loss', avg_cls, step)
    writer.add_scalar('train/reg_loss', avg_reg, step)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
    
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, cfg, device, epoch, writer):
    """Evaluate model and compute mAP."""
    model.eval()
    
    predictions = {}
    ground_truth = {}
    
    for features, masks, video_infos, video_ids in dataloader:
        features = features.to(device)
        masks = masks.to(device)
        
        # Forward pass
        output = model(features, masks)
        
        # Decode predictions for each video in batch
        for i in range(features.shape[0]):
            video_info = video_infos[i]
            video_id = video_ids[i]
            
            # Get per-video outputs
            cls_logits_i = [c[i:i+1].cpu() for c in output['cls_logits']]
            reg_preds_i = [r[i:i+1].cpu() for r in output['reg_preds']]
            masks_i = [m[i:i+1].cpu() for m in output['masks']]
            
            duration = video_info.get('duration', None)
            fps = video_info.get('fps', 30)
            
            segments, scores, labels = decode_predictions(
                cls_logits_i, reg_preds_i, masks_i,
                feat_stride=cfg['dataset']['feat_stride'],
                num_frames=cfg['dataset']['num_frames'],
                scale_factor=cfg['model']['scale_factor'],
                num_classes=cfg['dataset']['num_classes'],
                pre_nms_thresh=cfg['inference']['pre_nms_thresh'],
                pre_nms_topk=cfg['inference']['pre_nms_topk'],
                nms_sigma=cfg['inference']['nms_sigma'],
                nms_threshold=cfg['inference']['nms_threshold'],
                max_seg_num=cfg['inference']['max_seg_num'],
                duration=duration,
                fps=fps,
            )
            
            predictions[video_id] = [
                (seg[0], seg[1], lab, sc)
                for seg, sc, lab in zip(segments, scores, labels)
            ]
            
            # Collect ground truth
            gt_list = []
            if 'annotations' in video_info:
                for ann in video_info['annotations']:
                    label_id = ann['label_id']
                    if isinstance(label_id, str):
                        continue
                    gt_list.append((ann['segment'][0], ann['segment'][1], label_id))
            ground_truth[video_id] = gt_list
    
    # Compute mAP
    tiou_thresholds = cfg['eval']['tiou_thresholds']
    results = evaluate_mAP(predictions, ground_truth, tiou_thresholds)
    
    # Print results
    print(f"\n  Evaluation Results (Epoch {epoch}):")
    for tiou in tiou_thresholds:
        key = f'mAP@{tiou}'
        print(f"    {key}: {results[key]:.2f}%")
        writer.add_scalar(f'eval/{key}', results[key], epoch)
    
    print(f"    Average mAP: {results['avg_mAP']:.2f}%")
    writer.add_scalar('eval/avg_mAP', results['avg_mAP'], epoch)
    
    return results['avg_mAP']


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid BiMamba-Transformer TAL')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set seed
    set_seed(cfg['training']['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # Build model
    print("\nBuilding model...")
    model = HybridTALModel(cfg).to(device)
    
    num_params = count_parameters(model)
    print(f"Trainable parameters: {num_params / 1e6:.2f}M")
    
    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader = build_dataloader(cfg, split='training', is_training=True)
    test_loader = build_dataloader(cfg, split='testing', is_training=False)
    print(f"Training videos: {len(train_loader.dataset)}")
    print(f"Testing videos: {len(test_loader.dataset)}")
    
    if args.eval_only:
        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=device))
            print(f"Loaded checkpoint: {args.resume}")
        
        writer = SummaryWriter(log_dir='./logs/eval')
        evaluate(model, test_loader, cfg, device, 0, writer)
        writer.close()
        return
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    num_training_steps = len(train_loader) * cfg['training']['epochs']
    scheduler = build_scheduler(optimizer, cfg, num_training_steps)
    
    # Resume from checkpoint
    start_epoch = 0
    best_mAP = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint.get('best_mAP', 0.0)
        print(f"Resumed from epoch {start_epoch}, best mAP: {best_mAP:.2f}%")
    
    # Output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./ckpt/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=f'./logs/{timestamp}')
    
    print(f"\nStarting training for {cfg['training']['epochs']} epochs...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    for epoch in range(start_epoch, cfg['training']['epochs']):
        epoch_start = time.time()
        
        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer
        )
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s, avg loss: {avg_loss:.4f}")
        
        # Evaluate every 2 epochs (or every epoch after epoch 15)
        if (epoch + 1) % 2 == 0 or epoch >= 15:
            avg_mAP = evaluate(model, test_loader, cfg, device, epoch, writer)
            
            # Save best model
            if avg_mAP > best_mAP:
                best_mAP = avg_mAP
                save_path = os.path.join(output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mAP': best_mAP,
                }, save_path)
                print(f"  New best model saved! mAP: {best_mAP:.2f}%")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mAP': best_mAP,
            }, save_path)
        
        print("=" * 60)
    
    print(f"\nTraining complete! Best average mAP: {best_mAP:.2f}%")
    print(f"Best model saved at: {os.path.join(output_dir, 'best_model.pth')}")
    
    writer.close()


if __name__ == '__main__':
    main()
