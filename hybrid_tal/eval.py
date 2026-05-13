"""
Evaluation script for the trained model.

Usage:
    python eval.py configs/thumos14_bimamba_transformer.yaml ckpt/best_model.pth
"""

import sys
import yaml
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from models import HybridTALModel
from datasets import build_dataloader
from train import evaluate, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid BiMamba-Transformer')
    parser.add_argument('config', type=str, help='Path to config YAML file')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Build model
    model = HybridTALModel(cfg).to(device)
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Build test dataloader
    test_loader = build_dataloader(cfg, split='testing', is_training=False)
    print(f"Test videos: {len(test_loader.dataset)}")
    
    # Evaluate
    writer = SummaryWriter(log_dir='./logs/eval')
    avg_mAP = evaluate(model, test_loader, cfg, device, 0, writer)
    writer.close()
    
    print(f"\nFinal Average mAP: {avg_mAP:.2f}%")


if __name__ == '__main__':
    main()
