#!/bin/bash
# Setup script for RunPod environment
# Template: runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

set -e

echo "=== Installing Python dependencies ==="
pip install pyyaml tensorboard h5py joblib pandas scipy gdown

echo "=== Installing Mamba dependencies ==="
# causal-conv1d must be installed before mamba-ssm
pip install causal-conv1d==1.2.2.post1
pip install mamba-ssm==2.0.4

echo "=== Downloading THUMOS14 I3D features ==="
cd "$(dirname "$0")"
mkdir -p data/thumos

# Download from Google Drive (ActionFormer official link)
# File: thumos.tar.gz (md5sum 375f76ffbf7447af1035e694971ec9b2)
gdown 1zt2eoldshf99vJMDuu8jqxda55dCyhZP -O data/thumos/thumos.tar.gz

echo "=== Extracting dataset ==="
cd data/thumos
tar -xzf thumos.tar.gz
rm thumos.tar.gz
cd ../..

echo "=== Verifying dataset structure ==="
ls -la data/thumos/
echo ""
echo "Expected structure:"
echo "  data/thumos/annotations/thumos14.json"
echo "  data/thumos/i3d_features/*.npy"
echo ""

if [ -f "data/thumos/annotations/thumos14.json" ]; then
    echo "✓ Dataset ready!"
else
    echo "⚠ Dataset structure may differ. Check data/thumos/ and adjust config paths."
    find data/thumos -type f | head -20
fi

echo ""
echo "=== Setup complete ==="
echo "To train the baseline:  python train.py configs/thumos_i3d.yaml --output ckpt/baseline"
echo "To train parallel:      python train.py configs/thumos_bimamba_parallel.yaml --output ckpt/parallel"
echo "To train sequential:    python train.py configs/thumos_bimamba_sequential.yaml --output ckpt/sequential"
