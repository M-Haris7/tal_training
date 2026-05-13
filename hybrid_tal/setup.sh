#!/bin/bash
# =============================================================================
# Hybrid BiMamba-Transformer for Temporal Action Localization
# Setup script for RunPod with L40 GPU
# 
# REQUIRED TEMPLATE: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# DO NOT USE CUDA 11.8 templates - mamba-ssm will fail to compile
# =============================================================================

set -e

echo "============================================"
echo "Step 1: Verifying CUDA and PyTorch versions"
echo "============================================"

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)")
if [[ "$CUDA_VER" != 12.1* ]]; then
    echo "WARNING: CUDA version is $CUDA_VER, expected 12.1.x"
    echo "mamba-ssm may fail. Use a CUDA 12.1 template."
    echo "Recommended: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"
fi

echo "============================================"
echo "Step 2: Installing core dependencies"
echo "============================================"

pip install --upgrade pip
pip install packaging ninja wheel setuptools

# Install causal-conv1d first (mamba-ssm dependency)
# Version compatible with PyTorch 2.1.x + CUDA 12.1
pip install causal-conv1d==1.1.1

# Install mamba-ssm
pip install mamba-ssm==1.1.1

echo "============================================"
echo "Step 3: Installing additional packages"
echo "============================================"

pip install \
    numpy==1.24.4 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    pyyaml==6.0.1 \
    tensorboard==2.15.1 \
    fvcore==0.1.5.post20221221 \
    h5py==3.10.0 \
    joblib==1.3.2 \
    tqdm \
    einops

echo "============================================"
echo "Step 4: Downloading THUMOS14 I3D features"
echo "============================================"

mkdir -p data/thumos14

# Download I3D features (from ActionFormer's provided links)
# These are the standard I3D features pre-extracted on Kinetics
cd data/thumos14

if [ ! -f "i3d_features.tar.gz" ]; then
    echo "Downloading THUMOS14 I3D features..."
    # ActionFormer's official feature download
    wget -q --show-progress \
        https://github.com/happyharrycn/actionformer_release/releases/download/v1.0/thumos14_i3d_features.tar.gz \
        -O i3d_features.tar.gz
    
    echo "Extracting features..."
    tar -xzf i3d_features.tar.gz
    echo "Features extracted."
else
    echo "Features already downloaded."
fi

# Download annotations
if [ ! -d "annotations" ]; then
    echo "Downloading THUMOS14 annotations..."
    wget -q --show-progress \
        https://github.com/happyharrycn/actionformer_release/releases/download/v1.0/thumos14_annotations.tar.gz \
        -O annotations.tar.gz
    tar -xzf annotations.tar.gz
    echo "Annotations extracted."
else
    echo "Annotations already downloaded."
fi

cd ../..

echo "============================================"
echo "Step 5: Verifying installation"
echo "============================================"

python3 -c "
import torch
import mamba_ssm
import causal_conv1d
print(f'PyTorch: {torch.__version__}')
print(f'mamba-ssm: {mamba_ssm.__version__}')
print(f'causal-conv1d: {causal_conv1d.__version__}')
print(f'CUDA: {torch.version.cuda}')
print('All dependencies installed successfully!')
"

echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To train, run:"
echo "  python train.py configs/thumos14_bimamba_transformer.yaml"
echo ""
echo "To evaluate, run:"
echo "  python eval.py configs/thumos14_bimamba_transformer.yaml ckpt/best_model.pth"
