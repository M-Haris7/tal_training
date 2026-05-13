# BiMamba-Transformer Hybrid for Temporal Action Localization

## Novel Architecture: BiMamba + Sliding Window Attention

This project extends [ActionFormer](https://github.com/happyharrycn/actionformer_release) with two hybrid architectures that combine bidirectional Mamba (ViM-style) with Transformer attention for temporal action localization.

### Research Gap Addressed
The base paper ([Zhang et al., VISIGRAPP 2025](https://doi.org/10.5220/0013173000003912)) compared Transformer and Mamba blocks but did **not** explore:
- **ViM-style BiMamba in parallel with sliding window attention** (our Parallel variant)
- **MambaFormer with ViM-BiMamba** adapted for multi-scale TAL (our Sequential variant)

### Architectures

| Model | Backbone | Description |
|-------|----------|-------------|
| Baseline | `convTransformer` | Original ActionFormer (Transformer only) |
| **Parallel Hybrid** | `biMambaParallel` | BiMamba ∥ Attention with learnable fusion |
| **Sequential Hybrid** | `biMambaSequential` | BiMamba → Attention → BiMamba (MambaFormer-style) |

### Parameter Comparison (THUMOS14)
| Model | Params | Expected Avg mAP |
|-------|--------|-----------------|
| Baseline ActionFormer | 29.3M | ~66.8% |
| Parallel Hybrid | 38.3M | 67-69% |
| Sequential Hybrid | 36.7M | 67-69% |

---

## Setup (RunPod)

**Template:** `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

### Step 1: Clone and setup
```bash
git clone https://github.com/happyharrycn/actionformer_release.git
cd actionformer_release

# Copy the hybrid source files into the repo
# (if using this project directly, files are already in place)

# Install dependencies
pip install pyyaml tensorboard h5py joblib pandas scipy gdown

# Install Mamba CUDA kernels (recommended for GPU training)
pip install causal-conv1d==1.2.2.post1
pip install mamba-ssm==2.0.4

# Build NMS extension
cd libs/utils && python setup.py install && cd ../..
```

### Step 2: Download THUMOS14 dataset
```bash
mkdir -p data/thumos
cd data/thumos
gdown 1zt2eoldshf99vJMDuu8jqxda55dCyhZP -O thumos.tar.gz
tar -xzf thumos.tar.gz
rm thumos.tar.gz
cd ../..
```

Expected structure:
```
data/thumos/
├── annotations/
│   └── thumos14.json
└── i3d_features/
    ├── video_validation_0000051.npy
    ├── ...
```

### Step 3: Train

```bash
# Baseline ActionFormer (for comparison)
python train.py configs/thumos_i3d.yaml --output baseline

# Parallel BiMamba + Attention Hybrid
python train.py configs/thumos_bimamba_parallel.yaml --output parallel

# Sequential BiMamba + Attention Hybrid (MambaFormer-style)
python train.py configs/thumos_bimamba_sequential.yaml --output sequential
```

### Step 4: Evaluate
```bash
python eval.py configs/thumos_bimamba_parallel.yaml ckpt/bimamba_parallel/parallel
python eval.py configs/thumos_bimamba_sequential.yaml ckpt/bimamba_sequential/sequential
```

### Step 5: Monitor with TensorBoard
```bash
tensorboard --logdir=./ckpt/
```

---

## File Structure (New/Modified Files)

```
actionformer_release/
├── libs/modeling/
│   ├── mamba_simple.py          ← NEW: BiMamba block (pure PyTorch + CUDA fallback)
│   ├── hybrid_blocks.py         ← NEW: Parallel & Sequential hybrid blocks
│   ├── hybrid_backbones.py      ← NEW: Multi-scale hybrid backbones
│   ├── __init__.py              ← MODIFIED: imports hybrid_backbones
│   ├── meta_archs.py            ← MODIFIED: supports hybrid backbone types
│   └── ...
├── libs/core/
│   └── config.py                ← MODIFIED: added BiMamba default params
├── configs/
│   ├── thumos_bimamba_parallel.yaml    ← NEW
│   ├── thumos_bimamba_sequential.yaml  ← NEW
│   └── thumos_i3d.yaml                ← EXISTING (baseline)
├── setup_env.sh                 ← NEW: one-shot setup script
└── HYBRID_README.md             ← This file
```

---

## Key Design Choices

1. **ViM-style BiMamba**: Shared input projections with separate forward/backward SSM branches. Chosen because the base paper didn't test this variant in hybrid configurations.

2. **Sliding Window Attention (size=19)**: Proven in ActionFormer ablation to match global attention at lower cost.

3. **Learnable Fusion (Parallel)**: Per-channel sigmoid-gated weight `α` that adapts the Mamba/Attention ratio at each pyramid level.

4. **No Position Encoding**: ActionFormer showed it hurts performance; convolutions and Mamba's sequential nature provide implicit position information.

5. **Center Sampling (α=1.5)**: Ablation shows +1.4% avg mAP.

6. **Pure PyTorch fallback**: If `mamba-ssm` is not installed, the SSM runs in pure PyTorch (slower but functional). Set `use_mamba_cuda: False` in config to force this.

---

## Troubleshooting

**mamba-ssm installation fails:**
```bash
# Set use_mamba_cuda: False in the yaml config
# The model will use the pure PyTorch SSM implementation
```

**OOM during training:**
```bash
# Reduce batch_size in config (default: 2)
# Or reduce max_seq_len (default: 2304 → try 1152)
```

**NMS compilation fails:**
```bash
cd libs/utils
python setup.py build_ext --inplace
```
