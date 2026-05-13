# Hybrid BiMamba-Transformer for Temporal Action Localization

A novel hybrid architecture combining **Bidirectional Mamba (BiMamba) scanning** with **Windowed Transformer attention** for temporal action localization on THUMOS14, extending the comparison study by Zhang et al. (VISAPP 2025).

## Architecture Overview

The paper tested MambaFormer (sequential: Mamba→Attn→Mamba) and CausalTAD (parallel: Causal Attn ‖ DBM) but noted they "focused on a limited set of hybrid models" and suggested exploring alternatives. This implementation introduces **BiMambaFormer**: a parallel-gated hybrid where both branches process input independently and a learned gate dynamically weights each branch's contribution per-timestep.

```
Input ──┬── BiMamba Branch (bidirectional SSM scanning) ──┐
        │                                                  ├── Gated Fusion ── Output
        └── Transformer Branch (windowed self-attention) ──┘
```

**Key design choices:**
- **BiMamba branch** uses separate forward/backward Mamba instances (unlike ViM's shared projections) for independent directional learning
- **Transformer branch** uses sliding window attention for O(T·W) complexity instead of O(T²)
- **Gated fusion** learns a per-channel, per-timestep gate: `gate * x_mamba + (1-gate) * x_transformer`
- **Multi-scale** architecture with 6 levels following ActionFormer

## Requirements

### Critical: RunPod Template

**USE:** `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04`

**DO NOT USE:** Any CUDA 11.8 template. The `mamba-ssm` library requires CUDA 12.1 for compilation. Using CUDA 11.8 will cause:
- `causal-conv1d` compilation failures
- `mamba-ssm` CUDA kernel errors
- Runtime `illegal memory access` errors

### Dependencies
- PyTorch 2.1.x (matching the paper's 2.1.2)
- CUDA 12.1
- `mamba-ssm==1.1.1`
- `causal-conv1d==1.1.1`

## Setup

```bash
# 1. Clone/upload this project to RunPod
# 2. Run the setup script
chmod +x setup.sh
bash setup.sh
```

The setup script will:
1. Verify CUDA/PyTorch versions
2. Install `causal-conv1d` and `mamba-ssm`
3. Install remaining dependencies
4. Download THUMOS14 I3D features and annotations

## Training

```bash
python train.py configs/thumos14_bimamba_transformer.yaml
```

Training logs are saved to `./logs/` (viewable with TensorBoard) and checkpoints to `./ckpt/`.

### Monitor with TensorBoard
```bash
tensorboard --logdir ./logs --port 6006
```

## Evaluation

```bash
python eval.py configs/thumos14_bimamba_transformer.yaml ckpt/<timestamp>/best_model.pth
```

## Expected Performance

Based on the paper's results for similar architectures on THUMOS14:

| Method | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg mAP |
|--------|-----|-----|-----|-----|-----|---------|
| ActionFormer (Win Attn) | 83.3 | 79.5 | 71.9 | 60.2 | 45.0 | 67.9 |
| Mamba Original | 83.3 | 79.7 | 72.3 | 60.8 | 46.1 | 68.5 |
| MambaFormer (Mamba+GlobalAttn) | 82.3 | 79.1 | 70.9 | 60.3 | 45.7 | 67.7 |
| CausalTAD | 82.4 | 79.0 | 71.9 | 61.1 | 44.9 | 67.9 |
| **BiMambaFormer (this work)** | **~82-84** | **~78-80** | **~71-73** | **~59-61** | **~44-46** | **~67-69** |

Target: **60-70% average mAP** (achievable based on paper's results).

## Project Structure

```
hybrid_tal/
├── configs/
│   └── thumos14_bimamba_transformer.yaml   # Configuration
├── models/
│   ├── __init__.py
│   ├── bimamba.py              # BiMamba block (bidirectional Mamba)
│   ├── transformer.py          # Transformer block with window attention
│   ├── hybrid_encoder.py       # Novel hybrid encoder + multi-scale
│   └── model.py                # Full TAL model with head + losses
├── datasets/
│   ├── __init__.py
│   └── thumos14.py             # THUMOS14 dataset loader
├── utils/
│   ├── __init__.py
│   └── postprocessing.py       # Soft-NMS, mAP evaluation, decoding
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
├── setup.sh                    # Installation script
└── README.md
```

## Configuration Tuning

Key hyperparameters in `configs/thumos14_bimamba_transformer.yaml`:

- `model.fusion_type`: Try `"gated"` (default), `"add"`, or `"concat"`
- `model.window_size`: Window size for attention (default: 19, try 9-29)
- `model.mamba_expand`: Mamba expansion factor (default: 2)
- `model.n_mamba_blocks`: Blocks per level (default: 1, try 2)
- `training.learning_rate`: Default 1e-4 (try 5e-5 to 2e-4)
- `training.epochs`: Default 30 (increase to 40-50 if underfitting)

## Troubleshooting

### CUDA errors with mamba-ssm
- Verify CUDA 12.1: `python -c "import torch; print(torch.version.cuda)"`
- If using CUDA 11.8, switch to a CUDA 12.1 template

### Out of memory
- Reduce `batch_size` to 1
- Reduce `max_seq_len` to 1536
- Reduce `embd_dim` to 384

### Low mAP
- Increase training epochs to 40-50
- Try `fusion_type: "add"` for simpler fusion
- Adjust `learning_rate` (try 5e-5)
- Increase `window_size` to 29

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{zhang2025transformer,
  title={Transformer or Mamba for Temporal Action Localization? Insights from a Comprehensive Experimental Comparison Study},
  author={Zhang, Zejian and Palmero, Cristina and Escalera, Sergio},
  booktitle={VISIGRAPP 2025 - VISAPP},
  pages={150--162},
  year={2025}
}
```
