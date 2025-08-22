# β-VAE Experiments Guide

## 🎯 Understanding Continuous Latent Variables through β-VAE

You now have a complete β-VAE experimental framework! This demonstrates the most important concept in continuous latent variables: **disentanglement**.

## Quick Start

### 1. Test β-VAE Implementation

```bash
# Test the β-VAE code works
cd src/models
python beta_vae.py
```

### 2. Run Quick Comparison Experiment

```bash
# Compare β=1 vs β=4 (5 minutes)
python run_beta_experiments.py --quick --no-wandb
```

### 3. Full Experiment Suite

```bash
# Compare multiple β values (20-30 minutes)
python run_beta_experiments.py --beta-values 1.0 2.0 4.0 6.0 --epochs 25 --no-wandb
```

## What β-VAE Teaches Us

### The β Parameter Controls:

- **β = 1**: Standard VAE (good reconstruction, less organized latent space)
- **β > 1**: More disentangled latent space (each dimension captures specific features)
- **β >> 1**: Highly disentangled but poorer reconstruction

### Key Insights:

1. **Continuous latent variables** can be **regularized** to learn interpretable representations
2. **Higher β** encourages each latent dimension to specialize
3. **Trade-off** between reconstruction quality and interpretability
4. **Smooth interpolation** remains possible across all β values

## Experiment Commands

### Basic Experiments

```bash
# Quick test (2 β values, 10 epochs each)
python run_beta_experiments.py
```
