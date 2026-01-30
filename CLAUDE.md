# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GENREG-sine implements gradient-free evolutionary neural network training to demonstrate emergent hybrid computation. The core hypothesis: neural networks trained via evolutionary selection (not gradient descent) spontaneously develop hybrid digital-analog representations where saturated neurons (|activation| > 0.95) emerge as binary switches while continuous neurons provide fine-grained modulation.

## Commands

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run single training session
python sine_train.py

# Run full experimental sweep (13 configurations across 3 studies)
python sine_sweep.py

# Live visualization of training
python sine_visualize.py

# Analyze results from a completed session
python sine_analyze.py <session_dir>
```

## Architecture

### Parallel MLP + Protein Cascade
```
Input (scalar x)
  ├─→ [Expand to 256D: 16 true signals + 240 noise] → SineController (MLP)
  │   - 256 → 8 (tanh) → 1 (tanh)
  │   - Tracks saturation per neuron
  │
  └─→ Protein Cascade (parallel signal processor)
      - Computes "trust" signals that modify fitness
```

### Evolutionary Loop
1. **Evaluate**: All genomes predict sine wave, compute MSE
2. **Select**: Sort by fitness (fitness = -MSE × trust_multiplier)
3. **Reproduce**: Elite (20%) + Survivors (30%) + Clone+Mutate (40%) + Fresh (10%)
4. **Track**: Saturation metrics for theory validation

### Key Files
| File | Purpose |
|------|---------|
| `sine_config.py` | Central configuration (modify here to change experiments) |
| `sine_train.py` | Main training loop entry point |
| `sine_sweep.py` | Orchestrates experimental sweeps |
| `sine_population.py` | Population management and evolution |
| `sine_genome.py` | Individual genome container (MLP + proteins) |
| `sine_controller.py` | 2-layer MLP with input expansion |
| `sine_proteins.py` | Protein cascade trust mechanism |

### Input Expansion Strategy
- **True Signals (16)**: x, x², x³, sin(0.5x-3x), cos(0.5x-3x) - useful Fourier basis
- **Noise Signals (240)**: Wrong frequencies, inverted signals, polynomial products, distortions
- Creates 32:1 compression ratio through 8-neuron hidden layer bottleneck

## Configuration

All parameters in `sine_config.py`. Key settings:
- `HIDDEN_SIZE = 8` — Hidden layer neurons (bottleneck)
- `POPULATION_SIZE = 300` — Genomes per generation
- `GENERATIONS = 5000` — Training length
- `SATURATION_THRESHOLD = 0.95` — Defines when neuron is "saturated"
- `DEVICE = "cuda"` — GPU acceleration (falls back to CPU)

## Output

Results saved to `results/sweep_<timestamp>/` containing:
- Per-test configuration and saturation trajectories (JSON)
- Saturation heatmaps (NumPy arrays)
- Summary statistics in `sweep_summary.json`

## Implementation Notes

- All operations are vectorized with PyTorch tensors for GPU acceleration
- Saturation ratio = fraction of hidden neurons with |activation| > threshold
- Fitness modified by protein trust: `fitness = -MSE × max(0.1, 1.0 + trust × scale)`
- Seeds are set for reproducibility (`SEED = 42`)
