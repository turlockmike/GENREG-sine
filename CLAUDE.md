# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GENREG-sine implements gradient-free evolutionary neural network training to demonstrate emergent hybrid computation. The core hypothesis: neural networks trained via evolutionary selection (not gradient descent) spontaneously develop hybrid digital-analog representations where saturated neurons (|activation| > 0.95) emerge as binary switches while continuous neurons provide fine-grained modulation.

## Repository Structure

```
GENREG-sine/
├── core/                     # Shared utilities (USE THESE!)
│   ├── metrics.py            # compute_metrics() → MSE, Energy, Saturation
│   └── training.py           # train_sa(), train_hillclimb(), train_ga()
│
├── models/                   # Pre-trained checkpoints
│   ├── ultra_sparse_mse0.000303.pt   # BEST: 33 params, 0.44μs
│   ├── standard_sa_mse0.009155.pt    # 2065 params
│   ├── backprop_mse0.000003.pt       # Most accurate
│   └── README.md             # Usage guide with Numba examples
│
├── experiments/              # All experiment files
│   ├── ultra_sparse.py       # ⭐ BREAKTHROUGH experiment
│   ├── comprehensive_benchmark.py    # 20-trial comparison
│   ├── inference_engines.py  # PyTorch vs NumPy vs Numba
│   └── experiment_*.py       # Legacy experiments
│
├── sine_*.py                 # Original GENREG code
├── results/                  # Experiment outputs (JSON)
└── experiments_log.md        # Detailed results log
```

## Commands

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run single training session (original GENREG)
python sine_train.py

# Run full experimental sweep (13 configurations)
python sine_sweep.py

# Run a specific experiment
python experiments/sensory_bottleneck.py

# Live visualization of training
python sine_visualize.py
```

## Standard Metrics

**All experiments MUST report these three metrics:**

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **MSE** | Accuracy (lower is better) | `torch.mean((pred - y_true) ** 2)` |
| **Energy** | Inference cost | `activation_energy + weight_energy` |
| **Saturation** | % saturated neurons | `(activations.abs() > 0.95).mean()` |

Use `core.metrics.compute_metrics(controller, x_test, y_true)` for consistent measurement.

## Writing New Experiments

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from core.metrics import compute_metrics
from core.training import train_sa

# Create your controller class with forward(), mutate(), clone() methods

# Train
best, final_metrics, history = train_sa(
    controller, x_test, y_true,
    max_steps=15000,
    verbose=True
)

# Results automatically include MSE, Energy, Saturation
print(f"Final: {final_metrics}")
```

## Architecture

### Standard MLP (256 → 8 → 1)
```
Input (scalar x)
  └─→ [Expand to 256D: 16 true signals + 240 noise] → SineController
      - 256 → 8 (tanh) → 1 (tanh)
      - Tracks saturation per neuron
```

### Sensory Bottleneck (256 → N → 8 → 1)
```
Environment (256) → Sensory (N neurons) → Processing (8) → Output (1)
```

## Key Files

| File | Purpose |
|------|---------|
| `core/metrics.py` | Standard metrics calculation |
| `core/training.py` | SA, Hill Climbing, GA training loops |
| `sine_config.py` | Central configuration |
| `sine_controller.py` | 2-layer MLP with input expansion |
| `experiments_log.md` | All experiment results |

## Important Findings

1. **Original noise was correlated**: The 240 "noise" signals included `-sin(x)` (r=-1.0). Networks exploited this shortcut.

2. **True signals only works best**: 16 inputs → MSE 0.002 (vs 256 inputs → MSE 0.021)

3. **Saturation tradeoffs**:
   - 100% saturation → 170x more robust to noise, 2.26x faster inference
   - 0% saturation (backprop) → 100x better MSE

4. **Ultra-Sparse solves input selection**: Limiting each neuron to K=2 inputs creates selection pressure. Results:
   - **MSE 0.000303** (30x better than dense SA)
   - **33 parameters** (63x fewer than standard)
   - **4x better input selection** than random

5. **Inference engine matters**: PyTorch overhead dominates sparse models. With Numba JIT:
   - Ultra-Sparse: 0.44 μs (60x faster than PyTorch)
   - Dense models: 0.58-0.63 μs (5-6x faster than PyTorch)
   - Ultra-Sparse is fastest overall (1.3-1.4x faster than dense)

## Configuration

Key settings in `sine_config.py`:
- `HIDDEN_SIZE = 8` — Hidden layer neurons
- `TRUE_SIGNAL_SIZE = 16` — Useful input dimensions
- `NOISE_SIGNAL_SIZE = 240` — Noise input dimensions
- `SATURATION_THRESHOLD = 0.95` — Defines "saturated"
- `DEVICE = "cuda"` — GPU acceleration

## Output

Results saved to `results/<experiment_name>/`:
- `results.json` — Metrics and configuration
- Saturation trajectories and analysis
