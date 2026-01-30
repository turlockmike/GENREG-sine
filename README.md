# GENREG-sine

Gradient-free neural network training through evolutionary selection. Code for the paper "Emergent Hybrid Computation in Gradient-Free Evolutionary Networks."

## Overview

This repository demonstrates that neural networks trained via evolutionary selection (rather than gradient descent) spontaneously develop hybrid digital-analog representations. Saturated neurons emerge as discrete binary switches while continuous neurons provide fine-grained modulation. This is a computational structure that gradient-based methods systematically avoid due to vanishing gradients.

The key insight: **saturation is not a bug, it is a learned selective attention mechanism.**

When a network must attend to a subset of its inputs (filtering irrelevant dimensions, focusing on task-relevant signals), evolution discovers that saturating neurons into binary gates is an efficient solution. This creates a hybrid state space of 2^k discrete regions with continuous interpolation within each region.


### Doing More With Less

Standard deep learning uses massive networks because gradient descent requires smooth, high-dimensional landscapes to navigate. GENREG achieves comparable results with drastically smaller architectures:

| Task | Standard Approach | GENREG | Justification |
|------|-------------------|--------|---------------|
| Sine approximation | 2065 params | **33 params** | Ultra-Sparse: 63x smaller, 30x better MSE than dense SA |
| Humanoid locomotion | 256+ neurons | 16 neurons | Currently training |
| Function approximation | Millions of params | Thousands of params | Hybrid computation |

**NEW: Ultra-Sparse Breakthrough** - By limiting each neuron to only 2 inputs, we achieve:
- **MSE 0.000303** (30x better than dense gradient-free)
- **33 parameters** (63x fewer than standard)
- **0.44 μs inference** (1.4x faster than dense with Numba)

The secret is hybrid computation. With 8 neurons, a fully continuous network has limited representational power. But 8 saturated neurons create 256 discrete operational modes. A hybrid configuration (e.g., 6 saturated + 2 continuous) provides 64 discrete modes with smooth interpolation within each, combining the searchability of discrete spaces with the expressiveness of continuous spaces.

### Rethinking Neural Computation

Gradient-based training treats saturation as a failure mode. Techniques like batch normalization, careful initialization, and ReLU activations exist specifically to prevent it. This means an entire class of efficient hybrid solutions is systematically excluded from gradient-based discovery.

Evolution has no such constraint. Under selection pressure, networks naturally discover that:

- Clean continuous tasks (all inputs relevant) maintain k=0 saturation
- Tasks requiring selective attention develop partial saturation
- The ratio k/n is dynamically allocated based on task demands

This parallels biological neural systems, which use both discrete signaling (action potentials) and continuous signaling (graded potentials). The hybrid may represent a fundamental computational optimum.

## Results Summary

### Saturation Scales With Selective Attention Pressure

| Test | Irrelevant Dims | Total Input | Final k | k Ratio | MSE |
|------|-----------------|-------------|---------|---------|-----|
| N0 | 0 | 16 | 0/8 | 0% | 0.0017 |
| N1 | 16 | 32 | 0/8 | 0% | 0.0044 |
| N2 | 48 | 64 | 0/8 | 0% | 0.0131 |
| N3 | 112 | 128 | 6/8 | 75% | 0.0172 |
| N4 | 240 | 256 | 8/8 | 100% | 0.0168 |
| N5 | 496 | 512 | 8/8 | 100% | 0.0928 |

### Compression Alone Does Not Cause Saturation

| Test | Signal Dims | Compression | Final k |
|------|-------------|-------------|---------|
| C1 | 16 | 2:1 | 0/8 (0%) |
| C2 | 64 | 8:1 | 0/8 (0%) |
| C3 | 256 | 32:1 | 0/8 (0%) |

Even at 32:1 compression, networks remain fully continuous when all inputs are task-relevant.

### Excess Capacity Produces Hybrid Configurations

| Test | Hidden Size | Final k | k Ratio |
|------|-------------|---------|---------|
| H1 | 4 | 4/4 | 100% |
| H2 | 8 | 8/8 | 100% |
| H3 | 16 | 15/16 | 94% |
| H4 | 32 | 26/32 | 81% |

Given excess capacity, evolution preserves some continuous neurons for fine-grained modulation while allocating others to discrete gating. The ~75-80% saturation ratio appears to be a stable attractor.

### Visualizations

![Noise Dimensions vs Saturation](assets/noise_dimensions_chart.png)
*Saturation ratio (k/n) as a function of task-irrelevant input dimensions. A clear threshold exists around 100 dimensions where saturation becomes necessary.*

![Hidden Size vs Saturation](assets/hidden_size_chart.png)
*With excess capacity, networks maintain hybrid configurations (~80% saturated) rather than full saturation, preserving continuous neurons for fine-grained control.*

## Repository Structure

```
GENREG-sine/
├── README.md
├── CLAUDE.md                 # Claude Code guidance
├── experiments_log.md        # Detailed experiment results
├── paper.pdf                 # Full paper
│
├── core/                     # Shared utilities
│   ├── __init__.py
│   ├── metrics.py            # MSE, Energy, Saturation calculations
│   └── training.py           # SA, Hill Climbing, GA training loops
│
├── models/                   # Pre-trained model checkpoints
│   ├── ultra_sparse_mse0.000303.pt   # Best Ultra-Sparse model
│   ├── standard_sa_mse0.009155.pt    # Best Standard SA model
│   ├── backprop_mse0.000003.pt       # Best Backprop model
│   └── README.md             # Usage guide with Numba examples
│
├── experiments/              # All experiment files
│   ├── ultra_sparse.py       # Ultra-Sparse connectivity (breakthrough)
│   ├── comprehensive_benchmark.py    # 20-trial comparison
│   ├── inference_engines.py  # PyTorch vs NumPy vs Numba
│   └── experiment_*.py       # Various experiments
│
├── sine_config.py            # Configuration parameters
├── sine_train.py             # Main training script
├── sine_sweep.py             # Systematic sweep across configurations
├── sine_population.py        # Population management and evolution
├── sine_genome.py            # Individual genome (MLP + proteins)
├── sine_controller.py        # 2-layer MLP with input expansion
├── sine_proteins.py          # Protein cascade (trust mechanism)
│
├── assets/                   # Figures and visualizations
└── results/                  # Training outputs and checkpoints
```

## Setup

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- CUDA (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/A1CST/GENREG-sine.git
cd GENREG-sine
pip install torch numpy
```

## Usage

### Run a Single Training Session

```bash
python sine_train.py
```

This runs the default configuration (16 true signal + 240 noise dimensions, 8 hidden neurons) and outputs:
- Real-time logging of MSE and saturation levels
- Checkpoint of best genome
- Saturation trajectory JSON for analysis

### Run the Full Experimental Sweep

```bash
python sine_sweep.py
```

This reproduces all 13 configurations from the paper:
- N0-N5: Noise scaling experiments
- H1-H4: Hidden layer capacity experiments  
- C1-C3: Control experiments (compression without noise)

Results are saved to `sweep_summary.json`.

### Run Specific Configurations

```bash
python sine_sweep.py --configs N0 N4 C3
```

### Configuration Options

Edit `sine_config.py` to modify:

```python
# Input structure
TRUE_SIGNAL_SIZE = 16      # Task-relevant input dimensions
NOISE_SIGNAL_SIZE = 240    # Task-irrelevant input dimensions

# Architecture
HIDDEN_SIZE = 8            # Hidden layer neurons
OUTPUT_SIZE = 1            # Output dimensions

# Evolution
POPULATION_SIZE = 40       # Genomes per generation
GENERATIONS = 2000         # Training generations
ELITE_PCT = 0.20           # Top performers preserved unchanged
SURVIVE_PCT = 0.30         # Additional survivors
CLONE_MUTATE_PCT = 0.40    # Mutated copies of elites
RANDOM_PCT = 0.10          # Fresh random genomes

# Saturation tracking
SATURATION_THRESHOLD = 0.95  # |activation| > threshold = saturated
```

## How It Works

### Architecture

```
Input (true + irrelevant dims)
         │
         ▼
┌─────────────────────┐
│   Hidden Layer      │ ← tanh activation, saturation tracked
│   (n neurons)       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Output Layer      │ ← Task prediction
└─────────────────────┘
         │
         ▼
    MSE Fitness ──────► Selection Pressure
```

### Evolution Loop

1. **Evaluate**: All genomes predict sine wave, compute MSE
2. **Select**: Sort by fitness (lower MSE = higher fitness)
3. **Reproduce**:
   - Elite (20%): Preserved unchanged
   - Survivors (30%): Pass through
   - Clone+Mutate (40%): Copies of elites with weight mutations
   - Random (10%): Fresh random genomes for exploration
4. **Repeat**: Track saturation emergence over generations

### Saturation Measurement

A neuron is classified as "saturated" if its activation magnitude exceeds 0.95:

```python
saturated = (activation.abs() > 0.95).float().mean()
```

We track:
- `k_mlp`: Number of saturated hidden neurons
- `k_ratio`: k_mlp / total hidden neurons
- Trajectory over training generations

## Key Findings

1. **Saturation is selective attention**: Networks saturate neurons to gate off irrelevant inputs, not because of compression pressure.

2. **Threshold effect**: Below ~100 irrelevant dimensions, continuous processing suffices. Above this threshold, saturation becomes necessary.

3. **Hybrid equilibrium**: Given excess capacity, networks converge to ~75-80% saturation, preserving some continuous neurons for fine-tuning.

4. **Efficiency**: Hybrid networks achieve the representational power of much larger continuous networks by combining discrete mode selection with continuous interpolation.

## Ultra-Sparse Connectivity: The Breakthrough

The most significant discovery from our experiments: **Ultra-Sparse connectivity enables gradient-free input selection**.

### The Problem

Dense gradient-free networks cannot learn which inputs matter. Given 256 inputs (16 true signals + 240 noise), a standard evolutionary network distributes weights proportionally across all inputs (~6% to true signals = 16/256). It achieves reasonable MSE by averaging, but cannot selectively attend to useful inputs.

### The Solution

Limit each hidden neuron to only K inputs (we use K=2). This creates **selection pressure** that weight evolution alone cannot provide:

```
Standard Dense:     Each neuron connects to ALL 256 inputs
Ultra-Sparse (K=2): Each neuron connects to only 2 inputs (must choose wisely)
```

### Why It Works

1. **Forced Selection**: With only 2 input slots, the network MUST choose the most useful inputs
2. **Evolvable Indices**: We mutate both weights AND which inputs each neuron connects to
3. **Selection Pressure**: Bad input choices lead to high MSE, driving evolution toward true signals
4. **Emergent Feature Selection**: Input 4 (sin(2x)) was selected in 85% of trials across 20 runs

### The Real Insight: Evolution Enables Feature Selection

We tested backprop on the same sparse architecture to isolate the contribution:

| Method | Best MSE | Can Discover Inputs? |
|--------|----------|---------------------|
| Backprop + Optimal Indices | **0.000191** | No (hand-picked) |
| Evolution (Ultra-Sparse) | 0.000303 | **Yes** |
| Backprop + Learnable Attention | 0.001828 | Poorly |
| Backprop + Random Indices | 0.006920 | No |

**Backprop with hand-picked optimal indices beats evolution!** But backprop cannot discover which indices matter. Differentiable soft attention fails to select true inputs (only 2-4 out of 16 true signals).

**Evolution's value = automatic feature selection**, not better optimization. When you don't know which inputs matter, evolution finds them.

### Results

| Method | MSE | Params | Memory | Ops/sample | Inference |
|--------|-----|--------|--------|------------|-----------|
| Backprop + Optimal Indices | **0.000191** | 33 | 132 B | 24 | 0.44 μs |
| **Ultra-Sparse (evolution)** | 0.000303 | **33** | **132 B** | **24** | **0.44 μs** |
| Backprop (dense) | 0.000003 | 2065 | 8260 B | 2056 | 0.58 μs |
| Standard SA (dense) | 0.009155 | 2065 | 8260 B | 2056 | 0.63 μs |

**Ultra-Sparse achieves:**
- **63x fewer parameters** than dense architectures
- **86x fewer operations** per inference
- **30x better MSE** than Standard SA (gradient-free)
- **Automatic feature selection** (4x better than random at selecting true inputs)

**What the best model actually selected:**
```
Neuron 4: [15, 4] = cos(8x), sin(5x)  ← Both true signals!
Neuron 1: [117, 4] = noise, sin(5x)
Neuron 6: [189, 4] = noise, sin(5x)
... (other neurons use noise)

Result: 4/16 connections to true signals (25% vs 6% random = 4x selection)
```

### Quick Start

```python
import torch
from numba import jit
import numpy as np

# Load pre-trained model
data = torch.load('models/ultra_sparse_mse0.000303.pt', weights_only=False)
state = data['state_dict']

# Fast inference with Numba (0.44 μs per sample)
@jit(nopython=True, fastmath=True)
def inference(x_expanded, indices, w1, b1, w2, b2):
    hidden = np.empty(8, dtype=np.float32)
    for h in range(8):
        acc = b1[h]
        for k in range(2):
            acc += x_expanded[indices[h, k]] * w1[h, k]
        hidden[h] = np.tanh(acc)
    out = b2
    for h in range(8):
        out += hidden[h] * w2[h]
    return np.tanh(out)
```

See `models/README.md` for complete usage guide.

---

## Additional Experiments

Beyond the original paper, we conducted additional experiments comparing gradient-free methods:

### Standard Metrics

All experiments report three metrics:
- **MSE**: Mean Squared Error (accuracy)
- **Energy**: Inference cost (activation + weight energy)
- **Saturation**: Percentage of neurons with |activation| > 0.95

### Saturation Tradeoffs

| Property | Backprop (0% sat) | Saturated (100% sat) |
|----------|-------------------|----------------------|
| MSE | 0.0001 | 0.02-0.03 |
| Noise Robustness | 1x | **170x better** |
| Inference Speed | 1x | **2.26x faster** |
| Memory | 1x | **75% smaller** |

### Key Discoveries

1. **Noise Correlation Issue**: The original 240 "noise" signals contained `-sin(x)` (r=-1.0) and other correlated signals. Networks exploited these shortcuts. With truly uncorrelated noise, gradient-free methods cannot learn input selection.

2. **True Signals Only**: Using only the 16 true signal inputs (no noise) achieves 10x better MSE with 15x fewer parameters.

3. **Sensory Bottleneck**: Adding architectural bottlenecks doesn't force input selection - networks still distribute weights proportionally across all inputs.

4. **Ultra-Sparse Connectivity**: Limiting inputs per neuron creates selection pressure, enabling gradient-free feature selection for the first time.

5. **Inference Engine Matters**: PyTorch overhead dominates sparse models. Numba JIT provides 60x speedup for Ultra-Sparse, making it faster than dense models.

See `experiments_log.md` for detailed results from all experiments.

## Extending This Work

The same principles apply to other domains:

- **Classification**: Present inputs temporally (video-like sequences) rather than static snapshots to give evolution consistent signals
- **Control**: High-dimensional sensor inputs naturally create selective attention pressure
- **Reinforcement learning**: Hybrid state representations may improve policy search efficiency

## Citation

```bibtex
@article{miller2026emergent,
  title={Emergent Hybrid Computation in Gradient-Free Evolutionary Networks},
  author={Miller, Payton},
  year={2026}
}
```

## License

AGPL-3.0 - See LICENSE file for details.

## Acknowledgments

Experiments conducted using GENREG (Genetic Regulatory Networks), a gradient-free evolutionary learning framework.
