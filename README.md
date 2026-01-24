# GENREG-sine

Gradient-free neural network training through evolutionary selection. Code for the paper "Emergent Hybrid Computation in Gradient-Free Evolutionary Networks."

## Overview

This repository demonstrates that neural networks trained via evolutionary selection (rather than gradient descent) spontaneously develop hybrid digital-analog representations. Saturated neurons emerge as discrete binary switches while continuous neurons provide fine-grained modulation. This is a computational structure that gradient-based methods systematically avoid due to vanishing gradients.

The key insight: **saturation is not a bug, it is a learned selective attention mechanism.**

When a network must attend to a subset of its inputs (filtering irrelevant dimensions, focusing on task-relevant signals), evolution discovers that saturating neurons into binary gates is an efficient solution. This creates a hybrid state space of 2^k discrete regions with continuous interpolation within each region.


### Doing More With Less

Standard deep learning uses massive networks because gradient descent requires smooth, high-dimensional landscapes to navigate. GENREG achieves comparable results with drastically smaller architectures:

| Task | Standard Approach | GENREG |
|------|-------------------|--------|
| Sine approximation | 64+ neurons | 8 neurons |  -- yes I know gradients can probably do it with extreme fine tuning and drop out, I'm not saying they can't, I'm saying its difficult. 
| Humanoid locomotion | 256+ neurons | 16 neurons | -- This model is currently training and has reached 3 meters, evolution is slow as to why this github is not available yet. current training has been runing for 18 hours as of this publication. 
| Function approximation | Millions of params | Thousands of params |

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
├── paper.pdf                 # Full paper
├── sweep_summary.json        # All experimental results
├── sine_config.py            # Configuration parameters
├── sine_train.py             # Main training script
├── sine_sweep.py             # Systematic sweep across configurations
├── sine_population.py        # Population management and evolution
├── sine_genome.py            # Individual genome (MLP + proteins)
├── sine_mlp.py               # MLP controller
├── sine_proteins.py          # Protein cascade (trust mechanism)
├── assets/                   # Figures and visualizations
│   ├── noise_dimensions_chart.png
│   └── hidden_size_chart.png
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
