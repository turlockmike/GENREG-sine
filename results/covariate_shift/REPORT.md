# Covariate Shift Experiment: GENREG vs SGD

## Hypothesis

**GENREG networks with saturated neurons will show step-function degradation under covariate shift, while SGD-trained networks will degrade smoothly.**

The reasoning: GENREG's evolutionary training tends to produce saturated neurons (|activation| > 0.95) that act as binary gates. When input distributions shift, these binary decisions should flip suddenly once inputs cross decision boundaries—causing abrupt accuracy drops. In contrast, SGD-trained networks have continuous activations that should shift gradually, producing smoother degradation curves.

## Methodology

### Data

Synthetic 4-class Gaussian clusters in 16D space:
- Classes separated in first 2 dimensions (quadrant pattern)
- Remaining 14 dimensions are noise
- 600 train / 200 val / 200 test samples
- Cluster std: 0.5, separation: 2.0

### Models

| Model | Architecture | Connectivity | Training | Parameters |
|-------|--------------|--------------|----------|------------|
| **SGD (baseline)** | 16→32→4 | Dense (all inputs) | Adam optimizer | ~600 |
| **GENREG (sparse)** | 16→32→4 | Sparse (K=4 inputs/neuron) | GSA + evolvable indices | 292 |

**GENREG Training Details:**
- Population size: 50
- SA steps per member: 20
- Generations: up to 300 (early stopping at 90% accuracy)
- Selection: Roulette wheel + 5% elitism
- Mutation: Weight perturbation (σ=0.1) + index swaps (15% rate)

### Shift Protocol

Two shift types tested:
1. **Translation**: Shift data in +x, +y direction (diagonal)
2. **Rotation**: Rotate first 2 dimensions around origin

Evaluation:
- Shift magnitude swept from 0 to 3.0 in 25 steps
- At each level: measured accuracy, confidence, saturation, gate configurations

### Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **AUC** | Area under accuracy-vs-shift curve | Higher = more robust |
| **Max Derivative** | Steepest drop in accuracy curve | More negative = more step-like |
| **Gate Flips** | Neurons changing saturation state | Step-function indicator |

## Results

### Training Performance

| Metric | SGD (Dense) | GENREG (Sparse) |
|--------|-------------|-----------------|
| Training Accuracy | 100% | 86% |
| Validation Accuracy | 100% | 90.5% |
| Parameters | ~600 | 292 |
| Convergence | 1 epoch | 8 generations |
| **Saturation** | N/A | **2.6%** |

### Degradation Under Shift

| Shift Type | SGD AUC | GENREG AUC | SGD MaxDeriv | GENREG MaxDeriv | Gate Flips |
|------------|---------|------------|--------------|-----------------|------------|
| Translation | 2.56 | 2.24 | -0.460 | -0.280 | 0 |
| Rotation | 0.75 | 0.69 | **-1.740** | -0.960 | 0 |

### Visualization

Generated plots in this directory:
- `degradation_translation.png` - Accuracy curves under translation shift
- `degradation_rotation.png` - Accuracy curves under rotation shift
- `gate_changes_translation.png` - GENREG gate configuration heatmap
- `gate_changes_rotation.png` - GENREG gate configuration heatmap

## Key Finding

**The hypothesis was NOT supported.** Contrary to expectations:

1. **SGD showed steeper degradation** (more negative max_derivative) than GENREG
2. **GENREG degraded more smoothly** across both shift types
3. **Zero gate flips** occurred—no step-function behavior observed

## Analysis: Why the Hypothesis Failed

The critical issue: **GENREG only reached 2.6% saturation** on this problem.

Without saturated neurons acting as binary gates, there's no mechanism for step-function degradation. This occurred because:

1. **Problem too easy**: 4-class Gaussian clusters with clear separation
2. **Fast convergence**: GENREG reached 90% accuracy in only 8 generations
3. **No need for commitment**: The network didn't need binary decisions—continuous activations sufficed

### Comparison to Digits Experiments

In our digits classification experiments, GENREG achieved 60-90% saturation because:
- 10-class problem is harder
- 64 input features with complex structure
- Required 300+ generations to converge

## Implications

1. **Saturation is problem-dependent**: Easy problems don't induce saturation
2. **Sparse connectivity may provide robustness**: GENREG's smoother degradation suggests sparse networks might actually be MORE robust to distribution shift
3. **To properly test the hypothesis**: Need a harder problem where GENREG naturally develops high saturation

## How to Run

### Run the experiment

```bash
cd /path/to/GENREG-sine
python experiments/covariate_shift.py
```

### Monitor progress

```bash
tail -f results/covariate_shift/progress.log
```

### Configuration

Edit the `ExperimentConfig` at the bottom of `experiments/covariate_shift.py`:

```python
config = ExperimentConfig(
    hidden_dim=32,          # Hidden layer size
    K=4,                    # Inputs per neuron (GENREG sparsity)
    target_accuracy=0.90,   # Early stopping threshold
    genreg_generations=300, # Max generations
    genreg_pop_size=50,     # Population size
    genreg_sa_steps=20,     # SA steps per member
    shift_types=['translation', 'rotation'],
    shift_max_magnitude=3.0,
    shift_n_steps=25,
)
```

### To test with higher saturation

Option 1: Use a harder problem (modify `DataConfig`):
```python
data_config=DataConfig(
    n_classes=10,           # More classes
    n_dims=64,              # Higher dimensionality
    cluster_std=1.0,        # More overlap
)
```

Option 2: Lower the target accuracy to force longer training:
```python
target_accuracy=0.99,       # Train longer
genreg_generations=500,     # More generations
```

## Future Work

To properly test the step-function degradation hypothesis:

1. **Use pre-trained high-saturation model** from digits experiments
2. **Create harder synthetic problem** that requires binary-like decisions
3. **Test on sine regression** where saturation naturally emerges (we observed 60-100% saturation)
4. **Artificially induce saturation** via regularization or temperature scaling

## Files

```
results/covariate_shift/
├── REPORT.md                      # This file
├── progress.log                   # Training log
├── results_YYYYMMDD_HHMMSS.json   # Full results data
├── degradation_translation.png    # Accuracy vs shift plot
├── degradation_rotation.png       # Accuracy vs shift plot
├── gate_changes_translation.png   # Gate configuration heatmap
└── gate_changes_rotation.png      # Gate configuration heatmap
```

## Date

Experiment conducted: January 30, 2026
