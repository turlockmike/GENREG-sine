# Experiments Log

## Summary Table

| Method | Best MSE | Mean MSE | Saturation (k%) | Time | Evals | Notes |
|--------|----------|----------|-----------------|------|-------|-------|
| Backprop (Adam) | 0.00085 | - | 0% | <1s | ~100 epochs | Gold standard for accuracy |
| **Hill Climbing (20k)** | **0.0045** | 0.0062 | **12%** | ~30s | 20k | Surprising winner! Low saturation |
| Genetic Algorithm | 0.0127 | - | 100% | ~30s | 40×947 gens | Original GENREG approach |
| Simulated Annealing | 0.015 | 0.018 | 100% | 19s | 20k | Consistent saturation |
| sep-CMA-ES | 0.0247 | - | 100% | 7s | 20k | 63% better than standard CMA-ES |
| Hill Climbing (5k) | 0.042 | 0.060 | 50-100% | 1.4s | 5k | Gets stuck in local minima |
| CMA-ES (standard) | 0.067 | 0.077 | 100% | 100s | 20k | Too many params (2065) |

## Experiment Details

### 1. Genetic Algorithm (Original GENREG)
- **File**: `sine_sweep.py`
- **Config**: Population=40, max_gens=2000, elite=20%, mutate=40%
- **Result**: MSE=0.0127, k=8/8 (100% saturation)
- **Converged**: ~947 generations
- **Key finding**: Saturation emerges naturally under selection pressure

### 2. Hill Climbing
- **File**: `sine_hillclimb.py`
- **Config**: 5 restarts, 1000 steps/restart, patience=150
- **Result**: Best MSE=0.042, mean=0.060
- **Saturation**: Variable (50-100%)
- **Key finding**: Gets stuck in local minima, high variance between restarts

### 3. Simulated Annealing
- **File**: `sine_annealing.py`
- **Config**: T_initial=0.01, T_final=0.00001, 20k steps, exponential cooling
- **Result**: Best MSE=0.015, mean=0.018
- **Saturation**: 100%
- **Key finding**: Temperature tuning critical. T=0.1 too high (random walk), T=0.01 good

### 4. CMA-ES
- **File**: `sine_cmaes.py`
- **Config**: sigma0=0.5, 20k evals, auto popsize
- **Result**: Best MSE=0.067, mean=0.077
- **Saturation**: 100%
- **Key finding**: Failed due to high dimensionality (2065 params). Sigma never converged.

### 5. Compression Test
- **File**: `compression_test.py`
- **Result**: Backprop wins at all hidden sizes (8,6,4,3,2 neurons)
- **Key finding**: Backprop achieves MSE~0.0001 even with 2 neurons

## Network Architecture
- Input: 256D (16 true signals + 240 noise)
- Hidden: 8 neurons (tanh)
- Output: 1 neuron (tanh)
- Parameters: 2065 total (256×8 + 8 + 8×1 + 1)
- Compression ratio: 32:1

## Key Observations

1. **Backprop dominates on MSE** - 10-100x better than any gradient-free method
2. **Saturation is gradient-free signature** - All evolutionary methods → 100% saturation, Backprop → 0%
3. **Simpler optimizers work better** - SA beat CMA-ES despite being "dumber"
4. **Temperature/hyperparams critical** - SA failed with T=0.1, worked with T=0.01

## New Experiments (Priority Tests)

### Experiment 1: Robustness Test
- **File**: `experiment_robustness.py`
- **Hypothesis**: Saturated networks are more robust to input noise
- **Result**: **HYPOTHESIS SUPPORTED**
  - Backprop wins on absolute MSE at all noise levels
  - But degradation ratio tells a different story:
    - Backprop degrades **3650x** at noise=1.0 vs clean
    - Evolutionary degrades only **21x** at noise=1.0 vs clean
  - Saturated networks are **170x more robust** to noise!

### Experiment 2: Separable CMA-ES
- **File**: `experiment_sepcmaes.py`
- **Hypothesis**: sep-CMA-ES scales better to high dimensions
- **Result**: **HYPOTHESIS SUPPORTED**
  - sep-CMA-ES: MSE=0.0247, time=7s
  - standard CMA-ES: MSE=0.067, time=100s
  - **63% improvement**, 14x faster

### Experiment 3: Hybrid SA + Local Search
- **File**: `experiment_hybrid.py`
- **Hypothesis**: Combining SA exploration + HC refinement beats either alone
- **Result**: **HYPOTHESIS NOT SUPPORTED** - Pure HC won!
  - Pure HC: MSE=0.0045, saturation=12%
  - Pure SA: MSE=0.015, saturation=100%
  - Hybrid 70/30: MSE=0.013, saturation=100%
- **Key Insight**: When not "locked" into saturation early, HC finds better MSE.
  SA's temperature annealing pushes solutions into saturated regions that are
  robust but less accurate. This is a tradeoff, not a failure.

## Open Questions

1. ~~Does saturation provide any advantage (robustness, interpretability)?~~ **YES - 170x more robust to noise**
2. ~~Can we close the gap with backprop using better gradient-free methods?~~ **Partially - sep-CMA-ES helps**
3. What happens with different architectures (deeper, wider)?
4. Is there a task where saturated solutions outperform on accuracy (not just robustness)?
