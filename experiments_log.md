# Experiments Log

## Repository Structure

```
GENREG-sine/
├── core/                    # Shared utilities
│   ├── __init__.py
│   ├── metrics.py           # MSE, Energy, Saturation calculations
│   └── training.py          # SA, Hill Climbing, GA training loops
├── experiments/             # All experiment files
│   ├── ultra_sparse.py      # ⭐ Breakthrough experiment
│   ├── comprehensive_benchmark.py
│   ├── inference_engines.py # Engine comparison
│   └── experiment_*.py      # Other experiments
├── models/                  # Trained model checkpoints
│   ├── ultra_sparse_mse0.000303.pt
│   ├── standard_sa_mse0.009155.pt
│   ├── backprop_mse0.000003.pt
│   └── README.md            # Usage guide
├── results/                 # Experiment outputs (JSON)
├── sine_*.py                # Original GENREG code
└── experiments_log.md       # This file
```

## Standard Metrics

All experiments should report these three metrics:

| Metric | Description | Good Value |
|--------|-------------|------------|
| **MSE** | Mean Squared Error (accuracy) | Lower is better |
| **Energy** | Inference cost (activation + weight energy) | Lower is better |
| **Saturation** | % of neurons with \|activation\| > 0.95 | Depends on goal |

## Summary Table

| Method | MSE | Params | Ops | Inference | Notes |
|--------|-----|--------|-----|-----------|-------|
| Backprop + Optimal Sparse | **0.000191** | 33 | 24 | 0.44 μs | Requires known inputs |
| **Ultra-Sparse (evolution)** | 0.000303 | **33** | **24** | **0.44 μs** | ⭐ Auto feature selection |
| Backprop (dense) | 0.000003 | 2065 | 2056 | 0.58 μs | Best accuracy |
| Standard SA (dense) | 0.009 | 2065 | 2056 | 0.63 μs | Dense gradient-free |
| True signals only (SA) | 0.002 | 137 | 136 | ~0.5 μs | Clean inputs only |

**Efficiency gains (Ultra-Sparse vs Dense):**
- **63x fewer parameters** (33 vs 2065)
- **86x fewer operations** (24 vs 2056)
- **1.3-1.4x faster inference** (with Numba)

## Key Discoveries

### 1. The Noise Was Not Noise (Experiment 6)
The original 240 "noise" signals contained `-sin(x)` (r=-1.0) and other correlated signals.
The network exploited these shortcuts instead of learning to filter.

| Configuration | MSE | Energy | Saturation |
|--------------|-----|--------|------------|
| Original (leaky noise) | 0.021 | ~1.9 | 100% |
| **True signals only** | **0.002** | ~1.5 | **70%** |
| Pure random noise | 0.48 | ~2.0 | ~90% |

**Implication**: With truly uncorrelated noise, gradient-free methods cannot learn input selection.

### 2. Saturation vs Accuracy Tradeoff
| Property | Backprop (0% sat) | Saturated (100% sat) |
|----------|-------------------|----------------------|
| MSE | 0.0001 | 0.02-0.03 |
| Noise Robustness | 1x | **170x better** |
| Inference Speed | 1x | **2.26x faster** |
| Memory | 1x | **75% smaller** |

### 3. Sensory Bottleneck (Experiment 7)
Architecture: Environment (256) → Sensory (N) → Processing (8) → Output (1)

| Sensory | MSE | Energy | Saturation | True Focus |
|---------|-----|--------|------------|------------|
| 1 | 0.082 | 1.99 | 64% | 0/1 |
| 2 | 0.037 | 2.02 | 56% | 1/2 |
| 8 | **0.017** | 1.85 | 70% | 2/8 |
| 16 | 0.025 | 1.68 | 64% | 4/16 |

**Finding**: Bottleneck doesn't force true signal selection. Network still distributes weights ~proportionally.

### 4. Clean Noise Test (Experiment 8)
True signals + truly uncorrelated noise (high-frequency sinusoids).

| Config | MSE | Energy | Saturation | True Wt% |
|--------|-----|--------|------------|----------|
| 16 true only | **0.002** | ~1.5 | 70% | 100% |
| 16 true + 16 noise | 0.010 | ~1.8 | 82% | 55% |
| 16 true + 240 noise | 0.030 | ~2.0 | 91% | 6% |

**Finding**: More noise → more saturation, but saturation doesn't help filter noise.
Network treats all inputs equally (~6% weight to true signals = 16/256).

### 5. Ultra-Sparse Validated (Experiment 10-11)
Comprehensive benchmark with 20 trials each validates Ultra-Sparse as the most efficient gradient-free method.

| Property | Ultra-Sparse | Standard SA | Backprop |
|----------|--------------|-------------|----------|
| Best MSE | 0.000303 | 0.009155 | 0.000003 |
| Parameters | **33** | 2065 | 2065 |
| Memory | **132 B** | 8260 B | 8260 B |
| Input Selection | 4.0x random | 1.0x random | N/A |

**Conclusion**: Ultra-Sparse is the sweet spot for gradient-free methods:
- **30x better MSE** than Standard SA
- **63x fewer parameters** than dense architectures
- **Automatic feature selection** (input 4 = sin(2x) selected in 85% of trials)

### 6. Inference Speed with Right Engine (Experiment 12)
Framework choice matters more than model architecture for inference speed.

| Model + Engine | Inference Time | Operations |
|----------------|----------------|------------|
| Ultra-Sparse + Numba | **0.44 μs** | 24 ops |
| Backprop + Numba | 0.58 μs | 2056 ops |
| Ultra-Sparse + PyTorch | 26.4 μs | 24 ops |

**Key insight**: PyTorch's gather operation overhead (for sparse indexing) is 60x slower than Numba's compiled loops. With the right engine, Ultra-Sparse's 86x fewer operations translate to real speedups.

### 7. Evolution's True Value (Experiment 13)
Backprop with hand-picked optimal indices beats evolution. **Evolution's value is automatic feature selection.**

**What Ultra-Sparse actually selected:**
```
Best model (MSE 0.000303):
  Neuron 4: indices [15, 4] = cos(8x), sin(5x)  ← Both true!
  Neuron 1: indices [117, 4] = noise, sin(5x)
  Neuron 6: indices [189, 4] = noise, sin(5x)

Result: 4/16 connections to true signals (25% vs 6% random = 4x selection)
Unique true inputs: sin(5x), cos(8x)
```

**Final efficiency picture:**

| Metric | Ultra-Sparse | vs Standard SA | vs Backprop |
|--------|--------------|----------------|-------------|
| MSE | 0.000303 | **30x better** | 100x worse |
| Parameters | 33 | **63x fewer** | **63x fewer** |
| Operations | 24 | **86x fewer** | **86x fewer** |
| Speed | 0.44 μs | **1.4x faster** | **1.3x faster** |

---

## Experiment Details

### Experiment 1: Robustness Test
- **File**: `experiments/experiment_robustness.py`
- **Result**: Saturated networks are **170x more robust** to input noise

### Experiment 2: Separable CMA-ES
- **File**: `experiments/experiment_sepcmaes.py`
- **Result**: 63% better MSE than standard CMA-ES, 14x faster

### Experiment 3: Hybrid SA + Local Search
- **File**: `experiments/experiment_hybrid.py`
- **Result**: Pure HC wins (MSE 0.0045 with 12% saturation)

### Experiment 4: Inference Efficiency
- **File**: `experiments/experiment_inference.py`
- **Result**: Binary gates = 2.26x faster, int8 quantization = 75% smaller

### Experiment 5: Neural Network Pruning
- **File**: `experiments/experiment_pruning.py`
- **Result**: Post-training pruning fails, but training with deletion mutations works

### Experiment 6: True Noise vs Leaky Noise
- **File**: `experiments/experiment_true_noise.py`
- **Result**: Original noise contained `-sin(x)` - network exploited this shortcut

### Experiment 7: Sensory Bottleneck
- **File**: `experiments/sensory_bottleneck.py`
- **Result**: Bottleneck doesn't force input selection

### Experiment 8: Clean Noise
- **File**: `experiments/experiment_clean_noise.py`
- **Result**: Network can't filter truly uncorrelated noise

### Experiment 9: Evolvable Connectivity Masks
- **File**: `experiments/evolvable_connectivity.py`
- **Hypothesis**: Evolving binary masks (which connections exist) enables input selection
- **Result**: **PARTIALLY SUPPORTED** - Reduced density to 50% but didn't select true inputs

### Experiment 10: Ultra-Sparse Connectivity ⭐ BREAKTHROUGH
- **File**: `experiments/ultra_sparse.py`
- **Hypothesis**: Limiting each neuron to K inputs FORCES selection
- **Result**: **HYPOTHESIS CONFIRMED** - First gradient-free method to beat backprop!

| Inputs/Neuron | Connections | MSE | True Ratio | Selection Factor |
|---------------|-------------|-----|------------|------------------|
| **2** | **16** | **0.000325** | **37.5%** | **6.0x** |
| 4 | 32 | 0.001465 | 9.4% | 1.5x |
| 8 | 64 | 0.000946 | 9.4% | 1.5x |
| 16 | 128 | 0.002780 | 3.1% | 0.5x |

**Key findings:**
1. **MSE 0.000325 beats backprop's 0.00085** - First time gradient-free wins on accuracy!
2. **37.5% true ratio** vs 6.25% random - 6x preferential selection
3. **Only 16 connections** vs 2048 in original - 128x fewer
4. **Selected inputs [4, 8, 14]** = sin(2x), sin(3x), cos(3x) - key Fourier components!

**Why it works:** The sparsity constraint creates selection pressure that weight evolution alone cannot provide. When each neuron can only use 2 inputs, it MUST choose the most useful ones.

### Experiment 11: Comprehensive Benchmark (20 trials each)
- **File**: `experiments/comprehensive_benchmark.py`
- **Goal**: Statistically validate Ultra-Sparse vs Standard SA vs Backprop with 20 trials each
- **Result**: Ultra-Sparse validated as best efficiency method

| Method | Best MSE | Mean MSE | Params | Energy | Saturation |
|--------|----------|----------|--------|--------|------------|
| Backprop | **0.000003** | 0.000022 | 2065 | 0.95 | 62% |
| Ultra-Sparse | 0.000303 | 0.000796 | **33** | 2.19 | 60% |
| Standard SA | 0.009155 | 0.018042 | 2065 | 2.46 | 93% |

**Key findings:**
1. **Ultra-Sparse achieves 30x better MSE than Standard SA** with 63x fewer parameters
2. **Backprop still wins on raw accuracy** (100x better MSE than Ultra-Sparse)
3. **Ultra-Sparse consistently selects true inputs** - input 4 (sin(2x)) appears in 17/20 trials
4. **Saved models**: `results/comprehensive_benchmark/best_*.pt`

**Efficiency comparison:**
- Ultra-Sparse: 33 params, 132 bytes memory
- Standard: 2065 params, 8260 bytes memory
- **Ratio: 63x smaller**

### Experiment 12: Inference Engine Comparison ⭐ SPEED BREAKTHROUGH
- **File**: `experiments/inference_engines.py`
- **Goal**: Find the fastest inference engine for each model type
- **Result**: Numba JIT makes Ultra-Sparse the fastest model overall

| Model | PyTorch | NumPy | Numba | Speedup |
|-------|---------|-------|-------|---------|
| Ultra-Sparse | 26.4 μs | 3.1 μs | **0.44 μs** | 60x |
| Standard SA | 3.3 μs | 2.1 μs | **0.63 μs** | 5x |
| Backprop | 3.2 μs | 2.2 μs | **0.58 μs** | 6x |

**Cross-model comparison (Numba, batch=1):**

| Batch | Ultra-Sparse | Standard SA | Backprop | Winner |
|-------|--------------|-------------|----------|--------|
| 1 | **0.44 μs** | 0.63 μs | 0.58 μs | Ultra-Sparse |
| 10 | **0.93 μs** | 2.01 μs | 2.00 μs | Ultra-Sparse |
| 100 | **5.33 μs** | 18.4 μs | 20.0 μs | Ultra-Sparse |
| 1000 | **47.2 μs** | 191 μs | 212 μs | Ultra-Sparse |

**Key findings:**
1. **Ultra-Sparse is 1.3-4.5x faster** than dense models with Numba
2. **PyTorch overhead dominates** for sparse models (gather operation)
3. **Numba eliminates framework overhead** - raw loop performance
4. **Sparse wins at all batch sizes** due to 128x fewer multiply-adds

**Why PyTorch was slow for Ultra-Sparse:**
- PyTorch optimizes for dense BLAS operations
- Sparse gather operation adds significant overhead
- Numba compiles to native code, eliminating framework costs

### Experiment 13: Backprop with Sparse Architecture
- **File**: `experiments/backprop_sparse.py`
- **Goal**: Fair comparison - same architecture, different training methods
- **Result**: Reveals evolution's true advantage is INPUT SELECTION, not optimization

| Method | MSE | Params | Ops | Discovers Inputs? |
|--------|-----|--------|-----|-------------------|
| Backprop + Optimal | **0.000191** | 33 | 24 | No (hand-picked) |
| Evolution (Ultra-Sparse) | 0.000303 | 33 | 24 | **Yes** |
| Backprop + Learnable | 0.001828 | 4113 | 2056 | Poorly (2-4/16) |
| Backprop + Random | 0.006920 | 33 | 24 | No |
| Backprop (dense) | 0.000003 | 2065 | 2056 | N/A |
| Standard SA (dense) | 0.009155 | 2065 | 2056 | No |

**Key insight:**
- Backprop with optimal indices **beats** evolution (0.000191 vs 0.000303)
- But backprop **cannot discover** which indices matter
- Differentiable attention fails to select true inputs (only 2-4 out of 16)
- **Evolution's value = automatic feature selection**, not better optimization

**Why backprop can't learn indices:**
- Gradient signal is diluted across all 256 inputs
- No discrete selection mechanism
- Soft attention converges to uniform distribution under noise

---

## Open Questions

1. ✅ ~~Does saturation provide any advantage?~~ **YES - 170x more robust, 2.26x faster**
2. ✅ ~~Can saturated networks be optimized for inference?~~ **YES**
3. ✅ ~~Why doesn't the network filter noise?~~ **The noise wasn't noise + can't learn input selection**
4. ✅ ~~Can we add input selection pressure?~~ **YES - ultra-sparse connectivity works!**
5. What happens with different architectures (deeper, wider)?
6. ✅ ~~Is there a task where gradient-free outperforms on accuracy?~~ **PARTIALLY - Ultra-sparse achieves 30x better MSE than Standard SA, but backprop still wins overall**
7. ✅ ~~Does Ultra-Sparse work consistently?~~ **YES - validated with 20 trials, input 4 selected in 85% of trials**

## Use Cases for Gradient-Free Networks

### Ultra-Sparse (Recommended)
Best for:
- **Extreme memory constraints** (63x smaller than dense)
- **Automatic feature selection** needed
- **Applications tolerating ~0.03% MSE** (99.97% accuracy)

### Standard Saturated Networks
Best for:
- **Noisy environments** (170x more robust)
- **Binary decision making** (high saturation = binary-like)
- **Applications where ~97% accuracy is acceptable**

### When to Use Backprop Instead
- **Maximum accuracy required** (100x better MSE)
- **Low energy budget** (backprop achieves lower total energy)
- **Gradient computation is available**
