# TODO - Future Experiments

---

## High Priority

### 1. MNIST Benchmark ⭐ READY TO RUN
**Question**: Can GENREG match or beat standard MNIST benchmarks with a much smaller network?

**Status**: Code ready at `experiments/mnist_benchmark.py`

**Run commands** (for GPU machine):
```bash
# List all configs
uv run python experiments/mnist_benchmark.py --list

# Run all 18 experiments in parallel
for H in 64 128 256; do
  for K in 4 8 16; do
    for seed in 0 1; do
      uv run python experiments/mnist_benchmark.py --config mnist_H${H}_initK${K}_seed${seed} &
    done
  done
done

# Or run single experiment
uv run python experiments/mnist_benchmark.py --config mnist_H128_initK8_seed0
```

**Configs**: 18 total (H=[64,128,256] × init_K=[4,8,16] × 2 seeds)
- Uses VariableK (K can grow/shrink during training)
- pop=50, 1000 generations, 10k training samples
- Results output to `results/live/mnist_*.csv`

**Baseline targets to beat**:

| Model | MNIST Accuracy | Parameters | Target |
|-------|----------------|------------|--------|
| Logistic Regression | 92% | 7,850 | Beat with fewer params |
| MLP (784→128→10) | 97-98% | ~100k | Match with 10× fewer |
| MLP (784→32→10) | ~95% | ~25k | Match with 5× fewer |
| **GENREG target** | **>95%** | **<5,000** | 5× more efficient |

**Success criteria**:
- **Minimum**: >92% accuracy (beat logistic regression)
- **Good**: >95% with <5,000 params (5× more efficient than small MLP)
- **Excellent**: >97% with <10,000 params (10× more efficient than standard MLP)

---

### 2. Mini-Batch Fitness Evaluation
**Question**: Does noisy fitness (evaluating on random subset) help exploration?

**Current**: Evaluate on ALL training examples every time (exact but slow)
**Proposed**: Evaluate on random batch of 64-128 examples (noisy but fast)

| Aspect | Full Batch | Mini-Batch |
|--------|-----------|------------|
| Speed | 1x | ~10-20x faster |
| Noise | None | Stochastic |
| Exploration | Deterministic | May escape local optima |

**Biological analogy**: Real organisms don't get perfect fitness signals - noisy evaluation is more realistic.

---

### 3. Real-World Datasets
**Question**: Does Ultra-Sparse + SA work on real data with unknown feature importance?

**Candidates**:
- UCI California Housing (8 features)
- sklearn make_regression (configurable)
- Gene expression data (high-dim, few samples)

---

## Environmental Pressure & Adaptive Mutation

**Core Insight**: Fitness isn't absolute - it's relative to environmental pressure. Harsh environments create different selection dynamics than lenient ones.

### Experiment F: Survival Thresholds (Fixed)
**Question**: How does a fixed survival threshold affect final accuracy and efficiency?

| Condition | Threshold | Prediction |
|-----------|-----------|------------|
| Lenient | 0.0 (none) | Baseline - current behavior |
| Moderate | 0.50 | Faster convergence, less diversity |
| Harsh | 0.75 | Higher accuracy OR population collapse |
| Extreme | 0.85 | May fail to bootstrap |

### Experiment G: Extinction Events (Sudden Threshold Increases)
**Question**: Can populations survive and adapt to sudden increases in survival requirements?

### Experiment H: Stagnation-Adaptive Mutation
**Question**: Does increasing mutation rate during plateaus help escape local optima?

### Experiment I: Extinction-Adaptive Mutation (Hypermutation)
**Question**: Does hypermutation help populations survive extinction events?

### Experiment J: Combined Adaptive System
**Question**: What's the optimal combination of environmental pressure + adaptive mutation?

---

## Medium Priority

### Efficiency-Aware Fitness Function
**Question**: Can we explicitly reward efficiency, not just accuracy?

**Current approach**: Fitness = -MSE (accuracy only)

**Proposed approaches**:
```python
# Option 1: Penalty term
fitness = -MSE - λ * num_params

# Option 2: Efficiency ratio
fitness = accuracy / log(params)

# Option 3: Pareto optimization
# Track (accuracy, params) pairs, evolve toward Pareto frontier
```

---

### Compare to FS-NEAT Directly
**Question**: How does Ultra-Sparse + SA compare to FS-NEAT on the same problems?

**Rationale**: FS-NEAT is the closest related work - both use evolution for feature selection.

---

### Scaling SA to Larger Networks
**Question**: At what network size does SA become impractical?

**Experiment**: Increase hidden size: 8 → 16 → 32 → 64 → 128

---

## Low Priority / Exploratory

### Hybrid: SA for Indices, Backprop for Weights
**Question**: Can we get best of both worlds?

**Idea**: Use SA to evolve which inputs each neuron sees, but use backprop to optimize weights.

---

### Theoretical Analysis
**Question**: Can we prove why the K constraint creates selection pressure?

**Approach**: Information-theoretic analysis of sparse vs dense connectivity under gradient-free optimization.

---

## Completed (see docs/experiments/ for reports)

- Population-Based SA (GSA) - 95.3% on sklearn digits
- Data Augmentation - negative result, hurts small images
- Ablation Study - all 3 components proven essential
- Selection Pressure vs K - K=4 optimal
- Deeper Networks - depth hurts, L=1 optimal
- VariableK - discovers optimal K automatically
- Seed Exploration - more seeds beats larger population
