# TODO - Future Experiments

---

## Proving Novelty: The GENREG Hypothesis

### Core Hypothesis

**Fixed-K sparse connectivity + evolvable indices + gradient-free optimization creates selection pressure that existing methods cannot achieve.**

The three components work together:
1. **Fixed K constraint**: Creates scarcity - each neuron can only "afford" K inputs
2. **Evolvable indices**: Allows exploration of which inputs to use
3. **Gradient-free (SA)**: Evaluates whole-network fitness, not per-weight gradients

**Why this combination is novel:**
- SET/DST: Uses gradients + random regrowth (no fitness-guided index selection)
- Soft attention: Gradients diluted across all inputs (no hard constraint)
- Pruning: Starts dense, removes connections (no fixed K throughout)
- NEAT: Grows topology (opposite direction - adds complexity)

### What We Need to Prove

| Claim | Evidence Needed | Status |
|-------|-----------------|--------|
| Fixed K creates selection pressure | Compare K=2 vs K=32 on same problem | ✅ Done (K=4 is 6.5x better than K=32) |
| SA outperforms backprop with same architecture | Same sparse arch, different training | ✅ Done (3.7x on Friedman1) |
| Index evolution is essential | Ablation: freeze indices vs evolve | ✅ Done (10x better with evolution) |
| Combination beats SET-style random regrowth | Direct comparison | ✅ Done (13x better than random) |
| Scales to high-dimensional problems | Test 1000+ features | ✅ Done (5.7x better) |

### Experiments to Prove Novelty

#### Experiment A: Ablation Study (CRITICAL)
**Goal**: Prove each component is necessary

| Variant | Fixed K | Evolvable Indices | SA Weights | Expected Result |
|---------|---------|-------------------|------------|-----------------|
| Full GENREG | ✅ | ✅ | ✅ | Best selection |
| No index evolution | ✅ | ❌ (frozen) | ✅ | Poor selection (like random backprop) |
| Backprop weights | ✅ | ❌ | ❌ (backprop) | Moderate (can optimize weights but not indices) |
| Random regrowth (SET-style) | ✅ | ❌ (random) | ✅ | Worse than guided evolution |
| Large K (K=32) | Weak | ✅ | ✅ | Weak selection (approaches dense) |

**Key comparison**: Full GENREG vs "No index evolution" proves that evolvable indices are essential.

#### Experiment B: Index Evolution Dynamics
**Goal**: Show HOW indices evolve toward true features

Track across training:
- Which indices each neuron uses at step 0, 10k, 50k, 100k
- Probability of true feature in index set over time
- "Discovery events" - when a true feature first enters the network

**Expected**: True features should accumulate over time, not appear randomly.

#### Experiment C: Direct Comparison to SET
**Goal**: Prove GENREG's guided evolution beats SET's random regrowth

Setup:
- Same problem (1000 features, 10 true)
- Same sparsity level (32 connections)
- GENREG: SA + index mutations
- SET-style: SA + random drop/add (magnitude-based pruning + random regrowth)

**Expected**: GENREG finds more true features because evolution is fitness-guided.

#### Experiment D: The K Threshold
**Goal**: Find the critical K where selection pressure emerges

Test K = 1, 2, 4, 8, 16, 32, 64, 128 on same problem
Plot: Selection factor vs K

**Expected**: Sharp transition - selection pressure emerges below some critical K.

#### Experiment E: Gradient-Free is Essential
**Goal**: Prove SA enables index evolution in ways backprop cannot

Compare:
1. SA weights + evolving indices (GENREG)
2. Backprop weights + evolving indices (hybrid)
3. Backprop weights + learned attention (differentiable)

**Question**: Can backprop work WITH index evolution, or does SA's global fitness signal matter?

---

## High Priority

### 0. Training Improvements ⭐ TOP PRIORITY

Two techniques to try before MNIST:

#### A. Data Augmentation
**Question**: Does augmenting training data improve generalization without changing the algorithm?

**Technique**: Random transforms during fitness evaluation:
- Rotation: ±10°
- Shift: ±10% in x/y
- Zoom: ±10%
- (Optional) Elastic deformation

**Expected benefit**: +2-5% accuracy, more robust networks

**Implementation**:
```python
def augment_batch(X):
    # Apply random rotation, shift, zoom to each image
    return X_augmented

# During fitness eval
pred = controller.forward(augment_batch(X_train))
```

#### B. Mini-Batch Fitness Evaluation
**Question**: Does noisy fitness (evaluating on random subset) help exploration?

**Current**: Evaluate on ALL 1437 training examples every time (exact but slow)
**Proposed**: Evaluate on random batch of 64-128 examples (noisy but fast)

| Aspect | Full Batch | Mini-Batch |
|--------|-----------|------------|
| Speed | 1x | ~10-20x faster |
| Noise | None | Stochastic |
| Exploration | Deterministic | May escape local optima |

**Biological analogy**: Real organisms don't get perfect fitness signals - noisy evaluation is more realistic.

**Implementation**:
```python
batch_idx = np.random.choice(len(X_train), size=128, replace=False)
pred = controller.forward(X_train[batch_idx])
fitness = -torch.mean((pred - y_onehot[batch_idx]) ** 2).item()
```

**Experiment**: Compare full-batch vs mini-batch (64, 128, 256) on digits

---

### 1. Population-Based SA (Genetic Simulated Annealing) ⭐ IMPLEMENTED
**Question**: Can running multiple SA chains with selection pressure solve harder problems like digits?

**Status**: ✅ IMPLEMENTED - 87.2% accuracy (vs single SA's 64.7%)

**File**: `experiments/gsa_digits.py`

**Best Result So Far**:
- H=64, K=16, Pop=50, 300 generations
- **Test accuracy: 87.2%** (vs 64.7% single SA, 97% Dense)
- **1738 params** (5.2x fewer than Dense's 8970)
- +22.5 percentage point improvement over single SA!

**Next steps to reach 90%**:
- Try even longer training (500+ generations)
- Larger population (100+)
- Tune seed fraction and SA steps per generation

**Ablation: GSA vs Random Restarts** ✅ COMPLETE
- See: `docs/experiments/2026-01-30_gsa_ablation.md`
- Result: GSA beats random restarts 71.9% vs 68.3% (+3.6pp)

**Background**: Du et al. (2018) show that GSA outperforms both pure GA and pure SA for sparse network optimization because:
- SA is fast at local refinement (Monte Carlo acceptance)
- Selection pressure prevents wasting time on chains stuck in bad local optima

---

### 2. MNIST Benchmark ⭐ KEY GOAL
**Question**: Can GENREG match or beat standard MNIST benchmarks with a much smaller network?

**Why MNIST matters**:
- 784 input features (12× more than digits) → Tests sparse selection at scale
- Well-established benchmark → Easy comparison with literature
- If GENREG wins on params-per-accuracy, strong evidence for the approach

**Baseline targets to beat**:

| Model | MNIST Accuracy | Parameters | Target |
|-------|----------------|------------|--------|
| Logistic Regression | 92% | 7,850 | Beat with fewer params |
| MLP (784→128→10) | 97-98% | ~100k | Match with 10× fewer |
| MLP (784→32→10) | ~95% | ~25k | Match with 5× fewer |
| **GENREG target** | **>95%** | **<5,000** | 5× more efficient |

**Challenges**:
- 784 features → Larger search space for index selection
- Forward passes slower (784 vs 64 inputs)
- May need larger H to handle complexity
- Training time will scale up

**Experiment plan**:
1. Start with architecture sweep: H=[32, 64, 128], K=[8, 16, 32]
2. Use stopping criteria from extended training experiment (gen 500-900)
3. Compare params-per-accuracy against baselines
4. If plateau too low, try environmental pressure experiments (F-J)

**Success criteria**:
- **Minimum**: >92% accuracy (beat logistic regression)
- **Good**: >95% with <5,000 params (5× more efficient than small MLP)
- **Excellent**: >97% with <10,000 params (10× more efficient than standard MLP)

**File to create**: `experiments/mnist_benchmark.py`
- No need for complex crossover (which is hard to define for sparse masks)

**Algorithm from Paper (Algorithm 1 & 2)**:
```python
# Parameters from paper
population_size = 100
mutation_rate = 0.025
temperature = 0.01
cooling_rate = 0.98
iterations = 500

population = [UltraSparseController() for _ in range(population_size)]

for iteration in range(iterations):
    # 1. Evaluate fitness of all controllers
    scores = [evaluate(c) for c in population]

    # 2. Natural Selection
    #    - Seed selection: Keep best 5% unchanged
    #    - Roulette selection: Probabilistic for rest 95%
    sorted_pop = sorted(zip(population, scores), key=lambda x: -x[1])
    seeds = [c for c, s in sorted_pop[:int(0.05 * population_size)]]
    rest = roulette_select(sorted_pop, int(0.95 * population_size))
    population = seeds + rest

    # 3. Mutation with Monte Carlo acceptance (per gene)
    for controller in population:
        for gene in controller.genes:  # Each index or weight
            if random() < mutation_rate:
                mutant = mutate_gene(gene)
                delta = fitness(mutant) - fitness(original)
                if delta > 0 or random() < exp(delta / temperature):
                    accept_mutation()

    # 4. Cool temperature
    temperature *= cooling_rate
```

**Key Differences from Current GENREG**:
1. **Population**: We run 1 chain, paper uses 100
2. **Selection**: We have none, paper uses seed (5%) + roulette (95%)
3. **Per-gene mutation**: Paper mutates individual genes with Monte Carlo acceptance

**Expected Benefits**:
- Multiple chains explore different index combinations simultaneously
- Selection favors chains that find useful features
- Could solve digits (10-class) where single SA fails

**Reference**: Du et al. "A Genetic Simulated Annealing Algorithm to Optimize the Small-World Network Generating Process" (Complexity, 2018)

---

### 2. Extended Training on Optimal Config (K=4, 5000 generations)
**Question**: How far can we push accuracy with much longer training on the sweet spot config?

**Current status**:
- H=32, K=4, 300 gens → **85.6%** (490 params)
- H=32, K=4, 500 gens → Not yet tested

**Experiment**: Run H=32, K=4 with 5000 generations (10x current)
- Track learning curve - does it plateau or keep improving?
- Compare to K=2 with same compute budget
- Measure final saturation levels

**Expected insights**:
- Is 85-86% the asymptotic limit for this architecture?
- Does extended training increase or decrease saturation?
- Worth the 10x compute cost?

**File to create**: `experiments/very_extended_training.py`

---

### 3. Solve Digits (10-class classification) ✅ SOLVED (95.0%)
**Question**: What does GENREG need to achieve >90% on digits?

**Status**: ✅ ACHIEVED 95.0% - only 2pp below dense backprop!

**Phase 1 - Single Variable Ablation** (2026-01-30):
| Config | Accuracy | Key Change |
|--------|----------|------------|
| **pop100** | **92.2%** | Population 50→100 |
| **idx0.05** | **91.7%** | Index swap 0.1→0.05 |
| control | 88.6% | H=32, K=4, pop=50 |

**Phase 2 - Combination Experiments** (2026-01-30):
| Config | Accuracy | Key Change |
|--------|----------|------------|
| **minimal_mut** | **95.0%** | pop=100, idx=0.02, wt=0.02 |
| combo_top3 | 94.4% | pop=100, idx=0.05, wt=0.05 |
| pop200 | 93.6% | pop=200, idx=0.05, wt=0.05 |

**Key Finding**: The plateau was due to **over-aggressive mutation**, not architectural limits.
- Very low mutation rates (0.02) + population diversity = breaks through to 95%
- Weight mutation essential (idx_only at 70.3%)

**Result**: 95.0% accuracy with 490 params (18x fewer than dense backprop's 8970)

---

### 4. Selection Pressure vs K (inputs per neuron) ✅ TESTED
**Question**: How does the number of inputs per neuron (K) affect selection pressure and accuracy?

**Results from extreme_sparsity.py** (500 generations, digits dataset):

| Config | Params | Mean Acc | Coverage | vs K=4 |
|--------|--------|----------|----------|--------|
| K=1, H=32 | 394 | 74.2% | 41% | -11.4% |
| **K=2, H=32** | **426** | **81.5%** | 63% | **-4.1%** |
| K=1, H=64 | 778 | 63.6% | 65% | -22.0% |
| K=2, H=64 | 842 | 77.9% | 84% | -7.7% |

**Key Findings**:
1. **K=2 is viable**: Only 4% below K=4 with 13% fewer params
2. **K=1 hits a wall**: ~74% max - single-input neurons too limited
3. **Bigger H doesn't help extreme sparsity**: H=64 worse than H=32 at both K=1 and K=2
4. **Coverage scales with K**: K=1→41%, K=2→63%, K=4→~80%

**Conclusion**: K=4 remains sweet spot, but K=2 offers good efficiency tradeoff

---

### 5. Deeper Networks with Low H ✅ TESTED (Depth hurts)
**Question**: Can depth compensate for width? Does a narrow-deep network outperform wide-shallow?

**Results** (2026-01-30, `experiments/gsa_deep.py`):

| Config | Params | Accuracy | vs Baseline |
|--------|--------|----------|-------------|
| L=1, H=32 | 490 | **83.1%** | baseline |
| L=2, H=24 | 490 | 81.9% | -1.2pp |
| L=3, H=18 | 460 | 67.2% | **-15.8pp** |

**Conclusion**: ❌ Depth does NOT help. L=1 is optimal.
- L=2 is slightly worse but recoverable
- L=3 is severely bottlenecked - information can't flow through sparse deep layers
- Mutations in early layers disrupt all downstream computation

---

### 5. Real-World Datasets
**Question**: Does Ultra-Sparse + SA work on real data with unknown feature importance?

**Candidates**:
- UCI California Housing (8 features)
- sklearn make_regression (configurable)
- Gene expression data (high-dim, few samples)

---

## Combine Ablation Winners ✅ COMPLETE

**Goal**: Combine the best hyperparameters from ablation suite to maximize accuracy.

**Tested combinations** (all with pop=100):
| Config | idx_swap | wt_rate | Accuracy |
|--------|----------|---------|----------|
| **minimal_mut** | **0.02** | **0.02** | **95.0%** |
| combo_top3 | 0.05 | 0.05 | 94.4% |
| pop200 | 0.05 | 0.05 | 93.6% |
| combo_all | 0.05 | 0.05 (sa=5) | 93.3% |
| idx_only | 0.05 | 0.0 | 70.3% |

**Result**: 95.0% achieved with minimal_mut - exceeds prediction of >93%!

**Key insight**: Even lower mutation rates (0.02) beat the ablation winners (0.05). Weight mutation is essential.

---

## Environmental Pressure & Adaptive Mutation

**Core Insight**: Fitness isn't absolute - it's relative to environmental pressure. Harsh environments (high survival threshold) create different selection dynamics than lenient ones. Additionally, biological systems adapt mutation rates under stress.

### Experiment F: Survival Thresholds (Fixed)
**Question**: How does a fixed survival threshold affect final accuracy and efficiency?

**Variable**: Survival threshold (networks below threshold are replaced)
**Controlled**: Architecture (H=32, K=4), generations (1000), population (50), mutation rates

| Condition | Threshold | Prediction |
|-----------|-----------|------------|
| Lenient | 0.0 (none) | Baseline - current behavior |
| Moderate | 0.50 | Faster convergence, less diversity |
| Harsh | 0.75 | Higher accuracy OR population collapse |
| Extreme | 0.85 | May fail to bootstrap |

**Implementation**:
```python
survivors = [c for c in population if accuracy(c) >= threshold]
# Repopulate from survivors if needed
```

**Metrics**: Final accuracy, generations to plateau, extinction events, population diversity

---

### Experiment G: Extinction Events (Sudden Threshold Increases)
**Question**: Can populations survive and adapt to sudden increases in survival requirements?

**Variable**: Extinction event frequency and magnitude
**Controlled**: Architecture, base threshold (0.3), mutation rates

| Condition | Event Schedule | Prediction |
|-----------|----------------|------------|
| No events | Threshold stays 0.3 | Baseline |
| Gradual | +0.05 every 200 gens | Steady improvement |
| Sudden | +0.20 every 500 gens | Punctuated equilibrium |
| Catastrophic | +0.30 at gen 500 only | Mass extinction, then recovery? |

**Key question**: Does surviving extinction events produce more robust solutions?

---

### Experiment H: Stagnation-Adaptive Mutation
**Question**: Does increasing mutation rate during plateaus help escape local optima?

**Variable**: Mutation rate adaptation when stagnating
**Controlled**: Architecture, threshold (none), extinction events (none)

| Condition | Mutation Response | Prediction |
|-----------|-------------------|------------|
| Fixed | Always 0.15 | Baseline |
| 2x on plateau | 0.30 when no improvement for 50 gens | Escapes some local optima |
| 5x on plateau | 0.75 when stagnating | More exploration, maybe unstable |
| Adaptive decay | Increase then gradually return to base | Best of both worlds? |

**Stagnation detection**:
```python
is_stagnating = (best_fitness - best_fitness_50_gens_ago) < 0.001
```

---

### Experiment I: Extinction-Adaptive Mutation (Hypermutation)
**Question**: Does hypermutation help populations survive extinction events?

**Variable**: Mutation rate response to extinction pressure
**Controlled**: Architecture, threshold schedule (fixed events), stagnation response (none)

| Condition | Mutation Response | Prediction |
|-----------|-------------------|------------|
| Fixed | Always 0.15 | Many extinction failures |
| 3x under pressure | 0.45 when survival_rate < 0.5 | Better survival |
| 10x hypermutation | 1.5 when survival_rate < 0.3 | Desperate exploration |
| Graduated | Scale mutation inversely with survival rate | Proportional response |

**Extinction pressure detection**:
```python
survival_rate = len(survivors) / len(population)
if survival_rate < 0.3:
    mutation_rate = base_rate * 10  # Hypermutation
```

---

### Experiment J: Combined Adaptive System
**Question**: What's the optimal combination of environmental pressure + adaptive mutation?

**Run after F, G, H, I**: Use best settings from each to build full adaptive system.

| Component | Best Setting from Experiments |
|-----------|------------------------------|
| Survival threshold | From Exp F |
| Extinction schedule | From Exp G |
| Stagnation response | From Exp H |
| Extinction response | From Exp I |

**Compare**: Full adaptive system vs current fixed GSA on digits benchmark.

---

### Experiment Order (One Variable at a Time)

1. **F first**: Establish baseline with survival thresholds (no adaptation)
2. **G second**: Add extinction events (still fixed mutation)
3. **H third**: Test stagnation-adaptive mutation (no threshold)
4. **I fourth**: Test extinction-adaptive mutation (with threshold)
5. **J last**: Combine best settings

---

## Medium Priority

### Efficiency-Aware Fitness Function ⭐ NEW
**Question**: Can we explicitly reward efficiency, not just accuracy?

**Current approach**: Fitness = -MSE (accuracy only)
- Smaller networks won because they happened to perform better
- No explicit selection pressure for efficiency

**Proposed approaches**:
```python
# Option 1: Penalty term
fitness = -MSE - λ * num_params

# Option 2: Efficiency ratio
fitness = accuracy / log(params)

# Option 3: Pareto optimization
# Track (accuracy, params) pairs, evolve toward Pareto frontier

# Option 4: Multi-objective evolution (NSGA-II style)
# Maintain population diversity across accuracy/efficiency tradeoff
```

**Why this matters**:
- Current results may be problem-specific (digits happened to favor small nets)
- Explicit efficiency pressure would generalize to other problems
- Aligns with biological evolution (energy cost matters)
- Could discover optimal accuracy/efficiency tradeoff automatically

**Experiment**: Run arch search with efficiency term, compare Pareto frontiers

---

### 4. Evolvable K - Selection Pressure on Sparsity Itself
**Question**: Can K itself evolve, so the network learns how many inputs each neuron needs?

**Approaches**:
1. **Per-neuron K**: Each neuron has its own K that can mutate (add/remove connections)
2. **Global K with penalty**: Add sparsity term to fitness: `fitness = -MSE - λ * total_connections`
3. **Curriculum K**: Start with K=1, allow growth only if fitness improves
4. **Variable K neurons**: Some neurons K=2, others K=4, evolved together

**Expected Insight**:
- Does the network learn that some neurons need more inputs than others?
- Can we discover the "natural" K for a problem?
- Does this improve efficiency further (some neurons might need only K=1)?

**Implementation**:
```python
def mutate_k(self, add_rate=0.02, remove_rate=0.05):
    for h in range(self.hidden_size):
        if random() < remove_rate and self.k[h] > 1:
            # Remove least useful connection
            self.k[h] -= 1
        elif random() < add_rate and self.k[h] < max_k:
            # Add new random connection
            self.k[h] += 1
```

---

### 5. Adaptive K During Training (Curriculum)
**Question**: Can we start with small K (high pressure) and increase it as training progresses?

**Rationale**: Early training needs exploration (find good features), later training needs exploitation (fit the function).

---

### 6. Compare to FS-NEAT Directly
**Question**: How does Ultra-Sparse + SA compare to FS-NEAT on the same problems?

**Rationale**: FS-NEAT is the closest related work - both use evolution for feature selection.

---

### 7. Scaling SA to Larger Networks
**Question**: At what network size does SA become impractical?

**Experiment**: Increase hidden size: 8 → 16 → 32 → 64 → 128
- Measure convergence time
- Measure final accuracy vs backprop

---

## Low Priority / Exploratory

### 7. Hybrid: SA for Indices, Backprop for Weights
**Question**: Can we get best of both worlds?

**Idea**: Use SA to evolve which inputs each neuron sees, but use backprop to optimize weights.

---

### 8. Population-Based Training
**Question**: Does a population of Ultra-Sparse networks outperform single SA?

**Idea**: Evolve a population, share good index patterns across individuals.

---

### 9. Theoretical Analysis
**Question**: Can we prove why the K constraint creates selection pressure?

**Approach**: Information-theoretic analysis of sparse vs dense connectivity under gradient-free optimization.

---

## Completed ✅

- [x] Validate on standard benchmark (Friedman1)
- [x] Test high-dimensional scaling (1000 features) - 5.7x better than dense
- [x] Compare inference engines (Numba wins)
- [x] Literature review (SET, FS-NEAT, QuickSelection)
- [x] **Ablation study** - All 3 components proven essential:
  - Index evolution: 10x better than frozen
  - Guided vs random: 13x better than SET-style
  - K constraint: 6.5x better than weak K=32
- [ ] Parameter sweep across problem sizes (in progress)
