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

**Ablation: GSA vs Random Restarts** (`experiments/gsa_ablation.py`)
- Question: Is GSA's selection pressure actually helping, or just trying more starts?
- Partial results (60k SA steps each):
  - Random Restarts (10x6000): **68.3%** best, 60.3% mean
  - GSA (10 gens only): 36.7% - not enough generations
  - GSA (60 gens): TBD - run experiment to compare
- Run with: `python experiments/gsa_ablation.py`

**Background**: Du et al. (2018) show that GSA outperforms both pure GA and pure SA for sparse network optimization because:
- SA is fast at local refinement (Monte Carlo acceptance)
- Selection pressure prevents wasting time on chains stuck in bad local optima
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

### 2. Solve Digits (10-class classification)
**Question**: What does GENREG need to achieve >90% on digits?

**Current status**: H=32, K=8 → 64.7% (vs Dense 97%)

**Approaches to try**:
- Larger network (H=128, K=32)
- Population-based SA (see #1 above)
- Longer training
- Different temperature schedules

---

### 3. Selection Pressure vs K (inputs per neuron)
**Question**: How does the number of inputs per neuron (K) affect selection pressure and accuracy?

**Hypothesis**: Smaller K = stronger selection pressure but potentially lower accuracy due to limited expressiveness. There's likely an optimal K for each problem complexity.

**Experiment Design**:
- Fix problem: 1000 features, 10 true
- Fix hidden size: 8 neurons
- Vary K: 1, 2, 3, 4, 6, 8, 16, 32
- Measure: MSE, feature selection recall, selection factor

**Expected Insights**:
- K=1: Maximum pressure, but can each neuron only see one input?
- K=2: High pressure (early experiments showed 6x selection factor)
- K=4: Current default, moderate pressure
- K=8+: Approaches dense behavior, selection pressure diminishes

**Key Metric**: Selection Factor = (true features found / total connections) / (true features / total features)

---

### 4. Deeper Networks
**Question**: Does layer-wise sparsity compound the selection advantage?

**Experiment Design**:
- Architecture: Input → Sparse(8×4) → Sparse(8×4) → Output
- Compare: 1-layer vs 2-layer vs 3-layer
- Same total params budget

---

### 5. Real-World Datasets
**Question**: Does Ultra-Sparse + SA work on real data with unknown feature importance?

**Candidates**:
- UCI California Housing (8 features)
- sklearn make_regression (configurable)
- Gene expression data (high-dim, few samples)

---

## Medium Priority

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
