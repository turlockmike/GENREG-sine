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
├── experiments/              # Experiment code files
│   ├── ultra_sparse.py       # ⭐ BREAKTHROUGH experiment
│   ├── comprehensive_benchmark.py    # 20-trial comparison
│   ├── inference_engines.py  # PyTorch vs NumPy vs Numba
│   └── gsa_*.py              # GSA experiments
│
├── docs/
│   ├── experiments/          # Individual experiment reports (YYYY-MM-DD_title.md)
│   ├── experiments_log.md    # Index of all experiments with brief summaries
│   └── TODO.md               # Future work and open questions
│
├── results/                  # Raw experiment outputs (JSON, checkpoints)
└── legacy/                   # Original GENREG code (sine_*.py)
```

## Git Remotes

```bash
# Push to fork (turlockmike has write access)
git push fork main

# origin is A1CST/GENREG-sine (upstream, read-only for most users)
```

## Commands

This project uses `uv` for dependency management.

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Install with fast inference (Numba JIT)
uv sync --extra fast

# Install with dev tools (pytest, ruff)
uv sync --extra dev

# Run experiments (uv run uses the project's venv)
uv run python experiments/ultra_sparse.py
uv run python experiments/gsa_digits.py --hidden 64 --k 16 --pop 50 --gens 300
uv run python experiments/sklearn_benchmarks.py

# Run original GENREG code
uv run python legacy/sine_train.py
uv run python legacy/sine_sweep.py

# Run tests
uv run pytest tests/
```

## Standard Metrics

**All experiments MUST report these three metrics:**

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| **MSE** | Accuracy (lower is better) | `torch.mean((pred - y_true) ** 2)` |
| **Energy** | Inference cost | `activation_energy + weight_energy` |
| **Saturation** | % saturated neurons | `(activations.abs() > 0.95).mean()` |

Use `core.metrics.compute_metrics(controller, x_test, y_true)` for consistent measurement.

## Experiment Workflow

Follow this standard process for all experiments:

### 1. Create the Experiment Code

Create a file in `experiments/` with a descriptive name:

```bash
experiments/my_experiment.py
```

Include a header docstring with the question being investigated:

```python
"""
Experiment: Brief Title

Question: What specific question does this experiment answer?
Hypothesis: What do we expect to find?
"""
```

### 2. Run the Experiment

```bash
uv run python experiments/my_experiment.py
```

All experiments MUST report the standard metrics (MSE, Energy, Saturation) using `core.metrics.compute_metrics()`.

### 3. Create the Experiment Report

After running, create a detailed report:

```bash
docs/experiments/YYYY-MM-DD_experiment_name.md
```

Report template:

```markdown
# Experiment N: Title

**Date**: YYYY-MM-DD
**Code**: `experiments/experiment_name.py`

## Question

What specific question does this experiment answer?

## Setup

- Dataset, architecture, hyperparameters
- Matched compute budget (if comparing methods)

## Results

| Method | Metric 1 | Metric 2 | ... |
|--------|----------|----------|-----|
| ...    | ...      | ...      | ... |

## Key Findings

1. Finding with **bold** for emphasis
2. Quantified comparisons (e.g., "3.6pp improvement")

## Conclusion

Brief interpretation of results and implications.
```

### 4. Update the Experiments Log

Add a brief entry to `docs/experiments_log.md`:

```markdown
## Experiment N: Title

See: [docs/experiments/YYYY-MM-DD_experiment_name.md](experiments/YYYY-MM-DD_experiment_name.md)

**Result**: One-line summary of the key finding.
```

### 5. Update TODO.md (if applicable)

If the experiment was listed in TODO.md:
- Mark it ✅ COMPLETE
- Add a reference to the report file
- Keep TODO.md focused on future work, not detailed results

### Summary

| Step | Location | Content |
|------|----------|---------|
| Code | `experiments/*.py` | Runnable experiment |
| Report | `docs/experiments/YYYY-MM-DD_*.md` | Full results and analysis |
| Index | `docs/experiments_log.md` | Brief summary + link |
| Future | `docs/TODO.md` | Open questions only |

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
| `docs/experiments_log.md` | Index of all experiments |
| `docs/experiments/*.md` | Detailed experiment reports |
| `docs/TODO.md` | Future work and open questions |

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

6. **Ablation proves all components essential**:
   - Index evolution: 10x better than frozen indices
   - Guided evolution: 13x better than random regrowth (SET-style)
   - K constraint: 6.5x better selection with K=4 vs K=32

7. **GSA unlocks harder problems**: Population-based SA (Genetic Simulated Annealing) achieves 87.2% on digits vs 64.7% for single SA (+22.5pp improvement)

8. **sklearn benchmarks**: GENREG achieves near-backprop accuracy with massive efficiency gains:
   - Breast Cancer: 95.9% vs 97.1% (108x fewer params)
   - Wine: 97.2% vs 100% (78x fewer params)
   - Digits: 87.2% vs 97.0% (5.2x fewer params, using GSA)

## Configuration

Key settings in `sine_config.py`:
- `HIDDEN_SIZE = 8` — Hidden layer neurons
- `TRUE_SIGNAL_SIZE = 16` — Useful input dimensions
- `NOISE_SIGNAL_SIZE = 240` — Noise input dimensions
- `SATURATION_THRESHOLD = 0.95` — Defines "saturated"
- `DEVICE = "cuda"` — GPU acceleration

## Output

- **Raw data**: `results/<experiment_name>/` — JSON metrics, checkpoints
- **Reports**: `docs/experiments/YYYY-MM-DD_*.md` — Analysis and conclusions
- **Index**: `docs/experiments_log.md` — Quick reference to all experiments
