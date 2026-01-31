# Experiment Workflow

Guide for creating rigorous, reproducible experiments in GENREG-sine.

## Phase 1: Design

Before writing any code, answer these questions:

### Question
What specific question does this experiment answer? (One sentence)

### Hypothesis
What do we expect to find? Why?

### Variables
- **Independent**: What are we changing?
- **Dependent**: What are we measuring?
- **Controlled**: What stays constant?

### Comparison
What baseline proves the result is meaningful? (e.g., random baseline, existing method, ablation)

### Success Criteria
How do we know if the hypothesis is supported or refuted?

## Phase 2: Setup

### Dataset & Architecture
- Dataset:
- Architecture (H, K, L, Pop):
- Compute budget (generations, trials):

### Metrics
Standard metrics (always report):
- Accuracy or MSE
- Parameter count
- Saturation %

Experiment-specific metrics:

## Phase 3: Execute

### Reuse Core Functions

**IMPORTANT**: Always reuse functions from `core/` instead of reimplementing:

```python
from core.models import UltraSparseController
from core.training import train_sa, train_gsa, train_ga
from core.metrics import compute_metrics
```

### Available Training Functions

| Function | Use Case |
|----------|----------|
| `train_sa()` | Single-chain simulated annealing |
| `train_gsa()` | Population + SA refinement (best for classification) |
| `train_ga()` | Simple genetic algorithm |
| `train_hillclimb()` | Hill climbing with early stopping |

### Example: GSA Experiment

```python
"""
Experiment: [Title]

Question: [Your question]
Hypothesis: [Your hypothesis]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.models import UltraSparseController
from core.training import train_gsa

# Load data
data = load_digits()
X = StandardScaler().fit_transform(data.data)
X_train, X_test, y_train, y_test = train_test_split(X, data.target, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# One-hot encode for training
y_onehot = torch.zeros(len(y_train), 10)
y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
y_onehot = y_onehot * 1.6 - 0.8  # Scale for tanh

# Train using GSA
best, results, history = train_gsa(
    controller_factory=lambda: UltraSparseController(64, 32, 10, 4),
    x_train=X_train, y_train=y_onehot,
    x_test=X_test, y_test=y_test,
    generations=300,
    verbose=True
)

print(f"Test accuracy: {results['test_accuracy']:.1%}")
print(f"Parameters: {results['params']}")
```

### Create Experiment File

1. Create: `experiments/<descriptive_name>.py`
2. Include design answers in docstring header
3. Run: `uv run python experiments/<name>.py`
4. Save raw results to `results/`

## Phase 4: Document

After running:

1. Create report: `docs/experiments/YYYY-MM-DD_<name>.md`
   - Question, Setup, Results table, Key Findings, Conclusion
2. Add entry to `docs/experiments_log.md` (brief summary + link)
3. Update `docs/TODO.md` if experiment was listed there

## Checklist

- [ ] Question is specific and answerable
- [ ] Hypothesis is falsifiable
- [ ] Comparison/baseline is defined
- [ ] Compute budget matches comparison (fair test)
- [ ] **Reused core functions** (not reimplemented)
- [ ] Code created with docstring header
- [ ] Results saved
- [ ] Report created in docs/experiments/
- [ ] experiments_log.md updated
