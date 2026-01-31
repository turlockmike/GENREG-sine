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

1. Create experiment file: `experiments/<descriptive_name>.py`
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
- [ ] Code created with docstring header
- [ ] Results saved
- [ ] Report created in docs/experiments/
- [ ] experiments_log.md updated
