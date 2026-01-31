# Experiment 17: GSA vs Random Restarts Ablation

**Date**: 2026-01-30
**Code**: `experiments/gsa_ablation.py`

## Question

Is GSA's population + selection actually better than naive random restarts with matched compute?

## Setup

- **Dataset**: Digits (64 features, 10 classes)
- **Architecture**: H=64, K=16 ultra-sparse
- **Matched compute**: ~60,000 SA steps each

## Results

| Method | Best Acc | Mean Acc | SA Steps | Time |
|--------|----------|----------|----------|------|
| Random Restarts (10×6000) | 68.3% | 60.3% | 60,000 | 274s |
| GSA (Pop=50, Gens=10) | 36.7% | - | 10,000 | 49s |
| **GSA (Pop=50, Gens=60)** | **71.9%** | - | 60,000 | 296s |
| Random Restarts (50×1200) | 58.1% | 46.0% | 60,000 | 284s |

## Key Findings

1. **GSA beats random restarts**: 71.9% vs 68.3% (+3.6pp) with matched compute
2. **Chain length matters more than quantity**: 10×6000 (68.3%) >> 50×1200 (58.1%)
3. **GSA needs enough generations**: 10 gens only got 36.7% - far too few
4. **Selection pressure helps but isn't dramatic**: ~3-4pp improvement over random restarts

## Conclusion

GSA's selection pressure provides a modest but consistent improvement over naive random restarts. The benefit comes from:
- Killing off chains stuck in bad local optima
- Propagating good solutions to more of the population
- But the effect is incremental (~3-4pp), not transformative

The bigger factor is **chain length** - longer individual chains (6000 steps) outperform many short chains (1200 steps) even with 5x more random starts.
