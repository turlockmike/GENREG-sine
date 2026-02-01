# Experiment 23: Data Augmentation for GSA

**Date**: 2026-01-31
**Code**: `experiments/gsa_augmented.py`, `core/augmentation.py`

## Question

Does data augmentation improve GSA accuracy on sklearn digits?

## Background

Data augmentation is a standard technique in deep learning that typically improves generalization by 2-3%. We implemented rotation (±10°) and translation (±1px) augmentation for the 8×8 digit images.

## Methodology

**Augmentation types tested:**
1. **Static augmentation**: Pre-generate N× augmented training samples before training
2. **Online augmentation**: Apply random transforms during each fitness evaluation (abandoned - too slow)

**Transforms:**
- Rotation: ±10° using scipy.ndimage.rotate
- Shift: ±1 pixel using scipy.ndimage.shift

**Base config:** H=32, K=4, pop=100, gen=1000 (proven 95% config)

## Results

| Config | Training Samples | Accuracy | vs Baseline |
|--------|------------------|----------|-------------|
| **baseline** | 1,437 | **95.6%** | - |
| static_2x | 2,874 | 92.5% | **-3.1pp** |
| static_5x | 7,185 | 90.0% | **-5.6pp** |

**Online augmentation abandoned**: Applying transforms on every fitness evaluation was computationally infeasible (~2 billion image transforms required).

## Key Finding

**Data augmentation HURTS accuracy on sklearn digits.**

The more augmentation applied, the worse the results:
- No augmentation: 95.6%
- 2× augmentation: 92.5% (-3.1pp)
- 5× augmentation: 90.0% (-5.6pp)

## Analysis

Why augmentation hurts rather than helps:

1. **Small image size (8×8)**: Rotation and shifting at this resolution destroys important structural information. A 10° rotation on an 8×8 image significantly distorts the digit.

2. **Already clean data**: sklearn digits are pre-centered and normalized. Unlike raw MNIST, there's no natural variation to simulate.

3. **Evolutionary training has implicit regularization**: The population-based search with stochastic selection may already provide sufficient regularization, making explicit data augmentation redundant or harmful.

4. **Training set size**: With only 1,437 training samples, the network can memorize the augmented patterns as noise rather than learning invariances.

## Comparison to Deep Learning

In deep learning, augmentation typically helps because:
- Networks are prone to overfitting
- Gradient descent can memorize training data
- Larger images (28×28+) preserve structure under transforms

For GSA on small images:
- Population diversity provides implicit regularization
- 8×8 images lose structure under rotation
- The sparse K=4 constraint already limits overfitting

## Conclusion

**Data augmentation is NOT recommended for GSA on sklearn digits.**

The 8×8 image size and evolutionary training dynamics make augmentation counterproductive. The baseline config (H=32, K=4, pop=100, no augmentation) achieving 95.6% remains optimal.

For larger images (MNIST 28×28), augmentation may still be beneficial - this should be tested separately.

## Files Created

- `core/augmentation.py`: Augmentation utilities (augment_digits, augment_batch)
- `experiments/gsa_augmented.py`: Experiment with multiple augmentation configs
