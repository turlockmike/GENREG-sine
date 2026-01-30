"""
GENREG Core Module

Shared utilities for all experiments. All experiments should use these
for consistent metric reporting and training.

Standard Metrics (always report all three):
- MSE: Mean Squared Error (accuracy)
- Energy: Inference cost (activation + weight energy)
- Saturation: Percentage of saturated neurons (|activation| > 0.95)

Example usage:
    from core.metrics import compute_metrics
    from core.training import train_sa

    best, metrics, history = train_sa(controller, x_test, y_true)
    print(f"Results: {metrics}")  # MSE=0.021, Energy=1.85, Saturation=70%
"""

from .metrics import compute_metrics, Metrics, print_metrics, format_metrics_table
from .training import train_sa, train_hillclimb, train_ga

__all__ = [
    # Metrics
    'compute_metrics',
    'Metrics',
    'print_metrics',
    'format_metrics_table',
    # Training
    'train_sa',
    'train_hillclimb',
    'train_ga',
]
