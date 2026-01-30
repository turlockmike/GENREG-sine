"""
Metrics for evaluating USEN networks.
"""

import numpy as np
from typing import List, Optional
from .networks import SparseNet


def mse(net: SparseNet, X: np.ndarray, y: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    pred = net.predict(X)
    return float(np.mean((pred - y) ** 2))


def saturation(net: SparseNet, X: np.ndarray, threshold: float = 0.95) -> float:
    """
    Compute saturation ratio (% of neurons with |activation| > threshold).

    Args:
        net: Network to evaluate
        X: Input data
        threshold: Saturation threshold (default 0.95)

    Returns:
        Fraction of hidden activations that are saturated
    """
    h, _ = net.forward(X)
    return float((np.abs(h) > threshold).mean())


def selection_stats(
    net: SparseNet,
    true_features: Optional[List[int]] = None
) -> dict:
    """
    Analyze feature selection statistics.

    Args:
        net: Network to analyze
        true_features: List of ground-truth important features

    Returns:
        Dictionary with selection statistics
    """
    all_indices = net.indices.flatten().tolist()
    unique = set(all_indices)

    stats = {
        'total_connections': len(all_indices),
        'unique_inputs': len(unique),
        'selected_indices': sorted(unique),
    }

    if true_features is not None:
        true_set = set(true_features)
        true_selected = [i for i in unique if i in true_set]
        true_connections = sum(1 for i in all_indices if i in true_set)

        random_rate = len(true_set) / net.input_dim
        actual_rate = true_connections / len(all_indices) if all_indices else 0

        stats.update({
            'true_features_found': len(true_selected),
            'true_features_total': len(true_features),
            'true_connections': true_connections,
            'true_ratio': actual_rate,
            'expected_random_ratio': random_rate,
            'selection_factor': actual_rate / random_rate if random_rate > 0 else 0,
        })

    return stats


def accuracy(net: SparseNet, X: np.ndarray, y: np.ndarray) -> float:
    """
    Compute classification accuracy (for classification problems).

    Assumes network outputs are in range [-1, 1] and need to be
    converted to class predictions.

    Args:
        net: Network to evaluate
        X: Input data
        y: True labels (integer class labels)

    Returns:
        Accuracy as fraction correct
    """
    pred = net.predict(X)

    # For multi-class, assume output_dim > 1 and take argmax
    if net.output_dim > 1:
        pred_labels = np.argmax(pred, axis=1)
    else:
        # Binary: threshold at 0
        pred_labels = (pred > 0).astype(int)

    return float(np.mean(pred_labels == y))
