"""
Problem generators for USEN experiments.

Each function returns (X, y, true_features) where:
- X: Input features (n_samples, n_features)
- y: Target values (n_samples,)
- true_features: List of indices that are actually relevant
"""

import numpy as np
from typing import Tuple, List, Optional


def sine_problem(
    n_samples: int = 500,
    n_features: int = 16,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Sine wave approximation with harmonic features.

    Target: y = sin(x)
    Features: sin(kx), cos(kx) for various k

    Args:
        n_samples: Number of samples
        n_features: Number of features (should be even)
        seed: Random seed

    Returns:
        (X, y, true_features) where all features are true (no noise)
    """
    np.random.seed(seed)

    x = np.linspace(0, 2 * np.pi, n_samples).astype(np.float32)
    y = np.sin(x).astype(np.float32)

    # Create feature matrix with sin/cos at various frequencies
    X = np.column_stack([
        np.sin((i + 1) * x) if i % 2 == 0 else np.cos((i // 2 + 1) * x)
        for i in range(n_features)
    ]).astype(np.float32)

    true_features = list(range(n_features))

    return X, y, true_features


def highdim_problem(
    n_samples: int = 500,
    n_features: int = 1000,
    n_true: int = 10,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    High-dimensional regression with sparse true features.

    Features 0-4: Linear terms with decreasing weights
    Features 5-9: Nonlinear terms (interactions, quadratic, etc.)
    Features 10+: Pure noise

    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_true: Number of true (relevant) features
        seed: Random seed

    Returns:
        (X, y, true_features)
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    # Linear terms (features 0-4)
    weights = [10, 8, 6, 4, 2]
    for i, w in enumerate(weights):
        if i < n_true:
            y += w * X[:, i]

    # Nonlinear terms (features 5-9)
    if n_true > 5:
        y += 5 * np.sin(np.pi * X[:, 5] * X[:, 6])
    if n_true > 7:
        y += 3 * (X[:, 7] - 0.5) ** 2
    if n_true > 8:
        y += 2 * np.abs(X[:, 8] - 0.5)
    if n_true > 9:
        y += np.cos(2 * np.pi * X[:, 9])

    # Add noise
    y += np.random.randn(n_samples).astype(np.float32) * 0.5

    # Normalize for tanh output
    y = ((y - y.mean()) / (y.std() + 1e-8) * 0.8).astype(np.float32)

    true_features = list(range(n_true))

    return X, y, true_features


def digits_problem(
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    sklearn digits classification problem.

    Returns train/test split ready for use.

    Args:
        seed: Random seed

    Returns:
        (X_train, y_train, X_test, y_test, true_features)
        Note: All 64 features are considered "true" for digits
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(seed)

    digits = load_digits()
    X, y = digits.data, digits.target

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # All features are potentially relevant for digits
    true_features = list(range(64))

    return X_train, y_train.astype(np.int32), X_test, y_test.astype(np.int32), true_features


def friedman1_problem(
    n_samples: int = 500,
    n_features: int = 100,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Friedman #1 benchmark - classic ML problem.

    y = 10*sin(π*x₁*x₂) + 20*(x₃ - 0.5)² + 10*x₄ + 5*x₅ + noise

    Only features 0-4 matter, rest are noise.

    Args:
        n_samples: Number of samples
        n_features: Total features (5 true + rest noise)
        seed: Random seed

    Returns:
        (X, y, true_features)
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, n_features).astype(np.float32)

    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1]) +
        20 * (X[:, 2] - 0.5) ** 2 +
        10 * X[:, 3] +
        5 * X[:, 4] +
        np.random.randn(n_samples) * 0.5
    ).astype(np.float32)

    # Normalize
    y = ((y - y.mean()) / (y.std() + 1e-8) * 0.8).astype(np.float32)

    true_features = [0, 1, 2, 3, 4]

    return X, y, true_features
