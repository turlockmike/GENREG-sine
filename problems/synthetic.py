"""
Synthetic high-dimensional regression problems.

These problems have known ground truth features, making them ideal
for testing feature selection capabilities.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple

from .base import Problem, ProblemConfig


class SyntheticHighDim(Problem):
    """
    High-dimensional regression with sparse true features.

    Input: x ∈ R^d (d = feature_dim)
    Target: y = f(x[i1], x[i2], ...) where only a few indices matter

    The input IS the feature space (no expansion needed).
    Most features are noise, only true_features matter.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        true_features: List[int] = None,
        noise_std: float = 0.1,
        seed: int = 42
    ):
        super().__init__()

        if true_features is None:
            true_features = [3, 17, 42]  # Default: 3 sparse features

        self.feature_dim = feature_dim
        self.true_features = true_features
        self.noise_std = noise_std
        self.seed = seed

        self.config = ProblemConfig(
            name="synthetic_highdim",
            input_dim=feature_dim,
            feature_dim=feature_dim,
            output_dim=1,
            true_features=true_features,
            description=f"y = f(x[{true_features}]) + noise, {feature_dim}D input"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Generate random input features (all dimensions)
        x = torch.randn(n_samples, self.feature_dim)

        # Target depends only on true features
        # y = sin(x[3]) + cos(x[17]) + 0.5*x[42] (example)
        y = torch.zeros(n_samples)
        for i, idx in enumerate(self.true_features):
            if i == 0:
                y += torch.sin(x[:, idx])
            elif i == 1:
                y += torch.cos(x[:, idx])
            else:
                y += 0.5 * x[:, idx]

        # Add noise
        y += torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        """No expansion needed - input IS the feature space."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        names = []
        for i in range(self.feature_dim):
            if i in self.true_features:
                idx = self.true_features.index(i)
                if idx == 0:
                    names.append(f"x[{i}] (sin)")
                elif idx == 1:
                    names.append(f"x[{i}] (cos)")
                else:
                    names.append(f"x[{i}] (linear)")
            else:
                names.append(f"x[{i}] (noise)")
        return names


class SyntheticXOR(Problem):
    """
    XOR-like problem in high dimensions.

    Target: y = sign(x[i] * x[j]) - requires interaction between features
    """

    def __init__(
        self,
        feature_dim: int = 256,
        true_features: List[int] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()

        if true_features is None:
            true_features = [5, 23]  # Two features that interact

        self.feature_dim = feature_dim
        self.true_features = true_features
        self.noise_std = noise_std

        self.config = ProblemConfig(
            name="synthetic_xor",
            input_dim=feature_dim,
            feature_dim=feature_dim,
            output_dim=1,
            true_features=true_features,
            description=f"y = tanh(x[{true_features[0]}] * x[{true_features[1]}])"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.randn(n_samples, self.feature_dim)

        # XOR-like: product of two features
        i, j = self.true_features[0], self.true_features[1]
        y = torch.tanh(x[:, i] * x[:, j])
        y += torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        return [f"x[{i}]" for i in range(self.feature_dim)]


class SyntheticTimeSeries(Problem):
    """
    Time series with sparse lag dependencies.

    Input: [x(t-1), x(t-2), ..., x(t-max_lag)]
    Target: y(t) = f(x(t-k1), x(t-k2), ...) for specific lags
    """

    def __init__(
        self,
        max_lag: int = 100,
        true_lags: List[int] = None,
        noise_std: float = 0.1,
    ):
        super().__init__()

        if true_lags is None:
            true_lags = [1, 5, 20]  # Only these lags matter

        self.max_lag = max_lag
        self.true_lags = true_lags
        self.noise_std = noise_std

        self.config = ProblemConfig(
            name="synthetic_timeseries",
            input_dim=max_lag,
            feature_dim=max_lag,
            output_dim=1,
            true_features=[lag - 1 for lag in true_lags],  # 0-indexed
            description=f"y(t) depends on lags {true_lags}"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate a long time series
        total_len = n_samples + self.max_lag
        raw_series = torch.randn(total_len)

        # Create lagged features
        x = torch.zeros(n_samples, self.max_lag)
        for i in range(n_samples):
            for lag in range(self.max_lag):
                x[i, lag] = raw_series[self.max_lag + i - lag - 1]

        # Target depends only on true lags
        y = torch.zeros(n_samples)
        for i, lag in enumerate(self.true_lags):
            weight = 1.0 / (i + 1)  # Decreasing weights
            y += weight * x[:, lag - 1]  # lag-1 because 0-indexed

        y += torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        return [f"lag_{i+1}" for i in range(self.max_lag)]
