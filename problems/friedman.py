"""
Friedman benchmark problems - classic ML regression benchmarks.

These are standard benchmarks for testing feature selection and regression.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple

from .base import Problem, ProblemConfig


class Friedman1(Problem):
    """
    Friedman #1 benchmark problem.

    y = 10*sin(π*x₁*x₂) + 20*(x₃ - 0.5)² + 10*x₄ + 5*x₅ + noise

    Features:
    - x₁, x₂: Interact nonlinearly (sin of product)
    - x₃: Quadratic term
    - x₄: Linear term (weight 10)
    - x₅: Linear term (weight 5)
    - x₆...x_d: Noise features (irrelevant)

    This is a classic benchmark because:
    1. Known ground truth (features 0-4 matter)
    2. Nonlinear interaction (x₁ * x₂)
    3. Mix of nonlinear and linear terms
    4. Standard for comparing feature selection methods
    """

    def __init__(
        self,
        n_noise_features: int = 95,  # Total = 5 true + 95 noise = 100
        noise_std: float = 1.0,
    ):
        super().__init__()

        self.n_true = 5
        self.n_noise = n_noise_features
        self.noise_std = noise_std

        self.config = ProblemConfig(
            name="friedman1",
            input_dim=self.n_true + n_noise_features,
            feature_dim=self.n_true + n_noise_features,
            output_dim=1,
            true_features=[0, 1, 2, 3, 4],
            description=f"y = 10sin(πx₁x₂) + 20(x₃-0.5)² + 10x₄ + 5x₅ | {self.n_true} true + {n_noise_features} noise"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # All features uniform in [0, 1]
        x = torch.rand(n_samples, self.config.feature_dim)

        # Friedman #1 formula
        y = (
            10 * torch.sin(np.pi * x[:, 0] * x[:, 1]) +
            20 * (x[:, 2] - 0.5) ** 2 +
            10 * x[:, 3] +
            5 * x[:, 4]
        )

        # Add noise
        y = y + torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        """No expansion needed - input IS the feature space."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        names = [
            "x₁ (sin interaction)",
            "x₂ (sin interaction)",
            "x₃ (quadratic)",
            "x₄ (linear, w=10)",
            "x₅ (linear, w=5)",
        ]
        for i in range(self.n_noise):
            names.append(f"noise_{i+6}")
        return names


class Friedman2(Problem):
    """
    Friedman #2 benchmark problem.

    y = (x₁² + (x₂*x₃ - 1/(x₂*x₄))²)^0.5

    Features 0-3 matter, complex nonlinear interactions.
    """

    def __init__(
        self,
        n_noise_features: int = 96,
        noise_std: float = 125.0,  # Standard noise level for this problem
    ):
        super().__init__()

        self.n_true = 4
        self.n_noise = n_noise_features
        self.noise_std = noise_std

        self.config = ProblemConfig(
            name="friedman2",
            input_dim=self.n_true + n_noise_features,
            feature_dim=self.n_true + n_noise_features,
            output_dim=1,
            true_features=[0, 1, 2, 3],
            description=f"y = sqrt(x₁² + (x₂x₃ - 1/(x₂x₄))²) | Complex nonlinear"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x = torch.zeros(n_samples, self.config.feature_dim)

        # Specific ranges for Friedman #2
        x[:, 0] = torch.rand(n_samples) * 100  # [0, 100]
        x[:, 1] = torch.rand(n_samples) * (560 - 40) + 40  # [40, 560]
        x[:, 2] = torch.rand(n_samples)  # [0, 1]
        x[:, 3] = torch.rand(n_samples) * 10 + 1  # [1, 11]

        # Noise features uniform [0, 1]
        x[:, 4:] = torch.rand(n_samples, self.n_noise)

        # Friedman #2 formula
        y = torch.sqrt(
            x[:, 0] ** 2 +
            (x[:, 1] * x[:, 2] - 1 / (x[:, 1] * x[:, 3])) ** 2
        )

        y = y + torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        names = ["x₁", "x₂", "x₃", "x₄"]
        for i in range(self.n_noise):
            names.append(f"noise_{i+5}")
        return names


class Friedman3(Problem):
    """
    Friedman #3 benchmark problem.

    y = atan((x₂*x₃ - 1/(x₂*x₄)) / x₁)

    Features 0-3 matter, arctangent of ratio.
    """

    def __init__(
        self,
        n_noise_features: int = 96,
        noise_std: float = 0.1,
    ):
        super().__init__()

        self.n_true = 4
        self.n_noise = n_noise_features
        self.noise_std = noise_std

        self.config = ProblemConfig(
            name="friedman3",
            input_dim=self.n_true + n_noise_features,
            feature_dim=self.n_true + n_noise_features,
            output_dim=1,
            true_features=[0, 1, 2, 3],
            description=f"y = atan((x₂x₃ - 1/(x₂x₄)) / x₁)"
        )

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        x = torch.zeros(n_samples, self.config.feature_dim)

        # Specific ranges for Friedman #3
        x[:, 0] = torch.rand(n_samples) * 100  # [0, 100]
        x[:, 1] = torch.rand(n_samples) * (560 - 40) + 40  # [40, 560]
        x[:, 2] = torch.rand(n_samples)  # [0, 1]
        x[:, 3] = torch.rand(n_samples) * 10 + 1  # [1, 11]

        # Noise features
        x[:, 4:] = torch.rand(n_samples, self.n_noise)

        # Friedman #3 formula
        y = torch.atan(
            (x[:, 1] * x[:, 2] - 1 / (x[:, 1] * x[:, 3])) / x[:, 0]
        )

        y = y + torch.randn(n_samples) * self.noise_std

        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def get_feature_names(self) -> List[str]:
        names = ["x₁", "x₂", "x₃", "x₄"]
        for i in range(self.n_noise):
            names.append(f"noise_{i+5}")
        return names
