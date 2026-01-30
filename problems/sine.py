"""
Sine wave problem - the original GENREG benchmark.

Target: y = sin(x)
Features: 16 true (sin/cos at frequencies 1-8) + 240 noise
True features: indices 0-15
"""

import torch
import numpy as np
from typing import List, Optional, Tuple

from .base import Problem, ProblemConfig


class SineProblem(Problem):
    """
    Original sine wave approximation problem.

    Input: scalar x in [-2π, 2π]
    Target: sin(x)
    Features: 256 dimensions
      - [0-7]: sin(kx) for k=1..8
      - [8-15]: cos(kx) for k=1..8
      - [16-255]: high-frequency noise sinusoids
    """

    def __init__(self, n_true: int = 16, n_noise: int = 240):
        super().__init__()
        self.n_true = n_true
        self.n_noise = n_noise

        self.config = ProblemConfig(
            name="sine",
            input_dim=1,
            feature_dim=n_true + n_noise,
            output_dim=1,
            true_features=list(range(n_true)),
            description=f"y = sin(x) with {n_true} true + {n_noise} noise features"
        )

        # Pre-generate noise parameters (deterministic)
        np.random.seed(42)
        self.noise_freqs = 10 + np.random.random(n_noise) * 90
        self.noise_phases = np.random.random(n_noise) * 2 * np.pi

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples)
        y = torch.sin(x)
        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.view(1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (n_samples, 1)

        batch_size = x.shape[0]
        features = torch.zeros(batch_size, self.config.feature_dim)

        # True signals: sin(kx), cos(kx) for k=1..8
        for i in range(8):
            freq = i + 1
            features[:, i] = torch.sin(freq * x.squeeze())
            features[:, i + 8] = torch.cos(freq * x.squeeze())

        # Noise signals: high-frequency sinusoids
        for i in range(self.n_noise):
            features[:, self.n_true + i] = torch.sin(
                self.noise_freqs[i] * x.squeeze() + self.noise_phases[i]
            )

        return features

    def get_feature_names(self) -> List[str]:
        names = []
        for i in range(8):
            names.append(f"sin({i+1}x)")
        for i in range(8):
            names.append(f"cos({i+1}x)")
        for i in range(self.n_noise):
            names.append(f"noise[{i}]")
        return names


class MultiFrequencySineProblem(Problem):
    """
    Multi-frequency sine wave - slightly harder.

    Target: y = sin(x) + 0.3*sin(3x)
    Requires finding multiple true signals.
    """

    def __init__(self, n_noise: int = 240):
        super().__init__()
        self.n_true = 16
        self.n_noise = n_noise

        self.config = ProblemConfig(
            name="multi_freq_sine",
            input_dim=1,
            feature_dim=self.n_true + n_noise,
            output_dim=1,
            true_features=[0, 2],  # sin(x) and sin(3x)
            description="y = sin(x) + 0.3*sin(3x)"
        )

        np.random.seed(42)
        self.noise_freqs = 10 + np.random.random(n_noise) * 90
        self.noise_phases = np.random.random(n_noise) * 2 * np.pi

    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)

        x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples)
        y = torch.sin(x) + 0.3 * torch.sin(3 * x)
        return x, y

    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.view(1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        batch_size = x.shape[0]
        features = torch.zeros(batch_size, self.config.feature_dim)

        for i in range(8):
            freq = i + 1
            features[:, i] = torch.sin(freq * x.squeeze())
            features[:, i + 8] = torch.cos(freq * x.squeeze())

        for i in range(self.n_noise):
            features[:, self.n_true + i] = torch.sin(
                self.noise_freqs[i] * x.squeeze() + self.noise_phases[i]
            )

        return features

    def get_feature_names(self) -> List[str]:
        names = []
        for i in range(8):
            names.append(f"sin({i+1}x)")
        for i in range(8):
            names.append(f"cos({i+1}x)")
        for i in range(self.n_noise):
            names.append(f"noise[{i}]")
        return names
