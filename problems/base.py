"""
Base class for problems that Ultra-Sparse can solve.

A Problem defines:
1. How to generate input data (x)
2. How to compute target outputs (y)
3. How to expand inputs to high-dimensional features
4. Which features are "true" (for evaluation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import numpy as np


@dataclass
class ProblemConfig:
    """Configuration for a problem."""
    name: str
    input_dim: int          # Raw input dimensionality
    feature_dim: int        # Expanded feature dimensionality
    output_dim: int         # Output dimensionality
    true_features: List[int]  # Indices of informative features
    description: str = ""


class Problem(ABC):
    """Abstract base class for problems."""

    def __init__(self):
        self.config: ProblemConfig = None

    @abstractmethod
    def generate_data(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate training/test data.

        Returns:
            x: Input tensor of shape (n_samples, input_dim) or (n_samples,)
            y: Target tensor of shape (n_samples, output_dim) or (n_samples,)
        """
        pass

    @abstractmethod
    def expand_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand raw inputs to high-dimensional feature space.

        Args:
            x: Raw input tensor

        Returns:
            features: Expanded features of shape (n_samples, feature_dim)
        """
        pass

    def get_feature_names(self) -> List[str]:
        """Get human-readable names for each feature."""
        return [f"feature_{i}" for i in range(self.config.feature_dim)]

    def evaluate_selection(self, selected_indices: List[int]) -> dict:
        """
        Evaluate how well the selected indices match true features.

        Args:
            selected_indices: Indices selected by the model

        Returns:
            Dictionary with precision, recall, selection factor, etc.
        """
        selected_set = set(selected_indices)
        true_set = set(self.config.true_features)

        true_positives = len(selected_set & true_set)
        false_positives = len(selected_set - true_set)
        false_negatives = len(true_set - selected_set)

        precision = true_positives / len(selected_set) if selected_set else 0
        recall = true_positives / len(true_set) if true_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Selection factor: how much better than random?
        random_rate = len(true_set) / self.config.feature_dim
        actual_rate = true_positives / len(selected_set) if selected_set else 0
        selection_factor = actual_rate / random_rate if random_rate > 0 else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'selection_factor': selection_factor,
            'selected': sorted(selected_indices),
            'true_features': sorted(self.config.true_features),
        }
