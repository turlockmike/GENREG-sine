"""
USEN - Ultra-Sparse Evolvable Networks

A library for gradient-free neural network training with automatic feature selection.

Core components:
- SparseNet: Fixed-K sparse network with evolvable indices
- train_gsa: Genetic Simulated Annealing trainer
- train_sa: Single-chain Simulated Annealing trainer
- Problems: Sine, HighDim, Digits problem generators
"""

from .networks import SparseNet, DenseNet
from .trainers import train_gsa, train_sa, train_backprop
from .problems import sine_problem, highdim_problem, digits_problem, friedman1_problem
from .metrics import mse, saturation, selection_stats

__version__ = "0.1.0"

__all__ = [
    # Networks
    "SparseNet",
    "DenseNet",
    # Trainers
    "train_gsa",
    "train_sa",
    "train_backprop",
    # Problems
    "sine_problem",
    "highdim_problem",
    "digits_problem",
    "friedman1_problem",
    # Metrics
    "mse",
    "saturation",
    "selection_stats",
]
