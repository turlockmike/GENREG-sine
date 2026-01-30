"""
Problems module - defines various benchmark problems for Ultra-Sparse.

Each problem provides:
- Data generation
- Feature expansion
- Ground truth features (for evaluation)
"""

from .base import Problem, ProblemConfig
from .sine import SineProblem, MultiFrequencySineProblem
from .synthetic import SyntheticHighDim, SyntheticXOR, SyntheticTimeSeries
from .friedman import Friedman1, Friedman2, Friedman3

__all__ = [
    'Problem',
    'ProblemConfig',
    'SineProblem',
    'MultiFrequencySineProblem',
    'SyntheticHighDim',
    'SyntheticXOR',
    'SyntheticTimeSeries',
    'Friedman1',
    'Friedman2',
    'Friedman3',
]

# Registry for easy lookup
PROBLEMS = {
    'sine': SineProblem,
    'multi_freq': MultiFrequencySineProblem,
    'synthetic': SyntheticHighDim,
    'xor': SyntheticXOR,
    'timeseries': SyntheticTimeSeries,
    'friedman1': Friedman1,
    'friedman2': Friedman2,
    'friedman3': Friedman3,
}


def get_problem(name: str, **kwargs) -> Problem:
    """Get a problem by name."""
    if name not in PROBLEMS:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEMS.keys())}")
    return PROBLEMS[name](**kwargs)
