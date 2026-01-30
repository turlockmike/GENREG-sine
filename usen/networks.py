"""
Neural network architectures for USEN.

SparseNet: Fixed-K sparse connectivity with evolvable indices (NumPy-based)
DenseNet: Standard dense network for comparison
"""

import numpy as np
from typing import List, Optional, Tuple


class SparseNet:
    """
    Ultra-Sparse network where each hidden neuron connects to exactly K inputs.

    This creates selection pressure that enables automatic feature selection
    through evolutionary training (SA, GSA).

    Args:
        input_dim: Number of input features
        H: Number of hidden neurons
        K: Number of inputs per neuron (the sparsity constraint)
        output_dim: Number of outputs (default 1 for regression)
    """

    def __init__(self, input_dim: int, H: int = 8, K: int = 2, output_dim: int = 1):
        self.input_dim = input_dim
        self.H = H
        self.K = K
        self.output_dim = output_dim

        # Each hidden neuron selects K input indices
        self.indices = np.array([
            np.random.choice(input_dim, K, replace=False) for _ in range(H)
        ])

        # Weights (small initialization)
        self.W1 = np.random.randn(H, K).astype(np.float32) * 0.5
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = np.random.randn(output_dim, H).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        # Cache for saturation calculation
        self._last_hidden = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            x: Input array of shape (n_samples, input_dim)

        Returns:
            Tuple of (hidden_activations, output)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples = len(x)

        # Gather selected inputs for each hidden neuron
        h = np.zeros((n_samples, self.H), dtype=np.float32)
        for i in range(self.H):
            h[:, i] = x[:, self.indices[i]] @ self.W1[i] + self.b1[i]

        h_act = np.tanh(h)
        self._last_hidden = h_act

        # Output layer
        out = np.tanh(h_act @ self.W2.T + self.b2)

        if self.output_dim == 1:
            out = out.flatten()

        return h_act, out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning only output."""
        _, out = self.forward(x)
        return out

    def clone(self) -> 'SparseNet':
        """Create a deep copy."""
        new = SparseNet(self.input_dim, self.H, self.K, self.output_dim)
        new.indices = self.indices.copy()
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new

    def mutate(
        self,
        weight_rate: float = 0.2,
        weight_std: float = 0.1,
        index_rate: float = 0.1
    ):
        """
        Mutate weights and indices in-place.

        Args:
            weight_rate: Probability of mutating each weight
            weight_std: Standard deviation of weight mutations
            index_rate: Probability of swapping an input index per neuron
        """
        # Weight mutations
        for i in range(self.H):
            if np.random.random() < weight_rate:
                j = np.random.randint(self.K)
                self.W1[i, j] += np.random.randn() * weight_std
            if np.random.random() < weight_rate * 0.5:
                self.b1[i] += np.random.randn() * weight_std * 0.5

        if np.random.random() < weight_rate:
            j = np.random.randint(self.H)
            self.W2[0, j] += np.random.randn() * weight_std

        # Index mutations (the key to feature selection)
        for i in range(self.H):
            if np.random.random() < index_rate:
                j = np.random.randint(self.K)
                current = set(self.indices[i])
                available = [idx for idx in range(self.input_dim) if idx not in current]
                if available:
                    self.indices[i, j] = np.random.choice(available)
                    # Reset weight for new connection
                    self.W1[i, j] = np.random.randn() * 0.3

    def get_selected_indices(self) -> List[int]:
        """Get list of unique selected input indices."""
        return sorted(set(self.indices.flatten().tolist()))

    def num_params(self) -> int:
        """Total number of parameters."""
        return self.H * self.K + self.H + self.output_dim * self.H + self.output_dim

    def weight_stats(self) -> dict:
        """Get weight magnitude statistics."""
        return {
            'W1_max': float(np.abs(self.W1).max()),
            'W1_mean': float(np.abs(self.W1).mean()),
            'W2_max': float(np.abs(self.W2).max()),
            'W2_mean': float(np.abs(self.W2).mean()),
        }

    def save(self, path: str):
        """Save model to file."""
        np.savez(
            path,
            input_dim=self.input_dim,
            H=self.H,
            K=self.K,
            output_dim=self.output_dim,
            indices=self.indices,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )

    @classmethod
    def load(cls, path: str) -> 'SparseNet':
        """Load model from file."""
        data = np.load(path)
        net = cls(
            input_dim=int(data['input_dim']),
            H=int(data['H']),
            K=int(data['K']),
            output_dim=int(data['output_dim']),
        )
        net.indices = data['indices']
        net.W1 = data['W1']
        net.b1 = data['b1']
        net.W2 = data['W2']
        net.b2 = data['b2']
        return net


class DenseNet:
    """
    Standard dense network for comparison.

    Same interface as SparseNet but with full connectivity.
    """

    def __init__(self, input_dim: int, H: int = 8, output_dim: int = 1):
        self.input_dim = input_dim
        self.H = H
        self.output_dim = output_dim

        self.W1 = np.random.randn(H, input_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = np.random.randn(output_dim, H).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_dim, dtype=np.float32)

        self._last_hidden = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass returning (hidden, output)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        h = np.tanh(x @ self.W1.T + self.b1)
        self._last_hidden = h
        out = np.tanh(h @ self.W2.T + self.b2)

        if self.output_dim == 1:
            out = out.flatten()

        return h, out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning only output."""
        _, out = self.forward(x)
        return out

    def num_params(self) -> int:
        """Total number of parameters."""
        return self.H * self.input_dim + self.H + self.output_dim * self.H + self.output_dim
