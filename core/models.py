"""
Neural network models for gradient-free training.

These models are designed to work with any Problem from the problems module.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable


class UltraSparseController:
    """
    Ultra-Sparse neural network where each hidden neuron connects to only K inputs.

    This creates selection pressure that enables automatic feature selection
    through evolutionary training (SA, GA, etc.).

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden neurons
        output_size: Number of outputs
        inputs_per_neuron: How many inputs each hidden neuron can connect to (K)
        device: Torch device
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 8,
        output_size: int = 1,
        inputs_per_neuron: int = 2,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.inputs_per_neuron = inputs_per_neuron

        # Each hidden neuron selects K input indices
        self.input_indices = torch.zeros(
            hidden_size, inputs_per_neuron, dtype=torch.long, device=self.device
        )
        for h in range(hidden_size):
            self.input_indices[h] = torch.randperm(input_size)[:inputs_per_neuron]

        # Weights for selected connections only
        scale_w1 = np.sqrt(2.0 / inputs_per_neuron)
        self.w1 = torch.randn(hidden_size, inputs_per_neuron, device=self.device) * scale_w1
        self.b1 = torch.zeros(hidden_size, device=self.device)

        # Output layer
        scale_w2 = np.sqrt(2.0 / hidden_size)
        self.w2 = torch.randn(output_size, hidden_size, device=self.device) * scale_w2
        self.b2 = torch.zeros(output_size, device=self.device)

        self.last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_size) or (input_size,)

        Returns:
            Output of shape (batch_size, output_size) or (output_size,)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        batch_size = x.shape[0]

        # Gather selected inputs for each hidden neuron
        selected = torch.zeros(
            batch_size, self.hidden_size, self.inputs_per_neuron, device=self.device
        )
        for h in range(self.hidden_size):
            selected[:, h, :] = x[:, self.input_indices[h]]

        # Compute hidden activations
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        self.last_hidden = hidden.detach()

        # Output
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def mutate(
        self,
        weight_rate: float = 0.1,
        weight_scale: float = 0.1,
        index_swap_rate: float = 0.05
    ):
        """
        Mutate weights and input selections.

        Args:
            weight_rate: Probability of mutating each weight
            weight_scale: Std dev of weight mutations
            index_swap_rate: Probability of swapping an input index per neuron
        """
        # Weight mutations
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mask = torch.rand_like(param) < weight_rate
            noise = torch.randn_like(param) * weight_scale
            param.data += mask.float() * noise

        # Index mutations: swap one input for another
        for h in range(self.hidden_size):
            if np.random.random() < index_swap_rate:
                pos = np.random.randint(self.inputs_per_neuron)
                current = set(self.input_indices[h].tolist())
                available = [i for i in range(self.input_size) if i not in current]
                if available:
                    new_idx = np.random.choice(available)
                    self.input_indices[h, pos] = new_idx

    def clone(self) -> 'UltraSparseController':
        """Create a deep copy of this controller."""
        new = UltraSparseController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            inputs_per_neuron=self.inputs_per_neuron,
            device=self.device
        )
        new.input_indices = self.input_indices.clone()
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        return new

    def get_selected_indices(self) -> List[int]:
        """Get list of all selected input indices (unique)."""
        return sorted(set(self.input_indices.flatten().tolist()))

    def get_selection_stats(self, true_features: Optional[List[int]] = None) -> dict:
        """
        Analyze which inputs were selected.

        Args:
            true_features: Optional list of ground-truth important feature indices

        Returns:
            Dictionary with selection statistics
        """
        all_indices = self.input_indices.flatten().tolist()
        unique_indices = set(all_indices)

        stats = {
            'total_connections': len(all_indices),
            'unique_inputs': len(unique_indices),
            'selected_indices': sorted(unique_indices),
        }

        if true_features is not None:
            true_set = set(true_features)
            true_selected = sorted([i for i in unique_indices if i in true_set])
            noise_selected = [i for i in unique_indices if i not in true_set]

            true_connections = sum(1 for i in all_indices if i in true_set)
            noise_connections = sum(1 for i in all_indices if i not in true_set)

            random_rate = len(true_set) / self.input_size
            actual_rate = true_connections / len(all_indices) if all_indices else 0

            stats.update({
                'true_features_selected': true_selected,
                'noise_features_selected': len(noise_selected),
                'true_connections': true_connections,
                'noise_connections': noise_connections,
                'true_ratio': actual_rate,
                'expected_random_ratio': random_rate,
                'selection_factor': actual_rate / random_rate if random_rate > 0 else 0,
            })

        return stats

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return (
            self.w1.numel() + self.b1.numel() +
            self.w2.numel() + self.b2.numel()
        )

    def state_dict(self) -> dict:
        """Get state dictionary for saving."""
        return {
            'input_indices': self.input_indices,
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
        }

    def load_state_dict(self, state: dict):
        """Load state dictionary."""
        self.input_indices = state['input_indices'].to(self.device)
        self.w1 = state['w1'].to(self.device)
        self.b1 = state['b1'].to(self.device)
        self.w2 = state['w2'].to(self.device)
        self.b2 = state['b2'].to(self.device)


class DenseController(nn.Module):
    """
    Standard dense neural network for comparison.

    Same architecture as UltraSparse but with full connectivity.
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 8,
        output_size: int = 1,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = nn.Parameter(torch.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(output_size))

        self.last_hidden = None
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True

        hidden = torch.tanh(nn.functional.linear(x, self.w1, self.b1))
        self.last_hidden = hidden.detach()
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))

        if squeeze_output:
            output = output.squeeze(0)

        return output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
