"""
Metrics Module

Standard metrics for evaluating networks:
- MSE: Mean squared error (accuracy)
- Energy: Inference cost (activation energy + weight energy)
- Saturation: Percentage of saturated neurons (|activation| > 0.95)

All experiments should report these three metrics.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Metrics:
    """Standard metrics for network evaluation."""
    mse: float
    energy: float
    saturation: float

    # Detailed breakdowns
    activation_energy: float = 0.0
    weight_energy: float = 0.0
    saturated_neurons: int = 0
    total_neurons: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mse': self.mse,
            'energy': self.energy,
            'saturation': self.saturation,
            'activation_energy': self.activation_energy,
            'weight_energy': self.weight_energy,
            'saturated_neurons': self.saturated_neurons,
            'total_neurons': self.total_neurons,
        }

    def __str__(self) -> str:
        return f"MSE={self.mse:.6f}, Energy={self.energy:.4f}, Saturation={self.saturation*100:.1f}%"


def compute_metrics(
    controller,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    activation_threshold: float = 0.95
) -> Metrics:
    """
    Compute standard metrics for a controller.

    Args:
        controller: Network with forward() method and weight attributes
        x_test: Test inputs
        y_true: True outputs
        activation_threshold: Threshold for saturation (default 0.95)

    Returns:
        Metrics object with MSE, energy, and saturation
    """
    with torch.no_grad():
        # Forward pass
        pred = controller.forward(x_test)

        # MSE
        mse = torch.mean((pred - y_true) ** 2).item()

        # Get weights
        weights = []
        if hasattr(controller, 'w1'):
            weights.append(controller.w1)
        if hasattr(controller, 'w2'):
            weights.append(controller.w2)
        if hasattr(controller, 'w3'):
            weights.append(controller.w3)

        # Weight energy: average absolute weight magnitude
        if weights:
            total_weights = sum(w.abs().sum().item() for w in weights)
            total_params = sum(w.numel() for w in weights)
            weight_energy = total_weights / total_params
        else:
            weight_energy = 0.0

        # Activation energy and saturation
        activation_energy = 0.0
        saturated_count = 0
        total_neurons = 0

        # Check for stored activations
        activation_attrs = ['last_hidden', 'last_hidden_activations', 'last_sensory', 'last_processing']
        activations = []

        for attr in activation_attrs:
            if hasattr(controller, attr):
                act = getattr(controller, attr)
                if act is not None:
                    activations.append(act)

        if activations:
            for act in activations:
                activation_energy += act.abs().mean().item()
                saturated_count += (act.abs() > activation_threshold).float().sum().item()
                total_neurons += act.shape[-1]  # neurons per layer

            activation_energy /= len(activations)
            saturation = saturated_count / (total_neurons * act.shape[0]) if total_neurons > 0 else 0.0
        else:
            saturation = 0.0

        # Combined energy (weighted sum)
        energy = activation_energy + weight_energy

        return Metrics(
            mse=mse,
            energy=energy,
            saturation=saturation,
            activation_energy=activation_energy,
            weight_energy=weight_energy,
            saturated_neurons=int(saturated_count / act.shape[0]) if activations else 0,
            total_neurons=total_neurons,
        )


def print_metrics(metrics: Metrics, prefix: str = "") -> None:
    """Print metrics in a formatted way."""
    if prefix:
        print(f"{prefix}: {metrics}")
    else:
        print(str(metrics))


def format_metrics_table(results: list, headers: list = None) -> str:
    """
    Format a list of results as a table.

    Args:
        results: List of dicts with 'name' and Metrics or metric values
        headers: Optional custom headers

    Returns:
        Formatted table string
    """
    if not results:
        return ""

    if headers is None:
        headers = ['Name', 'MSE', 'Energy', 'Saturation']

    # Build rows
    rows = []
    for r in results:
        name = r.get('name', 'Unknown')
        if isinstance(r.get('metrics'), Metrics):
            m = r['metrics']
            rows.append([name, f"{m.mse:.6f}", f"{m.energy:.4f}", f"{m.saturation*100:.1f}%"])
        else:
            mse = r.get('mse', 0)
            energy = r.get('energy', 0)
            sat = r.get('saturation', 0)
            rows.append([name, f"{mse:.6f}", f"{energy:.4f}", f"{sat*100:.1f}%"])

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    # Format
    lines = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        lines.append(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))

    return "\n".join(lines)
