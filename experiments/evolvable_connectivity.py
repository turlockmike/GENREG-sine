"""
Experiment: Evolvable Connectivity Masks

Hypothesis: Evolving connectivity (which connections exist) alongside weights
will enable natural selection to discover input selection - something that
weight-only evolution cannot achieve.

Architecture:
    Genome = {
        weights: float[input × hidden],
        mask: bool[input × hidden],    # NEW: Does connection exist?
    }

    Forward: h = tanh((inputs ⊙ mask) @ weights)

The mask gives evolution direct control over which inputs to use.
A sparsity penalty creates pressure to disable useless connections.

Inspired by:
- NEAT (topology + weight evolution)
- Biological synaptic pruning
- Lottery Ticket Hypothesis (sparse subnetworks work)
- Our Experiment 5 (deletion mutations during training worked!)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
from typing import Tuple, Dict, List

from legacy.sine_controller import expand_input
from core.metrics import compute_metrics, Metrics

DEVICE = torch.device("cpu")


class ConnectivityMaskController:
    """
    Controller with evolvable connectivity masks.

    Each input-to-hidden connection has:
    - weight: float (strength of connection)
    - mask: bool (does connection exist?)

    Only masked connections contribute to the forward pass.
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 8,
        initial_density: float = 1.0,  # Start fully connected
        device=None
    ):
        self.device = device or DEVICE
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Layer 1: Input → Hidden (with mask)
        scale_w1 = np.sqrt(2.0 / input_size)
        self.w1 = torch.randn(hidden_size, input_size, device=self.device) * scale_w1
        self.b1 = torch.zeros(hidden_size, device=self.device)

        # Connectivity mask (True = connection exists)
        if initial_density >= 1.0:
            self.mask1 = torch.ones(hidden_size, input_size, dtype=torch.bool, device=self.device)
        else:
            self.mask1 = torch.rand(hidden_size, input_size, device=self.device) < initial_density

        # Layer 2: Hidden → Output (no mask, always fully connected)
        scale_w2 = np.sqrt(2.0 / hidden_size)
        self.w2 = torch.randn(1, hidden_size, device=self.device) * scale_w2
        self.b2 = torch.zeros(1, device=self.device)

        # For tracking
        self.last_hidden = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        # Expand input to 256D
        expanded = expand_input(x)

        # Apply mask to weights (masked weights become 0)
        masked_w1 = self.w1 * self.mask1.float()

        # Forward pass
        hidden = torch.tanh(torch.nn.functional.linear(expanded, masked_w1, self.b1))
        self.last_hidden = hidden.detach()

        output = torch.tanh(torch.nn.functional.linear(hidden, self.w2, self.b2))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def mutate(
        self,
        weight_rate: float = 0.1,
        weight_scale: float = 0.1,
        mask_flip_rate: float = 0.01,  # Probability of flipping each mask bit
    ):
        """
        Mutate both weights and connectivity mask.

        Args:
            weight_rate: Probability of mutating each weight
            weight_scale: Std dev of weight mutations
            mask_flip_rate: Probability of flipping each mask bit
        """
        # Weight mutations (only for active connections)
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mutation_mask = torch.rand_like(param) < weight_rate
            noise = torch.randn_like(param) * weight_scale
            param.data += mutation_mask.float() * noise

        # Mask mutations (flip bits)
        flip_mask = torch.rand_like(self.mask1.float()) < mask_flip_rate
        self.mask1 = self.mask1 ^ flip_mask  # XOR to flip

    def clone(self):
        new = ConnectivityMaskController(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            initial_density=1.0,  # Will be overwritten
            device=self.device
        )
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        new.mask1 = self.mask1.clone()
        return new

    def get_connectivity_stats(self) -> Dict:
        """Analyze which inputs are connected."""
        mask = self.mask1.cpu().numpy()

        # Total connectivity
        total_connections = mask.size
        active_connections = mask.sum()
        density = active_connections / total_connections

        # Per-input analysis (which inputs have ANY connection to hidden layer)
        input_connected = mask.any(axis=0)  # Shape: (256,)
        true_input_connected = input_connected[:16].sum()  # First 16 are true signals
        noise_input_connected = input_connected[16:].sum()  # Rest are noise

        # Connection count per input
        connections_per_input = mask.sum(axis=0)
        true_connections = connections_per_input[:16].sum()
        noise_connections = connections_per_input[16:].sum()

        return {
            'density': float(density),
            'active_connections': int(active_connections),
            'total_connections': int(total_connections),
            'true_inputs_connected': int(true_input_connected),
            'noise_inputs_connected': int(noise_input_connected),
            'true_inputs_total': 16,
            'noise_inputs_total': 240,
            'true_connection_count': int(true_connections),
            'noise_connection_count': int(noise_connections),
            'true_selection_ratio': float(true_input_connected / 16),
            'noise_selection_ratio': float(noise_input_connected / 240),
        }

    def num_parameters(self) -> int:
        """Count active parameters (only masked connections count)."""
        active_w1 = self.mask1.sum().item()
        return int(active_w1 + self.b1.numel() + self.w2.numel() + self.b2.numel())


def compute_fitness(
    controller: ConnectivityMaskController,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    sparsity_weight: float = 0.0,
) -> Tuple[float, float, Dict]:
    """
    Compute fitness with optional sparsity penalty.

    fitness = -MSE - sparsity_weight × density

    Higher sparsity_weight = more pressure to disable connections.
    """
    with torch.no_grad():
        pred = controller.forward(x_test)
        mse = torch.mean((pred - y_true) ** 2).item()

    stats = controller.get_connectivity_stats()
    density = stats['density']

    # Fitness: negative MSE (higher is better) minus density penalty
    fitness = -mse - sparsity_weight * density

    return fitness, mse, stats


def train_evolutionary(
    controller: ConnectivityMaskController,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_steps: int = 20000,
    sparsity_weight: float = 0.1,
    weight_rate: float = 0.1,
    weight_scale: float = 0.1,
    mask_flip_rate: float = 0.01,
    t_initial: float = 0.01,
    t_final: float = 0.00001,
    verbose: bool = True,
    report_interval: int = 2000,
) -> Tuple[ConnectivityMaskController, List[Dict]]:
    """
    Train using Simulated Annealing with connectivity mutations.
    """
    current = controller
    current_fitness, current_mse, _ = compute_fitness(current, x_test, y_true, sparsity_weight)

    best = current.clone()
    best_fitness = current_fitness
    best_mse = current_mse

    decay = (t_final / t_initial) ** (1.0 / max_steps)
    history = []

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        # Mutate
        mutant = current.clone()
        mutant.mutate(
            weight_rate=weight_rate,
            weight_scale=weight_scale,
            mask_flip_rate=mask_flip_rate,
        )

        # Evaluate
        mutant_fitness, mutant_mse, mutant_stats = compute_fitness(
            mutant, x_test, y_true, sparsity_weight
        )

        # SA acceptance
        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            current_mse = mutant_mse

            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
                best_mse = current_mse

        # Report
        if step % report_interval == 0:
            stats = best.get_connectivity_stats()
            metrics = compute_metrics(best, x_test, y_true)

            history.append({
                'step': step,
                'mse': best_mse,
                'fitness': best_fitness,
                'density': stats['density'],
                'true_connected': stats['true_inputs_connected'],
                'noise_connected': stats['noise_inputs_connected'],
                'energy': metrics.energy,
                'saturation': metrics.saturation,
            })

            if verbose:
                print(f"    Step {step}: MSE={best_mse:.6f}, Density={stats['density']:.1%}, "
                      f"True={stats['true_inputs_connected']}/16, Noise={stats['noise_inputs_connected']}/240, "
                      f"Sat={metrics.saturation*100:.0f}%")

    return best, history


def run_experiment():
    """Run the evolvable connectivity experiment."""
    print("=" * 70)
    print("EXPERIMENT: Evolvable Connectivity Masks")
    print("=" * 70)
    print("Testing if evolving connectivity enables input selection")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    results = {'configs': {}}

    # Test configurations
    configs = [
        # (name, initial_density, sparsity_weight, mask_flip_rate)
        ('baseline_no_mask', 1.0, 0.0, 0.0),      # No mask mutations (baseline)
        ('dense_light_pressure', 1.0, 0.01, 0.01),  # Start dense, light sparsity pressure
        ('dense_medium_pressure', 1.0, 0.1, 0.01),  # Start dense, medium pressure
        ('dense_heavy_pressure', 1.0, 0.5, 0.01),   # Start dense, heavy pressure
        ('sparse_start', 0.2, 0.1, 0.02),           # Start 20% connected
        ('sparse_start_heavy', 0.2, 0.5, 0.02),    # Start sparse, heavy pressure
    ]

    for name, density, sparsity_wt, flip_rate in configs:
        print(f"\n[{name}]")
        print(f"    Initial density: {density:.0%}, Sparsity weight: {sparsity_wt}, Mask flip rate: {flip_rate}")

        torch.manual_seed(42)
        np.random.seed(42)

        controller = ConnectivityMaskController(
            input_size=256,
            hidden_size=8,
            initial_density=density,
            device=DEVICE
        )

        initial_stats = controller.get_connectivity_stats()
        print(f"    Initial: {initial_stats['active_connections']} connections, "
              f"True={initial_stats['true_inputs_connected']}/16, Noise={initial_stats['noise_inputs_connected']}/240")

        best, history = train_evolutionary(
            controller, x_test, y_true,
            max_steps=20000,
            sparsity_weight=sparsity_wt,
            mask_flip_rate=flip_rate,
            verbose=True,
            report_interval=4000,
        )

        # Final analysis
        final_stats = best.get_connectivity_stats()
        final_metrics = compute_metrics(best, x_test, y_true)

        results['configs'][name] = {
            'initial_density': density,
            'sparsity_weight': sparsity_wt,
            'mask_flip_rate': flip_rate,
            'final_mse': final_metrics.mse,
            'final_energy': final_metrics.energy,
            'final_saturation': final_metrics.saturation,
            'final_density': final_stats['density'],
            'true_inputs_connected': final_stats['true_inputs_connected'],
            'noise_inputs_connected': final_stats['noise_inputs_connected'],
            'true_selection_ratio': final_stats['true_selection_ratio'],
            'noise_selection_ratio': final_stats['noise_selection_ratio'],
            'active_params': best.num_parameters(),
            'history': history,
        }

        print(f"\n    FINAL: MSE={final_metrics.mse:.6f}, Energy={final_metrics.energy:.4f}, "
              f"Sat={final_metrics.saturation*100:.0f}%")
        print(f"    Connectivity: {final_stats['density']:.1%} dense, "
              f"True={final_stats['true_inputs_connected']}/16 ({final_stats['true_selection_ratio']:.0%}), "
              f"Noise={final_stats['noise_inputs_connected']}/240 ({final_stats['noise_selection_ratio']:.0%})")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<25} | {'MSE':<10} | {'Energy':<8} | {'Sat':<6} | {'Density':<8} | {'True':<6} | {'Noise':<8} | {'Selection?'}")
    print("-" * 100)

    for name in [c[0] for c in configs]:
        r = results['configs'][name]
        # Check if selection happened: true_selection_ratio >> noise_selection_ratio
        selection_ratio = r['true_selection_ratio'] / max(r['noise_selection_ratio'], 0.001)
        selected = "YES!" if selection_ratio > 2.0 else ("partial" if selection_ratio > 1.2 else "no")
        print(f"{name:<25} | {r['final_mse']:<10.6f} | {r['final_energy']:<8.4f} | {r['final_saturation']*100:<5.0f}% | "
              f"{r['final_density']:<7.1%} | {r['true_inputs_connected']:<6}/16 | {r['noise_inputs_connected']:<8}/240 | {selected}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Find best configuration for input selection
    best_selection = max(results['configs'].items(),
                        key=lambda x: x[1]['true_selection_ratio'] / max(x[1]['noise_selection_ratio'], 0.001))
    best_mse_config = min(results['configs'].items(), key=lambda x: x[1]['final_mse'])

    print(f"""
Best input selection: {best_selection[0]}
  - True inputs connected: {best_selection[1]['true_inputs_connected']}/16 ({best_selection[1]['true_selection_ratio']:.0%})
  - Noise inputs connected: {best_selection[1]['noise_inputs_connected']}/240 ({best_selection[1]['noise_selection_ratio']:.0%})
  - Selection ratio: {best_selection[1]['true_selection_ratio'] / max(best_selection[1]['noise_selection_ratio'], 0.001):.1f}x

Best MSE: {best_mse_config[0]}
  - MSE: {best_mse_config[1]['final_mse']:.6f}
  - Active parameters: {best_mse_config[1]['active_params']}
""")

    # Check if hypothesis is supported
    baseline = results['configs']['baseline_no_mask']
    best_with_mask = min(
        [(k, v) for k, v in results['configs'].items() if v['mask_flip_rate'] > 0],
        key=lambda x: x[1]['final_mse']
    )

    if best_with_mask[1]['true_selection_ratio'] > best_with_mask[1]['noise_selection_ratio'] * 1.5:
        print("HYPOTHESIS SUPPORTED: Evolvable connectivity enables input selection!")
        print(f"  True signals are {best_with_mask[1]['true_selection_ratio'] / max(best_with_mask[1]['noise_selection_ratio'], 0.001):.1f}x more likely to remain connected")
    else:
        print("HYPOTHESIS NOT SUPPORTED: Connectivity masks didn't improve input selection")
        print("  Consider: higher sparsity pressure, more generations, or different mutation rates")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "evolvable_connectivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
