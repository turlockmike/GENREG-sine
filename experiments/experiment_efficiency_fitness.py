"""
Experiment: Energy-Efficient Fitness Function

Biological inspiration: Every neuron firing costs energy. Evolution optimizes
for BOTH accuracy AND efficiency - not just accuracy alone.

Fitness = accuracy_score - efficiency_weight * energy_cost

Where energy_cost can be measured as:
1. Total activation energy: sum(|activations|) - neurons that fire strongly cost more
2. Weight magnitude: sum(|weights|) - large weights = more energy to maintain
3. Active neurons: count of neurons that fire above threshold

The efficiency_weight parameter controls the tradeoff:
- 0.0 = only care about accuracy (standard training)
- 0.1 = slight efficiency pressure
- 1.0 = strong efficiency pressure
- 10.0 = extreme efficiency pressure (may sacrifice accuracy)

Goal: Find the Pareto frontier of accuracy vs efficiency.
"""

import torch
import numpy as np
import json
import time
from pathlib import Path

import legacy.sine_config as sine_config as cfg
from legacy.sine_controller import SineController, expand_input

DEVICE = torch.device("cpu")


class EnergyAwareController(SineController):
    """Controller that tracks energy usage."""

    def __init__(self, device=None):
        super().__init__(device)
        self.last_energy = None

    def forward_with_energy(self, x, track=True):
        """Forward pass that also computes energy cost."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        # Handle dimensions
        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        # Expand input
        if cfg.INPUT_EXPANSION:
            x = expand_input(x)

        # Hidden layer
        h_pre = torch.nn.functional.linear(x, self.w1, self.b1)
        h = torch.tanh(h_pre)

        # Track activations
        if track:
            self.last_hidden_activations = h.detach()

        # Output layer
        out_pre = torch.nn.functional.linear(h, self.w2, self.b2)
        output = torch.tanh(out_pre)

        # Compute energy metrics
        energy = {
            # Activation energy: how much neurons fire (biological cost)
            'activation_energy': h.abs().sum().item() / h.numel(),

            # Weight energy: cost of maintaining synapses
            'weight_energy': (self.w1.abs().sum() + self.w2.abs().sum()).item() / (self.w1.numel() + self.w2.numel()),

            # Sparsity: fraction of near-zero weights
            'weight_sparsity': ((self.w1.abs() < 0.01).sum() + (self.w2.abs() < 0.01).sum()).item() / (self.w1.numel() + self.w2.numel()),

            # Active neurons: neurons with significant activation
            'active_neurons': (h.abs().mean(dim=0) > 0.1).sum().item(),

            # Saturated neurons
            'saturated_neurons': (h.abs().mean(dim=0) > 0.95).sum().item(),
        }

        self.last_energy = energy

        # Return output
        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output, energy

    def clone(self):
        """Create a deep copy."""
        new = EnergyAwareController(device=self.device)
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        return new

    @staticmethod
    def random(device=None):
        return EnergyAwareController(device=device)


def compute_fitness(controller, x_test, y_true, efficiency_weight=0.0, energy_type='activation'):
    """
    Compute fitness with accuracy and efficiency components.

    Args:
        controller: The neural network
        x_test: Test inputs
        y_true: True outputs
        efficiency_weight: How much to penalize energy (0 = ignore, higher = more penalty)
        energy_type: 'activation' (neuron firing), 'weight' (synapse cost), or 'combined'

    Returns:
        fitness: Combined score (higher is better)
        mse: Raw accuracy
        energy: Energy cost
    """
    with torch.no_grad():
        pred, energy = controller.forward_with_energy(x_test, track=True)
        mse = torch.mean((pred - y_true) ** 2).item()

    # Select energy metric
    if energy_type == 'activation':
        energy_cost = energy['activation_energy']
    elif energy_type == 'weight':
        energy_cost = energy['weight_energy']
    elif energy_type == 'combined':
        energy_cost = energy['activation_energy'] + energy['weight_energy']
    else:
        energy_cost = energy['activation_energy']

    # Fitness: negative MSE (higher is better) minus energy penalty
    # Scale MSE to be comparable to energy (both roughly 0-1 range)
    accuracy_score = -mse * 10  # Scale up MSE so it's comparable to energy

    fitness = accuracy_score - efficiency_weight * energy_cost

    return fitness, mse, energy_cost, energy


def train_energy_efficient(
    x_test, y_true,
    efficiency_weight=0.0,
    energy_type='activation',
    max_steps=15000,
    verbose=True
):
    """
    Train with energy-efficient fitness function.
    """
    current = EnergyAwareController(device=DEVICE)
    current_fitness, current_mse, current_energy, _ = compute_fitness(
        current, x_test, y_true, efficiency_weight, energy_type
    )

    best = current.clone()
    best_fitness = current_fitness
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    history = []

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        # Mutate
        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)

        # Evaluate with energy-aware fitness
        mutant_fitness, mutant_mse, mutant_energy, _ = compute_fitness(
            mutant, x_test, y_true, efficiency_weight, energy_type
        )

        # SA acceptance
        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            current_mse = mutant_mse
            current_energy = mutant_energy

            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
                best_mse = current_mse

        if step % 500 == 0:
            _, mse, energy, energy_dict = compute_fitness(best, x_test, y_true, efficiency_weight, energy_type)
            history.append({
                'step': step,
                'mse': mse,
                'energy': energy,
                'fitness': best_fitness,
                'active_neurons': energy_dict['active_neurons'],
                'saturated_neurons': energy_dict['saturated_neurons'],
                'weight_sparsity': energy_dict['weight_sparsity'],
            })

            if verbose and step % 2000 == 0:
                print(f"    Step {step}: MSE={mse:.6f}, Energy={energy:.4f}, Active={energy_dict['active_neurons']}")

    return best, history


def run_experiment():
    """Run efficiency sweep experiment."""
    print("=" * 70)
    print("EXPERIMENT: ENERGY-EFFICIENT FITNESS FUNCTION")
    print("=" * 70)
    print("Testing different efficiency weights to find accuracy/efficiency tradeoff")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    # Test different efficiency weights
    efficiency_weights = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    results = {'runs': []}

    print(f"\n{'Eff Weight':<12} | {'MSE':<12} | {'Energy':<10} | {'Active N':<10} | {'Saturated':<10} | {'Sparsity':<10}")
    print("-" * 80)

    for eff_weight in efficiency_weights:
        print(f"\n[Training with efficiency_weight={eff_weight}]")

        torch.manual_seed(42)
        np.random.seed(42)

        best, history = train_energy_efficient(
            x_test, y_true,
            efficiency_weight=eff_weight,
            energy_type='activation',
            max_steps=15000,
            verbose=True
        )

        # Final evaluation
        _, final_mse, final_energy, energy_dict = compute_fitness(
            best, x_test, y_true, eff_weight, 'activation'
        )

        result = {
            'efficiency_weight': eff_weight,
            'mse': final_mse,
            'energy': final_energy,
            'active_neurons': energy_dict['active_neurons'],
            'saturated_neurons': energy_dict['saturated_neurons'],
            'weight_sparsity': energy_dict['weight_sparsity'],
            'history': history,
        }
        results['runs'].append(result)

        print(f"{eff_weight:<12} | {final_mse:<12.6f} | {final_energy:<10.4f} | {energy_dict['active_neurons']:<10} | {energy_dict['saturated_neurons']:<10} | {energy_dict['weight_sparsity']:<10.2%}")

    # Summary - Pareto frontier
    print("\n" + "=" * 70)
    print("PARETO FRONTIER: Accuracy vs Efficiency")
    print("=" * 70)

    # Sort by MSE
    sorted_results = sorted(results['runs'], key=lambda x: x['mse'])

    print(f"\n{'Eff Weight':<12} | {'MSE':<12} | {'Energy':<10} | {'MSE×Energy':<12} | {'Active/Total'}")
    print("-" * 70)

    for r in sorted_results:
        combined = r['mse'] * r['energy']
        active_ratio = f"{r['active_neurons']}/8"
        print(f"{r['efficiency_weight']:<12} | {r['mse']:<12.6f} | {r['energy']:<10.4f} | {combined:<12.6f} | {active_ratio}")

    # Find best combined score
    best_combined = min(results['runs'], key=lambda x: x['mse'] * x['energy'])
    best_accuracy = min(results['runs'], key=lambda x: x['mse'])
    best_efficiency = min(results['runs'], key=lambda x: x['energy'])

    print(f"\nBest accuracy: eff_weight={best_accuracy['efficiency_weight']}, MSE={best_accuracy['mse']:.6f}")
    print(f"Best efficiency: eff_weight={best_efficiency['efficiency_weight']}, Energy={best_efficiency['energy']:.4f}")
    print(f"Best combined (MSE×Energy): eff_weight={best_combined['efficiency_weight']}, Score={best_combined['mse']*best_combined['energy']:.6f}")

    results['summary'] = {
        'best_accuracy_weight': best_accuracy['efficiency_weight'],
        'best_accuracy_mse': best_accuracy['mse'],
        'best_efficiency_weight': best_efficiency['efficiency_weight'],
        'best_efficiency_energy': best_efficiency['energy'],
        'best_combined_weight': best_combined['efficiency_weight'],
        'best_combined_score': best_combined['mse'] * best_combined['energy'],
    }

    # Save results
    output_dir = Path("results/efficiency_fitness_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()
