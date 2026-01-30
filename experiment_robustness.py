"""
Experiment 1: Robustness Test

Hypothesis: Saturated networks (from evolutionary training) are more robust
to input noise than non-saturated networks (from backprop) because binary
gates are less sensitive to perturbations than continuous activations.

Test: Train networks with both methods, then evaluate with increasing noise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
from pathlib import Path

import sine_config as cfg
from sine_controller import SineController, expand_input
from sine_annealing import simulated_annealing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BackpropMLP(nn.Module):
    """Standard MLP for backprop baseline."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        return out.squeeze(-1)

    def get_hidden_activations(self, x):
        with torch.no_grad():
            return torch.tanh(self.fc1(x))

    def get_saturation(self, x):
        h = self.get_hidden_activations(x)
        return (h.abs() > 0.95).float().mean().item()


def train_backprop(x_expanded, y_true, hidden_size=8, max_epochs=500, lr=0.01):
    """Train network with backprop."""
    input_size = x_expanded.shape[1]
    model = BackpropMLP(input_size, hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_mse = float('inf')
    patience, no_improve = 50, 0

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = model(x_expanded)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()

        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience or mse < 0.0001:
            break

    return model, best_mse


def train_evolutionary(x_test, y_true, max_steps=15000):
    """Train network with simulated annealing."""
    best, history = simulated_annealing(
        x_test, y_true,
        max_steps=max_steps,
        t_initial=0.01,
        t_final=0.00001,
        verbose=False,
    )

    with torch.no_grad():
        pred = best.forward(x_test, track=True)
        mse = torch.mean((pred - y_true) ** 2).item()

    return best, mse


def evaluate_with_noise(model, x_raw, y_true, noise_std, is_backprop=True, n_trials=10):
    """Evaluate model with noisy inputs."""
    mses = []

    for _ in range(n_trials):
        # Add noise to raw input before expansion
        x_noisy = x_raw + torch.randn_like(x_raw) * noise_std

        with torch.no_grad():
            if is_backprop:
                x_expanded = expand_input(x_noisy)
                pred = model(x_expanded)
            else:
                pred = model.forward(x_noisy.squeeze(), track=False)

            mse = torch.mean((pred - y_true) ** 2).item()
            mses.append(mse)

    return np.mean(mses), np.std(mses)


def run_experiment():
    """Run robustness comparison."""
    print("=" * 70)
    print("EXPERIMENT 1: ROBUSTNESS TEST")
    print("=" * 70)
    print("Hypothesis: Saturated networks are more robust to input noise")
    print("=" * 70)

    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE).unsqueeze(-1)
    x_expanded = expand_input(x_raw)
    y_true = torch.sin(x_raw.squeeze())

    # Train both models
    print("\n[Training Backprop Model]")
    bp_model, bp_mse = train_backprop(x_expanded, y_true)
    bp_sat = bp_model.get_saturation(x_expanded)
    print(f"  Clean MSE: {bp_mse:.6f}, Saturation: {bp_sat:.1%}")

    print("\n[Training Evolutionary Model (SA)]")
    evo_model, evo_mse = train_evolutionary(x_raw.squeeze(), y_true)
    evo_sat = evo_model.get_k() / cfg.HIDDEN_SIZE
    print(f"  Clean MSE: {evo_mse:.6f}, Saturation: {evo_sat:.1%}")

    # Test with increasing noise
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    print("\n[Robustness Test]")
    print(f"{'Noise σ':>10} | {'Backprop MSE':>15} | {'Evo MSE':>15} | {'Winner':>10}")
    print("-" * 60)

    results = {
        'noise_levels': noise_levels,
        'backprop': {'clean_mse': bp_mse, 'saturation': bp_sat, 'noisy_mse': [], 'noisy_std': []},
        'evolutionary': {'clean_mse': evo_mse, 'saturation': evo_sat, 'noisy_mse': [], 'noisy_std': []},
    }

    bp_wins, evo_wins = 0, 0

    for noise_std in noise_levels:
        bp_noisy_mse, bp_noisy_std = evaluate_with_noise(
            bp_model, x_raw, y_true, noise_std, is_backprop=True
        )
        evo_noisy_mse, evo_noisy_std = evaluate_with_noise(
            evo_model, x_raw, y_true, noise_std, is_backprop=False
        )

        results['backprop']['noisy_mse'].append(bp_noisy_mse)
        results['backprop']['noisy_std'].append(bp_noisy_std)
        results['evolutionary']['noisy_mse'].append(evo_noisy_mse)
        results['evolutionary']['noisy_std'].append(evo_noisy_std)

        winner = "Backprop" if bp_noisy_mse < evo_noisy_mse else "Evo"
        if bp_noisy_mse < evo_noisy_mse:
            bp_wins += 1
        else:
            evo_wins += 1

        print(f"{noise_std:>10.1f} | {bp_noisy_mse:>12.6f} ± {bp_noisy_std:.3f} | "
              f"{evo_noisy_mse:>12.6f} ± {evo_noisy_std:.3f} | {winner:>10}")

    # Compute degradation ratios
    bp_degradation = results['backprop']['noisy_mse'][-1] / results['backprop']['noisy_mse'][0]
    evo_degradation = results['evolutionary']['noisy_mse'][-1] / results['evolutionary']['noisy_mse'][0]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Backprop wins: {bp_wins}/{len(noise_levels)}")
    print(f"  Evolutionary wins: {evo_wins}/{len(noise_levels)}")
    print(f"  Backprop degradation (noise=1.0 / noise=0): {bp_degradation:.1f}x")
    print(f"  Evolutionary degradation: {evo_degradation:.1f}x")

    if evo_degradation < bp_degradation:
        print("\n  >> HYPOTHESIS SUPPORTED: Evolutionary model degrades less")
    else:
        print("\n  >> HYPOTHESIS NOT SUPPORTED: Backprop model degrades less")

    results['summary'] = {
        'bp_wins': bp_wins,
        'evo_wins': evo_wins,
        'bp_degradation': bp_degradation,
        'evo_degradation': evo_degradation,
        'hypothesis_supported': bool(evo_degradation < bp_degradation),
    }

    # Save results
    output_dir = Path("results/robustness_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
