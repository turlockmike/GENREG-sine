"""
Experiment: Parameter Sweep - Finding the Efficiency Sweet Spot

Goal: Map the accuracy vs efficiency tradeoff across:
- Network sizes (params)
- Problem complexities (features, true signals)

This helps identify:
1. Minimum params needed for a given problem
2. Diminishing returns threshold
3. Optimal architecture for different problem scales
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from core.models import UltraSparseController, DenseController


@dataclass
class Config:
    """Experiment configuration."""
    n_features: int
    n_true: int
    hidden_size: int
    inputs_per_neuron: int

    @property
    def us_params(self) -> int:
        """Ultra-Sparse parameter count."""
        return self.hidden_size * self.inputs_per_neuron + self.hidden_size + self.hidden_size + 1

    @property
    def dense_params(self) -> int:
        """Dense parameter count."""
        return self.hidden_size * self.n_features + self.hidden_size + self.hidden_size + 1

    @property
    def total_connections(self) -> int:
        """Total sparse connections."""
        return self.hidden_size * self.inputs_per_neuron


def generate_data(n_samples: int, n_features: int, n_true: int, seed: int = 42):
    """Generate regression data with known true features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.rand(n_samples, n_features)
    y = torch.zeros(n_samples)

    # Linear terms with decreasing weights
    for i in range(min(n_true, 5)):
        weight = 10 - 2 * i  # 10, 8, 6, 4, 2
        y += weight * x[:, i]

    # Nonlinear terms for remaining true features
    if n_true > 5:
        y += 5 * torch.sin(np.pi * x[:, 5] * x[:, min(6, n_features-1)])
    if n_true > 7:
        y += 3 * (x[:, 7] - 0.5) ** 2
    if n_true > 8:
        y += 2 * torch.abs(x[:, 8] - 0.5)
    if n_true > 9:
        y += torch.cos(2 * np.pi * x[:, 9])

    y += torch.randn(n_samples) * 0.5

    # Normalize for tanh
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8) * 0.8

    return x, y_norm, list(range(n_true))


def train_ultra_sparse(x, y, true_features, config: Config, max_steps=50000):
    """Train Ultra-Sparse with SA."""
    controller = UltraSparseController(
        input_size=config.n_features,
        hidden_size=config.hidden_size,
        inputs_per_neuron=config.inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x).squeeze()
        current_mse = torch.mean((pred - y) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    # Adjust index swap rate based on search space
    index_swap_rate = min(0.2, 0.05 * (config.n_features / 100))

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=index_swap_rate)

        with torch.no_grad():
            pred = mutant.forward(x).squeeze()
            mutant_mse = torch.mean((pred - y) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

    stats = best.get_selection_stats(true_features)
    return best_mse, stats


def train_dense(x, y, config: Config, epochs=5000):
    """Train Dense with backprop."""
    model = DenseController(
        input_size=config.n_features,
        hidden_size=config.hidden_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x).squeeze()
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_mse:
            best_mse = loss.item()

    return best_mse


def run_sweep():
    """Run parameter sweep across problem sizes and network configs."""

    print("=" * 80)
    print("PARAMETER SWEEP: Finding the Efficiency Sweet Spot")
    print("=" * 80)

    # Problem complexities to test
    problems = [
        {"n_features": 100, "n_true": 5, "name": "Small (100/5)"},
        {"n_features": 500, "n_true": 10, "name": "Medium (500/10)"},
        {"n_features": 1000, "n_true": 10, "name": "Large (1000/10)"},
        {"n_features": 2000, "n_true": 15, "name": "XLarge (2000/15)"},
    ]

    # Network configurations to test
    # Format: (hidden_size, inputs_per_neuron)
    network_configs = [
        (4, 2),    # Tiny: 17 params
        (8, 2),    # Small: 33 params
        (8, 4),    # Medium: 49 params
        (16, 4),   # Large: 97 params
        (16, 8),   # XLarge: 161 params
        (32, 4),   # XXLarge: 193 params
        (32, 8),   # Huge: 321 params
    ]

    n_samples = 500
    n_trials = 3
    sa_steps = 50000
    bp_epochs = 5000

    results = []

    for prob in problems:
        print(f"\n{'='*80}")
        print(f"PROBLEM: {prob['name']}")
        print(f"  Features: {prob['n_features']}, True: {prob['n_true']}")
        print("=" * 80)

        # Generate data once per problem
        x, y, true_features = generate_data(
            n_samples, prob['n_features'], prob['n_true'], seed=42
        )

        problem_results = {
            'problem': prob['name'],
            'n_features': prob['n_features'],
            'n_true': prob['n_true'],
            'configs': []
        }

        # Test each network config
        for hidden, k in network_configs:
            config = Config(
                n_features=prob['n_features'],
                n_true=prob['n_true'],
                hidden_size=hidden,
                inputs_per_neuron=k
            )

            print(f"\n  Network: H={hidden}, K={k} ({config.us_params} params, {config.total_connections} connections)")

            us_mses = []
            us_true_ratios = []

            for trial in range(n_trials):
                torch.manual_seed(trial * 1000)
                np.random.seed(trial * 1000)

                mse, stats = train_ultra_sparse(x, y, true_features, config, max_steps=sa_steps)
                us_mses.append(mse)
                us_true_ratios.append(stats['true_ratio'])

            us_mean_mse = np.mean(us_mses)
            us_best_mse = min(us_mses)
            us_mean_true = np.mean(us_true_ratios)

            # Also run dense for comparison (only once per problem, smallest hidden)
            if hidden == 4:
                dense_config = Config(prob['n_features'], prob['n_true'], hidden, k)
                torch.manual_seed(42)
                dense_mse = train_dense(x, y, dense_config, epochs=bp_epochs)
                problem_results['dense_mse'] = dense_mse
                problem_results['dense_params'] = dense_config.dense_params

            config_result = {
                'hidden': hidden,
                'k': k,
                'params': config.us_params,
                'connections': config.total_connections,
                'mean_mse': us_mean_mse,
                'best_mse': us_best_mse,
                'mean_true_ratio': us_mean_true,
            }
            problem_results['configs'].append(config_result)

            print(f"    MSE: {us_mean_mse:.4f} (best: {us_best_mse:.4f})")
            print(f"    True ratio: {us_mean_true:.0%}")

        results.append(problem_results)

    # ================================================================
    # Summary Tables
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Accuracy vs Efficiency")
    print("=" * 80)

    for prob_result in results:
        print(f"\n--- {prob_result['problem']} ---")
        print(f"Dense baseline: MSE={prob_result.get('dense_mse', 'N/A'):.4f}, Params={prob_result.get('dense_params', 'N/A')}")
        print()
        print(f"{'Config':<12} | {'Params':<8} | {'MSE':<10} | {'True%':<8} | {'Efficiency':<12}")
        print("-" * 60)

        dense_mse = prob_result.get('dense_mse', 1.0)

        for cfg in prob_result['configs']:
            # Efficiency = (dense_params / us_params) / (us_mse / dense_mse)
            # Higher is better - more param savings per unit of accuracy loss
            if cfg['mean_mse'] > 0 and dense_mse > 0:
                param_ratio = prob_result.get('dense_params', cfg['params']) / cfg['params']
                mse_ratio = cfg['mean_mse'] / dense_mse if dense_mse < cfg['mean_mse'] else dense_mse / cfg['mean_mse']
                efficiency = param_ratio / max(mse_ratio, 1.0)
            else:
                efficiency = 0

            print(f"H={cfg['hidden']:2d},K={cfg['k']:<2d} | {cfg['params']:<8d} | {cfg['mean_mse']:<10.4f} | {cfg['mean_true_ratio']:<8.0%} | {efficiency:<12.1f}")

    # Find Pareto optimal configs
    print("\n" + "=" * 80)
    print("PARETO OPTIMAL CONFIGURATIONS")
    print("=" * 80)

    for prob_result in results:
        print(f"\n{prob_result['problem']}:")

        configs = prob_result['configs']
        pareto = []

        for cfg in configs:
            is_dominated = False
            for other in configs:
                # Check if other dominates cfg (better on both params AND mse)
                if (other['params'] <= cfg['params'] and
                    other['mean_mse'] < cfg['mean_mse'] and
                    (other['params'] < cfg['params'] or other['mean_mse'] < cfg['mean_mse'])):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(cfg)

        for cfg in sorted(pareto, key=lambda x: x['params']):
            print(f"  H={cfg['hidden']}, K={cfg['k']}: {cfg['params']} params, MSE={cfg['mean_mse']:.4f}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    for prob_result in results:
        configs = sorted(prob_result['configs'], key=lambda x: x['mean_mse'])
        best = configs[0]

        # Find smallest config within 2x of best MSE
        efficient = None
        for cfg in sorted(prob_result['configs'], key=lambda x: x['params']):
            if cfg['mean_mse'] < best['mean_mse'] * 2:
                efficient = cfg
                break

        print(f"\n{prob_result['problem']}:")
        print(f"  Best accuracy: H={best['hidden']}, K={best['k']} ({best['params']} params, MSE={best['mean_mse']:.4f})")
        if efficient and efficient != best:
            print(f"  Most efficient: H={efficient['hidden']}, K={efficient['k']} ({efficient['params']} params, MSE={efficient['mean_mse']:.4f})")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "param_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_dir / "results.json", "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_sweep()
