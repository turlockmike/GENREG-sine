"""
Experiment: High-Dimensional Scaling Test

Question: Can Ultra-Sparse + SA find 10 needles in 1000 features?

This is the critical test for the efficiency hypothesis:
- Dense Backprop: 8017 params (8×1000 + 8 + 8 + 1)
- Ultra-Sparse:   49 params   (8×4 + 8 + 8 + 1)

If Ultra-Sparse achieves comparable accuracy with 163x fewer parameters,
that validates the efficiency advantage at scale.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from collections import Counter

from core.models import UltraSparseController, DenseController


def generate_highdim_data(
    n_samples: int,
    n_features: int = 1000,
    n_true: int = 10,
    seed: int = 42
):
    """
    Generate high-dimensional regression data.

    y = sum(w_i * x_i) + nonlinear_interactions + noise

    Only the first n_true features matter.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # All features uniform [0, 1]
    x = torch.rand(n_samples, n_features)

    # True features: mix of linear and nonlinear effects
    # Features 0-4: linear terms with varying weights
    # Features 5-9: interaction/nonlinear terms

    y = torch.zeros(n_samples)

    # Linear terms (features 0-4)
    weights = [10, 8, 6, 4, 2]  # Decreasing importance
    for i, w in enumerate(weights):
        if i < n_true:
            y += w * x[:, i]

    # Nonlinear terms (features 5-9)
    if n_true > 5:
        y += 5 * torch.sin(np.pi * x[:, 5] * x[:, 6])  # Interaction
    if n_true > 7:
        y += 3 * (x[:, 7] - 0.5) ** 2  # Quadratic
    if n_true > 8:
        y += 2 * torch.abs(x[:, 8] - 0.5)  # V-shape
    if n_true > 9:
        y += 1 * torch.cos(2 * np.pi * x[:, 9])  # Periodic

    # Add noise
    y += torch.randn(n_samples) * 0.5

    true_features = list(range(n_true))

    return x, y, true_features


def train_ultra_sparse_sa(
    x_features, y_train, true_features,
    hidden_size=8, inputs_per_neuron=4,
    max_steps=100000, verbose=True
):
    """Train Ultra-Sparse with Simulated Annealing."""
    input_size = x_features.shape[1]

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x_features).squeeze()
        current_mse = torch.mean((pred - y_train) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    # Longer schedule for harder problem
    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    start_time = time.time()

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        # Higher index swap rate for larger search space
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(x_features).squeeze()
            mutant_mse = torch.mean((pred - y_train) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        if verbose and step % 20000 == 0:
            elapsed = time.time() - start_time
            stats = best.get_selection_stats(true_features)
            print(f"    Step {step}: MSE={best_mse:.4f}, "
                  f"True={stats['true_connections']}/{stats['total_connections']}, "
                  f"Time={elapsed:.1f}s")

    return best, best_mse


def train_dense_backprop(
    x_features, y_train,
    hidden_size=8, epochs=10000, verbose=True
):
    """Train Dense Backprop with Adam."""
    input_size = x_features.shape[1]

    model = DenseController(
        input_size=input_size,
        hidden_size=hidden_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')
    best_state = None

    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_features).squeeze()
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_mse:
            best_mse = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"    Epoch {epoch}: MSE={loss.item():.4f}, Time={elapsed:.1f}s")

    model.load_state_dict(best_state)
    return model, best_mse


def run_experiment(
    n_features=1000,
    n_true=10,
    n_samples=500,
    n_trials=5,
    hidden_size=8,
    inputs_per_neuron=4,
    sa_steps=100000,
    bp_epochs=10000
):
    """Run the high-dimensional scaling experiment."""

    print("=" * 70)
    print("EXPERIMENT: High-Dimensional Scaling Test")
    print("=" * 70)
    print(f"Features: {n_features} total, {n_true} true ({100*n_true/n_features:.1f}% signal)")
    print(f"Samples: {n_samples}")
    print(f"Trials: {n_trials}")
    print()

    # Calculate param counts
    us_params = hidden_size * inputs_per_neuron + hidden_size + hidden_size + 1
    dense_params = hidden_size * n_features + hidden_size + hidden_size + 1

    print(f"Ultra-Sparse: {us_params} params ({hidden_size}×{inputs_per_neuron} connections)")
    print(f"Dense:        {dense_params} params ({hidden_size}×{n_features} connections)")
    print(f"Ratio:        {dense_params/us_params:.0f}x fewer params (Ultra-Sparse)")
    print("=" * 70)

    # Generate data
    x_raw, y_true, true_features = generate_highdim_data(
        n_samples, n_features, n_true, seed=42
    )

    # Normalize targets for tanh output
    y_mean, y_std = y_true.mean(), y_true.std()
    y_train = (y_true - y_mean) / (y_std + 1e-8) * 0.8

    print(f"Target range: [{y_true.min():.2f}, {y_true.max():.2f}] -> normalized")
    print(f"True features: {true_features}")

    # Results storage
    results = {
        'config': {
            'n_features': n_features,
            'n_true': n_true,
            'n_samples': n_samples,
            'hidden_size': hidden_size,
            'inputs_per_neuron': inputs_per_neuron,
        },
        'ultra_sparse': [],
        'dense_backprop': [],
    }

    for trial in range(n_trials):
        print(f"\n{'='*70}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print("=" * 70)

        torch.manual_seed(trial * 1000)
        np.random.seed(trial * 1000)

        # 1. Ultra-Sparse SA
        print("\n[Ultra-Sparse SA]")
        best_us, mse_us = train_ultra_sparse_sa(
            x_raw, y_train, true_features,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            max_steps=sa_steps,
            verbose=True
        )

        stats_us = best_us.get_selection_stats(true_features)
        results['ultra_sparse'].append({
            'mse': mse_us,
            'params': best_us.num_parameters(),
            'selected': stats_us['selected_indices'],
            'true_ratio': stats_us['true_ratio'],
            'true_connections': stats_us['true_connections'],
            'total_connections': stats_us['total_connections'],
        })
        print(f"  Final: MSE={mse_us:.4f}, "
              f"True={stats_us['true_connections']}/{stats_us['total_connections']} "
              f"({stats_us['true_ratio']:.0%})")

        # 2. Dense Backprop
        print("\n[Dense Backprop]")
        torch.manual_seed(trial * 1000)
        best_bp, mse_bp = train_dense_backprop(
            x_raw, y_train,
            hidden_size=hidden_size,
            epochs=bp_epochs,
            verbose=True
        )

        results['dense_backprop'].append({
            'mse': mse_bp,
            'params': best_bp.num_parameters(),
        })
        print(f"  Final: MSE={mse_bp:.4f}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    us_mses = [r['mse'] for r in results['ultra_sparse']]
    bp_mses = [r['mse'] for r in results['dense_backprop']]

    us_params = results['ultra_sparse'][0]['params']
    bp_params = results['dense_backprop'][0]['params']

    print(f"\n{'Method':<20} | {'Best MSE':<10} | {'Mean MSE':<10} | {'Params':<8}")
    print("-" * 60)
    print(f"{'Ultra-Sparse SA':<20} | {min(us_mses):<10.4f} | {np.mean(us_mses):<10.4f} | {us_params:<8}")
    print(f"{'Dense Backprop':<20} | {min(bp_mses):<10.4f} | {np.mean(bp_mses):<10.4f} | {bp_params:<8}")

    # Feature selection analysis
    print(f"\n--- Feature Selection Analysis ---")
    print(f"True features: {true_features}")

    all_selected = []
    for r in results['ultra_sparse']:
        all_selected.extend(r['selected'])
    freq = Counter(all_selected)

    print(f"\nUltra-Sparse SA - Most frequently selected:")
    for idx, count in freq.most_common(15):
        is_true = "TRUE" if idx in true_features else ""
        print(f"  Feature {idx}: {count}/{n_trials} trials {is_true}")

    # Calculate recall
    true_found = sum(1 for f in true_features if freq.get(f, 0) >= 1)
    true_majority = sum(1 for f in true_features if freq.get(f, 0) >= n_trials // 2 + 1)

    print(f"\nTrue features found (at least once): {true_found}/{n_true}")
    print(f"True features found (majority): {true_majority}/{n_true}")

    # Efficiency analysis
    print(f"\n--- Efficiency Analysis ---")
    print(f"Parameter ratio: {bp_params/us_params:.0f}x fewer (Ultra-Sparse)")

    if np.mean(bp_mses) < np.mean(us_mses):
        mse_ratio = np.mean(us_mses) / np.mean(bp_mses)
        print(f"MSE ratio: Dense is {mse_ratio:.1f}x better accuracy")
        print(f"Efficiency: {bp_params/us_params / mse_ratio:.0f}x better params/accuracy (Ultra-Sparse)")
    else:
        mse_ratio = np.mean(bp_mses) / np.mean(us_mses)
        print(f"MSE ratio: Ultra-Sparse is {mse_ratio:.1f}x better accuracy")
        print(f"Efficiency: Ultra-Sparse wins on BOTH params AND accuracy!")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"\n{us_params} params vs {bp_params} params ({bp_params/us_params:.0f}x difference)")
    print(f"Ultra-Sparse MSE: {np.mean(us_mses):.4f}")
    print(f"Dense MSE:        {np.mean(bp_mses):.4f}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "highdim_scaling"
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
    # Main experiment: 1000 features, 10 true
    run_experiment(
        n_features=1000,
        n_true=10,
        n_samples=500,
        n_trials=5,
        hidden_size=8,
        inputs_per_neuron=4,
        sa_steps=100000,  # More steps for harder problem
        bp_epochs=10000
    )
