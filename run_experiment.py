#!/usr/bin/env python3
"""
Generic experiment runner for Ultra-Sparse vs Backprop comparison.

Usage:
    python run_experiment.py --problem sine
    python run_experiment.py --problem synthetic --feature_dim 256 --true_features 3,17,42
    python run_experiment.py --problem multi_freq --trials 10

This script:
1. Loads the specified problem
2. Trains Ultra-Sparse (SA) and Dense (Backprop) models
3. Compares accuracy, efficiency, and feature selection
4. Outputs a summary report
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np

from problems import get_problem, Problem
from core.models import UltraSparseController, DenseController


def train_ultra_sparse_sa(
    problem: Problem,
    x_features: torch.Tensor,
    y_true: torch.Tensor,
    hidden_size: int = 8,
    inputs_per_neuron: int = 2,
    max_steps: int = 50000,
    verbose: bool = True
) -> tuple:
    """Train Ultra-Sparse controller with Simulated Annealing."""

    controller = UltraSparseController(
        input_size=problem.config.feature_dim,
        hidden_size=hidden_size,
        output_size=problem.config.output_dim,
        inputs_per_neuron=inputs_per_neuron,
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x_features)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        current_mse = torch.mean((pred - y_true) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    start_time = time.time()

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.05)

        with torch.no_grad():
            pred = mutant.forward(x_features)
            if pred.dim() > 1:
                pred = pred.squeeze(-1)
            mutant_mse = torch.mean((pred - y_true) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        if verbose and step % 10000 == 0:
            elapsed = time.time() - start_time
            stats = best.get_selection_stats(problem.config.true_features)
            print(f"  Step {step}: MSE={best_mse:.6f}, "
                  f"True={stats['true_connections']}/{stats['total_connections']}, "
                  f"Time={elapsed:.1f}s")

    return best, best_mse


def train_dense_backprop(
    problem: Problem,
    x_features: torch.Tensor,
    y_true: torch.Tensor,
    hidden_size: int = 8,
    epochs: int = 5000,
    lr: float = 0.01,
    verbose: bool = True
) -> tuple:
    """Train Dense controller with Backprop."""

    model = DenseController(
        input_size=problem.config.feature_dim,
        hidden_size=hidden_size,
        output_size=problem.config.output_dim,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')
    best_state = None

    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_features)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        loss = torch.mean((pred - y_true) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}: MSE={mse:.6f}, Time={elapsed:.1f}s")

    model.load_state_dict(best_state)
    return model, best_mse


def run_comparison(
    problem: Problem,
    n_samples: int = 200,
    n_trials: int = 5,
    hidden_size: int = 8,
    inputs_per_neuron: int = 2,
    sa_steps: int = 50000,
    bp_epochs: int = 5000,
    verbose: bool = True
) -> dict:
    """Run full comparison between Ultra-Sparse SA and Dense Backprop."""

    print("=" * 70)
    print(f"PROBLEM: {problem.config.name}")
    print(f"  {problem.config.description}")
    print(f"  Features: {problem.config.feature_dim}, True: {problem.config.true_features}")
    print("=" * 70)

    # Generate data
    x_raw, y_true = problem.generate_data(n_samples, seed=42)
    x_features = problem.expand_features(x_raw)

    if y_true.dim() > 1:
        y_true = y_true.squeeze(-1)

    # Normalize targets to [-1, 1] range for tanh output
    y_mean = y_true.mean()
    y_std = y_true.std()
    y_normalized = (y_true - y_mean) / (y_std + 1e-8)
    # Scale to fit in tanh range with some margin
    y_normalized = y_normalized * 0.8  # Keep within [-0.8, 0.8] to avoid saturation

    print(f"  Target range: [{y_true.min():.2f}, {y_true.max():.2f}] -> normalized")

    # Use normalized targets for training
    y_train = y_normalized

    results = {
        'problem': problem.config.name,
        'feature_dim': problem.config.feature_dim,
        'true_features': problem.config.true_features,
        'ultra_sparse': [],
        'backprop': [],
    }

    # Run trials
    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")

        # Set seeds
        torch.manual_seed(trial * 1000)
        np.random.seed(trial * 1000)

        # Ultra-Sparse SA
        print("\n[Ultra-Sparse SA]")
        best_us, mse_us = train_ultra_sparse_sa(
            problem, x_features, y_train,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            max_steps=sa_steps,
            verbose=verbose
        )
        stats_us = best_us.get_selection_stats(problem.config.true_features)
        selection_eval = problem.evaluate_selection(best_us.get_selected_indices())

        results['ultra_sparse'].append({
            'mse': mse_us,
            'params': best_us.num_parameters(),
            'selected': stats_us['selected_indices'],
            'true_ratio': stats_us['true_ratio'],
            'selection_factor': stats_us['selection_factor'],
            'precision': selection_eval['precision'],
            'recall': selection_eval['recall'],
            'f1': selection_eval['f1'],
        })
        print(f"  Final: MSE={mse_us:.6f}, Selected={stats_us['selected_indices']}")
        print(f"  True ratio: {stats_us['true_ratio']:.1%} ({stats_us['selection_factor']:.1f}x random)")

        # Dense Backprop
        print("\n[Dense Backprop]")
        torch.manual_seed(trial * 1000)
        best_bp, mse_bp = train_dense_backprop(
            problem, x_features, y_train,
            hidden_size=hidden_size,
            epochs=bp_epochs,
            verbose=verbose
        )

        results['backprop'].append({
            'mse': mse_bp,
            'params': best_bp.num_parameters(),
        })
        print(f"  Final: MSE={mse_bp:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    us_mses = [r['mse'] for r in results['ultra_sparse']]
    bp_mses = [r['mse'] for r in results['backprop']]

    us_params = results['ultra_sparse'][0]['params']
    bp_params = results['backprop'][0]['params']

    print(f"\n{'Method':<20} | {'Best MSE':<12} | {'Mean MSE':<12} | {'Params':<8}")
    print("-" * 60)
    print(f"{'Ultra-Sparse SA':<20} | {min(us_mses):<12.6f} | {np.mean(us_mses):<12.6f} | {us_params:<8}")
    print(f"{'Dense Backprop':<20} | {min(bp_mses):<12.6f} | {np.mean(bp_mses):<12.6f} | {bp_params:<8}")

    # Feature selection analysis
    print(f"\n--- Feature Selection (Ultra-Sparse) ---")
    print(f"True features: {problem.config.true_features}")

    all_selected = []
    for r in results['ultra_sparse']:
        all_selected.extend(r['selected'])

    from collections import Counter
    freq = Counter(all_selected)
    print(f"Most frequently selected:")
    for idx, count in freq.most_common(10):
        name = problem.get_feature_names()[idx] if idx < len(problem.get_feature_names()) else f"[{idx}]"
        is_true = "TRUE" if idx in problem.config.true_features else ""
        print(f"  {idx}: {count}/{n_trials} times - {name} {is_true}")

    avg_precision = np.mean([r['precision'] for r in results['ultra_sparse']])
    avg_recall = np.mean([r['recall'] for r in results['ultra_sparse']])
    avg_f1 = np.mean([r['f1'] for r in results['ultra_sparse']])
    print(f"\nSelection quality: Precision={avg_precision:.2f}, Recall={avg_recall:.2f}, F1={avg_f1:.2f}")

    # Efficiency comparison
    print(f"\n--- Efficiency ---")
    print(f"Parameter ratio: {bp_params/us_params:.0f}x fewer (Ultra-Sparse)")
    ops_us = hidden_size * inputs_per_neuron + hidden_size
    ops_bp = hidden_size * problem.config.feature_dim + hidden_size
    print(f"Operations ratio: {ops_bp/ops_us:.0f}x fewer (Ultra-Sparse)")

    results['summary'] = {
        'us_best_mse': min(us_mses),
        'us_mean_mse': np.mean(us_mses),
        'bp_best_mse': min(bp_mses),
        'bp_mean_mse': np.mean(bp_mses),
        'us_params': us_params,
        'bp_params': bp_params,
        'param_ratio': bp_params / us_params,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Ultra-Sparse vs Backprop comparison")
    parser.add_argument('--problem', type=str, default='sine',
                        choices=['sine', 'multi_freq', 'synthetic', 'xor', 'timeseries',
                                 'friedman1', 'friedman2', 'friedman3'],
                        help='Problem to solve')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='Feature dimension (for synthetic problems)')
    parser.add_argument('--true_features', type=str, default='3,17,42',
                        help='Comma-separated true feature indices (for synthetic)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of training samples')
    parser.add_argument('--hidden', type=int, default=8,
                        help='Hidden layer size')
    parser.add_argument('--k', type=int, default=2,
                        help='Inputs per neuron (Ultra-Sparse)')
    parser.add_argument('--sa_steps', type=int, default=50000,
                        help='SA training steps')
    parser.add_argument('--bp_epochs', type=int, default=5000,
                        help='Backprop training epochs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    # Parse true features
    true_features = [int(x) for x in args.true_features.split(',')]

    # Get problem
    if args.problem == 'synthetic':
        problem = get_problem(args.problem,
                              feature_dim=args.feature_dim,
                              true_features=true_features)
    elif args.problem == 'xor':
        problem = get_problem(args.problem,
                              feature_dim=args.feature_dim,
                              true_features=true_features[:2])
    elif args.problem == 'timeseries':
        problem = get_problem(args.problem,
                              max_lag=args.feature_dim,
                              true_lags=[f+1 for f in true_features])
    elif args.problem.startswith('friedman'):
        # Friedman problems: use feature_dim to set noise features
        n_noise = args.feature_dim - 5 if args.problem == 'friedman1' else args.feature_dim - 4
        problem = get_problem(args.problem, n_noise_features=max(0, n_noise))
    else:
        problem = get_problem(args.problem)

    # Run comparison
    results = run_comparison(
        problem,
        n_samples=args.samples,
        n_trials=args.trials,
        hidden_size=args.hidden,
        inputs_per_neuron=args.k,
        sa_steps=args.sa_steps,
        bp_epochs=args.bp_epochs,
        verbose=not args.quiet
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON
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

        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
