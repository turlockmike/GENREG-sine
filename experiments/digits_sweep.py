"""
Experiment: Digits Parameter Sweep - Finding what GENREG needs for 10-class

Goal: Find the minimum GENREG config that achieves >90% on digits (vs Dense's 97%)

Current failure: H=32, K=8 → 64.7% accuracy (618 params)
Target: >90% accuracy with <2000 params (vs Dense's 8970)

Parameters to sweep:
- Hidden size: 16, 32, 64, 128
- K (inputs per neuron): 4, 8, 16, 32
- SA steps: 30000, 50000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List

from core.models import UltraSparseController


def load_and_prep_digits():
    """Load and preprocess digits dataset."""
    data = load_digits()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def train_genreg_config(args: Tuple) -> Dict:
    """Train a single GENREG configuration."""
    hidden_size, k, sa_steps, trial, X_train_np, X_test_np, y_train_np, y_test_np = args

    # Convert back to torch
    X_train = torch.from_numpy(X_train_np).float()
    X_test = torch.from_numpy(X_test_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    y_test = torch.from_numpy(y_test_np).long()

    input_size = 64  # digits has 64 features
    num_classes = 10

    # Set seeds
    torch.manual_seed(trial * 1000)
    np.random.seed(trial * 1000)

    # One-hot encode targets for training
    y_onehot = torch.zeros(len(y_train), num_classes)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8  # Scale to [-0.8, 0.8] for tanh

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        inputs_per_neuron=k
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(X_train)
        current_mse = torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / sa_steps)

    start = time.time()
    for step in range(sa_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_mse = torch.mean((pred - y_onehot) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

    train_time = time.time() - start

    # Evaluate
    with torch.no_grad():
        preds = best.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    params = best.num_parameters()

    return {
        'hidden': hidden_size,
        'k': k,
        'sa_steps': sa_steps,
        'trial': trial,
        'accuracy': accuracy,
        'params': params,
        'train_time': train_time,
    }


def run_sweep():
    """Run parameter sweep in parallel."""

    print("=" * 70)
    print("DIGITS PARAMETER SWEEP")
    print("=" * 70)
    print("\nGoal: Find GENREG config that achieves >90% accuracy")
    print("Dense baseline: 97% accuracy, 8970 params")
    print()

    # Load data once
    X_train, X_test, y_train, y_test = load_and_prep_digits()
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    print(f"Data: {len(X_train)} train, {len(X_test)} test, 64 features, 10 classes")

    # Configurations to test
    hidden_sizes = [16, 32, 64, 128]
    k_values = [4, 8, 16, 32]
    sa_steps_list = [30000]
    n_trials = 2  # Quick test

    # Build tasks
    tasks = []
    for hidden in hidden_sizes:
        for k in k_values:
            # Skip invalid configs (k > input_size)
            if k > 64:
                continue
            for sa_steps in sa_steps_list:
                for trial in range(n_trials):
                    tasks.append((
                        hidden, k, sa_steps, trial,
                        X_train_np, X_test_np, y_train_np, y_test_np
                    ))

    print(f"\nTesting {len(tasks)} configurations...")
    print(f"Configs: H in {hidden_sizes}, K in {k_values}")

    n_workers = min(cpu_count() - 1, 12)
    print(f"Using {n_workers} parallel workers")
    print("=" * 70)

    start = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(train_genreg_config, tasks)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.1f}s")

    # Aggregate results by config
    configs = {}
    for r in results:
        key = (r['hidden'], r['k'])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Hidden':<8} | {'K':<4} | {'Params':<8} | {'Accuracy':<10} | {'vs Dense':<10}")
    print("-" * 55)

    best_config = None
    best_accuracy = 0

    for (hidden, k), trials in sorted(configs.items()):
        mean_acc = np.mean([t['accuracy'] for t in trials])
        params = trials[0]['params']

        vs_dense = f"{8970/params:.0f}x fewer"

        marker = ""
        if mean_acc > 0.9:
            marker = " ✅"
        elif mean_acc > 0.8:
            marker = " 🔶"

        print(f"{hidden:<8} | {k:<4} | {params:<8} | {mean_acc:<10.1%} | {vs_dense:<10}{marker}")

        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_config = (hidden, k, params)

    print("-" * 55)
    print(f"Dense baseline: 97.0% accuracy, 8970 params")

    if best_config:
        h, k, p = best_config
        print(f"\nBest GENREG: H={h}, K={k} → {best_accuracy:.1%} accuracy, {p} params")
        print(f"Efficiency: {8970/p:.0f}x fewer params, {97-best_accuracy*100:.1f}% accuracy gap")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "digits_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_sweep()
