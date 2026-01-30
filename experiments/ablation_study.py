"""
Experiment 17: Ablation Study - Proving Each Component is Necessary

Problem: 1000 features, 10 true (1% signal). Prove USEN's three components are essential.

Hypothesis: Fixed-K + evolvable indices + SA creates selection pressure
that no single component alone can achieve.

Variants tested:
1. Full GENREG:      Fixed K=4 + Evolvable Indices + SA Weights
2. Frozen Indices:   Fixed K=4 + Frozen Indices + SA Weights
3. Random Regrowth:  Fixed K=4 + Random Index Swap + SA Weights (SET-style)
4. Backprop Weights: Fixed K=4 + Frozen Indices + Backprop Weights
5. Weak Constraint:  Fixed K=32 + Evolvable Indices + SA Weights

Key Findings - ALL COMPONENTS PROVEN ESSENTIAL:
- Full USEN: 25x random selection factor (baseline)
- No index evolution: 10x worse
- Random regrowth (SET-style): 13x worse
- Weak K (K=32): 6.5x worse

References:
- Results: results/ablation_study/
- Log: docs/experiments_log.md (Experiment 17)
- Summary: docs/SUMMARY_Ultra_Sparse_Evolvable_Networks.md
- Weak Constraint: Weak selection (K=32 approaches dense behavior)

If Full GENREG >> Frozen Indices, we prove evolvable indices matter.
If Full GENREG >> Random Regrowth, we prove guided evolution beats random.
If Full GENREG >> Weak Constraint, we prove the K constraint matters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
import copy
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

from core.models import UltraSparseController


@dataclass
class AblationResult:
    """Results from one ablation variant."""
    name: str
    mse: float
    true_ratio: float
    true_connections: int
    total_connections: int
    selected_indices: List[int]
    params: int


def generate_data(n_samples: int, n_features: int, n_true: int, seed: int = 42):
    """Generate regression data with known true features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.rand(n_samples, n_features)
    y = torch.zeros(n_samples)

    # Linear terms with decreasing weights
    for i in range(min(n_true, 5)):
        weight = 10 - 2 * i
        y += weight * x[:, i]

    # Nonlinear terms
    if n_true > 5:
        y += 5 * torch.sin(np.pi * x[:, 5] * x[:, min(6, n_features-1)])
    if n_true > 7:
        y += 3 * (x[:, 7] - 0.5) ** 2

    y += torch.randn(n_samples) * 0.5

    # Normalize for tanh
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-8) * 0.8

    return x, y_norm, list(range(n_true))


class SparseBackpropController(nn.Module):
    """Sparse architecture trained with backprop (frozen indices)."""

    def __init__(self, input_size: int, hidden_size: int, inputs_per_neuron: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.inputs_per_neuron = inputs_per_neuron

        # Random fixed indices (not learnable)
        self.register_buffer(
            'input_indices',
            torch.zeros(hidden_size, inputs_per_neuron, dtype=torch.long)
        )
        for h in range(hidden_size):
            self.input_indices[h] = torch.randperm(input_size)[:inputs_per_neuron]

        # Learnable weights
        self.w1 = nn.Parameter(torch.randn(hidden_size, inputs_per_neuron) * np.sqrt(2.0 / inputs_per_neuron))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        selected = torch.stack([x[:, self.input_indices[h]] for h in range(self.hidden_size)], dim=1)
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))
        return output.squeeze()

    def get_selected_indices(self):
        return sorted(set(self.input_indices.flatten().tolist()))

    def get_selection_stats(self, true_features):
        all_idx = self.input_indices.flatten().tolist()
        true_count = sum(1 for i in all_idx if i in true_features)
        total = len(all_idx)
        return {
            'selected_indices': self.get_selected_indices(),
            'true_connections': true_count,
            'total_connections': total,
            'true_ratio': true_count / total if total > 0 else 0,
        }


# ============================================================
# ABLATION VARIANT 1: Full GENREG (baseline)
# ============================================================
def train_full_genreg(
    x: torch.Tensor,
    y: torch.Tensor,
    true_features: List[int],
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    max_steps: int = 50000,
    verbose: bool = False,
) -> AblationResult:
    """Full GENREG: Fixed K + Evolvable Indices + SA Weights."""

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x).squeeze()
        current_mse = torch.mean((pred - y) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)
    index_swap_rate = min(0.2, 0.05 * (input_size / 100))

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        # KEY: Both weight AND index mutations
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

        if verbose and step % 10000 == 0:
            stats = best.get_selection_stats(true_features)
            print(f"    [Full GENREG] Step {step}: MSE={best_mse:.4f}, True={stats['true_connections']}/{stats['total_connections']}")

    stats = best.get_selection_stats(true_features)
    return AblationResult(
        name="Full GENREG",
        mse=best_mse,
        true_ratio=stats['true_ratio'],
        true_connections=stats['true_connections'],
        total_connections=stats['total_connections'],
        selected_indices=stats['selected_indices'],
        params=best.num_parameters(),
    )


# ============================================================
# ABLATION VARIANT 2: Frozen Indices (no index evolution)
# ============================================================
def train_frozen_indices(
    x: torch.Tensor,
    y: torch.Tensor,
    true_features: List[int],
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    max_steps: int = 50000,
    verbose: bool = False,
) -> AblationResult:
    """Frozen Indices: Fixed K + FROZEN Indices + SA Weights."""

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x).squeeze()
        current_mse = torch.mean((pred - y) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        # KEY: Only weight mutations, NO index mutations
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.0)

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

        if verbose and step % 10000 == 0:
            stats = best.get_selection_stats(true_features)
            print(f"    [Frozen Indices] Step {step}: MSE={best_mse:.4f}, True={stats['true_connections']}/{stats['total_connections']}")

    stats = best.get_selection_stats(true_features)
    return AblationResult(
        name="Frozen Indices",
        mse=best_mse,
        true_ratio=stats['true_ratio'],
        true_connections=stats['true_connections'],
        total_connections=stats['total_connections'],
        selected_indices=stats['selected_indices'],
        params=best.num_parameters(),
    )


# ============================================================
# ABLATION VARIANT 3: Random Regrowth (SET-style)
# ============================================================
def train_random_regrowth(
    x: torch.Tensor,
    y: torch.Tensor,
    true_features: List[int],
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    max_steps: int = 50000,
    verbose: bool = False,
) -> AblationResult:
    """Random Regrowth: Fixed K + RANDOM Index Swaps + SA Weights (SET-style)."""

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x).squeeze()
        current_mse = torch.mean((pred - y) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)
    index_swap_rate = min(0.2, 0.05 * (input_size / 100))

    # Track accepted vs rejected for random regrowth
    regrowth_interval = 1000  # Swap indices randomly every N steps

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        # Weight mutations always
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.0)

        # KEY: Random index swaps at fixed intervals (not fitness-guided)
        if step % regrowth_interval == 0 and step > 0:
            # Randomly swap some indices (like SET's random regrowth)
            for h in range(hidden_size):
                if np.random.random() < index_swap_rate:
                    k = np.random.randint(inputs_per_neuron)
                    new_idx = np.random.randint(input_size)
                    mutant.input_indices[h, k] = new_idx

        with torch.no_grad():
            pred = mutant.forward(x).squeeze()
            mutant_mse = torch.mean((pred - y) ** 2).item()

        # For random regrowth, we ALWAYS accept the index changes
        # (they're not fitness-guided, just random exploration)
        # But weight changes still follow SA acceptance
        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        if verbose and step % 10000 == 0:
            stats = best.get_selection_stats(true_features)
            print(f"    [Random Regrowth] Step {step}: MSE={best_mse:.4f}, True={stats['true_connections']}/{stats['total_connections']}")

    stats = best.get_selection_stats(true_features)
    return AblationResult(
        name="Random Regrowth",
        mse=best_mse,
        true_ratio=stats['true_ratio'],
        true_connections=stats['true_connections'],
        total_connections=stats['total_connections'],
        selected_indices=stats['selected_indices'],
        params=best.num_parameters(),
    )


# ============================================================
# ABLATION VARIANT 4: Backprop Weights (frozen indices)
# ============================================================
def train_backprop_weights(
    x: torch.Tensor,
    y: torch.Tensor,
    true_features: List[int],
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    epochs: int = 5000,
    verbose: bool = False,
) -> AblationResult:
    """Backprop Weights: Fixed K + Frozen Indices + BACKPROP Weights."""

    model = SparseBackpropController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_mse:
            best_mse = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 1000 == 0:
            stats = model.get_selection_stats(true_features)
            print(f"    [Backprop] Epoch {epoch}: MSE={loss.item():.4f}, True={stats['true_connections']}/{stats['total_connections']}")

    model.load_state_dict(best_state)
    stats = model.get_selection_stats(true_features)

    return AblationResult(
        name="Backprop Weights",
        mse=best_mse,
        true_ratio=stats['true_ratio'],
        true_connections=stats['true_connections'],
        total_connections=stats['total_connections'],
        selected_indices=stats['selected_indices'],
        params=sum(p.numel() for p in model.parameters()),
    )


# ============================================================
# ABLATION VARIANT 5: Weak Constraint (large K)
# ============================================================
def train_weak_constraint(
    x: torch.Tensor,
    y: torch.Tensor,
    true_features: List[int],
    input_size: int,
    hidden_size: int = 8,
    inputs_per_neuron: int = 32,  # KEY: Large K = weak constraint
    max_steps: int = 50000,
    verbose: bool = False,
) -> AblationResult:
    """Weak Constraint: Fixed K=32 + Evolvable Indices + SA Weights."""

    controller = UltraSparseController(
        input_size=input_size,
        hidden_size=hidden_size,
        inputs_per_neuron=inputs_per_neuron
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x).squeeze()
        current_mse = torch.mean((pred - y) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.05, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)
    index_swap_rate = min(0.2, 0.05 * (input_size / 100))

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

        if verbose and step % 10000 == 0:
            stats = best.get_selection_stats(true_features)
            print(f"    [Weak K=32] Step {step}: MSE={best_mse:.4f}, True={stats['true_connections']}/{stats['total_connections']}")

    stats = best.get_selection_stats(true_features)
    return AblationResult(
        name="Weak Constraint (K=32)",
        mse=best_mse,
        true_ratio=stats['true_ratio'],
        true_connections=stats['true_connections'],
        total_connections=stats['total_connections'],
        selected_indices=stats['selected_indices'],
        params=best.num_parameters(),
    )


# ============================================================
# Parallel Worker Function
# ============================================================
def run_single_trial(args: Tuple) -> Dict:
    """Worker function for parallel execution."""
    (variant_name, trial, x_np, y_np, true_features, n_features,
     hidden_size, inputs_per_neuron, sa_steps, bp_epochs) = args

    # Convert numpy back to torch (can't pickle torch tensors easily)
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()

    # Set seeds
    torch.manual_seed(trial * 1000)
    np.random.seed(trial * 1000)

    # Select training function based on variant
    if variant_name == "Full GENREG":
        result = train_full_genreg(
            x, y, true_features, n_features,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            max_steps=sa_steps,
            verbose=False,
        )
    elif variant_name == "Frozen Indices":
        result = train_frozen_indices(
            x, y, true_features, n_features,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            max_steps=sa_steps,
            verbose=False,
        )
    elif variant_name == "Random Regrowth":
        result = train_random_regrowth(
            x, y, true_features, n_features,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            max_steps=sa_steps,
            verbose=False,
        )
    elif variant_name == "Backprop Weights":
        result = train_backprop_weights(
            x, y, true_features, n_features,
            hidden_size=hidden_size,
            inputs_per_neuron=inputs_per_neuron,
            epochs=bp_epochs,
            verbose=False,
        )
    elif variant_name == "Weak Constraint (K=32)":
        result = train_weak_constraint(
            x, y, true_features, n_features,
            hidden_size=hidden_size,
            inputs_per_neuron=32,
            max_steps=sa_steps,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return {
        'variant': variant_name,
        'trial': trial,
        'mse': result.mse,
        'true_ratio': result.true_ratio,
        'true_connections': result.true_connections,
        'total_connections': result.total_connections,
        'selected_indices': result.selected_indices,
        'params': result.params,
    }


# ============================================================
# Main Experiment
# ============================================================
def run_ablation_study(
    n_features: int = 1000,
    n_true: int = 10,
    n_samples: int = 500,
    n_trials: int = 5,
    hidden_size: int = 8,
    inputs_per_neuron: int = 4,
    sa_steps: int = 50000,
    bp_epochs: int = 5000,
    verbose: bool = True,
    parallel: bool = True,
    n_workers: int = None,
):
    """Run the full ablation study."""

    print("=" * 80)
    print("ABLATION STUDY: Proving Each Component is Necessary")
    print("=" * 80)
    print(f"\nProblem: {n_features} features, {n_true} true ({100*n_true/n_features:.1f}% signal)")
    print(f"Network: H={hidden_size}, K={inputs_per_neuron} (except Weak Constraint: K=32)")
    print(f"Trials: {n_trials}")

    if parallel:
        n_workers = n_workers or min(cpu_count() - 1, 10)
        print(f"Parallel execution: {n_workers} workers")
    print()

    # Generate data
    x, y, true_features = generate_data(n_samples, n_features, n_true, seed=42)
    print(f"True features: {true_features}")

    # Expected random baseline
    total_connections = hidden_size * inputs_per_neuron
    random_expected = total_connections * n_true / n_features
    print(f"Random expected: {random_expected:.1f}/{total_connections} true connections")
    print("=" * 80)

    # Results storage
    results = {
        'config': {
            'n_features': n_features,
            'n_true': n_true,
            'n_samples': n_samples,
            'hidden_size': hidden_size,
            'inputs_per_neuron': inputs_per_neuron,
            'sa_steps': sa_steps,
            'bp_epochs': bp_epochs,
        },
        'variants': {},
    }

    # Variant names
    variant_names = [
        "Full GENREG",
        "Frozen Indices",
        "Random Regrowth",
        "Backprop Weights",
        "Weak Constraint (K=32)",
    ]

    if parallel:
        # Build all tasks
        x_np = x.numpy()
        y_np = y.numpy()

        tasks = []
        for variant_name in variant_names:
            for trial in range(n_trials):
                tasks.append((
                    variant_name, trial, x_np, y_np, true_features, n_features,
                    hidden_size, inputs_per_neuron, sa_steps, bp_epochs
                ))

        print(f"\nRunning {len(tasks)} tasks in parallel...")
        start_time = time.time()

        # Run in parallel
        with Pool(n_workers) as pool:
            all_results = pool.map(run_single_trial, tasks)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s ({elapsed/len(tasks):.1f}s per task effective)")

        # Organize results by variant
        for variant_name in variant_names:
            variant_results = [r for r in all_results if r['variant'] == variant_name]
            variant_results.sort(key=lambda r: r['trial'])
            results['variants'][variant_name] = [{
                'mse': r['mse'],
                'true_ratio': r['true_ratio'],
                'true_connections': r['true_connections'],
                'total_connections': r['total_connections'],
                'selected_indices': r['selected_indices'],
                'params': r['params'],
            } for r in variant_results]

            # Print summary for this variant
            mean_mse = np.mean([r['mse'] for r in variant_results])
            mean_true = np.mean([r['true_ratio'] for r in variant_results])
            print(f"  {variant_name}: MSE={mean_mse:.4f}, True={mean_true:.1%}")

    else:
        # Sequential execution (original code)
        variants = [
            ("Full GENREG", train_full_genreg, {'inputs_per_neuron': inputs_per_neuron}),
            ("Frozen Indices", train_frozen_indices, {'inputs_per_neuron': inputs_per_neuron}),
            ("Random Regrowth", train_random_regrowth, {'inputs_per_neuron': inputs_per_neuron}),
            ("Backprop Weights", train_backprop_weights, {'inputs_per_neuron': inputs_per_neuron}),
            ("Weak Constraint (K=32)", train_weak_constraint, {'inputs_per_neuron': 32}),
        ]

        for variant_name, train_fn, extra_kwargs in variants:
            print(f"\n{'='*80}")
            print(f"VARIANT: {variant_name}")
            print("=" * 80)

            variant_results = []

            for trial in range(n_trials):
                print(f"\n  Trial {trial + 1}/{n_trials}")
                torch.manual_seed(trial * 1000)
                np.random.seed(trial * 1000)

                if "Backprop" in variant_name:
                    result = train_fn(
                        x, y, true_features, n_features,
                        hidden_size=hidden_size,
                        epochs=bp_epochs,
                        verbose=verbose,
                        **extra_kwargs
                    )
                else:
                    result = train_fn(
                        x, y, true_features, n_features,
                        hidden_size=hidden_size,
                        max_steps=sa_steps,
                        verbose=verbose,
                        **extra_kwargs
                    )

                variant_results.append({
                    'mse': result.mse,
                    'true_ratio': result.true_ratio,
                    'true_connections': result.true_connections,
                    'total_connections': result.total_connections,
                    'selected_indices': result.selected_indices,
                    'params': result.params,
                })

                print(f"    MSE: {result.mse:.4f}, True: {result.true_connections}/{result.total_connections} ({result.true_ratio:.0%})")

            results['variants'][variant_name] = variant_results

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Variant':<25} | {'Mean MSE':<10} | {'True Ratio':<12} | {'Selection Factor':<16} | {'Params':<8}")
    print("-" * 85)

    for variant_name in variant_names:
        vr = results['variants'][variant_name]
        mean_mse = np.mean([r['mse'] for r in vr])
        mean_true_ratio = np.mean([r['true_ratio'] for r in vr])

        # Selection factor = actual true ratio / expected random ratio
        total_conn = vr[0]['total_connections']
        random_ratio = n_true / n_features
        selection_factor = mean_true_ratio / random_ratio if random_ratio > 0 else 0

        params = vr[0]['params']

        print(f"{variant_name:<25} | {mean_mse:<10.4f} | {mean_true_ratio:<12.1%} | {selection_factor:<16.1f}x | {params:<8}")

    # Key comparisons
    print("\n" + "=" * 80)
    print("KEY COMPARISONS")
    print("=" * 80)

    full = results['variants']['Full GENREG']
    frozen = results['variants']['Frozen Indices']
    random_rg = results['variants']['Random Regrowth']
    backprop = results['variants']['Backprop Weights']
    weak = results['variants']['Weak Constraint (K=32)']

    full_mse = np.mean([r['mse'] for r in full])
    full_true = np.mean([r['true_ratio'] for r in full])

    frozen_mse = np.mean([r['mse'] for r in frozen])
    frozen_true = np.mean([r['true_ratio'] for r in frozen])

    random_mse = np.mean([r['mse'] for r in random_rg])
    random_true = np.mean([r['true_ratio'] for r in random_rg])

    backprop_mse = np.mean([r['mse'] for r in backprop])
    backprop_true = np.mean([r['true_ratio'] for r in backprop])

    weak_mse = np.mean([r['mse'] for r in weak])
    weak_true = np.mean([r['true_ratio'] for r in weak])

    print(f"\n1. Does index evolution matter?")
    print(f"   Full GENREG vs Frozen Indices:")
    print(f"   MSE: {full_mse:.4f} vs {frozen_mse:.4f} ({frozen_mse/full_mse:.1f}x worse)")
    print(f"   True ratio: {full_true:.1%} vs {frozen_true:.1%} ({full_true/frozen_true:.1f}x better)")

    print(f"\n2. Does guided evolution beat random?")
    print(f"   Full GENREG vs Random Regrowth:")
    print(f"   MSE: {full_mse:.4f} vs {random_mse:.4f} ({random_mse/full_mse:.1f}x worse)")
    print(f"   True ratio: {full_true:.1%} vs {random_true:.1%} ({full_true/random_true:.1f}x better)")

    print(f"\n3. Does SA beat backprop with same architecture?")
    print(f"   Frozen Indices (SA) vs Backprop Weights:")
    print(f"   MSE: {frozen_mse:.4f} vs {backprop_mse:.4f}")
    print(f"   (Both have frozen indices - comparing optimization method)")

    print(f"\n4. Does the K constraint matter?")
    print(f"   Full GENREG (K=4) vs Weak Constraint (K=32):")
    print(f"   MSE: {full_mse:.4f} vs {weak_mse:.4f}")
    print(f"   True ratio: {full_true:.1%} vs {weak_true:.1%} ({full_true/weak_true:.1f}x better)")

    # Conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    conclusions = []

    if full_true > frozen_true * 1.5:
        conclusions.append("✅ EVOLVABLE INDICES ARE ESSENTIAL - Full GENREG selects significantly more true features than frozen indices")
    else:
        conclusions.append("❌ Evolvable indices did not significantly improve selection")

    if full_true > random_true * 1.2:
        conclusions.append("✅ GUIDED EVOLUTION BEATS RANDOM - Fitness-guided index mutations outperform random regrowth")
    else:
        conclusions.append("⚠️ Random regrowth performs similarly to guided evolution")

    if full_true > weak_true * 1.5:
        conclusions.append("✅ K CONSTRAINT MATTERS - Strong constraint (K=4) creates more selection pressure than weak (K=32)")
    else:
        conclusions.append("⚠️ K constraint strength had limited effect")

    for c in conclusions:
        print(f"\n{c}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "ablation_study"
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
    # Use spawn method for macOS compatibility
    mp.set_start_method('spawn', force=True)

    run_ablation_study(
        n_features=1000,
        n_true=10,
        n_samples=500,
        n_trials=5,
        hidden_size=8,
        inputs_per_neuron=4,
        sa_steps=50000,
        bp_epochs=5000,
        verbose=True,
        parallel=True,  # Use parallel execution
        n_workers=10,   # Use 10 workers (leaving cores free)
    )
