"""
Experiment 15: Friedman1 Benchmark - SA vs Backprop (Same Architecture)

Problem: Friedman #1 - 100 features, only 5 true (classic ML benchmark)
    y = 10*sin(π*x₁*x₂) + 20*(x₃ - 0.5)² + 10*x₄ + 5*x₅ + noise

Question: With identical 49-param architecture, does SA beat backprop?

Key Findings - SA WINS 3.7x ON SAME ARCHITECTURE:
- Ultra-Sparse SA: MSE=0.12, finds ALL 5 true features (100% recall)
- Sparse Backprop: MSE=0.45, finds ~1.4/5 (random chance)
- Dense Backprop: MSE=0.09, uses all 100 features (17x more params)

Conclusion: SA's evolvable indices enable feature discovery that backprop cannot achieve.
Backprop is stuck with whatever random indices it starts with.

References:
- Results: results/friedman1_comparison/
- Log: docs/experiments_log.md (Experiment 15)
- Related: experiments/highdim_scaling.py (1000 features version)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
from collections import Counter

from problems import Friedman1
from core.models import UltraSparseController, DenseController


class SparseBackpropController(nn.Module):
    """Sparse architecture trained with backprop (random fixed indices)."""

    def __init__(self, input_size=100, hidden_size=8, inputs_per_neuron=4):
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

        # Gather selected inputs
        selected = torch.stack([x[:, self.input_indices[h]] for h in range(self.hidden_size)], dim=1)

        # Forward pass
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))

        return output.squeeze()

    def get_selected_indices(self):
        return sorted(set(self.input_indices.flatten().tolist()))

    def get_true_count(self, true_features):
        all_idx = self.input_indices.flatten().tolist()
        return sum(1 for i in all_idx if i in true_features)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train_ultra_sparse_sa(x_features, y_train, true_features, max_steps=50000, verbose=True):
    """Train Ultra-Sparse with Simulated Annealing."""
    controller = UltraSparseController(
        input_size=x_features.shape[1],
        hidden_size=8,
        inputs_per_neuron=4
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(x_features).squeeze()
        current_mse = torch.mean((pred - y_train) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.000001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.05)

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

        if verbose and step % 10000 == 0:
            stats = best.get_selection_stats(true_features)
            print(f"    Step {step}: MSE={best_mse:.4f}, True={stats['true_connections']}/32")

    return best, best_mse


def train_sparse_backprop(x_features, y_train, true_features, epochs=5000, verbose=True):
    """Train Sparse Backprop with Adam."""
    model = SparseBackpropController(
        input_size=x_features.shape[1],
        hidden_size=8,
        inputs_per_neuron=4
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_features)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_mse:
            best_mse = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 1000 == 0:
            true_count = model.get_true_count(true_features)
            print(f"    Epoch {epoch}: MSE={loss.item():.4f}, True={true_count}/32 (fixed)")

    model.load_state_dict(best_state)
    return model, best_mse


def train_dense_backprop(x_features, y_train, epochs=5000, verbose=True):
    """Train Dense Backprop with Adam."""
    model = DenseController(
        input_size=x_features.shape[1],
        hidden_size=8
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_mse = float('inf')
    best_state = None

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

        if verbose and epoch % 1000 == 0:
            print(f"    Epoch {epoch}: MSE={loss.item():.4f}")

    model.load_state_dict(best_state)
    return model, best_mse


def run_experiment(n_trials=5, n_samples=500, sa_steps=50000, bp_epochs=5000):
    """Run the full comparison experiment."""

    print("=" * 70)
    print("EXPERIMENT: Friedman1 - Ultra-Sparse SA vs Sparse Backprop vs Dense")
    print("=" * 70)

    # Setup problem
    problem = Friedman1(n_noise_features=95)  # 100 total features
    true_features = problem.config.true_features  # [0, 1, 2, 3, 4]

    print(f"Problem: {problem.config.description}")
    print(f"Features: {problem.config.feature_dim} ({len(true_features)} true)")
    print(f"True features: {true_features}")
    print("=" * 70)

    # Generate data
    x_raw, y_true = problem.generate_data(n_samples, seed=42)
    x_features = problem.expand_features(x_raw)

    # Normalize targets for tanh output
    y_mean, y_std = y_true.mean(), y_true.std()
    y_train = (y_true - y_mean) / (y_std + 1e-8) * 0.8
    print(f"Target range: [{y_true.min():.2f}, {y_true.max():.2f}] -> normalized")

    # Results storage
    results = {
        'ultra_sparse_sa': [],
        'sparse_backprop': [],
        'dense_backprop': [],
    }

    for trial in range(n_trials):
        print(f"\n{'='*70}")
        print(f"TRIAL {trial + 1}/{n_trials}")
        print("=" * 70)

        # Set seeds
        torch.manual_seed(trial * 1000)
        np.random.seed(trial * 1000)

        # 1. Ultra-Sparse SA
        print("\n[Ultra-Sparse SA]")
        best_us, mse_us = train_ultra_sparse_sa(
            x_features, y_train, true_features,
            max_steps=sa_steps, verbose=True
        )
        stats_us = best_us.get_selection_stats(true_features)
        results['ultra_sparse_sa'].append({
            'mse': mse_us,
            'params': best_us.num_parameters(),
            'selected': stats_us['selected_indices'],
            'true_ratio': stats_us['true_ratio'],
            'true_connections': stats_us['true_connections'],
        })
        print(f"  Final: MSE={mse_us:.4f}, True={stats_us['true_connections']}/32 ({stats_us['true_ratio']:.0%})")

        # 2. Sparse Backprop (same seed = same random indices to start)
        print("\n[Sparse Backprop]")
        torch.manual_seed(trial * 1000)
        np.random.seed(trial * 1000)
        best_sp, mse_sp = train_sparse_backprop(
            x_features, y_train, true_features,
            epochs=bp_epochs, verbose=True
        )
        true_count_sp = best_sp.get_true_count(true_features)
        results['sparse_backprop'].append({
            'mse': mse_sp,
            'params': best_sp.num_parameters(),
            'selected': best_sp.get_selected_indices(),
            'true_count': true_count_sp,
        })
        print(f"  Final: MSE={mse_sp:.4f}, True={true_count_sp}/32 (random fixed)")

        # 3. Dense Backprop
        print("\n[Dense Backprop]")
        torch.manual_seed(trial * 1000)
        best_bp, mse_bp = train_dense_backprop(
            x_features, y_train,
            epochs=bp_epochs, verbose=True
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

    us_mses = [r['mse'] for r in results['ultra_sparse_sa']]
    sp_mses = [r['mse'] for r in results['sparse_backprop']]
    bp_mses = [r['mse'] for r in results['dense_backprop']]

    us_params = results['ultra_sparse_sa'][0]['params']
    sp_params = results['sparse_backprop'][0]['params']
    bp_params = results['dense_backprop'][0]['params']

    print(f"\n{'Method':<20} | {'Best MSE':<10} | {'Mean MSE':<10} | {'Params':<8} | {'Feature Selection'}")
    print("-" * 85)
    print(f"{'Ultra-Sparse SA':<20} | {min(us_mses):<10.4f} | {np.mean(us_mses):<10.4f} | {us_params:<8} | YES (evolves)")
    print(f"{'Sparse Backprop':<20} | {min(sp_mses):<10.4f} | {np.mean(sp_mses):<10.4f} | {sp_params:<8} | NO (random fixed)")
    print(f"{'Dense Backprop':<20} | {min(bp_mses):<10.4f} | {np.mean(bp_mses):<10.4f} | {bp_params:<8} | N/A (uses all)")

    # Feature selection analysis
    print(f"\n--- Feature Selection Analysis ---")
    print(f"True features: {true_features}")

    all_selected = []
    for r in results['ultra_sparse_sa']:
        all_selected.extend(r['selected'])
    freq = Counter(all_selected)

    print(f"\nUltra-Sparse SA - Most frequently selected:")
    for idx, count in freq.most_common(10):
        is_true = "TRUE" if idx in true_features else ""
        print(f"  Feature {idx}: {count}/{n_trials} trials {is_true}")

    # Calculate recall
    true_found = sum(1 for f in true_features if freq.get(f, 0) >= n_trials // 2 + 1)
    print(f"\nTrue features found (majority of trials): {true_found}/{len(true_features)}")

    avg_true_sp = np.mean([r['true_count'] for r in results['sparse_backprop']])
    expected_random = 32 * len(true_features) / problem.config.feature_dim
    print(f"\nSparse Backprop: avg {avg_true_sp:.1f}/32 true (expected random: {expected_random:.1f})")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"\nSame architecture (49 params), different training:")
    print(f"  Ultra-Sparse SA:  MSE={np.mean(us_mses):.4f} - DISCOVERS features")
    print(f"  Sparse Backprop:  MSE={np.mean(sp_mses):.4f} - STUCK with random indices")

    if np.mean(sp_mses) > np.mean(us_mses):
        ratio = np.mean(sp_mses) / np.mean(us_mses)
        print(f"\n  SA is {ratio:.1f}x BETTER than Backprop with same architecture!")
    else:
        ratio = np.mean(us_mses) / np.mean(sp_mses)
        print(f"\n  Backprop is {ratio:.1f}x better (but can't discover features)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "friedman1_comparison"
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
    run_experiment(n_trials=5, n_samples=500, sa_steps=50000, bp_epochs=5000)
