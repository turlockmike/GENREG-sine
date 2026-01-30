"""
Inference Benchmark: Ultra-Sparse vs Standard SA vs Backprop

Train all three models, save them, and compare:
- MSE (accuracy)
- Inference speed
- Memory footprint
- Energy (activation + weight)
- Saturation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
import pickle

from legacy.sine_controller import expand_input

DEVICE = torch.device("cpu")
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "inference_benchmark"


# ============================================================================
# Model Definitions
# ============================================================================

class UltraSparseModel(nn.Module):
    """Ultra-sparse: each hidden neuron connects to only K inputs."""

    def __init__(self, hidden_size=8, inputs_per_neuron=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.inputs_per_neuron = inputs_per_neuron

        # Input indices (which inputs each hidden neuron uses)
        self.register_buffer(
            'input_indices',
            torch.zeros(hidden_size, inputs_per_neuron, dtype=torch.long)
        )
        for h in range(hidden_size):
            self.input_indices[h] = torch.randperm(256)[:inputs_per_neuron]

        # Weights
        self.w1 = nn.Parameter(torch.randn(hidden_size, inputs_per_neuron) * np.sqrt(2.0 / inputs_per_neuron))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))

        self.last_hidden = None

    def forward(self, x):
        # x is already expanded (batch, 256)
        batch_size = x.shape[0]

        # Gather selected inputs
        selected = torch.zeros(batch_size, self.hidden_size, self.inputs_per_neuron, device=x.device)
        for h in range(self.hidden_size):
            selected[:, h, :] = x[:, self.input_indices[h]]

        # Hidden layer
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        self.last_hidden = hidden.detach()

        # Output
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))
        return output.squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_selection_stats(self):
        all_idx = self.input_indices.flatten().tolist()
        true_count = sum(1 for i in all_idx if i < 16)
        return {
            'true_ratio': true_count / len(all_idx),
            'true_inputs': sorted(set(i for i in all_idx if i < 16)),
        }


class StandardModel(nn.Module):
    """Standard dense model: 256 -> 8 -> 1."""

    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size

        self.w1 = nn.Parameter(torch.randn(hidden_size, 256) * np.sqrt(2.0 / 256))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))

        self.last_hidden = None

    def forward(self, x):
        # x is already expanded (batch, 256)
        hidden = torch.tanh(nn.functional.linear(x, self.w1, self.b1))
        self.last_hidden = hidden.detach()
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))
        return output.squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Training Functions
# ============================================================================

def train_backprop(model, x_expanded, y_true, epochs=1000, lr=0.01):
    """Train with Adam optimizer."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_expanded)
        loss = nn.functional.mse_loss(pred, y_true)
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0:
            print(f"    Epoch {epoch}: MSE = {loss.item():.6f}")

    return loss.item()


def train_sa_sparse(model, x_expanded, y_true, max_steps=20000):
    """Train ultra-sparse model with SA (mutating indices + weights)."""

    def get_mse():
        with torch.no_grad():
            pred = model(x_expanded)
            return nn.functional.mse_loss(pred, y_true).item()

    def mutate_model(m, weight_rate=0.1, weight_scale=0.1, index_swap_rate=0.05):
        with torch.no_grad():
            for param in [m.w1, m.b1, m.w2, m.b2]:
                mask = torch.rand_like(param) < weight_rate
                noise = torch.randn_like(param) * weight_scale
                param.data += mask.float() * noise

            for h in range(m.hidden_size):
                if np.random.random() < index_swap_rate:
                    pos = np.random.randint(m.inputs_per_neuron)
                    current = set(m.input_indices[h].tolist())
                    available = [i for i in range(256) if i not in current]
                    if available:
                        m.input_indices[h, pos] = np.random.choice(available)

    def clone_model(m):
        new = UltraSparseModel(m.hidden_size, m.inputs_per_neuron)
        new.load_state_dict(m.state_dict())
        new.input_indices = m.input_indices.clone()
        return new

    current_mse = get_mse()
    best_model = clone_model(model)
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        # Clone and mutate
        mutant = clone_model(model)
        mutate_model(mutant)

        # Evaluate
        with torch.no_grad():
            pred = mutant(x_expanded)
            mutant_mse = nn.functional.mse_loss(pred, y_true).item()

        # Accept?
        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            model.load_state_dict(mutant.state_dict())
            model.input_indices = mutant.input_indices.clone()
            current_mse = mutant_mse

            if current_mse < best_mse:
                best_model = clone_model(model)
                best_mse = current_mse

        if step % 5000 == 0:
            print(f"    Step {step}: MSE = {best_mse:.6f}")

    model.load_state_dict(best_model.state_dict())
    model.input_indices = best_model.input_indices.clone()
    return best_mse


def train_sa_standard(model, x_expanded, y_true, max_steps=20000):
    """Train standard model with SA (weights only)."""

    def get_mse():
        with torch.no_grad():
            pred = model(x_expanded)
            return nn.functional.mse_loss(pred, y_true).item()

    def mutate_model(m, rate=0.1, scale=0.1):
        with torch.no_grad():
            for param in m.parameters():
                mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * scale
                param.data += mask.float() * noise

    def clone_model(m):
        new = StandardModel(m.hidden_size)
        new.load_state_dict(m.state_dict())
        return new

    current_mse = get_mse()
    best_model = clone_model(model)
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = clone_model(model)
        mutate_model(mutant)

        with torch.no_grad():
            pred = mutant(x_expanded)
            mutant_mse = nn.functional.mse_loss(pred, y_true).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            model.load_state_dict(mutant.state_dict())
            current_mse = mutant_mse

            if current_mse < best_mse:
                best_model = clone_model(model)
                best_mse = current_mse

        if step % 5000 == 0:
            print(f"    Step {step}: MSE = {best_mse:.6f}")

    model.load_state_dict(best_model.state_dict())
    return best_mse


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_inference(model, x_expanded, n_iters=1000, warmup=100):
    """Benchmark inference speed."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_expanded)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = model(x_expanded)
            times.append(time.perf_counter() - start)

    return {
        'mean_us': np.mean(times) * 1e6,
        'std_us': np.std(times) * 1e6,
        'min_us': np.min(times) * 1e6,
        'max_us': np.max(times) * 1e6,
    }


def get_model_stats(model, x_expanded, y_true):
    """Get comprehensive model statistics."""
    model.eval()

    with torch.no_grad():
        pred = model(x_expanded)
        mse = nn.functional.mse_loss(pred, y_true).item()

    # Saturation
    if model.last_hidden is not None:
        saturation = (model.last_hidden.abs() > 0.95).float().mean().item()
        activation_energy = model.last_hidden.abs().mean().item()
    else:
        saturation = 0.0
        activation_energy = 0.0

    # Weight energy
    weight_sum = sum(p.abs().sum().item() for p in model.parameters())
    weight_count = sum(p.numel() for p in model.parameters())
    weight_energy = weight_sum / weight_count

    # Memory
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        'mse': mse,
        'saturation': saturation,
        'activation_energy': activation_energy,
        'weight_energy': weight_energy,
        'total_energy': activation_energy + weight_energy,
        'parameters': model.num_parameters(),
        'memory_bytes': param_bytes,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment():
    print("=" * 70)
    print("INFERENCE BENCHMARK: Ultra-Sparse vs Standard SA vs Backprop")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare data
    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100)
    x_expanded = expand_input(x_raw.unsqueeze(-1))  # (100, 256)
    y_true = torch.sin(x_raw)

    results = {}

    # ========================================================================
    # Train Ultra-Sparse (SA)
    # ========================================================================
    print("\n[1] Training Ultra-Sparse (SA)...")
    torch.manual_seed(42)
    np.random.seed(42)

    ultra_sparse = UltraSparseModel(hidden_size=8, inputs_per_neuron=2)
    mse_sparse = train_sa_sparse(ultra_sparse, x_expanded, y_true, max_steps=20000)

    # Save model
    torch.save({
        'state_dict': ultra_sparse.state_dict(),
        'input_indices': ultra_sparse.input_indices,
        'config': {'hidden_size': 8, 'inputs_per_neuron': 2},
    }, OUTPUT_DIR / "ultra_sparse.pt")

    stats_sparse = get_model_stats(ultra_sparse, x_expanded, y_true)
    selection = ultra_sparse.get_selection_stats()
    print(f"  Final MSE: {stats_sparse['mse']:.6f}")
    print(f"  True inputs selected: {selection['true_inputs']}")

    # ========================================================================
    # Train Standard (SA)
    # ========================================================================
    print("\n[2] Training Standard (SA)...")
    torch.manual_seed(42)
    np.random.seed(42)

    standard_sa = StandardModel(hidden_size=8)
    mse_sa = train_sa_standard(standard_sa, x_expanded, y_true, max_steps=20000)

    torch.save({
        'state_dict': standard_sa.state_dict(),
        'config': {'hidden_size': 8},
    }, OUTPUT_DIR / "standard_sa.pt")

    stats_sa = get_model_stats(standard_sa, x_expanded, y_true)
    print(f"  Final MSE: {stats_sa['mse']:.6f}")

    # ========================================================================
    # Train Standard (Backprop)
    # ========================================================================
    print("\n[3] Training Standard (Backprop)...")
    torch.manual_seed(42)
    np.random.seed(42)

    standard_bp = StandardModel(hidden_size=8)
    mse_bp = train_backprop(standard_bp, x_expanded, y_true, epochs=1000, lr=0.01)

    torch.save({
        'state_dict': standard_bp.state_dict(),
        'config': {'hidden_size': 8},
    }, OUTPUT_DIR / "standard_backprop.pt")

    stats_bp = get_model_stats(standard_bp, x_expanded, y_true)
    print(f"  Final MSE: {stats_bp['mse']:.6f}")

    # ========================================================================
    # Inference Benchmarks
    # ========================================================================
    print("\n" + "=" * 70)
    print("INFERENCE BENCHMARKS")
    print("=" * 70)

    batch_sizes = [1, 10, 100, 1000]

    for batch_size in batch_sizes:
        x_batch = expand_input(torch.randn(batch_size, 1) * 2 * np.pi)

        bench_sparse = benchmark_inference(ultra_sparse, x_batch)
        bench_sa = benchmark_inference(standard_sa, x_batch)
        bench_bp = benchmark_inference(standard_bp, x_batch)

        print(f"\nBatch size {batch_size}:")
        print(f"  Ultra-Sparse: {bench_sparse['mean_us']:>8.2f} μs")
        print(f"  Standard SA:  {bench_sa['mean_us']:>8.2f} μs")
        print(f"  Backprop:     {bench_bp['mean_us']:>8.2f} μs")

        if batch_size == 100:
            results['inference_100'] = {
                'ultra_sparse': bench_sparse,
                'standard_sa': bench_sa,
                'backprop': bench_bp,
            }

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"""
| Metric              | Ultra-Sparse   | Standard SA    | Backprop       |
|---------------------|----------------|----------------|----------------|
| MSE                 | {stats_sparse['mse']:<14.6f} | {stats_sa['mse']:<14.6f} | {stats_bp['mse']:<14.6f} |
| Parameters          | {stats_sparse['parameters']:<14} | {stats_sa['parameters']:<14} | {stats_bp['parameters']:<14} |
| Memory (bytes)      | {stats_sparse['memory_bytes']:<14} | {stats_sa['memory_bytes']:<14} | {stats_bp['memory_bytes']:<14} |
| Saturation          | {stats_sparse['saturation']*100:<13.1f}% | {stats_sa['saturation']*100:<13.1f}% | {stats_bp['saturation']*100:<13.1f}% |
| Activation Energy   | {stats_sparse['activation_energy']:<14.4f} | {stats_sa['activation_energy']:<14.4f} | {stats_bp['activation_energy']:<14.4f} |
| Weight Energy       | {stats_sparse['weight_energy']:<14.4f} | {stats_sa['weight_energy']:<14.4f} | {stats_bp['weight_energy']:<14.4f} |
| Total Energy        | {stats_sparse['total_energy']:<14.4f} | {stats_sa['total_energy']:<14.4f} | {stats_bp['total_energy']:<14.4f} |
""")

    # Winner analysis
    print("WINNERS:")
    print(f"  Best MSE: {'Ultra-Sparse' if stats_sparse['mse'] < min(stats_sa['mse'], stats_bp['mse']) else ('Backprop' if stats_bp['mse'] < stats_sa['mse'] else 'Standard SA')}")
    print(f"  Fewest params: Ultra-Sparse ({stats_sparse['parameters']} vs {stats_sa['parameters']})")
    print(f"  Lowest energy: {'Ultra-Sparse' if stats_sparse['total_energy'] < min(stats_sa['total_energy'], stats_bp['total_energy']) else ('Backprop' if stats_bp['total_energy'] < stats_sa['total_energy'] else 'Standard SA')}")

    # Ratios
    print(f"\nUltra-Sparse vs Backprop:")
    print(f"  MSE ratio: {stats_bp['mse'] / stats_sparse['mse']:.1f}x {'better' if stats_sparse['mse'] < stats_bp['mse'] else 'worse'}")
    print(f"  Parameter ratio: {stats_bp['parameters'] / stats_sparse['parameters']:.0f}x fewer")
    print(f"  Memory ratio: {stats_bp['memory_bytes'] / stats_sparse['memory_bytes']:.0f}x smaller")

    # Save results
    results['models'] = {
        'ultra_sparse': {**stats_sparse, 'selection': selection},
        'standard_sa': stats_sa,
        'backprop': stats_bp,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("  - ultra_sparse.pt")
    print("  - standard_sa.pt")
    print("  - standard_backprop.pt")

    return results


if __name__ == "__main__":
    run_experiment()
