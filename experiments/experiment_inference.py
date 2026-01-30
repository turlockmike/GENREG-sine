"""
Experiment: Inference Efficiency Comparison

Compare inference cost between:
1. Backprop network (low saturation) - standard float32 operations
2. Saturated network (from SA) - optimized with binary gates

Optimizations for saturated networks:
- Replace tanh() with sign() for saturated neurons
- Quantize weights to int8
- Skip activation computation entirely for always-saturated neurons

Metrics:
- Inference time (microseconds per batch)
- Memory footprint
- MSE accuracy (to verify optimization doesn't break things)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

import legacy.sine_config as sine_config as cfg
from legacy.sine_controller import SineController, expand_input
from sine_annealing import simulated_annealing

DEVICE = torch.device("cpu")  # CPU for fair timing comparison


class BackpropMLP(nn.Module):
    """Standard backprop-trained MLP."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        return out.squeeze(-1)


class OptimizedSaturatedMLP(nn.Module):
    """
    Optimized MLP for saturated networks.

    Optimizations:
    1. Use sign() instead of tanh() for hidden layer (binary output)
    2. Quantize weights to int8
    3. Use fused operations where possible
    """
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        # Store weights as int8 (quantized)
        self.w1_scale = w1.abs().max().item()
        self.w1 = nn.Parameter((w1 / self.w1_scale * 127).to(torch.int8), requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)

        self.w2_scale = w2.abs().max().item()
        self.w2 = nn.Parameter((w2 / self.w2_scale * 127).to(torch.int8), requires_grad=False)
        self.b2 = nn.Parameter(b2, requires_grad=False)

    def forward(self, x):
        # Dequantize and compute hidden layer
        w1_float = self.w1.float() * self.w1_scale / 127
        h_pre = torch.nn.functional.linear(x, w1_float, self.b1)

        # Binary activation: sign() instead of tanh()
        h = torch.sign(h_pre)

        # Output layer
        w2_float = self.w2.float() * self.w2_scale / 127
        out = torch.tanh(torch.nn.functional.linear(h, w2_float, self.b2))
        return out.squeeze(-1)


class BinaryGateMLP(nn.Module):
    """
    Fully optimized binary gate network.

    For 100% saturated networks, hidden neurons are just +1 or -1.
    We can precompute which neurons fire positive vs negative for speed.
    """
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = nn.Parameter(w1, requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.b2 = nn.Parameter(b2, requires_grad=False)

    def forward(self, x):
        # Hidden layer with sign activation (no tanh)
        h_pre = torch.nn.functional.linear(x, self.w1, self.b1)
        h = torch.sign(h_pre)  # Binary: -1 or +1

        # Output layer (keep tanh for bounded output)
        out = torch.tanh(torch.nn.functional.linear(h, self.w2, self.b2))
        return out.squeeze(-1)


class StandardMLP(nn.Module):
    """Standard MLP from SineController weights (for fair comparison)."""
    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = nn.Parameter(w1, requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.b2 = nn.Parameter(b2, requires_grad=False)

    def forward(self, x):
        h = torch.tanh(torch.nn.functional.linear(x, self.w1, self.b1))
        out = torch.tanh(torch.nn.functional.linear(h, self.w2, self.b2))
        return out.squeeze(-1)


def train_backprop_network(x_expanded, y_true, hidden_size=8):
    """Train standard backprop network."""
    model = BackpropMLP(x_expanded.shape[1], hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(x_expanded)
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.0001:
            break

    return model


def train_saturated_network(x_test, y_true, steps=15000):
    """Train network with simulated annealing (produces saturation)."""
    best, _ = simulated_annealing(
        x_test, y_true,
        max_steps=steps,
        t_initial=0.01,
        t_final=0.00001,
        verbose=False,
    )
    return best


def benchmark_inference(model, x_input, n_warmup=100, n_runs=1000):
    """Benchmark inference time."""
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x_input)

    # Timed runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x_input)
            end = time.perf_counter()
            times.append((end - start) * 1e6)  # Convert to microseconds

    return {
        'mean_us': np.mean(times),
        'std_us': np.std(times),
        'min_us': np.min(times),
        'max_us': np.max(times),
        'median_us': np.median(times),
    }


def get_model_size(model):
    """Get model memory footprint in bytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_size + buffer_size


def check_saturation(model, x_input):
    """Check saturation level of a model."""
    with torch.no_grad():
        if hasattr(model, 'fc1'):
            h = torch.tanh(model.fc1(x_input))
        else:
            h = torch.tanh(torch.nn.functional.linear(x_input, model.w1, model.b1))
        saturated = (h.abs() > 0.95).float().mean().item()
    return saturated


def run_experiment():
    """Run inference efficiency comparison."""
    print("=" * 70)
    print("EXPERIMENT: INFERENCE EFFICIENCY COMPARISON")
    print("=" * 70)
    print("Comparing backprop (low saturation) vs optimized saturated networks")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare data
    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    x_raw_2d = x_raw.unsqueeze(-1)
    x_expanded = expand_input(x_raw_2d)
    y_true = torch.sin(x_raw)

    # Batch sizes to test
    batch_sizes = [1, 10, 100, 1000]

    results = {'batch_tests': {}}

    # Train networks
    print("\n[Training Networks]")

    print("  Training backprop network...")
    bp_model = train_backprop_network(x_expanded, y_true)
    bp_mse = nn.MSELoss()(bp_model(x_expanded), y_true).item()
    bp_sat = check_saturation(bp_model, x_expanded)
    print(f"    MSE: {bp_mse:.6f}, Saturation: {bp_sat:.1%}")

    print("  Training saturated network (SA)...")
    sa_controller = train_saturated_network(x_raw, y_true)
    sa_pred = sa_controller.forward(x_raw, track=True)
    sa_mse = torch.mean((sa_pred - y_true) ** 2).item()
    sa_sat = sa_controller.get_k() / cfg.HIDDEN_SIZE
    print(f"    MSE: {sa_mse:.6f}, Saturation: {sa_sat:.1%}")

    # Create optimized versions of saturated network
    print("\n[Creating Optimized Networks]")

    # Standard version (same as SA but in nn.Module form)
    standard_model = StandardMLP(
        sa_controller.w1.clone(),
        sa_controller.b1.clone(),
        sa_controller.w2.clone(),
        sa_controller.b2.clone(),
    )

    # Binary gate version (sign instead of tanh)
    binary_model = BinaryGateMLP(
        sa_controller.w1.clone(),
        sa_controller.b1.clone(),
        sa_controller.w2.clone(),
        sa_controller.b2.clone(),
    )

    # Quantized version
    quantized_model = OptimizedSaturatedMLP(
        sa_controller.w1.clone(),
        sa_controller.b1.clone(),
        sa_controller.w2.clone(),
        sa_controller.b2.clone(),
    )

    # Verify accuracy of optimized versions
    print("\n[Verifying Optimized Networks]")

    with torch.no_grad():
        standard_pred = standard_model(x_expanded)
        standard_mse = torch.mean((standard_pred - y_true) ** 2).item()

        binary_pred = binary_model(x_expanded)
        binary_mse = torch.mean((binary_pred - y_true) ** 2).item()

        quantized_pred = quantized_model(x_expanded)
        quantized_mse = torch.mean((quantized_pred - y_true) ** 2).item()

    print(f"  Standard (tanh):    MSE={standard_mse:.6f}")
    print(f"  Binary (sign):      MSE={binary_mse:.6f}")
    print(f"  Quantized (int8):   MSE={quantized_mse:.6f}")

    # Model sizes
    print("\n[Model Sizes]")
    models = {
        'Backprop (float32)': bp_model,
        'SA Standard (float32)': standard_model,
        'SA Binary (float32)': binary_model,
        'SA Quantized (int8)': quantized_model,
    }

    for name, model in models.items():
        size = get_model_size(model)
        print(f"  {name}: {size:,} bytes")

    results['model_sizes'] = {name: get_model_size(model) for name, model in models.items()}

    # Benchmark inference at different batch sizes
    print("\n[Inference Benchmarks]")

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        print(f"  {'Model':<25} | {'Mean (μs)':<12} | {'Std (μs)':<10} | {'Speedup':<10}")
        print("  " + "-" * 65)

        # Create batch input
        x_batch = x_expanded[:batch_size] if batch_size <= 100 else x_expanded.repeat(batch_size // 100 + 1, 1)[:batch_size]

        batch_results = {}
        baseline_time = None

        for name, model in models.items():
            timing = benchmark_inference(model, x_batch)
            batch_results[name] = timing

            if baseline_time is None:
                baseline_time = timing['mean_us']
                speedup = "1.00x (baseline)"
            else:
                speedup = f"{baseline_time / timing['mean_us']:.2f}x"

            print(f"  {name:<25} | {timing['mean_us']:<12.2f} | {timing['std_us']:<10.2f} | {speedup:<10}")

        results['batch_tests'][batch_size] = batch_results

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nAccuracy Comparison:")
    print(f"  Backprop:           MSE={bp_mse:.6f} (saturation={bp_sat:.1%})")
    print(f"  SA Standard:        MSE={standard_mse:.6f} (saturation={sa_sat:.1%})")
    print(f"  SA Binary (sign):   MSE={binary_mse:.6f}")
    print(f"  SA Quantized:       MSE={quantized_mse:.6f}")

    print("\nMemory Savings:")
    bp_size = get_model_size(bp_model)
    quant_size = get_model_size(quantized_model)
    print(f"  Backprop: {bp_size:,} bytes")
    print(f"  Quantized: {quant_size:,} bytes")
    print(f"  Savings: {(1 - quant_size/bp_size)*100:.1f}%")

    print("\nSpeed Comparison (batch=100):")
    batch_100 = results['batch_tests'].get(100, results['batch_tests'].get(10))
    bp_time = batch_100['Backprop (float32)']['mean_us']
    binary_time = batch_100['SA Binary (float32)']['mean_us']
    print(f"  Backprop: {bp_time:.2f} μs")
    print(f"  Binary:   {binary_time:.2f} μs")
    print(f"  Speedup:  {bp_time/binary_time:.2f}x")

    results['summary'] = {
        'backprop_mse': bp_mse,
        'backprop_saturation': bp_sat,
        'sa_mse': sa_mse,
        'sa_saturation': sa_sat,
        'binary_mse': binary_mse,
        'quantized_mse': quantized_mse,
        'memory_savings_pct': (1 - quant_size/bp_size) * 100,
        'speedup_binary': bp_time / binary_time,
    }

    # Save results
    output_dir = Path("results/inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()
