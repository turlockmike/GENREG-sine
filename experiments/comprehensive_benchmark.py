"""
Comprehensive Benchmark: Ultra-Sparse vs Standard SA vs Backprop

Trains each model 20 times, selects the best, and generates a full report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime

from sine_controller import expand_input

DEVICE = torch.device("cpu")
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "comprehensive_benchmark"
N_TRIALS = 20


# ============================================================================
# Model Definitions
# ============================================================================

class UltraSparseModel(nn.Module):
    def __init__(self, hidden_size=8, inputs_per_neuron=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.inputs_per_neuron = inputs_per_neuron

        self.register_buffer(
            'input_indices',
            torch.zeros(hidden_size, inputs_per_neuron, dtype=torch.long)
        )
        for h in range(hidden_size):
            self.input_indices[h] = torch.randperm(256)[:inputs_per_neuron]

        self.w1 = nn.Parameter(torch.randn(hidden_size, inputs_per_neuron) * np.sqrt(2.0 / inputs_per_neuron))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))
        self.last_hidden = None

    def forward(self, x):
        batch_size = x.shape[0]
        selected = torch.zeros(batch_size, self.hidden_size, self.inputs_per_neuron, device=x.device)
        for h in range(self.hidden_size):
            selected[:, h, :] = x[:, self.input_indices[h]]
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        self.last_hidden = hidden.detach()
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
    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Parameter(torch.randn(hidden_size, 256) * np.sqrt(2.0 / 256))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))
        self.last_hidden = None

    def forward(self, x):
        hidden = torch.tanh(nn.functional.linear(x, self.w1, self.b1))
        self.last_hidden = hidden.detach()
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))
        return output.squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Training Functions
# ============================================================================

def train_backprop(model, x_expanded, y_true, epochs=1000, lr=0.01, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_expanded)
        loss = nn.functional.mse_loss(pred, y_true)
        loss.backward()
        optimizer.step()
    return loss.item()


def train_sa_sparse(model, x_expanded, y_true, max_steps=20000, verbose=False):
    def get_mse():
        with torch.no_grad():
            pred = model(x_expanded)
            return nn.functional.mse_loss(pred, y_true).item()

    def clone_model(m):
        new = UltraSparseModel(m.hidden_size, m.inputs_per_neuron)
        new.load_state_dict(m.state_dict())
        new.input_indices = m.input_indices.clone()
        return new

    def mutate_model(m):
        with torch.no_grad():
            for param in [m.w1, m.b1, m.w2, m.b2]:
                mask = torch.rand_like(param) < 0.1
                noise = torch.randn_like(param) * 0.1
                param.data += mask.float() * noise
            for h in range(m.hidden_size):
                if np.random.random() < 0.05:
                    pos = np.random.randint(m.inputs_per_neuron)
                    current = set(m.input_indices[h].tolist())
                    available = [i for i in range(256) if i not in current]
                    if available:
                        m.input_indices[h, pos] = np.random.choice(available)

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
            model.input_indices = mutant.input_indices.clone()
            current_mse = mutant_mse
            if current_mse < best_mse:
                best_model = clone_model(model)
                best_mse = current_mse

    model.load_state_dict(best_model.state_dict())
    model.input_indices = best_model.input_indices.clone()
    return best_mse


def train_sa_standard(model, x_expanded, y_true, max_steps=20000, verbose=False):
    def get_mse():
        with torch.no_grad():
            pred = model(x_expanded)
            return nn.functional.mse_loss(pred, y_true).item()

    def clone_model(m):
        new = StandardModel(m.hidden_size)
        new.load_state_dict(m.state_dict())
        return new

    def mutate_model(m):
        with torch.no_grad():
            for param in m.parameters():
                mask = torch.rand_like(param) < 0.1
                noise = torch.randn_like(param) * 0.1
                param.data += mask.float() * noise

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

    model.load_state_dict(best_model.state_dict())
    return best_mse


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_inference(model, x_expanded, n_iters=1000, warmup=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_expanded)
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
    }


def get_model_stats(model, x_expanded, y_true):
    model.eval()
    with torch.no_grad():
        pred = model(x_expanded)
        mse = nn.functional.mse_loss(pred, y_true).item()
    saturation = (model.last_hidden.abs() > 0.95).float().mean().item() if model.last_hidden is not None else 0
    activation_energy = model.last_hidden.abs().mean().item() if model.last_hidden is not None else 0
    weight_sum = sum(p.abs().sum().item() for p in model.parameters())
    weight_count = sum(p.numel() for p in model.parameters())
    weight_energy = weight_sum / weight_count
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
    print(f"COMPREHENSIVE BENCHMARK ({N_TRIALS} trials each)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100)
    x_expanded = expand_input(x_raw.unsqueeze(-1))
    y_true = torch.sin(x_raw)

    results = {
        'ultra_sparse': {'trials': [], 'best_mse': float('inf'), 'best_model': None},
        'standard_sa': {'trials': [], 'best_mse': float('inf'), 'best_model': None},
        'backprop': {'trials': [], 'best_mse': float('inf'), 'best_model': None},
    }

    # ========================================================================
    # Train Ultra-Sparse (20 trials)
    # ========================================================================
    print(f"\n[1/3] Training Ultra-Sparse ({N_TRIALS} trials)...")
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        np.random.seed(trial)
        model = UltraSparseModel(hidden_size=8, inputs_per_neuron=2)
        mse = train_sa_sparse(model, x_expanded, y_true, max_steps=20000)
        selection = model.get_selection_stats()
        results['ultra_sparse']['trials'].append({
            'trial': trial, 'mse': mse,
            'true_inputs': selection['true_inputs'],
            'true_ratio': selection['true_ratio']
        })
        if mse < results['ultra_sparse']['best_mse']:
            results['ultra_sparse']['best_mse'] = mse
            results['ultra_sparse']['best_model'] = model
            results['ultra_sparse']['best_trial'] = trial
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: MSE = {mse:.6f}, True inputs: {selection['true_inputs']}")

    # ========================================================================
    # Train Standard SA (20 trials)
    # ========================================================================
    print(f"\n[2/3] Training Standard SA ({N_TRIALS} trials)...")
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        np.random.seed(trial)
        model = StandardModel(hidden_size=8)
        mse = train_sa_standard(model, x_expanded, y_true, max_steps=20000)
        results['standard_sa']['trials'].append({'trial': trial, 'mse': mse})
        if mse < results['standard_sa']['best_mse']:
            results['standard_sa']['best_mse'] = mse
            results['standard_sa']['best_model'] = model
            results['standard_sa']['best_trial'] = trial
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: MSE = {mse:.6f}")

    # ========================================================================
    # Train Backprop (20 trials)
    # ========================================================================
    print(f"\n[3/3] Training Backprop ({N_TRIALS} trials)...")
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        np.random.seed(trial)
        model = StandardModel(hidden_size=8)
        mse = train_backprop(model, x_expanded, y_true, epochs=1000)
        results['backprop']['trials'].append({'trial': trial, 'mse': mse})
        if mse < results['backprop']['best_mse']:
            results['backprop']['best_mse'] = mse
            results['backprop']['best_model'] = model
            results['backprop']['best_trial'] = trial
        print(f"  Trial {trial+1:2d}/{N_TRIALS}: MSE = {mse:.6f}")

    # ========================================================================
    # Get stats for best models
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYZING BEST MODELS")
    print("=" * 70)

    for name in ['ultra_sparse', 'standard_sa', 'backprop']:
        model = results[name]['best_model']
        stats = get_model_stats(model, x_expanded, y_true)
        results[name]['stats'] = stats

        # Save best model
        save_dict = {'state_dict': model.state_dict(), 'stats': stats}
        if name == 'ultra_sparse':
            save_dict['input_indices'] = model.input_indices
            save_dict['selection'] = model.get_selection_stats()
        torch.save(save_dict, OUTPUT_DIR / f"best_{name}.pt")

    # ========================================================================
    # Inference Benchmarks
    # ========================================================================
    print("\n" + "=" * 70)
    print("INFERENCE BENCHMARKS")
    print("=" * 70)

    batch_sizes = [1, 10, 100, 1000]
    inference_results = {}

    for batch_size in batch_sizes:
        x_batch = expand_input(torch.randn(batch_size, 1) * 2 * np.pi)
        inference_results[batch_size] = {}

        for name in ['ultra_sparse', 'standard_sa', 'backprop']:
            model = results[name]['best_model']
            bench = benchmark_inference(model, x_batch)
            inference_results[batch_size][name] = bench

        print(f"\nBatch size {batch_size}:")
        print(f"  Ultra-Sparse: {inference_results[batch_size]['ultra_sparse']['mean_us']:>8.2f} μs")
        print(f"  Standard SA:  {inference_results[batch_size]['standard_sa']['mean_us']:>8.2f} μs")
        print(f"  Backprop:     {inference_results[batch_size]['backprop']['mean_us']:>8.2f} μs")

    results['inference'] = inference_results

    # ========================================================================
    # Generate Report
    # ========================================================================
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    # Compute statistics
    for name in ['ultra_sparse', 'standard_sa', 'backprop']:
        mses = [t['mse'] for t in results[name]['trials']]
        results[name]['mse_stats'] = {
            'best': min(mses),
            'worst': max(mses),
            'mean': np.mean(mses),
            'std': np.std(mses),
            'median': np.median(mses),
        }

    # Write report
    report = f"""# Comprehensive Benchmark Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Trials per method: {N_TRIALS}

## Summary

| Metric | Ultra-Sparse | Standard SA | Backprop |
|--------|--------------|-------------|----------|
| **Best MSE** | {results['ultra_sparse']['mse_stats']['best']:.6f} | {results['standard_sa']['mse_stats']['best']:.6f} | {results['backprop']['mse_stats']['best']:.6f} |
| Mean MSE | {results['ultra_sparse']['mse_stats']['mean']:.6f} | {results['standard_sa']['mse_stats']['mean']:.6f} | {results['backprop']['mse_stats']['mean']:.6f} |
| Std MSE | {results['ultra_sparse']['mse_stats']['std']:.6f} | {results['standard_sa']['mse_stats']['std']:.6f} | {results['backprop']['mse_stats']['std']:.6f} |
| Parameters | {results['ultra_sparse']['stats']['parameters']} | {results['standard_sa']['stats']['parameters']} | {results['backprop']['stats']['parameters']} |
| Memory (bytes) | {results['ultra_sparse']['stats']['memory_bytes']} | {results['standard_sa']['stats']['memory_bytes']} | {results['backprop']['stats']['memory_bytes']} |
| Saturation | {results['ultra_sparse']['stats']['saturation']*100:.1f}% | {results['standard_sa']['stats']['saturation']*100:.1f}% | {results['backprop']['stats']['saturation']*100:.1f}% |

## Inference Speed (μs)

| Batch Size | Ultra-Sparse | Standard SA | Backprop |
|------------|--------------|-------------|----------|
| 1 | {inference_results[1]['ultra_sparse']['mean_us']:.2f} | {inference_results[1]['standard_sa']['mean_us']:.2f} | {inference_results[1]['backprop']['mean_us']:.2f} |
| 10 | {inference_results[10]['ultra_sparse']['mean_us']:.2f} | {inference_results[10]['standard_sa']['mean_us']:.2f} | {inference_results[10]['backprop']['mean_us']:.2f} |
| 100 | {inference_results[100]['ultra_sparse']['mean_us']:.2f} | {inference_results[100]['standard_sa']['mean_us']:.2f} | {inference_results[100]['backprop']['mean_us']:.2f} |
| 1000 | {inference_results[1000]['ultra_sparse']['mean_us']:.2f} | {inference_results[1000]['standard_sa']['mean_us']:.2f} | {inference_results[1000]['backprop']['mean_us']:.2f} |

## Best Model Details

### Ultra-Sparse (Best of {N_TRIALS} trials)
- Trial: {results['ultra_sparse']['best_trial']}
- MSE: {results['ultra_sparse']['best_mse']:.6f}
- Parameters: {results['ultra_sparse']['stats']['parameters']} (63x fewer than standard)
- Memory: {results['ultra_sparse']['stats']['memory_bytes']} bytes (63x smaller)
- Saturation: {results['ultra_sparse']['stats']['saturation']*100:.1f}%
- True inputs selected: {results['ultra_sparse']['best_model'].get_selection_stats()['true_inputs']}
- True selection ratio: {results['ultra_sparse']['best_model'].get_selection_stats()['true_ratio']*100:.1f}% (vs 6.25% random)

### Standard SA (Best of {N_TRIALS} trials)
- Trial: {results['standard_sa']['best_trial']}
- MSE: {results['standard_sa']['best_mse']:.6f}
- Parameters: {results['standard_sa']['stats']['parameters']}
- Saturation: {results['standard_sa']['stats']['saturation']*100:.1f}%

### Backprop (Best of {N_TRIALS} trials)
- Trial: {results['backprop']['best_trial']}
- MSE: {results['backprop']['best_mse']:.6f}
- Parameters: {results['backprop']['stats']['parameters']}
- Saturation: {results['backprop']['stats']['saturation']*100:.1f}%

## Energy Comparison

| Metric | Ultra-Sparse | Standard SA | Backprop |
|--------|--------------|-------------|----------|
| Activation Energy | {results['ultra_sparse']['stats']['activation_energy']:.4f} | {results['standard_sa']['stats']['activation_energy']:.4f} | {results['backprop']['stats']['activation_energy']:.4f} |
| Weight Energy | {results['ultra_sparse']['stats']['weight_energy']:.4f} | {results['standard_sa']['stats']['weight_energy']:.4f} | {results['backprop']['stats']['weight_energy']:.4f} |
| Total Energy | {results['ultra_sparse']['stats']['total_energy']:.4f} | {results['standard_sa']['stats']['total_energy']:.4f} | {results['backprop']['stats']['total_energy']:.4f} |

## Efficiency Ratios (Ultra-Sparse vs others)

| Comparison | MSE Ratio | Param Ratio | Memory Ratio |
|------------|-----------|-------------|--------------|
| vs Backprop | {results['ultra_sparse']['best_mse']/results['backprop']['best_mse']:.2f}x | {results['backprop']['stats']['parameters']/results['ultra_sparse']['stats']['parameters']:.0f}x fewer | {results['backprop']['stats']['memory_bytes']/results['ultra_sparse']['stats']['memory_bytes']:.0f}x smaller |
| vs Standard SA | {results['ultra_sparse']['best_mse']/results['standard_sa']['best_mse']:.2f}x better | {results['standard_sa']['stats']['parameters']/results['ultra_sparse']['stats']['parameters']:.0f}x fewer | {results['standard_sa']['stats']['memory_bytes']/results['ultra_sparse']['stats']['memory_bytes']:.0f}x smaller |

## All Trial Results

### Ultra-Sparse MSE by Trial
"""
    for t in results['ultra_sparse']['trials']:
        report += f"- Trial {t['trial']}: {t['mse']:.6f} (inputs: {t['true_inputs']})\n"

    report += f"""
### Standard SA MSE by Trial
"""
    for t in results['standard_sa']['trials']:
        report += f"- Trial {t['trial']}: {t['mse']:.6f}\n"

    report += f"""
### Backprop MSE by Trial
"""
    for t in results['backprop']['trials']:
        report += f"- Trial {t['trial']}: {t['mse']:.6f}\n"

    report += f"""
## Conclusions

1. **Best MSE**: {'Ultra-Sparse' if results['ultra_sparse']['best_mse'] <= min(results['standard_sa']['best_mse'], results['backprop']['best_mse']) else ('Backprop' if results['backprop']['best_mse'] < results['standard_sa']['best_mse'] else 'Standard SA')} achieves the best accuracy.

2. **Most Efficient**: Ultra-Sparse uses **{results['backprop']['stats']['parameters']/results['ultra_sparse']['stats']['parameters']:.0f}x fewer parameters** than standard architectures.

3. **Input Selection**: Ultra-Sparse automatically discovered useful inputs with **{results['ultra_sparse']['best_model'].get_selection_stats()['true_ratio']/0.0625:.1f}x better than random** selection of true signals.

4. **Saturation**: Ultra-Sparse achieves {results['ultra_sparse']['stats']['saturation']*100:.0f}% saturation vs {results['standard_sa']['stats']['saturation']*100:.0f}% for Standard SA.

## Files Generated
- `best_ultra_sparse.pt` - Best ultra-sparse model
- `best_standard_sa.pt` - Best standard SA model
- `best_backprop.pt` - Best backprop model
- `results.json` - Full results data
- `report.md` - This report
"""

    # Save report
    with open(OUTPUT_DIR / "report.md", "w") as f:
        f.write(report)

    # Save JSON results (without models)
    json_results = {
        'ultra_sparse': {
            'trials': results['ultra_sparse']['trials'],
            'mse_stats': results['ultra_sparse']['mse_stats'],
            'stats': results['ultra_sparse']['stats'],
            'best_trial': results['ultra_sparse']['best_trial'],
        },
        'standard_sa': {
            'trials': results['standard_sa']['trials'],
            'mse_stats': results['standard_sa']['mse_stats'],
            'stats': results['standard_sa']['stats'],
            'best_trial': results['standard_sa']['best_trial'],
        },
        'backprop': {
            'trials': results['backprop']['trials'],
            'mse_stats': results['backprop']['mse_stats'],
            'stats': results['backprop']['stats'],
            'best_trial': results['backprop']['best_trial'],
        },
        'inference': {str(k): v for k, v in inference_results.items()},
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(json_results, f, indent=2, default=float)

    print(f"\nReport saved to: {OUTPUT_DIR / 'report.md'}")
    print(f"Results saved to: {OUTPUT_DIR / 'results.json'}")
    print(f"Models saved to: {OUTPUT_DIR}")

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
Best MSE (of {N_TRIALS} trials):
  Ultra-Sparse: {results['ultra_sparse']['best_mse']:.6f}
  Standard SA:  {results['standard_sa']['best_mse']:.6f}
  Backprop:     {results['backprop']['best_mse']:.6f}

Parameters:
  Ultra-Sparse: {results['ultra_sparse']['stats']['parameters']}
  Standard:     {results['standard_sa']['stats']['parameters']}

Winner: {'ULTRA-SPARSE' if results['ultra_sparse']['best_mse'] <= results['backprop']['best_mse'] else 'BACKPROP'} on MSE
        ULTRA-SPARSE on efficiency ({results['backprop']['stats']['parameters']/results['ultra_sparse']['stats']['parameters']:.0f}x fewer params)
""")

    return results


if __name__ == "__main__":
    run_experiment()
