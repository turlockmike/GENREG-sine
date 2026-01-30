"""
Experiment 14: Inference Engine Comparison

Problem: PyTorch has significant overhead for small sparse networks
Question: Can alternative backends (NumPy, Numba JIT) improve inference speed?

Key Findings - Numba JIT is 60x faster than PyTorch:
- PyTorch Ultra-Sparse: 26.5 μs (overhead dominates)
- NumPy Ultra-Sparse: 2.1 μs (12x faster)
- Numba Ultra-Sparse: 0.44 μs (60x faster than PyTorch)

- PyTorch Dense: 3.1 μs
- Numba Dense: 0.58-0.63 μs (5-6x faster)

Key insight: Ultra-Sparse with Numba is faster than Dense with Numba (1.3-1.4x)
because sparse models have fewer computations despite indexing overhead.

Requires comprehensive_benchmark.py models to exist in results/.

References:
- Results: results/inference_engines/ (if exists)
- Log: docs/experiments_log.md (Experiment 14)
- Depends on: experiments/comprehensive_benchmark.py (for model files)
- Models: models/README.md (Numba inference examples)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time

from legacy.sine_controller import expand_input

# Try to import optional accelerators
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - skipping JIT tests")


def load_ultra_sparse_model(path):
    """Load saved Ultra-Sparse model."""
    data = torch.load(path, weights_only=False)
    # Extract state_dict if nested
    if 'state_dict' in data:
        state = data['state_dict']
        return {
            'input_indices': state['input_indices'],
            'w1': state['w1'],
            'b1': state['b1'],
            'w2': state['w2'],
            'b2': state['b2'],
        }
    return data


def load_standard_model(path):
    """Load saved Standard SA or Backprop model (dense 256->8->1)."""
    data = torch.load(path, weights_only=False)
    if 'state_dict' in data:
        state = data['state_dict']
        return {
            'w1': state['w1'],
            'b1': state['b1'],
            'w2': state['w2'],
            'b2': state['b2'],
        }
    return data


def create_test_data(n_samples=1000):
    """Create test input data."""
    x = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).astype(np.float32)
    return x


def expand_input_numpy(x):
    """Expand single input to 256 features (numpy version)."""
    # x shape: (batch,) or scalar
    x = np.atleast_1d(x).astype(np.float32)
    batch_size = len(x)

    expanded = np.zeros((batch_size, 256), dtype=np.float32)
    for i in range(8):
        freq = i + 1
        expanded[:, i] = np.sin(freq * x)
        expanded[:, i + 8] = np.cos(freq * x)

    # Noise dimensions (16-255) - random but deterministic
    np.random.seed(42)
    for i in range(16, 256):
        phase = np.random.random() * 2 * np.pi
        freq = 10 + np.random.random() * 90
        expanded[:, i] = np.sin(freq * x + phase)

    return expanded


# ============================================================
# PyTorch Implementation - Ultra-Sparse
# ============================================================
def inference_pytorch_sparse(model_data, x_expanded):
    """PyTorch inference for Ultra-Sparse model."""
    input_indices = model_data['input_indices']
    w1 = model_data['w1']
    b1 = model_data['b1']
    w2 = model_data['w2']
    b2 = model_data['b2']

    hidden_size = w1.shape[0]
    inputs_per_neuron = w1.shape[1]
    batch_size = x_expanded.shape[0]

    # Gather selected inputs
    selected = torch.zeros(batch_size, hidden_size, inputs_per_neuron)
    for h in range(hidden_size):
        selected[:, h, :] = x_expanded[:, input_indices[h]]

    # Forward pass
    pre_act = (selected * w1.unsqueeze(0)).sum(dim=2) + b1
    hidden = torch.tanh(pre_act)
    output = torch.tanh(torch.nn.functional.linear(hidden, w2, b2))

    return output


# ============================================================
# PyTorch Implementation - Dense (Standard SA / Backprop)
# ============================================================
def inference_pytorch_dense(model_data, x_expanded):
    """PyTorch inference for dense model (256->8->1)."""
    w1 = model_data['w1']
    b1 = model_data['b1']
    w2 = model_data['w2']
    b2 = model_data['b2']

    # Forward pass - standard dense layers
    hidden = torch.tanh(torch.nn.functional.linear(x_expanded, w1, b1))
    output = torch.tanh(torch.nn.functional.linear(hidden, w2, b2))

    return output


# ============================================================
# NumPy Implementation - Ultra-Sparse
# ============================================================
def inference_numpy_sparse(model_data, x_expanded):
    """Pure NumPy inference for Ultra-Sparse."""
    input_indices = model_data['input_indices'].numpy()
    w1 = model_data['w1'].numpy()
    b1 = model_data['b1'].numpy()
    w2 = model_data['w2'].numpy().flatten()
    b2 = model_data['b2'].numpy().item()

    # Gather selected inputs using advanced indexing
    selected = x_expanded[:, input_indices]

    # Forward pass
    pre_act = np.sum(selected * w1, axis=2) + b1
    hidden = np.tanh(pre_act)
    output = np.tanh(np.dot(hidden, w2) + b2)

    return output


# ============================================================
# NumPy Implementation - Dense
# ============================================================
def inference_numpy_dense(model_data, x_expanded):
    """Pure NumPy inference for dense model."""
    w1 = model_data['w1'].numpy()
    b1 = model_data['b1'].numpy()
    w2 = model_data['w2'].numpy().flatten()
    b2 = model_data['b2'].numpy().item()

    # Forward pass
    hidden = np.tanh(x_expanded @ w1.T + b1)
    output = np.tanh(hidden @ w2 + b2)

    return output


# ============================================================
# NumPy Optimized - Ultra-Sparse
# ============================================================
class NumPyOptimizedSparse:
    """Optimized NumPy model for Ultra-Sparse."""

    def __init__(self, model_data):
        self.indices = model_data['input_indices'].numpy()
        self.w1 = model_data['w1'].numpy()
        self.b1 = model_data['b1'].numpy()
        self.w2 = model_data['w2'].numpy().flatten()
        self.b2 = model_data['b2'].numpy().item()

    def __call__(self, x_expanded):
        selected = x_expanded[:, self.indices]
        pre_act = np.einsum('bhk,hk->bh', selected, self.w1) + self.b1
        hidden = np.tanh(pre_act)
        return np.tanh(hidden @ self.w2 + self.b2)


# ============================================================
# NumPy Optimized - Dense
# ============================================================
class NumPyOptimizedDense:
    """Optimized NumPy model for dense."""

    def __init__(self, model_data):
        self.w1 = model_data['w1'].numpy()
        self.b1 = model_data['b1'].numpy()
        self.w2 = model_data['w2'].numpy().flatten()
        self.b2 = model_data['b2'].numpy().item()

    def __call__(self, x_expanded):
        hidden = np.tanh(x_expanded @ self.w1.T + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)


# ============================================================
# Numba JIT Implementation - Ultra-Sparse
# ============================================================
if HAS_NUMBA:
    @jit(nopython=True, fastmath=True, cache=True)
    def inference_numba_sparse(x_expanded, indices, w1, b1, w2, b2):
        """Numba-compiled inference for Ultra-Sparse."""
        batch_size = x_expanded.shape[0]
        hidden_size = w1.shape[0]
        inputs_per_neuron = w1.shape[1]

        output = np.empty(batch_size, dtype=np.float32)

        for b in range(batch_size):
            # Compute hidden layer
            hidden = np.empty(hidden_size, dtype=np.float32)
            for h in range(hidden_size):
                acc = b1[h]
                for k in range(inputs_per_neuron):
                    acc += x_expanded[b, indices[h, k]] * w1[h, k]
                hidden[h] = np.tanh(acc)

            # Compute output
            out_acc = b2
            for h in range(hidden_size):
                out_acc += hidden[h] * w2[h]
            output[b] = np.tanh(out_acc)

        return output

    @jit(nopython=True, fastmath=True, cache=True)
    def inference_numba_dense(x_expanded, w1, b1, w2, b2):
        """Numba-compiled inference for dense model."""
        batch_size = x_expanded.shape[0]
        hidden_size = w1.shape[0]
        input_size = w1.shape[1]

        output = np.empty(batch_size, dtype=np.float32)

        for b in range(batch_size):
            # Compute hidden layer (256 -> 8)
            hidden = np.empty(hidden_size, dtype=np.float32)
            for h in range(hidden_size):
                acc = b1[h]
                for i in range(input_size):
                    acc += x_expanded[b, i] * w1[h, i]
                hidden[h] = np.tanh(acc)

            # Compute output (8 -> 1)
            out_acc = b2
            for h in range(hidden_size):
                out_acc += hidden[h] * w2[h]
            output[b] = np.tanh(out_acc)

        return output


# ============================================================
# torch.compile Implementation (PyTorch 2.0+)
# ============================================================
def get_torch_compiled(model_data):
    """Get torch.compile version if available."""
    if not hasattr(torch, 'compile'):
        return None

    input_indices = model_data['input_indices']
    w1 = model_data['w1']
    b1 = model_data['b1']
    w2 = model_data['w2']
    b2 = model_data['b2']

    def forward(x_expanded):
        hidden_size = w1.shape[0]
        inputs_per_neuron = w1.shape[1]
        batch_size = x_expanded.shape[0]

        selected = torch.zeros(batch_size, hidden_size, inputs_per_neuron)
        for h in range(hidden_size):
            selected[:, h, :] = x_expanded[:, input_indices[h]]

        pre_act = (selected * w1.unsqueeze(0)).sum(dim=2) + b1
        hidden = torch.tanh(pre_act)
        return torch.tanh(torch.nn.functional.linear(hidden, w2, b2))

    try:
        compiled = torch.compile(forward, mode="reduce-overhead")
        return compiled
    except Exception as e:
        print(f"torch.compile failed: {e}")
        return None


def benchmark(name, fn, x, n_warmup=10, n_runs=100):
    """Benchmark a function."""
    # Warmup
    for _ in range(n_warmup):
        _ = fn(x)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = fn(x)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1e6)  # Convert to microseconds

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def run_experiment():
    """Run inference engine comparison for all three model types."""
    print("=" * 70)
    print("EXPERIMENT: Inference Engine Comparison")
    print("=" * 70)

    base_path = Path(__file__).parent.parent / "results" / "comprehensive_benchmark"

    # Load all three models
    models = {}

    # Ultra-Sparse
    sparse_path = base_path / "best_ultra_sparse.pt"
    if sparse_path.exists():
        models['ultra_sparse'] = load_ultra_sparse_model(sparse_path)
        print(f"Loaded Ultra-Sparse: {models['ultra_sparse']['w1'].shape}")
    else:
        print(f"Ultra-Sparse model not found: {sparse_path}")

    # Standard SA
    sa_path = base_path / "best_standard_sa.pt"
    if sa_path.exists():
        models['standard_sa'] = load_standard_model(sa_path)
        print(f"Loaded Standard SA: {models['standard_sa']['w1'].shape}")
    else:
        print(f"Standard SA model not found: {sa_path}")

    # Backprop
    bp_path = base_path / "best_backprop.pt"
    if bp_path.exists():
        models['backprop'] = load_standard_model(bp_path)
        print(f"Loaded Backprop: {models['backprop']['w1'].shape}")
    else:
        print(f"Backprop model not found: {bp_path}")

    if not models:
        print("No models found. Run comprehensive_benchmark.py first.")
        return

    # Test different batch sizes
    batch_sizes = [1, 10, 100, 1000]

    all_results = {model: {} for model in models}

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"BATCH SIZE: {batch_size}")
        print("=" * 70)

        # Create test data
        x_raw = create_test_data(batch_size)
        x_numpy = expand_input_numpy(x_raw)
        x_torch = torch.tensor(x_numpy)

        for model_name, model_data in models.items():
            print(f"\n--- {model_name.upper()} ---")
            all_results[model_name][batch_size] = {}

            is_sparse = 'input_indices' in model_data

            # PyTorch
            if is_sparse:
                fn = lambda x, m=model_data: inference_pytorch_sparse(m, x)
            else:
                fn = lambda x, m=model_data: inference_pytorch_dense(m, x)

            r = benchmark("PyTorch", fn, x_torch)
            all_results[model_name][batch_size]['pytorch'] = r
            print(f"  PyTorch: {r['mean']:.2f} μs")

            # NumPy Optimized
            if is_sparse:
                np_model = NumPyOptimizedSparse(model_data)
            else:
                np_model = NumPyOptimizedDense(model_data)

            r = benchmark("NumPy", lambda x: np_model(x), x_numpy)
            all_results[model_name][batch_size]['numpy_opt'] = r
            print(f"  NumPy:   {r['mean']:.2f} μs")

            # Numba JIT
            if HAS_NUMBA:
                w1 = model_data['w1'].numpy().astype(np.float32)
                b1 = model_data['b1'].numpy().astype(np.float32)
                w2 = model_data['w2'].numpy().flatten().astype(np.float32)
                b2 = np.float32(model_data['b2'].numpy().item())

                if is_sparse:
                    indices = model_data['input_indices'].numpy().astype(np.int64)
                    # Compile
                    _ = inference_numba_sparse(x_numpy, indices, w1, b1, w2, b2)
                    r = benchmark("Numba", lambda x: inference_numba_sparse(x, indices, w1, b1, w2, b2), x_numpy)
                else:
                    # Compile
                    _ = inference_numba_dense(x_numpy, w1, b1, w2, b2)
                    r = benchmark("Numba", lambda x: inference_numba_dense(x, w1, b1, w2, b2), x_numpy)

                all_results[model_name][batch_size]['numba'] = r
                print(f"  Numba:   {r['mean']:.2f} μs")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Best Engine per Model (μs)")
    print("=" * 70)

    print(f"\n{'Model':<15} | {'Batch':<6} | {'PyTorch':<10} | {'NumPy':<10} | {'Numba':<10} | {'Best':<10}")
    print("-" * 75)

    for model_name in models:
        for bs in batch_sizes:
            res = all_results[model_name][bs]
            pt = res.get('pytorch', {}).get('mean', 0)
            np_opt = res.get('numpy_opt', {}).get('mean', 0)
            nb = res.get('numba', {}).get('mean', 0)

            best_val = min(pt, np_opt, nb) if nb else min(pt, np_opt)
            best_name = 'Numba' if nb == best_val else ('NumPy' if np_opt == best_val else 'PyTorch')

            print(f"{model_name:<15} | {bs:<6} | {pt:<10.2f} | {np_opt:<10.2f} | {nb:<10.2f} | {best_name:<10}")

    # ================================================================
    # CROSS-MODEL COMPARISON (Numba for all)
    # ================================================================
    print("\n" + "=" * 70)
    print("CROSS-MODEL COMPARISON (Best Engine)")
    print("=" * 70)

    print(f"\n{'Batch':<8} | {'Ultra-Sparse':<15} | {'Standard SA':<15} | {'Backprop':<15} | {'Winner':<15}")
    print("-" * 75)

    for bs in batch_sizes:
        times = {}
        for model_name in models:
            res = all_results[model_name][bs]
            # Use best available engine
            if 'numba' in res:
                times[model_name] = res['numba']['mean']
            else:
                times[model_name] = res['numpy_opt']['mean']

        winner = min(times, key=times.get)
        row = f"{bs:<8}"
        for model_name in ['ultra_sparse', 'standard_sa', 'backprop']:
            if model_name in times:
                row += f" | {times[model_name]:<15.2f}"
            else:
                row += f" | {'N/A':<15}"
        row += f" | {winner:<15}"
        print(row)

    # ================================================================
    # EFFICIENCY ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("EFFICIENCY ANALYSIS (Batch=1, Numba)")
    print("=" * 70)

    if 'ultra_sparse' in models and 'standard_sa' in models:
        sparse_time = all_results['ultra_sparse'][1].get('numba', {}).get('mean', 0)
        sa_time = all_results['standard_sa'][1].get('numba', {}).get('mean', 0)
        bp_time = all_results.get('backprop', {}).get(1, {}).get('numba', {}).get('mean', 0)

        print(f"\nUltra-Sparse: {sparse_time:.2f} μs (33 params)")
        print(f"Standard SA:  {sa_time:.2f} μs (2065 params)")
        if bp_time:
            print(f"Backprop:     {bp_time:.2f} μs (2065 params)")

        if sparse_time and sa_time:
            speedup = sa_time / sparse_time
            print(f"\nUltra-Sparse is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Standard SA")

        if sparse_time and bp_time:
            speedup = bp_time / sparse_time
            print(f"Ultra-Sparse is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Backprop")

    return all_results


if __name__ == "__main__":
    run_experiment()
