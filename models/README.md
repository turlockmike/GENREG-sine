# Trained Models

Pre-trained models from comprehensive benchmark (20 trials each, best selected).

## Available Models

| Model | MSE | Params | Memory | Inference (Numba) |
|-------|-----|--------|--------|-------------------|
| `ultra_sparse_mse0.000303.pt` | 0.000303 | 33 | 132 B | 0.44 μs |
| `standard_sa_mse0.009155.pt` | 0.009155 | 2065 | 8260 B | 0.63 μs |
| `backprop_mse0.000003.pt` | 0.000003 | 2065 | 8260 B | 0.58 μs |

## Loading Models

### Ultra-Sparse Model

```python
import torch
import numpy as np

# Load
data = torch.load('models/ultra_sparse_mse0.000303.pt', weights_only=False)
state = data['state_dict']

input_indices = state['input_indices']  # Shape: (8, 2) - which inputs each neuron uses
w1 = state['w1']                        # Shape: (8, 2) - weights for selected inputs
b1 = state['b1']                        # Shape: (8,)
w2 = state['w2']                        # Shape: (1, 8)
b2 = state['b2']                        # Shape: (1,)

# Metadata
print(data['stats'])      # MSE, saturation, energy, etc.
print(data['selection'])  # Which true inputs were selected
```

### Standard SA / Backprop Models

```python
import torch

# Load
data = torch.load('models/standard_sa_mse0.009155.pt', weights_only=False)
state = data['state_dict']

w1 = state['w1']  # Shape: (8, 256) - dense layer
b1 = state['b1']  # Shape: (8,)
w2 = state['w2']  # Shape: (1, 8)
b2 = state['b2']  # Shape: (1,)
```

## Fast Inference with Numba

For maximum speed, use Numba JIT compilation:

```python
import numpy as np
from numba import jit

@jit(nopython=True, fastmath=True)
def inference_ultra_sparse(x_expanded, indices, w1, b1, w2, b2):
    """0.44 μs per sample - 1.4x faster than dense models."""
    batch_size = x_expanded.shape[0]
    output = np.empty(batch_size, dtype=np.float32)

    for b in range(batch_size):
        # Hidden layer (sparse: only 16 multiply-adds)
        hidden = np.empty(8, dtype=np.float32)
        for h in range(8):
            acc = b1[h]
            for k in range(2):
                acc += x_expanded[b, indices[h, k]] * w1[h, k]
            hidden[h] = np.tanh(acc)

        # Output layer
        out_acc = b2
        for h in range(8):
            out_acc += hidden[h] * w2[h]
        output[b] = np.tanh(out_acc)

    return output

@jit(nopython=True, fastmath=True)
def inference_dense(x_expanded, w1, b1, w2, b2):
    """0.58-0.63 μs per sample."""
    batch_size = x_expanded.shape[0]
    output = np.empty(batch_size, dtype=np.float32)

    for b in range(batch_size):
        # Hidden layer (dense: 2048 multiply-adds)
        hidden = np.empty(8, dtype=np.float32)
        for h in range(8):
            acc = b1[h]
            for i in range(256):
                acc += x_expanded[b, i] * w1[h, i]
            hidden[h] = np.tanh(acc)

        # Output layer
        out_acc = b2
        for h in range(8):
            out_acc += hidden[h] * w2[h]
        output[b] = np.tanh(out_acc)

    return output

# Prepare weights (convert to numpy float32)
indices = state['input_indices'].numpy().astype(np.int64)
w1 = state['w1'].numpy().astype(np.float32)
b1 = state['b1'].numpy().astype(np.float32)
w2 = state['w2'].numpy().flatten().astype(np.float32)
b2 = np.float32(state['b2'].numpy().item())

# Run inference
output = inference_ultra_sparse(x_expanded, indices, w1, b1, w2, b2)
```

## Input Expansion

All models expect 256-dimensional expanded input:

```python
def expand_input(x):
    """Expand scalar x to 256 features."""
    batch_size = len(np.atleast_1d(x))
    expanded = np.zeros((batch_size, 256), dtype=np.float32)

    # True signals: sin(kx), cos(kx) for k=1..8
    for i in range(8):
        freq = i + 1
        expanded[:, i] = np.sin(freq * x)
        expanded[:, i + 8] = np.cos(freq * x)

    # Noise signals: high-frequency sinusoids (indices 16-255)
    np.random.seed(42)
    for i in range(16, 256):
        phase = np.random.random() * 2 * np.pi
        freq = 10 + np.random.random() * 90
        expanded[:, i] = np.sin(freq * x + phase)

    return expanded
```

## Performance Comparison

Inference time for single sample (batch=1):

| Model | PyTorch | NumPy | Numba | Best |
|-------|---------|-------|-------|------|
| Ultra-Sparse | 26.4 μs | 3.1 μs | **0.44 μs** | Numba |
| Standard SA | 3.3 μs | 2.1 μs | **0.63 μs** | Numba |
| Backprop | 3.2 μs | 2.2 μs | **0.58 μs** | Numba |

Ultra-Sparse with Numba is **1.3-1.4x faster** than dense models while using **63x fewer parameters**.
