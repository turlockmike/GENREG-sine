"""
Experiment: Backprop with Ultra-Sparse Architecture

Fair comparison: Train the SAME sparse architecture (8 neurons × 2 inputs)
using backprop instead of evolutionary methods.

This answers: Is Ultra-Sparse good because of the architecture or the training method?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
import json

from sine_controller import expand_input
from core.metrics import compute_metrics

DEVICE = torch.device("cpu")


class SparseBackpropController(nn.Module):
    """
    Sparse architecture trained with backprop.

    Each hidden neuron connects to only K inputs (same as Ultra-Sparse),
    but trained with gradient descent instead of evolution.
    """

    def __init__(self, hidden_size=8, inputs_per_neuron=2, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.hidden_size = hidden_size
        self.inputs_per_neuron = inputs_per_neuron
        self.input_size = 256

        # Fixed input indices (same as Ultra-Sparse)
        # We'll try both random and optimized indices
        self.register_buffer(
            'input_indices',
            torch.zeros(hidden_size, inputs_per_neuron, dtype=torch.long, device=self.device)
        )
        for h in range(hidden_size):
            self.input_indices[h] = torch.randperm(256)[:inputs_per_neuron]

        # Learnable weights for selected connections
        self.w1 = nn.Parameter(torch.randn(hidden_size, inputs_per_neuron) * np.sqrt(2.0 / inputs_per_neuron))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))

        self.last_hidden = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        expanded = expand_input(x)
        batch_size = expanded.shape[0]

        # Gather selected inputs for each hidden neuron
        selected = torch.zeros(batch_size, self.hidden_size, self.inputs_per_neuron, device=self.device)
        for h in range(self.hidden_size):
            selected[:, h, :] = expanded[:, self.input_indices[h]]

        # Compute hidden activations
        pre_act = (selected * self.w1.unsqueeze(0)).sum(dim=2) + self.b1
        hidden = torch.tanh(pre_act)
        self.last_hidden = hidden.detach()

        # Output
        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def set_indices(self, indices):
        """Set specific input indices (for testing with known good indices)."""
        self.input_indices.copy_(torch.tensor(indices, dtype=torch.long, device=self.device))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class SparseBackpropControllerDifferentiable(nn.Module):
    """
    Sparse architecture with LEARNABLE indices using soft attention.

    Instead of hard indices, use soft attention over all inputs,
    with sparsity regularization to encourage selection.
    """

    def __init__(self, hidden_size=8, inputs_per_neuron=2, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.hidden_size = hidden_size
        self.inputs_per_neuron = inputs_per_neuron

        # Attention logits for each hidden neuron over all 256 inputs
        # Will be converted to sparse attention via top-k or softmax
        self.attention_logits = nn.Parameter(torch.randn(hidden_size, 256) * 0.1)

        # Weights for selected connections (applied after attention)
        self.w1 = nn.Parameter(torch.randn(hidden_size, 256) * np.sqrt(2.0 / inputs_per_neuron))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, hidden_size) * np.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(1))

        self.last_hidden = None
        self.temperature = 1.0

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        expanded = expand_input(x)  # (batch, 256)

        # Soft attention (approaches hard selection as temperature → 0)
        attention = torch.softmax(self.attention_logits / self.temperature, dim=1)  # (hidden, 256)

        # Weighted input selection
        # expanded: (batch, 256), attention: (hidden, 256), w1: (hidden, 256)
        weighted_inputs = expanded.unsqueeze(1) * attention.unsqueeze(0) * self.w1.unsqueeze(0)
        pre_act = weighted_inputs.sum(dim=2) + self.b1  # (batch, hidden)

        hidden = torch.tanh(pre_act)
        self.last_hidden = hidden.detach()

        output = torch.tanh(nn.functional.linear(hidden, self.w2, self.b2))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def get_selected_indices(self, k=2):
        """Get top-k selected indices per neuron."""
        with torch.no_grad():
            _, indices = torch.topk(self.attention_logits, k, dim=1)
            return indices

    def sparsity_loss(self, k=2):
        """Encourage sparsity in attention weights."""
        # Entropy loss: encourage low entropy (peaked distribution)
        attention = torch.softmax(self.attention_logits, dim=1)
        entropy = -(attention * torch.log(attention + 1e-10)).sum(dim=1).mean()
        return entropy


def train_backprop_sparse(controller, x_train, y_train, epochs=2000, lr=0.01, verbose=True):
    """Train sparse controller with backprop."""
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = []
    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = controller(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.clone() for k, v in controller.state_dict().items()}

        if epoch % 500 == 0:
            sat = (controller.last_hidden.abs() > 0.95).float().mean().item()
            history.append({'epoch': epoch, 'mse': mse, 'saturation': sat})
            if verbose:
                print(f"  Epoch {epoch}: MSE={mse:.6f}, Sat={sat*100:.0f}%")

    controller.load_state_dict(best_state)
    return best_mse, history


def train_backprop_differentiable(controller, x_train, y_train, epochs=2000, lr=0.01,
                                   sparsity_weight=0.1, verbose=True):
    """Train differentiable sparse controller with backprop."""
    optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = []
    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        # Anneal temperature for harder attention
        controller.temperature = max(0.1, 1.0 - epoch / epochs)

        optimizer.zero_grad()
        pred = controller(x_train)
        mse_loss = torch.mean((pred - y_train) ** 2)
        sparse_loss = controller.sparsity_loss(k=2)
        loss = mse_loss + sparsity_weight * sparse_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        mse = mse_loss.item()
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.clone() for k, v in controller.state_dict().items()}

        if epoch % 500 == 0:
            sat = (controller.last_hidden.abs() > 0.95).float().mean().item()
            selected = controller.get_selected_indices(k=2)
            true_selected = [(selected[h] < 16).sum().item() for h in range(8)]
            history.append({'epoch': epoch, 'mse': mse, 'saturation': sat})
            if verbose:
                print(f"  Epoch {epoch}: MSE={mse:.6f}, Sat={sat*100:.0f}%, "
                      f"True inputs: {sum(true_selected)}/16")

    controller.load_state_dict(best_state)
    return best_mse, history


def run_experiment():
    """Compare backprop vs evolution on sparse architecture."""
    print("=" * 70)
    print("EXPERIMENT: Backprop with Ultra-Sparse Architecture")
    print("=" * 70)
    print("Testing if the architecture or training method matters more")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    results = {}
    n_trials = 5

    # ================================================================
    # Test 1: Backprop with RANDOM fixed indices
    # ================================================================
    print("\n[1] Backprop + Random Sparse Indices")
    print("-" * 50)

    mses = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        controller = SparseBackpropController(hidden_size=8, inputs_per_neuron=2)
        mse, _ = train_backprop_sparse(controller, x_test, y_true, epochs=2000, verbose=False)
        mses.append(mse)
        print(f"  Trial {trial}: MSE={mse:.6f}")

    results['backprop_random'] = {
        'mses': mses,
        'best': min(mses),
        'mean': np.mean(mses),
        'std': np.std(mses),
    }
    print(f"  Best: {min(mses):.6f}, Mean: {np.mean(mses):.6f}")

    # ================================================================
    # Test 2: Backprop with OPTIMAL indices (true signals only)
    # ================================================================
    print("\n[2] Backprop + Optimal Sparse Indices (true signals)")
    print("-" * 50)

    # Optimal: each neuron gets 2 true signal indices
    optimal_indices = torch.tensor([
        [0, 1],   # sin(x), sin(2x)
        [2, 3],   # sin(3x), sin(4x)
        [4, 5],   # sin(5x), sin(6x)
        [6, 7],   # sin(7x), sin(8x)
        [8, 9],   # cos(x), cos(2x)
        [10, 11], # cos(3x), cos(4x)
        [12, 13], # cos(5x), cos(6x)
        [14, 15], # cos(7x), cos(8x)
    ], dtype=torch.long)

    mses = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        controller = SparseBackpropController(hidden_size=8, inputs_per_neuron=2)
        controller.set_indices(optimal_indices)
        mse, _ = train_backprop_sparse(controller, x_test, y_true, epochs=2000, verbose=False)
        mses.append(mse)
        print(f"  Trial {trial}: MSE={mse:.6f}")

    results['backprop_optimal'] = {
        'mses': mses,
        'best': min(mses),
        'mean': np.mean(mses),
        'std': np.std(mses),
    }
    print(f"  Best: {min(mses):.6f}, Mean: {np.mean(mses):.6f}")

    # ================================================================
    # Test 3: Backprop with LEARNABLE indices (differentiable attention)
    # ================================================================
    print("\n[3] Backprop + Learnable Sparse Indices (soft attention)")
    print("-" * 50)

    mses = []
    for trial in range(n_trials):
        torch.manual_seed(trial)
        controller = SparseBackpropControllerDifferentiable(hidden_size=8, inputs_per_neuron=2)
        mse, _ = train_backprop_differentiable(controller, x_test, y_true, epochs=2000, verbose=False)

        # Check which indices were selected
        selected = controller.get_selected_indices(k=2)
        true_count = (selected < 16).sum().item()

        mses.append(mse)
        print(f"  Trial {trial}: MSE={mse:.6f}, True inputs selected: {true_count}/16")

    results['backprop_learnable'] = {
        'mses': mses,
        'best': min(mses),
        'mean': np.mean(mses),
        'std': np.std(mses),
    }
    print(f"  Best: {min(mses):.6f}, Mean: {np.mean(mses):.6f}")

    # ================================================================
    # Load Ultra-Sparse (evolutionary) results for comparison
    # ================================================================
    print("\n[4] Ultra-Sparse (Evolution) - from saved model")
    print("-" * 50)

    model_path = Path(__file__).parent.parent / "models" / "ultra_sparse_mse0.000303.pt"
    if model_path.exists():
        data = torch.load(model_path, weights_only=False)
        results['evolution'] = {
            'best': data['stats']['mse'],
            'mean': 0.000796,  # from comprehensive benchmark
            'std': 0.000371,
        }
        print(f"  Best: {data['stats']['mse']:.6f}")
        print(f"  Selected indices: {data['selection']['true_inputs']}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Same Architecture (8×2=16 connections), Different Training")
    print("=" * 70)

    print(f"\n{'Method':<35} | {'Best MSE':<12} | {'Mean MSE':<12}")
    print("-" * 65)

    for name, r in results.items():
        best = r['best']
        mean = r.get('mean', r['best'])
        print(f"{name:<35} | {best:<12.6f} | {mean:<12.6f}")

    # Key insight
    print(f"""
======================================================================
KEY INSIGHT
======================================================================

With RANDOM indices:
  - Backprop: {results['backprop_random']['best']:.6f}
  - Evolution: {results.get('evolution', {}).get('best', 'N/A')}

With OPTIMAL indices (backprop only):
  - Backprop: {results['backprop_optimal']['best']:.6f}

CONCLUSION:
""")

    if results['backprop_optimal']['best'] < results['backprop_random']['best']:
        print("  Backprop with optimal indices beats random indices.")
        print("  → Index selection matters more than training method!")
        print("  → Evolution's advantage is discovering good indices, not better optimization.")

    if 'evolution' in results:
        if results['evolution']['best'] < results['backprop_random']['best']:
            print(f"\n  Evolution ({results['evolution']['best']:.6f}) beats Backprop+Random ({results['backprop_random']['best']:.6f})")
            print("  → Evolution successfully discovers good input indices!")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "backprop_sparse"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items() if kk != 'mses'}
                   for k, v in results.items()}, f, indent=2)

    return results


if __name__ == "__main__":
    run_experiment()
