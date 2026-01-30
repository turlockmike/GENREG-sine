"""
Experiment: Neural Network Pruning

Simulate genetic "gene deletion" by pruning useless neurons and weights.

Pruning strategies:
1. Weight pruning: Zero out small weights
2. Neuron pruning: Remove neurons with negligible output contribution
3. Constant folding: If a saturated neuron always outputs ±1, fold into bias
4. Evolutionary pruning: Add deletion mutations during training

Goal: Find the minimal network that achieves similar performance.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import copy
from pathlib import Path

import legacy.sine_config as sine_config as cfg
from legacy.sine_controller import SineController, expand_input
from sine_annealing import simulated_annealing

DEVICE = torch.device("cpu")


class PrunableMLP(nn.Module):
    """MLP that supports pruning operations."""

    def __init__(self, w1, b1, w2, b2):
        super().__init__()
        self.w1 = nn.Parameter(w1.clone(), requires_grad=False)
        self.b1 = nn.Parameter(b1.clone(), requires_grad=False)
        self.w2 = nn.Parameter(w2.clone(), requires_grad=False)
        self.b2 = nn.Parameter(b2.clone(), requires_grad=False)

        # Track which neurons are active
        self.active_neurons = torch.ones(w1.shape[0], dtype=torch.bool)

    def forward(self, x):
        # Only use active neurons
        w1_active = self.w1[self.active_neurons]
        b1_active = self.b1[self.active_neurons]
        w2_active = self.w2[:, self.active_neurons]

        h = torch.tanh(torch.nn.functional.linear(x, w1_active, b1_active))
        out = torch.tanh(torch.nn.functional.linear(h, w2_active, self.b2))
        return out.squeeze(-1)

    def forward_binary(self, x):
        """Forward pass with sign() activation for saturated networks."""
        w1_active = self.w1[self.active_neurons]
        b1_active = self.b1[self.active_neurons]
        w2_active = self.w2[:, self.active_neurons]

        h = torch.sign(torch.nn.functional.linear(x, w1_active, b1_active))
        out = torch.tanh(torch.nn.functional.linear(h, w2_active, self.b2))
        return out.squeeze(-1)

    def get_hidden_activations(self, x):
        """Get raw hidden activations for analysis."""
        w1_active = self.w1[self.active_neurons]
        b1_active = self.b1[self.active_neurons]
        pre_act = torch.nn.functional.linear(x, w1_active, b1_active)
        return pre_act, torch.tanh(pre_act)

    def num_active_neurons(self):
        return self.active_neurons.sum().item()

    def num_active_weights(self):
        n_active = self.active_neurons.sum().item()
        input_size = self.w1.shape[1]
        # w1: input_size * n_active, b1: n_active, w2: n_active, b2: 1
        return input_size * n_active + n_active + n_active + 1

    def prune_neuron(self, idx):
        """Mark a neuron as inactive."""
        active_indices = torch.where(self.active_neurons)[0]
        if idx < len(active_indices):
            real_idx = active_indices[idx]
            self.active_neurons[real_idx] = False

    def get_neuron_importance(self, x):
        """
        Compute importance score for each neuron.
        Importance = abs(output_weight) * variance(activation)
        """
        with torch.no_grad():
            pre_act, h = self.get_hidden_activations(x)

            # Get active output weights
            w2_active = self.w2[:, self.active_neurons].squeeze(0)

            # Importance = |w2| * std(activation)
            activation_std = h.std(dim=0)
            importance = w2_active.abs() * activation_std

            return importance

    def get_constant_neurons(self, x, threshold=0.99):
        """
        Find neurons that are essentially constant (always same output).
        These can be folded into bias.
        """
        with torch.no_grad():
            _, h = self.get_hidden_activations(x)

            # Check if neuron output is nearly constant
            h_min = h.min(dim=0)[0]
            h_max = h.max(dim=0)[0]
            h_range = h_max - h_min

            # Constant if range is very small (always ~same value)
            is_constant = h_range < (1 - threshold) * 2  # tanh range is 2

            # Also check if always saturated to same sign
            always_positive = (h > 0.95).all(dim=0)
            always_negative = (h < -0.95).all(dim=0)
            always_saturated_same = always_positive | always_negative

            return is_constant | always_saturated_same, h.mean(dim=0)


def prune_by_weight_magnitude(model, x_test, threshold_pct=10):
    """Prune neurons whose output weights are in bottom X percentile."""
    with torch.no_grad():
        w2_active = model.w2[:, model.active_neurons].squeeze(0)
        threshold = np.percentile(w2_active.abs().numpy(), threshold_pct)

        pruned = 0
        for i in range(len(w2_active) - 1, -1, -1):  # Reverse to handle index shifts
            if w2_active[i].abs() < threshold and model.num_active_neurons() > 1:
                model.prune_neuron(i)
                pruned += 1

        return pruned


def prune_by_importance(model, x_test, threshold_pct=10):
    """Prune neurons with lowest importance scores."""
    importance = model.get_neuron_importance(x_test)
    threshold = np.percentile(importance.numpy(), threshold_pct)

    pruned = 0
    for i in range(len(importance) - 1, -1, -1):
        if importance[i] < threshold and model.num_active_neurons() > 1:
            model.prune_neuron(i)
            pruned += 1

    return pruned


def prune_constant_neurons(model, x_test):
    """
    Remove neurons that always output the same value.
    Fold their contribution into the output bias.
    """
    is_constant, mean_values = model.get_constant_neurons(x_test)

    with torch.no_grad():
        w2_active = model.w2[:, model.active_neurons].squeeze(0)

        pruned = 0
        # Process in reverse to handle index shifts
        active_indices = torch.where(model.active_neurons)[0]

        for i in range(len(is_constant) - 1, -1, -1):
            if is_constant[i] and model.num_active_neurons() > 1:
                # Fold constant neuron's contribution into bias
                contribution = w2_active[i] * mean_values[i]
                model.b2.data += contribution

                # Prune the neuron
                model.prune_neuron(i)
                pruned += 1

                # Update w2_active for next iteration
                w2_active = model.w2[:, model.active_neurons].squeeze(0)

        return pruned


def iterative_pruning(model, x_test, y_true, max_mse_increase=0.01, use_binary=False):
    """
    Iteratively prune neurons until MSE degrades too much.
    Returns pruning history.
    """
    history = []

    # Initial evaluation
    with torch.no_grad():
        if use_binary:
            pred = model.forward_binary(x_test)
        else:
            pred = model.forward(x_test)
        initial_mse = torch.mean((pred - y_true) ** 2).item()

    history.append({
        'step': 0,
        'neurons': model.num_active_neurons(),
        'weights': model.num_active_weights(),
        'mse': initial_mse,
        'action': 'initial',
    })

    step = 1
    max_allowed_mse = initial_mse * (1 + max_mse_increase)

    while model.num_active_neurons() > 1:
        # Try pruning constant neurons first
        n_pruned = prune_constant_neurons(model, x_test)

        if n_pruned == 0:
            # Try pruning by importance
            n_pruned = prune_by_importance(model, x_test, threshold_pct=20)

        if n_pruned == 0:
            # Force prune least important
            importance = model.get_neuron_importance(x_test)
            min_idx = importance.argmin().item()
            model.prune_neuron(min_idx)
            n_pruned = 1

        # Evaluate
        with torch.no_grad():
            if use_binary:
                pred = model.forward_binary(x_test)
            else:
                pred = model.forward(x_test)
            mse = torch.mean((pred - y_true) ** 2).item()

        history.append({
            'step': step,
            'neurons': model.num_active_neurons(),
            'weights': model.num_active_weights(),
            'mse': mse,
            'action': f'pruned_{n_pruned}',
        })

        # Stop if MSE degrades too much
        if mse > max_allowed_mse and model.num_active_neurons() < 8:
            break

        step += 1

    return history


def train_with_pruning_pressure(x_test, y_true, steps=15000, prune_every=1000):
    """
    Train with periodic pruning - simulates evolutionary gene deletion.
    """
    # Initialize
    current = SineController(device=DEVICE)

    def evaluate(ctrl):
        with torch.no_grad():
            pred = ctrl.forward(x_test, track=True)
            return -torch.mean((pred - y_true) ** 2).item()

    current_fitness = evaluate(current)
    best = current.clone()
    best_fitness = current_fitness

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / steps)

    history = []

    for step in range(steps):
        temperature = t_initial * (decay ** step)

        # Standard SA mutation
        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)

        # Occasionally add "deletion" mutation - zero out random weights
        if np.random.random() < 0.05:  # 5% chance
            # Zero out a random weight
            layer = np.random.choice(['w1', 'b1', 'w2', 'b2'])
            param = getattr(mutant, layer)
            idx = tuple(np.random.randint(0, s) for s in param.shape)
            param.data[idx] = 0.0

        mutant_fitness = evaluate(mutant)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

        if step % 500 == 0:
            # Count near-zero weights
            total_weights = sum(p.numel() for p in [best.w1, best.b1, best.w2, best.b2])
            zero_weights = sum((p.abs() < 0.01).sum().item() for p in [best.w1, best.b1, best.w2, best.b2])

            history.append({
                'step': step,
                'mse': -best_fitness,
                'zero_weights': zero_weights,
                'total_weights': total_weights,
                'sparsity': zero_weights / total_weights,
            })

    return best, history


def train_with_l1_sparsity(x_test, y_true, steps=15000, l1_weight=0.001):
    """
    Train with L1 regularization - penalize weight magnitudes.
    This encourages sparsity (many weights go to zero).
    """
    current = SineController(device=DEVICE)

    def evaluate_with_l1(ctrl, l1_w):
        with torch.no_grad():
            pred = ctrl.forward(x_test, track=True)
            mse = torch.mean((pred - y_true) ** 2).item()

            # L1 penalty on weights
            l1_penalty = sum(p.abs().sum().item() for p in [ctrl.w1, ctrl.w2])
            l1_penalty = l1_penalty / (ctrl.w1.numel() + ctrl.w2.numel())  # Normalize

            # Combined fitness (negative because we maximize)
            return -(mse + l1_w * l1_penalty), mse, l1_penalty

    current_fitness, _, _ = evaluate_with_l1(current, l1_weight)
    best = current.clone()
    best_fitness = current_fitness
    best_mse = float('inf')

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / steps)

    history = []

    for step in range(steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)

        mutant_fitness, mse, l1_pen = evaluate_with_l1(mutant, l1_weight)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness

            # Track best by MSE only (not L1-penalized fitness)
            if mse < best_mse:
                best = current.clone()
                best_fitness = current_fitness
                best_mse = mse

        if step % 500 == 0:
            _, curr_mse, curr_l1 = evaluate_with_l1(best, l1_weight)
            total_weights = best.w1.numel() + best.w2.numel()
            small_weights = (best.w1.abs() < 0.01).sum().item() + (best.w2.abs() < 0.01).sum().item()

            history.append({
                'step': step,
                'mse': curr_mse,
                'l1_penalty': curr_l1,
                'small_weights': small_weights,
                'total_weights': total_weights,
                'sparsity': small_weights / total_weights,
            })

    return best, history


def weight_pruning(model, threshold_pct):
    """Zero out weights below threshold percentile."""
    with torch.no_grad():
        all_weights = torch.cat([model.w1.flatten(), model.w2.flatten()])
        threshold = np.percentile(all_weights.abs().numpy(), threshold_pct)

        mask1 = model.w1.abs() >= threshold
        mask2 = model.w2.abs() >= threshold

        model.w1.data *= mask1.float()
        model.w2.data *= mask2.float()

        total = model.w1.numel() + model.w2.numel()
        remaining = mask1.sum().item() + mask2.sum().item()
        return total - remaining, remaining


def benchmark_weight_pruning(controller, x_expanded, y_true):
    """Test different levels of weight pruning."""
    results = []

    for prune_pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        # Fresh copy each time
        model = PrunableMLP(
            controller.w1.clone(), controller.b1.clone(),
            controller.w2.clone(), controller.b2.clone()
        )

        pruned, remaining = weight_pruning(model, prune_pct)

        with torch.no_grad():
            pred = model.forward_binary(x_expanded)
            mse = torch.mean((pred - y_true) ** 2).item()

        results.append({
            'prune_pct': prune_pct,
            'pruned_weights': pruned,
            'remaining_weights': remaining,
            'mse': mse,
        })

    return results


def run_experiment():
    """Run pruning experiments."""
    print("=" * 70)
    print("EXPERIMENT: NEURAL NETWORK PRUNING")
    print("=" * 70)
    print("Simulating genetic gene deletion to find minimal networks")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare data
    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    x_expanded = expand_input(x_raw.unsqueeze(-1))
    y_true = torch.sin(x_raw)

    results = {}

    # Train saturated network
    print("\n[Training Saturated Network (SA)]")
    sa_controller, _ = simulated_annealing(
        x_raw, y_true, max_steps=15000,
        t_initial=0.01, t_final=0.00001, verbose=False
    )
    sa_pred = sa_controller.forward(x_raw, track=True)
    sa_mse = torch.mean((sa_pred - y_true) ** 2).item()
    sa_k = sa_controller.get_k()
    print(f"  MSE: {sa_mse:.6f}, Saturation: {sa_k}/{cfg.HIDDEN_SIZE}")

    # Create prunable model
    print("\n[Pruning Experiment - Standard (tanh)]")
    model_standard = PrunableMLP(
        sa_controller.w1, sa_controller.b1,
        sa_controller.w2, sa_controller.b2
    )

    history_standard = iterative_pruning(model_standard, x_expanded, y_true, max_mse_increase=0.5, use_binary=False)

    print(f"  {'Neurons':<10} | {'Weights':<10} | {'MSE':<12} | {'Action':<15}")
    print("  " + "-" * 55)
    for h in history_standard:
        print(f"  {h['neurons']:<10} | {h['weights']:<10} | {h['mse']:<12.6f} | {h['action']:<15}")

    results['standard_pruning'] = history_standard

    # Pruning with binary activations
    print("\n[Pruning Experiment - Binary (sign)]")
    model_binary = PrunableMLP(
        sa_controller.w1, sa_controller.b1,
        sa_controller.w2, sa_controller.b2
    )

    history_binary = iterative_pruning(model_binary, x_expanded, y_true, max_mse_increase=0.5, use_binary=True)

    print(f"  {'Neurons':<10} | {'Weights':<10} | {'MSE':<12} | {'Action':<15}")
    print("  " + "-" * 55)
    for h in history_binary:
        print(f"  {h['neurons']:<10} | {h['weights']:<10} | {h['mse']:<12.6f} | {h['action']:<15}")

    results['binary_pruning'] = history_binary

    # Training with pruning pressure
    print("\n[Training with Deletion Mutations]")
    sparse_controller, sparse_history = train_with_pruning_pressure(x_raw, y_true, steps=15000)

    sparse_pred = sparse_controller.forward(x_raw, track=True)
    sparse_mse = torch.mean((sparse_pred - y_true) ** 2).item()
    sparse_k = sparse_controller.get_k()

    print(f"  Final MSE: {sparse_mse:.6f}, Saturation: {sparse_k}/{cfg.HIDDEN_SIZE}")
    print(f"  Sparsity progression:")
    for h in sparse_history[::3]:  # Every 3rd entry
        print(f"    Step {h['step']}: {h['sparsity']:.1%} zero weights, MSE={h['mse']:.6f}")

    results['sparse_training'] = sparse_history

    # Training with L1 regularization
    print("\n[Training with L1 Sparsity Pressure]")
    l1_controller, l1_history = train_with_l1_sparsity(x_raw, y_true, steps=15000, l1_weight=0.001)

    l1_pred = l1_controller.forward(x_raw, track=True)
    l1_mse = torch.mean((l1_pred - y_true) ** 2).item()
    l1_k = l1_controller.get_k()

    # Count sparse weights
    total_w = l1_controller.w1.numel() + l1_controller.w2.numel()
    sparse_w = (l1_controller.w1.abs() < 0.01).sum().item() + (l1_controller.w2.abs() < 0.01).sum().item()

    print(f"  Final MSE: {l1_mse:.6f}, Saturation: {l1_k}/{cfg.HIDDEN_SIZE}")
    print(f"  Sparsity: {sparse_w}/{total_w} ({sparse_w/total_w:.1%}) weights near zero")
    print(f"  Progression:")
    for h in l1_history[::3]:
        print(f"    Step {h['step']}: {h['sparsity']:.1%} sparse, MSE={h['mse']:.6f}")

    results['l1_training'] = l1_history

    # Weight pruning (zero out small weights, keep all neurons)
    print("\n[Weight Pruning Experiment]")
    print("  Zeroing small weights while keeping all neurons")

    weight_prune_results = benchmark_weight_pruning(sa_controller, x_expanded, y_true)

    print(f"  {'Prune %':<10} | {'Remaining':<12} | {'MSE':<12} | {'MSE Change':<12}")
    print("  " + "-" * 55)
    baseline_mse = weight_prune_results[0]['mse']
    for r in weight_prune_results:
        mse_change = (r['mse'] / baseline_mse - 1) * 100
        print(f"  {r['prune_pct']:<10} | {r['remaining_weights']:<12} | {r['mse']:<12.6f} | {mse_change:+.1f}%")

    results['weight_pruning'] = weight_prune_results

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Find minimal network that maintains reasonable MSE
    initial_mse = history_standard[0]['mse']
    for h in history_standard:
        if h['mse'] < initial_mse * 1.5:  # Allow 50% MSE increase
            best_standard = h

    for h in history_binary:
        if h['mse'] < history_binary[0]['mse'] * 1.5:
            best_binary = h

    print(f"\nOriginal network: 8 neurons, {8*256 + 8 + 8 + 1} weights")
    print(f"  Standard MSE: {initial_mse:.6f}")

    print(f"\nPruned (standard, <50% MSE increase):")
    print(f"  Neurons: {best_standard['neurons']}, Weights: {best_standard['weights']}")
    print(f"  MSE: {best_standard['mse']:.6f}")
    print(f"  Compression: {(1 - best_standard['weights']/(8*256+8+8+1))*100:.1f}% fewer weights")

    print(f"\nPruned (binary, <50% MSE increase):")
    print(f"  Neurons: {best_binary['neurons']}, Weights: {best_binary['weights']}")
    print(f"  MSE: {best_binary['mse']:.6f}")
    print(f"  Compression: {(1 - best_binary['weights']/(8*256+8+8+1))*100:.1f}% fewer weights")

    # Find best weight pruning level
    best_weight_prune = None
    for r in weight_prune_results:
        if r['mse'] < baseline_mse * 1.5:  # Allow 50% MSE increase
            best_weight_prune = r

    if best_weight_prune:
        print(f"\nWeight pruning (binary, <50% MSE increase):")
        print(f"  Pruned: {best_weight_prune['prune_pct']}% of weights")
        print(f"  Remaining: {best_weight_prune['remaining_weights']} weights")
        print(f"  MSE: {best_weight_prune['mse']:.6f}")
        total_weights = 8*256 + 8
        print(f"  Compression: {best_weight_prune['prune_pct']}% fewer weights")

    results['summary'] = {
        'original_neurons': 8,
        'original_weights': 8*256 + 8 + 8 + 1,
        'original_mse': initial_mse,
        'pruned_standard_neurons': best_standard['neurons'],
        'pruned_standard_weights': best_standard['weights'],
        'pruned_standard_mse': best_standard['mse'],
        'pruned_binary_neurons': best_binary['neurons'],
        'pruned_binary_weights': best_binary['weights'],
        'pruned_binary_mse': best_binary['mse'],
    }

    # Save results
    output_dir = Path("results/pruning_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_experiment()
