"""
Experiment: Deep Networks for Digits (2D Spatial Reasoning)

Hypothesis: For 2D spatial data (8x8 images), we need hierarchical representations.
A single hidden layer may not capture spatial relationships effectively.

Test architectures:
1. Shallow: 64 → H → 10
2. Deep (2 layers): 64 → H → H → 10
3. Deep (3 layers): 64 → H → H → H → 10

Each layer is ultra-sparse (K inputs per neuron).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DeepUltraSparseController:
    """Multi-layer ultra-sparse network."""

    def __init__(self, input_size, hidden_sizes, output_size, inputs_per_neuron):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes, e.g., [64, 64] for 2 layers
            output_size: Number of output classes
            inputs_per_neuron: K - how many inputs each hidden neuron sees
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.k = inputs_per_neuron
        self.n_layers = len(hidden_sizes)

        # Initialize layers
        self.layers = []

        prev_size = input_size
        for h_size in hidden_sizes:
            layer = {
                'indices': np.random.randint(0, prev_size, (h_size, inputs_per_neuron)),
                'weights': np.random.randn(h_size, inputs_per_neuron).astype(np.float32) * 0.5,
                'bias': np.zeros(h_size, dtype=np.float32)
            }
            self.layers.append(layer)
            prev_size = h_size

        # Output layer (fully connected to last hidden)
        self.output_weights = np.random.randn(output_size, hidden_sizes[-1]).astype(np.float32) * 0.5
        self.output_bias = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        """Forward pass through all layers."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        batch_size = x.shape[0]
        h = x

        # Pass through each hidden layer
        for layer in self.layers:
            # Gather inputs for this layer
            gathered = h[:, layer['indices']]  # (batch, neurons, k)
            # Weighted sum
            pre_act = np.einsum('bnk,nk->bn', gathered, layer['weights']) + layer['bias']
            h = np.tanh(pre_act)

        # Output layer
        out = h @ self.output_weights.T + self.output_bias
        out = np.tanh(out)

        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1):
        """Mutate weights and indices across all layers."""
        for layer_idx, layer in enumerate(self.layers):
            # Weight mutations
            mask = np.random.random(layer['weights'].shape) < weight_rate
            layer['weights'] += mask * np.random.randn(*layer['weights'].shape).astype(np.float32) * weight_scale

            # Bias mutations
            mask = np.random.random(layer['bias'].shape) < weight_rate
            layer['bias'] += mask * np.random.randn(*layer['bias'].shape).astype(np.float32) * weight_scale

            # Index mutations - determine input size for this layer
            if layer_idx == 0:
                max_idx = self.input_size
            else:
                max_idx = self.hidden_sizes[layer_idx - 1]

            for h in range(layer['indices'].shape[0]):
                if np.random.random() < index_swap_rate:
                    idx_to_swap = np.random.randint(0, self.k)
                    layer['indices'][h, idx_to_swap] = np.random.randint(0, max_idx)

        # Output layer mutations
        mask = np.random.random(self.output_weights.shape) < weight_rate
        self.output_weights += mask * np.random.randn(*self.output_weights.shape).astype(np.float32) * weight_scale

        mask = np.random.random(self.output_bias.shape) < weight_rate
        self.output_bias += mask * np.random.randn(*self.output_bias.shape).astype(np.float32) * weight_scale

    def clone(self):
        """Deep copy."""
        new = DeepUltraSparseController(
            self.input_size, self.hidden_sizes, self.output_size, self.k
        )
        for i, layer in enumerate(self.layers):
            new.layers[i]['indices'] = layer['indices'].copy()
            new.layers[i]['weights'] = layer['weights'].copy()
            new.layers[i]['bias'] = layer['bias'].copy()
        new.output_weights = self.output_weights.copy()
        new.output_bias = self.output_bias.copy()
        return new

    def num_parameters(self):
        """Count trainable parameters."""
        count = 0
        for layer in self.layers:
            count += layer['weights'].size + layer['bias'].size
        count += self.output_weights.size + self.output_bias.size
        return count


def load_digits_data():
    data = load_digits()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def run_gsa(X_train, y_onehot, X_test, y_test, hidden_sizes, k, pop_size=50, generations=200):
    """Run GSA with deep network."""

    population = []
    for i in range(pop_size):
        np.random.seed(i * 10)
        controller = DeepUltraSparseController(
            input_size=64, hidden_sizes=hidden_sizes, output_size=10, inputs_per_neuron=k
        )
        with torch.no_grad():
            pred = controller.forward(X_train)
            fitness = -torch.mean((pred - y_onehot) ** 2).item()
        population.append((controller, fitness))

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        n_seeds = max(1, int(0.05 * pop_size))

        fitnesses = np.array([f for _, f in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, 20)
            new_population.append((improved, new_fitness))

        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, 20)
            new_population.append((improved, new_fitness))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        temperature *= decay

        if gen % 50 == 0:
            with torch.no_grad():
                preds = best_controller.forward(X_test).argmax(dim=1)
                acc = (preds == y_test).float().mean().item()
            print(f"    Gen {gen}: {acc:.1%}")

    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller.num_parameters()


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_fitness = -torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_fitness = -torch.mean((pred - y_onehot) ** 2).item()

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def main():
    print("=" * 70)
    print("DEEP NETWORKS FOR DIGITS (2D SPATIAL REASONING)")
    print("=" * 70)
    print("\nHypothesis: 2D spatial data needs hierarchical representations")
    print("Digits are 8x8 images - depth may matter more than width\n")

    X_train, X_test, y_train, y_test = load_digits_data()

    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Input: 64 features (8x8 flattened), 10 classes\n")

    # Test different depths
    # Keep total params roughly similar by adjusting H
    configs = [
        # (hidden_sizes, k, description)
        ([64], 16, "1 layer (baseline)"),
        ([48, 48], 12, "2 layers"),
        ([64, 64], 16, "2 layers (wider)"),
        ([32, 32, 32], 8, "3 layers"),
        ([64, 64, 64], 16, "3 layers (wider)"),
    ]

    results = []

    for hidden_sizes, k, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {desc}")
        print(f"Architecture: 64 → {' → '.join(map(str, hidden_sizes))} → 10, K={k}")
        print("=" * 60)

        start = time.time()
        accuracy, params = run_gsa(X_train, y_onehot, X_test, y_test,
                                   hidden_sizes=hidden_sizes, k=k,
                                   pop_size=50, generations=200)
        elapsed = time.time() - start

        results.append({
            'desc': desc,
            'hidden_sizes': hidden_sizes,
            'k': k,
            'accuracy': accuracy,
            'params': params,
            'time': elapsed,
            'depth': len(hidden_sizes)
        })

        print(f"\nResult: {accuracy:.1%} accuracy, {params} params, {elapsed:.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does depth help for 2D spatial reasoning?")
    print("=" * 70)
    print(f"\n{'Architecture':<30} | {'Depth':<6} | {'Params':<8} | {'Accuracy':<10}")
    print("-" * 65)

    for r in results:
        arch = f"64→{'→'.join(map(str, r['hidden_sizes']))}→10"
        marker = " ✅" if r['accuracy'] > 0.9 else ""
        print(f"{arch:<30} | {r['depth']:<6} | {r['params']:<8} | {r['accuracy']:<10.1%}{marker}")

    print("-" * 65)
    print("Dense backprop: 97.0% accuracy, 8970 params")

    # Compare by depth
    print("\n" + "=" * 70)
    print("DEPTH COMPARISON (best per depth)")
    print("=" * 70)
    for depth in [1, 2, 3]:
        depth_results = [r for r in results if r['depth'] == depth]
        if depth_results:
            best = max(depth_results, key=lambda x: x['accuracy'])
            print(f"{depth} layer(s): {best['accuracy']:.1%} (config: {best['desc']})")


if __name__ == "__main__":
    main()
