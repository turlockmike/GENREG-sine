"""
Experiment: Narrow-Deep vs Wide-Shallow (Controlled Comparison)

Question: Can depth compensate for width at a fixed parameter budget?

Hypothesis: Deeper networks may enable better feature composition through
hierarchical representations, potentially improving accuracy on digits.

Design (controlled experiment):
- Independent variable: Depth (L=1, 2, 3)
- Controlled variables: K=4 (fixed), ~490 params (matched), GSA settings
- Dependent variable: Test accuracy

Configs at ~490 params with K=4:
- L=1, H=32: 32*4 + 32 + 10*32 + 10 = 490 params (baseline)
- L=2, H=24: 24*4 + 24 + 24*4 + 24 + 10*24 + 10 = 490 params
- L=3, H=18: 18*4 + 18 + 18*4 + 18 + 18*4 + 18 + 10*18 + 10 = 460 params

Success criteria: Any deep config beats shallow baseline (84.7% from extended training).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress matmul warnings
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


def run_gsa(X_train, y_onehot, X_test, y_test, hidden_sizes, k, pop_size=50, generations=200, csv_path=None):
    """Run GSA with deep network."""
    start_time = time.time()

    # Initialize CSV for dashboard
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s\n")

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

        # Calculate accuracy and log
        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        elapsed = time.time() - start_time

        # Write to CSV for dashboard
        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f}\n")
                f.flush()

        if gen % 50 == 0:
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
    print("NARROW-DEEP vs WIDE-SHALLOW (Controlled Comparison)")
    print("=" * 70)
    print("\nQuestion: Can depth compensate for width at fixed param budget?")
    print("Control: K=4 fixed, ~490 params matched, only depth varies\n")

    X_train, X_test, y_train, y_test = load_digits_data()

    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Input: 64 features (8x8 flattened), 10 classes\n")

    # Controlled comparison: K=4 throughout, ~490 params matched
    # Only varying depth (L=1, 2, 3)
    K = 4  # Fixed across all configs
    configs = [
        # (hidden_sizes, k, description)
        ([32], K, "L=1, H=32 (490 params) - BASELINE"),
        ([24, 24], K, "L=2, H=24 (490 params)"),
        ([18, 18, 18], K, "L=3, H=18 (460 params)"),
    ]

    results = []

    for hidden_sizes, k, desc in configs:
        print(f"\n{'='*60}")
        print(f"Config: {desc}")
        print(f"Architecture: 64 → {' → '.join(map(str, hidden_sizes))} → 10, K={k}")
        print("=" * 60)

        # Dashboard CSV path
        depth = len(hidden_sizes)
        h = hidden_sizes[0]
        csv_path = Path("results/live") / f"deep_L{depth}_H{h}_K{k}.csv"

        start = time.time()
        accuracy, params = run_gsa(X_train, y_onehot, X_test, y_test,
                                   hidden_sizes=hidden_sizes, k=k,
                                   pop_size=50, generations=500,
                                   csv_path=csv_path)
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
    print("RESULTS: Does depth help at fixed param budget?")
    print("=" * 70)
    print(f"\n{'Config':<35} | {'Params':<8} | {'Accuracy':<10}")
    print("-" * 60)

    baseline_acc = None
    for r in results:
        arch = f"64→{'→'.join(map(str, r['hidden_sizes']))}→10 (K={r['k']})"
        if r['depth'] == 1:
            baseline_acc = r['accuracy']
            marker = " (baseline)"
        elif baseline_acc and r['accuracy'] > baseline_acc:
            marker = " ✅ BETTER"
        elif baseline_acc and r['accuracy'] < baseline_acc:
            marker = f" ({(r['accuracy'] - baseline_acc)*100:+.1f}pp)"
        else:
            marker = ""
        print(f"{arch:<35} | {r['params']:<8} | {r['accuracy']:<.1%}{marker}")

    print("-" * 60)
    print("Extended training baseline (H=32, K=4, L=1): 84.7%")
    print("Dense backprop: 97.0% accuracy, 8970 params")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    best = max(results, key=lambda x: x['accuracy'])
    shallow = [r for r in results if r['depth'] == 1][0]
    if best['depth'] > 1:
        print(f"✅ Depth HELPS: L={best['depth']} ({best['accuracy']:.1%}) > L=1 ({shallow['accuracy']:.1%})")
    else:
        print(f"❌ Depth does NOT help: L=1 ({shallow['accuracy']:.1%}) is best")
        deep_results = [r for r in results if r['depth'] > 1]
        if deep_results:
            worst_deep = min(deep_results, key=lambda x: x['accuracy'])
            print(f"   Deeper networks hurt: L={worst_deep['depth']} = {worst_deep['accuracy']:.1%}")


if __name__ == "__main__":
    main()
