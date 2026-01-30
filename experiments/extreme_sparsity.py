"""
Experiment: Extreme Sparsity (K=1, K=2)

Test whether even more constrained architectures can work with extended training.
Hypothesis: With K=1 or K=2, each neuron becomes a specialist on 1-2 inputs.
This is the "ant brain" limit - maximum selection pressure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
from datetime import datetime
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Progressive logging setup
LOG_DIR = Path(__file__).parent.parent / "results" / "extreme_sparsity"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    """Print and write to log file with flush."""
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


class ExtremeSparseController:
    """Ultra-sparse controller with K inputs per hidden neuron."""

    def __init__(self, H, K, input_dim=64, output_dim=10):
        self.H = H
        self.K = K
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Hidden layer: each neuron connects to K random inputs
        self.hidden_indices = np.array([
            np.random.choice(input_dim, size=K, replace=False)
            for _ in range(H)
        ])
        self.hidden_weights = np.random.randn(H, K).astype(np.float32) * 0.5
        self.hidden_bias = np.zeros(H, dtype=np.float32)

        # Output layer: fully connected from hidden
        self.output_weights = np.random.randn(output_dim, H).astype(np.float32) * 0.5
        self.output_bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        """Forward pass with sparse connectivity."""
        if isinstance(x, torch.Tensor):
            x_np = x.numpy()
        else:
            x_np = x

        batch_size = x_np.shape[0]

        # Hidden layer with sparse inputs
        hidden = np.zeros((batch_size, self.H), dtype=np.float32)
        for i in range(self.H):
            selected = x_np[:, self.hidden_indices[i]]  # (batch, K)
            hidden[:, i] = selected @ self.hidden_weights[i] + self.hidden_bias[i]
        hidden = np.tanh(hidden)

        # Output layer
        output = hidden @ self.output_weights.T + self.output_bias
        output = np.tanh(output)

        return torch.tensor(output, dtype=torch.float32)

    def mutate(self, index_swap_rate=0.15, weight_rate=0.2):
        """Mutate weights and potentially swap input indices."""
        # Weight mutations
        for i in range(self.H):
            if np.random.random() < weight_rate:
                idx = np.random.randint(self.K)
                self.hidden_weights[i, idx] += np.random.randn() * 0.3

            if np.random.random() < weight_rate * 0.5:
                self.hidden_bias[i] += np.random.randn() * 0.2

        # Output weight mutations
        if np.random.random() < weight_rate:
            i, j = np.random.randint(self.output_dim), np.random.randint(self.H)
            self.output_weights[i, j] += np.random.randn() * 0.3

        # Index swaps - the key mechanism for feature discovery
        for i in range(self.H):
            if np.random.random() < index_swap_rate:
                old_idx = np.random.randint(self.K)
                # Find a new input not currently used by this neuron
                available = list(set(range(self.input_dim)) - set(self.hidden_indices[i]))
                if available:
                    new_input = np.random.choice(available)
                    self.hidden_indices[i, old_idx] = new_input
                    # Reset the weight for the new connection
                    self.hidden_weights[i, old_idx] = np.random.randn() * 0.3

    def clone(self):
        """Create a deep copy."""
        c = ExtremeSparseController(self.H, self.K, self.input_dim, self.output_dim)
        c.hidden_indices = self.hidden_indices.copy()
        c.hidden_weights = self.hidden_weights.copy()
        c.hidden_bias = self.hidden_bias.copy()
        c.output_weights = self.output_weights.copy()
        c.output_bias = self.output_bias.copy()
        return c

    def num_params(self):
        """Count parameters."""
        hidden_params = self.H * self.K + self.H  # weights + bias
        output_params = self.output_dim * self.H + self.output_dim
        return hidden_params + output_params


def load_data():
    """Load and preprocess digits dataset."""
    data = load_digits()
    X = StandardScaler().fit_transform(data.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data.target, test_size=0.2, random_state=42
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def run_gsa(H, K, X_train, X_test, y_train, y_test, y_onehot,
            generations=500, pop_size=50, seed=42, verbose=True):
    """Run GSA with extreme sparsity."""

    np.random.seed(seed)

    # Initialize population
    population = []
    for i in range(pop_size):
        c = ExtremeSparseController(H, K)
        with torch.no_grad():
            f = -torch.mean((c.forward(X_train) - y_onehot) ** 2).item()
        population.append((c, f))

    best = population[0][0].clone()
    best_f = population[0][1]

    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    history = []

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        n_elite = max(1, pop_size // 20)
        new_pop = [(c.clone(), f) for c, f in population[:n_elite]]

        probs = np.array([f for _, f in population])
        probs = probs - probs.min() + 1e-8
        probs = probs / probs.sum()

        for _ in range(pop_size - n_elite):
            idx = np.random.choice(len(population), p=probs)
            c = population[idx][0].clone()
            current_f = population[idx][1]
            best_c, best_inner_f = c, current_f

            # 20 SA steps per member
            for _ in range(20):
                mutant = c.clone()
                mutant.mutate()
                with torch.no_grad():
                    f = -torch.mean((mutant.forward(X_train) - y_onehot) ** 2).item()
                delta = f - current_f
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    c = mutant
                    current_f = f
                    if f > best_inner_f:
                        best_c, best_inner_f = mutant.clone(), f

            new_pop.append((best_c, best_inner_f))

        population = new_pop

        # Track best
        gb = max(population, key=lambda x: x[1])
        if gb[1] > best_f:
            best, best_f = gb[0].clone(), gb[1]

        temp *= decay

        # Checkpoint
        if gen % 100 == 0 or gen == generations - 1:
            with torch.no_grad():
                acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()
            history.append((gen, acc))
            if verbose:
                log(f"    Gen {gen}: {acc:.1%}")

    # Final accuracy
    with torch.no_grad():
        final_acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()

    return final_acc, best.num_params(), history, best


def analyze_feature_usage(controller):
    """Analyze which input features are being used."""
    all_indices = controller.hidden_indices.flatten()
    unique, counts = np.unique(all_indices, return_counts=True)

    # Sort by usage count
    sorted_idx = np.argsort(-counts)
    top_features = [(unique[i], counts[i]) for i in sorted_idx[:10]]

    return {
        'unique_features': len(unique),
        'total_connections': len(all_indices),
        'top_10': top_features,
        'coverage': len(unique) / 64 * 100  # % of input features used
    }


def main():
    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 65)
    log("EXTREME SPARSITY TEST: K=1 and K=2")
    log("=" * 65)

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    log(f"Data: {len(X_train)} train, {len(X_test)} test")
    log(f"Extended training: 500 generations, pop=50")

    configs = [
        {"H": 32, "K": 1, "name": "Extreme (K=1)"},
        {"H": 32, "K": 2, "name": "Very Sparse (K=2)"},
        {"H": 64, "K": 1, "name": "Wide Extreme (H=64, K=1)"},
        {"H": 64, "K": 2, "name": "Wide Sparse (H=64, K=2)"},
    ]

    all_results = []

    for config in configs:
        H, K = config["H"], config["K"]
        log(f"\n{'='*65}")
        log(f"Config: {config['name']} (H={H}, K={K})")
        log(f"{'='*65}")

        results = []
        for trial in range(2):  # 2 trials per config
            log(f"\n  Trial {trial + 1}/2 (seed={trial * 1000}):")
            start = time.time()

            acc, params, history, best_model = run_gsa(
                H=H, K=K,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                y_onehot=y_onehot,
                generations=500,
                pop_size=50,
                seed=trial * 1000,
                verbose=True
            )

            elapsed = time.time() - start
            feature_analysis = analyze_feature_usage(best_model)

            results.append({
                'trial': trial + 1,
                'seed': trial * 1000,
                'accuracy': acc,
                'params': params,
                'time': elapsed,
                'history': history,
                'feature_analysis': feature_analysis
            })

            log(f"    → {acc:.1%} accuracy, {params} params, {elapsed:.0f}s")
            log(f"    → Using {feature_analysis['unique_features']}/64 features ({feature_analysis['coverage']:.0f}% coverage)")

        accs = [r['accuracy'] for r in results]
        all_results.append({
            'config': config,
            'H': H,
            'K': K,
            'params': results[0]['params'],
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'best_acc': max(accs),
            'trials': results
        })

    # Summary
    log("\n" + "=" * 65)
    log("SUMMARY")
    log("=" * 65)

    log(f"\n{'Config':<25} {'Params':>8} {'Mean':>8} {'Best':>8} {'Coverage':>10}")
    log("-" * 65)
    for r in all_results:
        coverage = np.mean([t['feature_analysis']['coverage'] for t in r['trials']])
        log(f"{r['config']['name']:<25} {r['params']:>8} {r['mean_acc']:>7.1%} {r['best_acc']:>7.1%} {coverage:>9.0f}%")

    # Compare to baseline
    log("\n" + "-" * 65)
    log("Comparison to H=32, K=4 baseline (85.6% with 490 params):")
    for r in all_results:
        diff = r['mean_acc'] - 0.856
        log(f"  {r['config']['name']}: {diff:+.1%} ({r['params']} params)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "extreme_sparsity"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    log(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
