"""
Experiment: Constraint Boundary Test

Question: Where is the sweet spot? At what point does too much constraint hurt?

Tests configs from very constrained (H=16, K=4) to less constrained (H=64, K=16)
to find the optimal selection pressure.
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


class MinimalController:
    """Minimal ultra-sparse controller for fast testing."""

    def __init__(self, H, K):
        self.H, self.K = H, K
        self.indices = np.random.randint(0, 64, (H, K))
        self.weights = np.random.randn(H, K).astype(np.float32) * 0.5
        self.bias = np.zeros(H, dtype=np.float32)
        self.out_w = np.random.randn(10, H).astype(np.float32) * 0.5
        self.out_b = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        gathered = x[:, self.indices]
        h = np.tanh(np.einsum('bnk,nk->bn', gathered, self.weights) + self.bias)
        return torch.from_numpy(np.tanh(h @ self.out_w.T + self.out_b))

    def mutate(self, wr=0.1, isr=0.2):
        mask = np.random.random(self.weights.shape) < wr
        self.weights += mask * np.random.randn(*self.weights.shape).astype(np.float32) * wr
        mask = np.random.random(self.bias.shape) < wr
        self.bias += mask * np.random.randn(*self.bias.shape).astype(np.float32) * wr
        for h in range(self.H):
            if np.random.random() < isr:
                self.indices[h, np.random.randint(0, self.K)] = np.random.randint(0, 64)
        mask = np.random.random(self.out_w.shape) < wr
        self.out_w += mask * np.random.randn(*self.out_w.shape).astype(np.float32) * wr

    def clone(self):
        c = MinimalController(self.H, self.K)
        c.indices = self.indices.copy()
        c.weights = self.weights.copy()
        c.bias = self.bias.copy()
        c.out_w = self.out_w.copy()
        c.out_b = self.out_b.copy()
        return c

    def num_params(self):
        return self.weights.size + self.bias.size + self.out_w.size + self.out_b.size


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
            generations=150, pop_size=40, verbose=True):
    """Run GSA for a given config."""

    np.random.seed(42)

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 100)
        c = MinimalController(H, K)
        with torch.no_grad():
            f = -torch.mean((c.forward(X_train) - y_onehot) ** 2).item()
        population.append((c, f))

    best = population[0][0].clone()
    best_f = population[0][1]

    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    checkpoints = []

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        # Selection
        n_elite = max(1, pop_size // 20)
        new_pop = [(c.clone(), f) for c, f in population[:n_elite]]

        probs = np.array([f for _, f in population])
        probs = probs - probs.min() + 1e-8
        probs = probs / probs.sum()

        for _ in range(pop_size - n_elite):
            idx = np.random.choice(len(population), p=probs)
            c = population[idx][0].clone()
            current_f = population[idx][1]

            # Run 20 SA steps (matching arch_search)
            best_c, best_f = c, current_f
            for _ in range(20):
                mutant = c.clone()
                mutant.mutate()
                with torch.no_grad():
                    f = -torch.mean((mutant.forward(X_train) - y_onehot) ** 2).item()
                delta = f - current_f
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    c = mutant
                    current_f = f
                    if f > best_f:
                        best_c, best_f = mutant.clone(), f

            new_pop.append((best_c, best_f))

        population = new_pop

        # Track best
        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_f:
            best = gen_best[0].clone()
            best_f = gen_best[1]

        temp *= decay

        # Checkpoint
        if gen % 50 == 0 or gen == generations - 1:
            with torch.no_grad():
                acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()
            checkpoints.append((gen, acc))
            if verbose:
                print(f"    Gen {gen}: {acc:.1%}")

    # Final accuracy
    with torch.no_grad():
        final_acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()

    return final_acc, best.num_params(), checkpoints


def main():
    print("=" * 65)
    print("CONSTRAINT BOUNDARY TEST")
    print("=" * 65)
    print("\nQuestion: Where is the sweet spot for selection pressure?")
    print("Too constrained → not enough capacity")
    print("Too unconstrained → no selection pressure\n")

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"Data: {len(X_train)} train, {len(X_test)} test\n")

    # Configs from very constrained to less constrained
    configs = [
        (16, 4, "Very constrained"),
        (16, 8, "Constrained"),
        (32, 4, "Medium-tight"),
        (32, 8, "WINNER (from search)"),
        (32, 16, "Medium-loose"),
        (64, 8, "Large-sparse"),
        (64, 16, "Large-medium"),
        (64, 32, "Large-loose"),
    ]

    results = []

    for H, K, desc in configs:
        print(f"\n{'-'*60}")
        print(f"Config: {desc} (H={H}, K={K})")
        print("-" * 60)

        start = time.time()
        acc, params, checkpoints = run_gsa(
            H, K, X_train, X_test, y_train, y_test, y_onehot,
            generations=150, pop_size=40
        )
        elapsed = time.time() - start

        results.append({
            'desc': desc,
            'H': H,
            'K': K,
            'params': params,
            'accuracy': acc,
            'time': elapsed,
            'checkpoints': checkpoints
        })

        print(f"  → {acc:.1%} accuracy, {params} params, {elapsed:.0f}s")

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\n{'Config':<25} | {'H':<4} | {'K':<4} | {'Params':<8} | {'Accuracy':<10} | {'Eff'}")
    print("-" * 75)

    for r in results:
        efficiency = r['accuracy'] / (r['params'] / 100)  # acc per 100 params
        marker = " ←" if "WINNER" in r['desc'] else ""
        print(f"{r['desc']:<25} | {r['H']:<4} | {r['K']:<4} | {r['params']:<8} | "
              f"{r['accuracy']:<10.1%} | {efficiency:.2f}%/100p{marker}")

    print("-" * 75)

    # Find sweet spot
    best = max(results, key=lambda x: x['accuracy'])
    most_efficient = max(results, key=lambda x: x['accuracy'] / x['params'])

    print(f"\nBest accuracy: {best['desc']} → {best['accuracy']:.1%}")
    print(f"Most efficient: {most_efficient['desc']} → {most_efficient['accuracy']:.1%} "
          f"({most_efficient['params']} params)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "constraint_boundary"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
