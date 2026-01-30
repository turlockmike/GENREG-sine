"""
Experiment: Hidden Dimension Scaling for Digits

Hypothesis: Harder problems (like 10-class digits) need more hidden neurons.
"16 is enough for locomotion, but for spatial reasoning you need more."

Test: Sweep hidden sizes [32, 64, 128, 256] with GSA
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

from core.models import UltraSparseController


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


def run_gsa(X_train, y_onehot, X_test, y_test, hidden_size, k, pop_size=50, generations=200):
    """Run GSA with given config."""

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10)
        torch.manual_seed(i * 10)
        controller = UltraSparseController(
            input_size=64, hidden_size=hidden_size, output_size=10, inputs_per_neuron=k
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

        # Roulette selection
        fitnesses = np.array([f for _, f in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        # Seeds do SA
        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, 20)
            new_population.append((improved, new_fitness))

        # Roulette selected do SA
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

    # Final eval
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
    print("HIDDEN DIMENSION SWEEP FOR DIGITS")
    print("=" * 70)
    print("\nHypothesis: Harder problems need more hidden neurons")
    print("'16 is enough for locomotion, spatial reasoning needs more'\n")

    X_train, X_test, y_train, y_test = load_digits_data()

    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"Data: {len(X_train)} train, {len(X_test)} test, 64 features, 10 classes\n")

    # Test different hidden sizes
    # K scales with H to maintain reasonable connectivity
    configs = [
        (32, 8),    # Small: 32 hidden, 8 inputs each
        (64, 16),   # Medium: 64 hidden, 16 inputs each (previous best)
        (128, 16),  # Large: 128 hidden, 16 inputs each
        (256, 16),  # XL: 256 hidden, 16 inputs each
        (128, 32),  # Large + more connectivity
    ]

    results = []

    for hidden, k in configs:
        print(f"\n{'='*60}")
        print(f"Config: H={hidden}, K={k}")
        print("=" * 60)

        start = time.time()
        accuracy, params = run_gsa(X_train, y_onehot, X_test, y_test,
                                   hidden_size=hidden, k=k, pop_size=50, generations=200)
        elapsed = time.time() - start

        results.append({
            'hidden': hidden,
            'k': k,
            'accuracy': accuracy,
            'params': params,
            'time': elapsed
        })

        print(f"\nResult: {accuracy:.1%} accuracy, {params} params, {elapsed:.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Does more hidden capacity help?")
    print("=" * 70)
    print(f"\n{'Hidden':<8} | {'K':<4} | {'Params':<8} | {'Accuracy':<10} | {'vs Dense'}")
    print("-" * 60)

    for r in results:
        vs_dense = f"{8970/r['params']:.1f}x fewer"
        marker = " ✅" if r['accuracy'] > 0.9 else ""
        print(f"{r['hidden']:<8} | {r['k']:<4} | {r['params']:<8} | {r['accuracy']:<10.1%} | {vs_dense}{marker}")

    print("-" * 60)
    print("Dense backprop: 97.0% accuracy, 8970 params")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest: H={best['hidden']}, K={best['k']} → {best['accuracy']:.1%}")


if __name__ == "__main__":
    main()
