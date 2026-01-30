"""
Experiment: Extended Training on Winner Config

Run the winning configuration (H=32, K=4) with more generations
to see how far we can push accuracy.
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

from experiments.constraint_boundary import MinimalController


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
            generations=300, pop_size=50, seed=42, verbose=True):
    """Run GSA with proper seeding."""

    np.random.seed(seed)

    # Initialize population
    population = []
    for i in range(pop_size):
        c = MinimalController(H, K)
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
        if gen % 50 == 0 or gen == generations - 1:
            with torch.no_grad():
                acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()
            history.append((gen, acc))
            if verbose:
                print(f"    Gen {gen}: {acc:.1%}")

    # Final accuracy
    with torch.no_grad():
        final_acc = (best.forward(X_test).argmax(dim=1) == y_test).float().mean().item()

    return final_acc, best.num_params(), history


def main():
    print("=" * 65)
    print("EXTENDED TRAINING: H=32, K=4")
    print("=" * 65)

    # Load data
    X_train, X_test, y_train, y_test = load_data()
    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"\nData: {len(X_train)} train, {len(X_test)} test")
    print(f"Config: H=32, K=4, 300 gens, pop=50")
    print(f"Running 3 trials with different seeds...\n")

    results = []

    for trial in range(3):
        print(f"Trial {trial + 1}/3 (seed={trial * 1000}):")
        start = time.time()

        acc, params, history = run_gsa(
            H=32, K=4,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            y_onehot=y_onehot,
            generations=300,
            pop_size=50,
            seed=trial * 1000,
            verbose=True
        )

        elapsed = time.time() - start
        results.append({
            'trial': trial + 1,
            'seed': trial * 1000,
            'accuracy': acc,
            'params': params,
            'time': elapsed,
            'history': history
        })

        print(f"    → {acc:.1%} accuracy, {elapsed:.0f}s\n")

    # Summary
    accs = [r['accuracy'] for r in results]

    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"\nConfig: H=32, K=4 (490 params)")
    print(f"\nTrial results:")
    for r in results:
        print(f"  Trial {r['trial']}: {r['accuracy']:.1%}")

    print(f"\nMean: {np.mean(accs):.1%}")
    print(f"Std:  {np.std(accs):.1%}")
    print(f"Best: {max(accs):.1%}")
    print(f"Worst: {min(accs):.1%}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "extended_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
