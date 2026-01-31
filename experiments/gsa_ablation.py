"""
Ablation: GSA vs Naive Random Restarts

Question: Is GSA's population + selection actually better than just trying
multiple random initializations?

Comparison:
1. Random Restarts: Run N independent SA chains, pick best at end
2. GSA (10 gens): Population with selection for only 10 generations
3. GSA (full): Population with selection for 300 generations

This tests whether selection pressure matters or if it's just "try more things"
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
    """Load and preprocess digits dataset."""
    data = load_digits()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.long),
    )


def run_single_sa(X_train, y_onehot, X_test, y_test,
                  hidden_size=64, k=16, sa_steps=6000, seed=0):
    """Run a single SA chain."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    controller = UltraSparseController(
        input_size=64,
        hidden_size=hidden_size,
        output_size=10,
        inputs_per_neuron=k
    )

    current = controller
    with torch.no_grad():
        pred = current.forward(X_train)
        current_mse = torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / sa_steps)

    for step in range(sa_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1)

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_mse = torch.mean((pred - y_onehot) ** 2).item()

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

    # Evaluate
    with torch.no_grad():
        preds = best.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return best, accuracy, best_mse


def run_random_restarts(X_train, y_onehot, X_test, y_test,
                        n_restarts=10, sa_steps_each=6000):
    """Run N independent SA chains, return best."""
    print(f"\nRandom Restarts: {n_restarts} chains x {sa_steps_each} steps each", flush=True)
    print(f"Total SA steps: {n_restarts * sa_steps_each}", flush=True)

    best_controller = None
    best_accuracy = 0
    all_accuracies = []

    start = time.time()
    for i in range(n_restarts):
        controller, accuracy, mse = run_single_sa(
            X_train, y_onehot, X_test, y_test,
            sa_steps=sa_steps_each, seed=i*100
        )
        all_accuracies.append(accuracy)
        print(f"  Chain {i+1}: {accuracy:.1%}", flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_controller = controller

    elapsed = time.time() - start

    return {
        'method': f'Random Restarts ({n_restarts}x{sa_steps_each})',
        'best_accuracy': best_accuracy,
        'mean_accuracy': np.mean(all_accuracies),
        'std_accuracy': np.std(all_accuracies),
        'time': elapsed,
        'total_sa_steps': n_restarts * sa_steps_each
    }


def run_gsa(X_train, y_onehot, X_test, y_test,
            population_size=50, generations=10, sa_steps_per_gen=20):
    """Run GSA with population + selection."""
    print(f"\nGSA: Pop={population_size}, Gens={generations}, SA/gen={sa_steps_per_gen}", flush=True)
    print(f"Total SA steps: ~{population_size * generations * sa_steps_per_gen}", flush=True)

    # Initialize population
    population = []
    for i in range(population_size):
        np.random.seed(i * 10)
        torch.manual_seed(i * 10)
        controller = UltraSparseController(
            input_size=64, hidden_size=64, output_size=10, inputs_per_neuron=16
        )
        with torch.no_grad():
            pred = controller.forward(X_train)
            fitness = -torch.mean((pred - y_onehot) ** 2).item()
        population.append((controller, fitness))

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / max(generations, 1))
    temperature = t_initial

    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    start = time.time()

    for gen in range(generations):
        # Sort by fitness
        population.sort(key=lambda x: x[1], reverse=True)

        # Selection: top 5% seeds + roulette for rest
        n_seeds = max(1, int(0.05 * population_size))

        # Roulette selection
        fitnesses = np.array([f for _, f in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        # Seeds do SA
        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, sa_steps_per_gen)
            new_population.append((improved, new_fitness))

        # Roulette selected do SA
        indices = np.random.choice(len(population), size=population_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, sa_steps_per_gen)
            new_population.append((improved, new_fitness))

        population = new_population

        # Track best
        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        # Progress logging every 10 generations
        if (gen + 1) % 10 == 0 or gen == 0:
            print(f"    Gen {gen+1:3d}/{generations}: best_fitness={best_fitness:.6f}", flush=True)

        temperature *= decay

    elapsed = time.time() - start

    # Evaluate
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return {
        'method': f'GSA (Pop={population_size}, Gens={generations})',
        'best_accuracy': accuracy,
        'time': elapsed,
        'total_sa_steps': population_size * generations * sa_steps_per_gen
    }


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps):
    """Run n SA steps on a controller."""
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
    print("=" * 70, flush=True)
    print("GSA vs RANDOM RESTARTS ABLATION", flush=True)
    print("=" * 70, flush=True)
    print("\nQuestion: Is GSA's selection pressure helping, or is it just", flush=True)
    print("          the benefit of trying multiple random starts?", flush=True)

    # Load data
    X_train, X_test, y_train, y_test = load_digits_data()

    # One-hot encode
    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    print(f"\nData: {len(X_train)} train, {len(X_test)} test", flush=True)

    results = []

    # Matched compute budget: ~60,000 total SA steps

    # 1. Random restarts: 10 chains x 6000 steps = 60,000 steps
    result = run_random_restarts(X_train, y_onehot, X_test, y_test,
                                  n_restarts=10, sa_steps_each=6000)
    results.append(result)
    print(f"  Best: {result['best_accuracy']:.1%}, Mean: {result['mean_accuracy']:.1%}", flush=True)

    # 2. GSA 10 generations: 50 pop x 10 gen x 20 steps = 10,000 steps (less compute)
    result = run_gsa(X_train, y_onehot, X_test, y_test,
                     population_size=50, generations=10, sa_steps_per_gen=20)
    results.append(result)
    print(f"  Accuracy: {result['best_accuracy']:.1%}", flush=True)

    # 3. GSA matched compute: 50 pop x 60 gen x 20 steps = 60,000 steps
    result = run_gsa(X_train, y_onehot, X_test, y_test,
                     population_size=50, generations=60, sa_steps_per_gen=20)
    results.append(result)
    print(f"  Accuracy: {result['best_accuracy']:.1%}", flush=True)

    # 4. Random restarts with more chains: 50 x 1200 = 60,000 steps
    result = run_random_restarts(X_train, y_onehot, X_test, y_test,
                                  n_restarts=50, sa_steps_each=1200)
    results.append(result)
    print(f"  Best: {result['best_accuracy']:.1%}, Mean: {result['mean_accuracy']:.1%}", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'Method':<45} | {'Accuracy':<10} | {'SA Steps':<12} | {'Time':<8}", flush=True)
    print("-" * 85, flush=True)

    for r in results:
        acc = f"{r['best_accuracy']:.1%}"
        if 'mean_accuracy' in r:
            acc += f" (mean {r['mean_accuracy']:.1%})"
        print(f"{r['method']:<45} | {acc:<20} | {r['total_sa_steps']:<12} | {r['time']:.1f}s", flush=True)

    print("-" * 85, flush=True)
    print("\nConclusion: ", end="", flush=True)

    gsa_acc = results[2]['best_accuracy']  # GSA 60 gens
    restart_acc = results[0]['best_accuracy']  # Random 10x6000

    if gsa_acc > restart_acc + 0.05:
        print(f"GSA's selection pressure helps! (+{(gsa_acc-restart_acc)*100:.1f}pp)", flush=True)
    elif gsa_acc < restart_acc - 0.05:
        print(f"Random restarts are better! (+{(restart_acc-gsa_acc)*100:.1f}pp)", flush=True)
    else:
        print("Similar performance - selection pressure has marginal effect", flush=True)


if __name__ == "__main__":
    main()
