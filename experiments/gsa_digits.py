"""
Experiment: Genetic Simulated Annealing (GSA) for Digits Classification

Goal: Solve 10-class digits problem (>90% accuracy) where single-chain SA fails (64.7%)

Algorithm from Du et al. (2018):
1. Population of controllers (100)
2. Natural Selection:
   - Seed selection: Keep best 5% unchanged
   - Roulette selection: Probabilistic for remaining 95%
3. Mutation with Monte Carlo acceptance
4. Temperature cooling

Key insight: Multiple chains explore different index combinations simultaneously,
with selection pressure favoring chains that find useful features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from dataclasses import dataclass

from core.models import UltraSparseController


@dataclass
class GSAConfig:
    """Configuration for Genetic Simulated Annealing."""
    population_size: int = 50  # Smaller than paper's 100 for speed
    generations: int = 200
    seed_fraction: float = 0.05  # Top 5% kept unchanged
    mutation_rate: float = 0.15  # Per-gene mutation probability
    weight_scale: float = 0.15
    index_swap_rate: float = 0.1
    t_initial: float = 0.1
    t_final: float = 0.0001


def load_and_prep_digits():
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


def evaluate_controller(controller: UltraSparseController,
                       X: torch.Tensor,
                       y_onehot: torch.Tensor) -> float:
    """Evaluate a controller's fitness (negative MSE, higher is better)."""
    with torch.no_grad():
        pred = controller.forward(X)
        mse = torch.mean((pred - y_onehot) ** 2).item()
    return -mse  # Negative because we want to maximize fitness


def roulette_select(population: List[Tuple[UltraSparseController, float]],
                    n_select: int) -> List[UltraSparseController]:
    """Roulette wheel selection based on fitness."""
    # Shift fitness to be positive
    fitnesses = np.array([f for _, f in population])
    min_fitness = fitnesses.min()
    shifted = fitnesses - min_fitness + 1e-8

    # Compute probabilities
    probs = shifted / shifted.sum()

    # Select indices with replacement
    indices = np.random.choice(len(population), size=n_select, p=probs, replace=True)

    return [population[i][0].clone() for i in indices]


def mutate_with_monte_carlo(controller: UltraSparseController,
                            X: torch.Tensor,
                            y_onehot: torch.Tensor,
                            temperature: float,
                            config: GSAConfig) -> UltraSparseController:
    """Mutate controller with Monte Carlo acceptance for each gene."""
    # Clone to avoid modifying original
    mutant = controller.clone()

    # Get current fitness
    current_fitness = evaluate_controller(mutant, X, y_onehot)

    # Mutate the controller
    mutant.mutate(
        weight_rate=config.mutation_rate,
        weight_scale=config.weight_scale,
        index_swap_rate=config.index_swap_rate
    )

    # Get new fitness
    new_fitness = evaluate_controller(mutant, X, y_onehot)

    # Monte Carlo acceptance
    delta = new_fitness - current_fitness
    if delta > 0 or np.random.random() < np.exp(delta / temperature):
        return mutant
    else:
        return controller.clone()


def run_sa_step(controller: UltraSparseController,
                X: torch.Tensor,
                y_onehot: torch.Tensor,
                temperature: float,
                config: GSAConfig,
                n_steps: int = 10) -> Tuple[UltraSparseController, float]:
    """Run multiple SA steps on a single controller."""
    current = controller.clone()
    current_fitness = evaluate_controller(current, X, y_onehot)

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(
            weight_rate=config.mutation_rate,
            weight_scale=config.weight_scale,
            index_swap_rate=config.index_swap_rate
        )
        mutant_fitness = evaluate_controller(mutant, X, y_onehot)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def run_gsa(hidden_size: int = 64,
            inputs_per_neuron: int = 16,
            config: GSAConfig = None,
            verbose: bool = True) -> dict:
    """Run GSA training on digits dataset."""

    if config is None:
        config = GSAConfig()

    # Load data
    X_train, X_test, y_train, y_test = load_and_prep_digits()

    input_size = 64
    num_classes = 10

    # One-hot encode targets
    y_onehot = torch.zeros(len(y_train), num_classes)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8  # Scale for tanh

    if verbose:
        print(f"\nGSA Configuration:")
        print(f"  Population: {config.population_size}")
        print(f"  Generations: {config.generations}")
        print(f"  Network: H={hidden_size}, K={inputs_per_neuron}")
        print(f"  Seed fraction: {config.seed_fraction:.0%}")
        print()

    # Initialize population
    np.random.seed(42)
    torch.manual_seed(42)

    population = []
    for i in range(config.population_size):
        controller = UltraSparseController(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            inputs_per_neuron=inputs_per_neuron
        )
        fitness = evaluate_controller(controller, X_train, y_onehot)
        population.append((controller, fitness))

    # Temperature schedule
    decay = (config.t_final / config.t_initial) ** (1.0 / config.generations)
    temperature = config.t_initial

    # Track best
    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    # Track history
    history = {
        'best_fitness': [],
        'mean_fitness': [],
        'best_accuracy': []
    }

    start_time = time.time()

    # SA steps per generation (more local refinement)
    sa_steps_per_gen = 20

    for gen in range(config.generations):
        # Sort by fitness (descending)
        population.sort(key=lambda x: x[1], reverse=True)

        # Natural Selection
        n_seeds = max(1, int(config.seed_fraction * config.population_size))
        n_roulette = config.population_size - n_seeds

        # Seeds: Keep top performers but still do SA on them
        seeds = []
        for c, _ in population[:n_seeds]:
            improved, new_fitness = run_sa_step(
                c, X_train, y_onehot, temperature, config, sa_steps_per_gen
            )
            seeds.append((improved, new_fitness))

        # Roulette: Select from full population, then do SA
        roulette_controllers = roulette_select(population, n_roulette)

        new_population = seeds.copy()
        for controller in roulette_controllers:
            improved, new_fitness = run_sa_step(
                controller, X_train, y_onehot, temperature, config, sa_steps_per_gen
            )
            new_population.append((improved, new_fitness))

        population = new_population

        # Update best
        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        # Cool temperature
        temperature *= decay

        # Track metrics
        mean_fitness = np.mean([f for _, f in population])
        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(mean_fitness)

        # Evaluate accuracy periodically
        if gen % 20 == 0 or gen == config.generations - 1:
            with torch.no_grad():
                preds = best_controller.forward(X_test).argmax(dim=1)
                accuracy = (preds == y_test).float().mean().item()
            history['best_accuracy'].append((gen, accuracy))

            if verbose:
                print(f"Gen {gen:3d}: Best fitness={best_fitness:.4f}, "
                      f"Mean={mean_fitness:.4f}, Accuracy={accuracy:.1%}, T={temperature:.6f}")

    train_time = time.time() - start_time

    # Final evaluation
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        final_accuracy = (preds == y_test).float().mean().item()

        train_preds = best_controller.forward(X_train).argmax(dim=1)
        train_accuracy = (train_preds == y_train).float().mean().item()

    params = best_controller.num_parameters()

    if verbose:
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_accuracy:.1%}")
        print(f"Test Accuracy:  {final_accuracy:.1%}")
        print(f"Parameters:     {params}")
        print(f"Train Time:     {train_time:.1f}s")
        print(f"{'='*60}")

    return {
        'hidden_size': hidden_size,
        'inputs_per_neuron': inputs_per_neuron,
        'population_size': config.population_size,
        'generations': config.generations,
        'train_accuracy': train_accuracy,
        'test_accuracy': final_accuracy,
        'params': params,
        'train_time': train_time,
        'history': history
    }


def run_gsa_sweep():
    """Run GSA with different configurations to find best setup."""

    print("=" * 70)
    print("GSA PARAMETER SWEEP FOR DIGITS")
    print("=" * 70)
    print("\nGoal: Find GSA config that achieves >90% accuracy")
    print("Single SA baseline: 64.7% accuracy")
    print("Dense backprop: 97% accuracy, 8970 params")
    print()

    configs = [
        # (hidden_size, k, population, generations)
        (32, 8, 30, 100),   # Small, fast test
        (64, 16, 50, 150),  # Medium
        (64, 16, 100, 200), # Large population
        (128, 16, 50, 200), # Larger network
    ]

    results = []

    for hidden, k, pop, gens in configs:
        print(f"\n{'='*60}")
        print(f"Config: H={hidden}, K={k}, Pop={pop}, Gens={gens}")
        print("=" * 60)

        config = GSAConfig(
            population_size=pop,
            generations=gens
        )

        result = run_gsa(
            hidden_size=hidden,
            inputs_per_neuron=k,
            config=config,
            verbose=True
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'H':<6} | {'K':<4} | {'Pop':<5} | {'Gens':<5} | {'Accuracy':<10} | {'Params':<8} | {'vs Dense'}")
    print("-" * 70)

    for r in results:
        vs_dense = f"{8970/r['params']:.0f}x fewer"
        marker = " ✅" if r['test_accuracy'] > 0.9 else ""
        print(f"{r['hidden_size']:<6} | {r['inputs_per_neuron']:<4} | {r['population_size']:<5} | "
              f"{r['generations']:<5} | {r['test_accuracy']:<10.1%} | {r['params']:<8} | {vs_dense}{marker}")

    print("-" * 70)
    print("Single SA: 64.7%, Dense backprop: 97.0%")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "gsa_digits"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert history to JSON-serializable format
    json_results = []
    for r in results:
        jr = r.copy()
        jr['history'] = {
            'best_fitness': list(r['history']['best_fitness']),
            'mean_fitness': list(r['history']['mean_fitness']),
            'best_accuracy': list(r['history']['best_accuracy'])
        }
        json_results.append(jr)

    with open(output_dir / "results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GSA for Digits Classification")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--k", type=int, default=16, help="Inputs per neuron")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--gens", type=int, default=200, help="Generations")

    args = parser.parse_args()

    if args.sweep:
        run_gsa_sweep()
    else:
        config = GSAConfig(
            population_size=args.pop,
            generations=args.gens
        )
        run_gsa(
            hidden_size=args.hidden,
            inputs_per_neuron=args.k,
            config=config,
            verbose=True
        )
