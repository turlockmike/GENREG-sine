"""
Experiment: Extended Training for GSA on Digits

Question: Does extended training (5000 generations) push GSA digits accuracy
past 90%, or does it plateau?

Hypothesis: The current 87% ceiling is due to insufficient training time,
not architecture limits. Longer training will show continued improvement,
or reveal a clear plateau indicating the architecture's limit.

Variables:
- Independent: Number of generations (5000), random seed
- Dependent: Test accuracy, learning curve shape
- Controlled: Architecture, dataset (digits), SA steps per gen

Comparison:
- Baseline: 87.2% at 300 gens (H=64, K=16, Pop=50)
- Compare H=32/K=4 vs H=64/K=16 with extended training

Success Criteria:
- >90% accuracy achieved, OR
- Clear plateau identified (proves architecture limit)

Configs:
- H=32, K=4, Pop=50 (490 params) - "sweet spot" from ablation
- H=64, K=16, Pop=50 (1738 params) - current best accuracy

3 seeds per config, 5000 generations each, log every generation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
import argparse
from datetime import datetime
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

from core.models import UltraSparseController


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
    return -mse


def roulette_select(population: List[Tuple[UltraSparseController, float]],
                    n_select: int) -> List[UltraSparseController]:
    """Roulette wheel selection based on fitness."""
    fitnesses = np.array([f for _, f in population])
    min_fitness = fitnesses.min()
    shifted = fitnesses - min_fitness + 1e-8
    probs = shifted / shifted.sum()
    indices = np.random.choice(len(population), size=n_select, p=probs, replace=True)
    return [population[i][0].clone() for i in indices]


def run_sa_step(controller: UltraSparseController,
                X: torch.Tensor,
                y_onehot: torch.Tensor,
                temperature: float,
                n_steps: int = 20) -> Tuple[UltraSparseController, float]:
    """Run multiple SA steps on a single controller."""
    current = controller.clone()
    current_fitness = evaluate_controller(current, X, y_onehot)

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(
            weight_rate=0.15,
            weight_scale=0.15,
            index_swap_rate=0.1
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


def run_extended_training(hidden_size: int,
                          inputs_per_neuron: int,
                          seed: int,
                          generations: int,
                          pop_size: int,
                          output_file: Path) -> dict:
    """Run extended GSA training with per-generation logging."""

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    X_train, X_test, y_train, y_test = load_and_prep_digits()

    input_size = 64
    num_classes = 10

    # One-hot encode targets
    y_onehot = torch.zeros(len(y_train), num_classes)
    y_onehot.scatter_(1, y_train.unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8  # Scale for tanh

    print(f"\n{'='*60}")
    print(f"Extended Training: H={hidden_size}, K={inputs_per_neuron}, Seed={seed}")
    print(f"Generations: {generations}, Population: {pop_size}")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")

    # Initialize population
    population = []
    for _ in range(pop_size):
        controller = UltraSparseController(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            inputs_per_neuron=inputs_per_neuron
        )
        fitness = evaluate_controller(controller, X_train, y_onehot)
        population.append((controller, fitness))

    # Temperature schedule
    t_initial = 0.1
    t_final = 0.0001
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    # Track best
    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    # Open output file for streaming writes
    with open(output_file, 'w') as f:
        # Write header
        f.write("gen,best_fitness,mean_fitness,test_accuracy,temperature,elapsed_s\n")

        start_time = time.time()
        sa_steps_per_gen = 20
        seed_fraction = 0.05

        for gen in range(generations):
            # Sort by fitness
            population.sort(key=lambda x: x[1], reverse=True)

            # Natural Selection
            n_seeds = max(1, int(seed_fraction * pop_size))
            n_roulette = pop_size - n_seeds

            # Seeds with SA
            seeds = []
            for c, _ in population[:n_seeds]:
                improved, new_fitness = run_sa_step(
                    c, X_train, y_onehot, temperature, sa_steps_per_gen
                )
                seeds.append((improved, new_fitness))

            # Roulette selection with SA
            roulette_controllers = roulette_select(population, n_roulette)

            new_population = seeds.copy()
            for controller in roulette_controllers:
                improved, new_fitness = run_sa_step(
                    controller, X_train, y_onehot, temperature, sa_steps_per_gen
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

            # Calculate metrics
            mean_fitness = np.mean([fit for _, fit in population])
            elapsed = time.time() - start_time

            # Evaluate test accuracy
            with torch.no_grad():
                preds = best_controller.forward(X_test).argmax(dim=1)
                test_accuracy = (preds == y_test).float().mean().item()

            # Write to file (every generation)
            f.write(f"{gen},{best_fitness:.6f},{mean_fitness:.6f},{test_accuracy:.4f},{temperature:.8f},{elapsed:.1f}\n")
            f.flush()  # Ensure data is written

            # Print progress periodically
            if gen % 100 == 0 or gen == generations - 1:
                print(f"Gen {gen:4d}: Acc={test_accuracy:.1%}, Fitness={best_fitness:.4f}, T={temperature:.6f}, {elapsed:.0f}s")

    # Final evaluation
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        final_accuracy = (preds == y_test).float().mean().item()

        train_preds = best_controller.forward(X_train).argmax(dim=1)
        train_accuracy = (train_preds == y_train).float().mean().item()

    total_time = time.time() - start_time
    params = best_controller.num_parameters()

    print(f"\n{'='*60}")
    print(f"FINAL: H={hidden_size}, K={inputs_per_neuron}, Seed={seed}")
    print(f"Train Accuracy: {train_accuracy:.1%}")
    print(f"Test Accuracy:  {final_accuracy:.1%}")
    print(f"Parameters:     {params}")
    print(f"Total Time:     {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}\n")

    return {
        'hidden_size': hidden_size,
        'inputs_per_neuron': inputs_per_neuron,
        'seed': seed,
        'generations': generations,
        'population_size': pop_size,
        'train_accuracy': train_accuracy,
        'test_accuracy': final_accuracy,
        'params': params,
        'total_time': total_time,
        'output_file': str(output_file)
    }


def main():
    parser = argparse.ArgumentParser(description="Extended GSA Training")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden size")
    parser.add_argument("--k", type=int, default=4, help="Inputs per neuron")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gens", type=int, default=5000, help="Generations")
    parser.add_argument("--pop", type=int, default=50, help="Population size")
    parser.add_argument("--all", action="store_true", help="Run all configs (2 configs × 3 seeds)")

    args = parser.parse_args()

    # Output directory
    output_dir = Path(__file__).parent.parent / "results" / "extended_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # Run all configurations
        configs = [
            (32, 4),   # Sweet spot config
            (64, 16),  # Current best accuracy config
        ]
        seeds = [42, 123, 456]

        all_results = []

        for hidden, k in configs:
            for seed in seeds:
                output_file = output_dir / f"H{hidden}_K{k}_seed{seed}.csv"
                result = run_extended_training(
                    hidden, k, seed, args.gens, args.pop, output_file
                )
                all_results.append(result)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: Extended Training (5000 generations)")
        print("=" * 70)
        print(f"\n{'Config':<12} | {'Seed':<6} | {'Test Acc':<10} | {'Params':<8} | {'Time'}")
        print("-" * 70)

        for r in all_results:
            config_str = f"H{r['hidden_size']}/K{r['inputs_per_neuron']}"
            print(f"{config_str:<12} | {r['seed']:<6} | {r['test_accuracy']:<10.1%} | {r['params']:<8} | {r['total_time']/60:.1f}min")

        # Aggregate by config
        print("\n" + "-" * 70)
        print("Aggregated by config:")
        for hidden, k in configs:
            config_results = [r for r in all_results if r['hidden_size'] == hidden and r['inputs_per_neuron'] == k]
            accs = [r['test_accuracy'] for r in config_results]
            print(f"  H={hidden}, K={k}: {np.mean(accs):.1%} +/- {np.std(accs):.1%} (n={len(accs)})")

        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    else:
        # Single run
        output_file = output_dir / f"H{args.hidden}_K{args.k}_seed{args.seed}.csv"
        run_extended_training(
            args.hidden, args.k, args.seed, args.gens, args.pop, output_file
        )


if __name__ == "__main__":
    main()
