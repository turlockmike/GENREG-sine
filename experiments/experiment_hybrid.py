"""
Experiment 3: Hybrid SA + Local Search (Memetic Algorithm)

Idea: Use simulated annealing for global exploration to find good regions,
then switch to hill climbing for local exploitation/fine-tuning.

This combines:
- SA's ability to escape local minima (exploration)
- Hill climbing's efficiency at converging (exploitation)
"""

import torch
import numpy as np
import json
import time
from pathlib import Path

import legacy.sine_config as sine_config as cfg
from legacy.sine_controller import SineController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(controller, x_test, y_true):
    """Compute negative MSE (higher is better)."""
    with torch.no_grad():
        pred = controller.forward(x_test, track=True)
        mse = torch.mean((pred - y_true) ** 2).item()
    return -mse


def simulated_annealing_phase(x_test, y_true, steps, t_initial, t_final):
    """SA exploration phase."""
    current = SineController(device=DEVICE)
    current_fitness = evaluate(current, x_test, y_true)

    best = current.clone()
    best_fitness = current_fitness

    decay = (t_final / t_initial) ** (1.0 / steps)

    for step in range(steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)
        mutant_fitness = evaluate(mutant, x_test, y_true)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, -best_fitness


def hill_climbing_phase(controller, x_test, y_true, steps, mutation_rate=0.05, mutation_scale=0.05):
    """Local search phase with smaller mutations."""
    current = controller.clone()
    current_fitness = evaluate(current, x_test, y_true)

    best = current.clone()
    best_fitness = current_fitness

    no_improve = 0
    patience = steps // 5  # Early stopping

    for step in range(steps):
        mutant = current.clone()
        mutant.mutate(rate=mutation_rate, scale=mutation_scale)
        mutant_fitness = evaluate(mutant, x_test, y_true)

        if mutant_fitness > current_fitness:
            current = mutant
            current_fitness = mutant_fitness
            no_improve = 0
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best, -best_fitness, step + 1


def run_hybrid(x_test, y_true, sa_steps, hc_steps, t_initial=0.01, t_final=0.0001, verbose=True):
    """Run hybrid SA + hill climbing."""
    # Phase 1: SA exploration
    if verbose:
        print("  Phase 1: Simulated Annealing (exploration)")
    sa_best, sa_mse = simulated_annealing_phase(x_test, y_true, sa_steps, t_initial, t_final)
    if verbose:
        print(f"    -> MSE after SA: {sa_mse:.6f}")

    # Phase 2: Hill climbing refinement
    if verbose:
        print("  Phase 2: Hill Climbing (refinement)")
    final_best, final_mse, hc_actual = hill_climbing_phase(sa_best, x_test, y_true, hc_steps)
    if verbose:
        print(f"    -> MSE after HC: {final_mse:.6f} (ran {hc_actual} steps)")

    return final_best, final_mse, sa_mse


def run_experiment():
    """Compare hybrid vs pure methods."""
    print("=" * 70)
    print("EXPERIMENT 3: HYBRID SA + LOCAL SEARCH")
    print("=" * 70)
    print("Hypothesis: Combining SA exploration + HC refinement beats either alone")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    total_budget = 20000  # Total evaluations to compare fairly

    configs = [
        ("Pure SA", {'sa_steps': total_budget, 'hc_steps': 0}),
        ("Pure HC", {'sa_steps': 0, 'hc_steps': total_budget}),
        ("Hybrid 50/50", {'sa_steps': 10000, 'hc_steps': 10000}),
        ("Hybrid 70/30", {'sa_steps': 14000, 'hc_steps': 6000}),
        ("Hybrid 80/20", {'sa_steps': 16000, 'hc_steps': 4000}),
        ("Hybrid 90/10", {'sa_steps': 18000, 'hc_steps': 2000}),
    ]

    results = {}
    n_runs = 3

    for name, cfg_dict in configs:
        print(f"\n[{name}]")
        run_mses = []
        run_k_ratios = []

        for run in range(n_runs):
            np.random.seed(42 + run)
            torch.manual_seed(42 + run)

            if cfg_dict['sa_steps'] == 0:
                # Pure HC
                controller = SineController(device=DEVICE)
                best, mse, _ = hill_climbing_phase(controller, x_test, y_true, cfg_dict['hc_steps'])
            elif cfg_dict['hc_steps'] == 0:
                # Pure SA
                best, mse = simulated_annealing_phase(
                    x_test, y_true, cfg_dict['sa_steps'], 0.01, 0.00001
                )
            else:
                # Hybrid
                best, mse, _ = run_hybrid(
                    x_test, y_true,
                    cfg_dict['sa_steps'],
                    cfg_dict['hc_steps'],
                    verbose=(run == 0)
                )

            _ = best.forward(x_test, track=True)
            k_ratio = best.get_k() / cfg.HIDDEN_SIZE

            run_mses.append(mse)
            run_k_ratios.append(k_ratio)

        mean_mse = np.mean(run_mses)
        std_mse = np.std(run_mses)
        mean_k = np.mean(run_k_ratios)

        results[name] = {
            'sa_steps': cfg_dict['sa_steps'],
            'hc_steps': cfg_dict['hc_steps'],
            'mean_mse': mean_mse,
            'std_mse': std_mse,
            'min_mse': min(run_mses),
            'mean_k_ratio': mean_k,
            'all_mses': run_mses,
        }

        print(f"  MSE: {mean_mse:.6f} ± {std_mse:.6f} (best: {min(run_mses):.6f}), k={mean_k:.0%}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} | {'SA Steps':>10} | {'HC Steps':>10} | {'Mean MSE':>12} | {'Best MSE':>12}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_mse'])

    for name, r in sorted_results:
        print(f"{name:<20} | {r['sa_steps']:>10} | {r['hc_steps']:>10} | {r['mean_mse']:>12.6f} | {r['min_mse']:>12.6f}")

    winner = sorted_results[0][0]
    print(f"\n  >> WINNER: {winner}")

    if "Hybrid" in winner:
        print("  >> HYPOTHESIS SUPPORTED: Hybrid approach beats pure methods")
    else:
        print("  >> HYPOTHESIS NOT SUPPORTED: Pure method wins")

    results['summary'] = {
        'winner': winner,
        'total_budget': total_budget,
        'n_runs': n_runs,
        'hypothesis_supported': "Hybrid" in winner,
    }

    # Save results
    output_dir = Path("results/hybrid_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
