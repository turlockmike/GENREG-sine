"""
Simulated Annealing for Sine Approximation

Like hill climbing, but accepts worse solutions with decreasing probability.
This allows escaping local minima early, then converging as temperature cools.

Acceptance probability for worse solution:
    p = exp((new_fitness - old_fitness) / temperature)

Temperature schedule:
    T(t) = T_initial * (1 - t/max_steps)  [linear cooling]
    or
    T(t) = T_initial * 0.99^t             [exponential cooling]
"""

import torch
import numpy as np
import math
import time
import json
import os
from pathlib import Path

from . import sine_config as cfg
from sine_controller import SineController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(controller: SineController, x_test: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute MSE fitness (lower is better, so we return negative MSE)."""
    with torch.no_grad():
        y_pred = controller.forward(x_test, track=True)
        mse = torch.mean((y_pred - y_true) ** 2).item()
    return -mse


def simulated_annealing(
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_steps: int = 5000,
    t_initial: float = 0.1,
    t_final: float = 0.001,
    cooling: str = "exponential",
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    verbose: bool = True,
) -> tuple[SineController, list[dict]]:
    """
    Simulated annealing from a random start.

    Args:
        x_test: Test inputs
        y_true: True sine values
        max_steps: Total steps
        t_initial: Starting temperature
        t_final: Ending temperature
        cooling: "linear" or "exponential"
        mutation_rate: Fraction of weights to mutate
        mutation_scale: Std dev of mutation noise
        verbose: Print progress

    Returns:
        best_controller: The best network found
        history: List of dicts with step, mse, k_ratio, temperature, acceptance_rate
    """
    # Initialize
    current = SineController(device=DEVICE)
    current_fitness = evaluate(current, x_test, y_true)

    best = current.clone()
    best_fitness = current_fitness

    # For exponential cooling: T(t) = T_initial * decay^t
    # Solve for decay: T_final = T_initial * decay^max_steps
    if cooling == "exponential":
        decay = (t_final / t_initial) ** (1.0 / max_steps)

    history = []
    accepts_worse = 0
    accepts_total = 0
    window_accepts = []

    for step in range(max_steps):
        # Compute temperature
        if cooling == "linear":
            progress = step / max_steps
            temperature = t_initial * (1 - progress) + t_final * progress
        else:  # exponential
            temperature = t_initial * (decay ** step)

        # Create mutant
        mutant = current.clone()
        mutant.mutate(rate=mutation_rate, scale=mutation_scale)
        mutant_fitness = evaluate(mutant, x_test, y_true)

        # Acceptance decision
        delta = mutant_fitness - current_fitness

        if delta > 0:
            # Better solution - always accept
            accept = True
        else:
            # Worse solution - accept with probability
            p_accept = math.exp(delta / temperature) if temperature > 0 else 0
            accept = np.random.random() < p_accept
            if accept:
                accepts_worse += 1

        if accept:
            current = mutant
            current_fitness = mutant_fitness
            accepts_total += 1
            window_accepts.append(1)

            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
        else:
            window_accepts.append(0)

        # Keep window of last 100 for acceptance rate
        if len(window_accepts) > 100:
            window_accepts.pop(0)

        # Log periodically
        if step % 100 == 0 or step == max_steps - 1:
            mse = -current_fitness
            best_mse = -best_fitness
            k = current.get_k()
            k_ratio = k / cfg.HIDDEN_SIZE
            accept_rate = sum(window_accepts) / len(window_accepts) if window_accepts else 0

            history.append({
                'step': step,
                'mse': mse,
                'best_mse': best_mse,
                'k': k,
                'k_ratio': k_ratio,
                'temperature': temperature,
                'accept_rate': accept_rate,
            })

            if verbose:
                print(f"  Step {step:5d} | MSE={mse:.6f} | best={best_mse:.6f} | "
                      f"k={k}/{cfg.HIDDEN_SIZE} | T={temperature:.5f} | accept={accept_rate:.0%}")

    if verbose:
        print(f"  -> Accepted {accepts_worse} worse solutions out of {max_steps} steps")

    return best, history


def run_experiment(
    num_runs: int = 5,
    max_steps: int = 5000,
    t_initial: float = 0.1,
    t_final: float = 0.001,
    cooling: str = "exponential",
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    noise_dims: int = None,
    hidden_size: int = None,
    seed: int = 42,
    output_dir: str = None,
):
    """Run simulated annealing with multiple runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if noise_dims is not None:
        cfg.NOISE_SIGNAL_SIZE = noise_dims
        cfg.EXPANSION_SIZE = cfg.TRUE_SIGNAL_SIZE + noise_dims
    if hidden_size is not None:
        cfg.HIDDEN_SIZE = hidden_size

    print("=" * 70)
    print("SIMULATED ANNEALING EXPERIMENT")
    print("=" * 70)
    print(f"  Runs:            {num_runs}")
    print(f"  Steps/run:       {max_steps}")
    print(f"  T_initial:       {t_initial}")
    print(f"  T_final:         {t_final}")
    print(f"  Cooling:         {cooling}")
    print(f"  Mutation rate:   {mutation_rate}")
    print(f"  Mutation scale:  {mutation_scale}")
    print(f"  True signals:    {cfg.TRUE_SIGNAL_SIZE}")
    print(f"  Noise signals:   {cfg.NOISE_SIGNAL_SIZE}")
    print(f"  Hidden size:     {cfg.HIDDEN_SIZE}")
    print(f"  Compression:     {cfg.EXPANSION_SIZE}:{cfg.HIDDEN_SIZE}")
    print(f"  Device:          {DEVICE}")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, cfg.NUM_TEST_POINTS, device=DEVICE)
    y_true = torch.sin(x_test)

    best_ever = None
    best_ever_fitness = float('-inf')
    all_results = []

    start_time = time.time()

    for run in range(num_runs):
        print(f"\n[Run {run + 1}/{num_runs}]")

        # Reseed each run differently but reproducibly
        torch.manual_seed(seed + run)
        np.random.seed(seed + run)

        best, history = simulated_annealing(
            x_test, y_true,
            max_steps=max_steps,
            t_initial=t_initial,
            t_final=t_final,
            cooling=cooling,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            verbose=True,
        )

        fitness = evaluate(best, x_test, y_true)
        mse = -fitness
        k = best.get_k()
        k_ratio = k / cfg.HIDDEN_SIZE

        all_results.append({
            'run': run,
            'final_mse': mse,
            'final_k': k,
            'final_k_ratio': k_ratio,
            'history': history,
        })

        if fitness > best_ever_fitness:
            best_ever = best.clone()
            best_ever_fitness = fitness
            print(f"  *** New best! MSE={mse:.6f}, k={k}")

    elapsed = time.time() - start_time

    final_mse = -best_ever_fitness
    final_k = best_ever.get_k()
    final_k_ratio = final_k / cfg.HIDDEN_SIZE

    all_mses = [r['final_mse'] for r in all_results]
    all_k_ratios = [r['final_k_ratio'] for r in all_results]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best MSE:        {final_mse:.6f}")
    print(f"  Best k:          {final_k}/{cfg.HIDDEN_SIZE} ({final_k_ratio:.1%})")
    print(f"  Mean MSE:        {np.mean(all_mses):.6f} ± {np.std(all_mses):.6f}")
    print(f"  Mean k_ratio:    {np.mean(all_k_ratios):.1%} ± {np.std(all_k_ratios):.1%}")
    print(f"  Time:            {elapsed:.1f}s")
    print("=" * 70)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = {
            'method': 'simulated_annealing',
            'num_runs': num_runs,
            'max_steps': max_steps,
            't_initial': t_initial,
            't_final': t_final,
            'cooling': cooling,
            'mutation_rate': mutation_rate,
            'mutation_scale': mutation_scale,
            'true_signal_size': cfg.TRUE_SIGNAL_SIZE,
            'noise_signal_size': cfg.NOISE_SIGNAL_SIZE,
            'hidden_size': cfg.HIDDEN_SIZE,
            'compression_ratio': cfg.EXPANSION_SIZE / cfg.HIDDEN_SIZE,
            'seed': seed,
            'best_mse': final_mse,
            'best_k': final_k,
            'best_k_ratio': final_k_ratio,
            'mean_mse': float(np.mean(all_mses)),
            'std_mse': float(np.std(all_mses)),
            'mean_k_ratio': float(np.mean(all_k_ratios)),
            'std_k_ratio': float(np.std(all_k_ratios)),
            'elapsed_seconds': elapsed,
            'all_results': all_results,
        }

        with open(os.path.join(output_dir, 'annealing_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return best_ever, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulated annealing for sine approximation")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--steps", type=int, default=5000, help="Steps per run")
    parser.add_argument("--t-initial", type=float, default=0.1, help="Initial temperature")
    parser.add_argument("--t-final", type=float, default=0.001, help="Final temperature")
    parser.add_argument("--cooling", type=str, default="exponential", choices=["linear", "exponential"])
    parser.add_argument("--noise", type=int, default=None, help="Override noise dimensions")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/annealing_{timestamp}"

    run_experiment(
        num_runs=args.runs,
        max_steps=args.steps,
        t_initial=args.t_initial,
        t_final=args.t_final,
        cooling=args.cooling,
        noise_dims=args.noise,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=args.output,
    )
