"""
Hill Climbing with Random Restarts for Sine Approximation

A simpler alternative to genetic algorithms for gradient-free optimization.
Reuses SineController and saturation tracking from the main experiment.

Algorithm:
    for each restart:
        net = random_network()
        for step in range(max_steps):
            mutant = mutate(net)
            if fitness(mutant) > fitness(net):
                net = mutant
        track best_ever across restarts
"""

import torch
import numpy as np
import time
import json
import os
from pathlib import Path

import sine_config as cfg
from sine_controller import SineController

# Use CPU by default for simplicity, override with CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(controller: SineController, x_test: torch.Tensor, y_true: torch.Tensor) -> float:
    """Compute MSE fitness (lower is better, so we return negative MSE)."""
    with torch.no_grad():
        y_pred = controller.forward(x_test, track=True)
        mse = torch.mean((y_pred - y_true) ** 2).item()
    return -mse  # Negative because we maximize fitness


def hill_climb(
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_steps: int = 2000,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    patience: int = 200,
    verbose: bool = True,
) -> tuple[SineController, list[dict]]:
    """
    Single hill climbing run from a random start.

    Args:
        x_test: Test inputs
        y_true: True sine values
        max_steps: Maximum mutation attempts
        mutation_rate: Fraction of weights to mutate
        mutation_scale: Std dev of mutation noise
        patience: Stop if no improvement for this many steps
        verbose: Print progress

    Returns:
        best_controller: The best network found
        history: List of dicts with step, mse, k_ratio
    """
    # Initialize random network
    current = SineController(device=DEVICE)
    current_fitness = evaluate(current, x_test, y_true)

    best = current.clone()
    best_fitness = current_fitness

    history = []
    steps_without_improvement = 0

    for step in range(max_steps):
        # Create mutant
        mutant = current.clone()
        mutant.mutate(rate=mutation_rate, scale=mutation_scale)

        # Evaluate
        mutant_fitness = evaluate(mutant, x_test, y_true)

        # Accept if better
        if mutant_fitness > current_fitness:
            current = mutant
            current_fitness = mutant_fitness
            steps_without_improvement = 0

            # Track global best
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
        else:
            steps_without_improvement += 1

        # Log periodically
        if step % 50 == 0 or step == max_steps - 1:
            mse = -current_fitness
            k = current.get_k()
            k_ratio = k / cfg.HIDDEN_SIZE

            history.append({
                'step': step,
                'mse': mse,
                'k': k,
                'k_ratio': k_ratio,
            })

            if verbose:
                print(f"  Step {step:5d} | MSE={mse:.6f} | k={k}/{cfg.HIDDEN_SIZE} | k_ratio={k_ratio:.2f}")

        # Early stopping
        if steps_without_improvement >= patience:
            if verbose:
                print(f"  -> Converged at step {step} (no improvement for {patience} steps)")
            break

    return best, history


def run_experiment(
    num_restarts: int = 10,
    steps_per_restart: int = 2000,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    patience: int = 200,
    noise_dims: int = None,
    hidden_size: int = None,
    seed: int = 42,
    output_dir: str = None,
):
    """
    Run hill climbing with multiple random restarts.

    Args:
        num_restarts: Number of independent restarts
        steps_per_restart: Max steps per restart
        mutation_rate: Fraction of weights to mutate
        mutation_scale: Std dev of mutation noise
        patience: Early stopping patience per restart
        noise_dims: Override cfg.NOISE_SIGNAL_SIZE
        hidden_size: Override cfg.HIDDEN_SIZE
        seed: Random seed
        output_dir: Directory to save results
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Override config if specified
    if noise_dims is not None:
        cfg.NOISE_SIGNAL_SIZE = noise_dims
        cfg.EXPANSION_SIZE = cfg.TRUE_SIGNAL_SIZE + noise_dims
    if hidden_size is not None:
        cfg.HIDDEN_SIZE = hidden_size

    print("=" * 60)
    print("HILL CLIMBING EXPERIMENT")
    print("=" * 60)
    print(f"  Restarts:        {num_restarts}")
    print(f"  Steps/restart:   {steps_per_restart}")
    print(f"  Patience:        {patience}")
    print(f"  Mutation rate:   {mutation_rate}")
    print(f"  Mutation scale:  {mutation_scale}")
    print(f"  True signals:    {cfg.TRUE_SIGNAL_SIZE}")
    print(f"  Noise signals:   {cfg.NOISE_SIGNAL_SIZE}")
    print(f"  Hidden size:     {cfg.HIDDEN_SIZE}")
    print(f"  Compression:     {cfg.EXPANSION_SIZE}:{cfg.HIDDEN_SIZE}")
    print(f"  Device:          {DEVICE}")
    print("=" * 60)

    # Create test data
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, cfg.NUM_TEST_POINTS, device=DEVICE)
    y_true = torch.sin(x_test)

    # Track results across restarts
    best_ever = None
    best_ever_fitness = float('-inf')
    all_results = []

    start_time = time.time()

    for restart in range(num_restarts):
        print(f"\n[Restart {restart + 1}/{num_restarts}]")

        best, history = hill_climb(
            x_test, y_true,
            max_steps=steps_per_restart,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
            patience=patience,
            verbose=True,
        )

        fitness = evaluate(best, x_test, y_true)
        mse = -fitness
        k = best.get_k()
        k_ratio = k / cfg.HIDDEN_SIZE

        all_results.append({
            'restart': restart,
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

    # Final results
    final_mse = -best_ever_fitness
    final_k = best_ever.get_k()
    final_k_ratio = final_k / cfg.HIDDEN_SIZE

    all_mses = [r['final_mse'] for r in all_results]
    all_k_ratios = [r['final_k_ratio'] for r in all_results]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Best MSE:        {final_mse:.6f}")
    print(f"  Best k:          {final_k}/{cfg.HIDDEN_SIZE} ({final_k_ratio:.1%})")
    print(f"  Mean MSE:        {np.mean(all_mses):.6f} ± {np.std(all_mses):.6f}")
    print(f"  Mean k_ratio:    {np.mean(all_k_ratios):.1%} ± {np.std(all_k_ratios):.1%}")
    print(f"  Time:            {elapsed:.1f}s")
    print("=" * 60)

    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = {
            'method': 'hill_climbing',
            'num_restarts': num_restarts,
            'steps_per_restart': steps_per_restart,
            'patience': patience,
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

        with open(os.path.join(output_dir, 'hillclimb_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return best_ever, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hill climbing for sine approximation")
    parser.add_argument("--restarts", type=int, default=10, help="Number of random restarts")
    parser.add_argument("--steps", type=int, default=2000, help="Max steps per restart")
    parser.add_argument("--patience", type=int, default=200, help="Early stopping patience")
    parser.add_argument("--noise", type=int, default=None, help="Override noise dimensions")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/hillclimb_{timestamp}"

    run_experiment(
        num_restarts=args.restarts,
        steps_per_restart=args.steps,
        patience=args.patience,
        noise_dims=args.noise,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=args.output,
    )
