"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Sine Approximation

CMA-ES learns the covariance structure of the fitness landscape and adapts
mutation directions accordingly. It's the gold standard for black-box
optimization of continuous functions.

Key idea: Instead of random mutations, CMA-ES mutates along directions
that have been successful in the past.
"""

import torch
import numpy as np
import cma
import time
import json
import os
from pathlib import Path

import sine_config as cfg
from sine_controller import SineController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def params_to_vector(controller: SineController) -> np.ndarray:
    """Flatten controller weights to a 1D numpy array."""
    params = [
        controller.w1.cpu().numpy().flatten(),
        controller.b1.cpu().numpy().flatten(),
        controller.w2.cpu().numpy().flatten(),
        controller.b2.cpu().numpy().flatten(),
    ]
    return np.concatenate(params)


def vector_to_controller(vec: np.ndarray, device=None) -> SineController:
    """Create controller from flattened parameter vector."""
    if device is None:
        device = DEVICE

    controller = SineController(device=device)

    # Calculate sizes
    w1_size = controller.w1.numel()
    b1_size = controller.b1.numel()
    w2_size = controller.w2.numel()
    b2_size = controller.b2.numel()

    # Unpack vector
    idx = 0
    controller.w1 = torch.tensor(
        vec[idx:idx+w1_size].reshape(controller.w1.shape),
        device=device, dtype=torch.float32
    )
    idx += w1_size

    controller.b1 = torch.tensor(
        vec[idx:idx+b1_size],
        device=device, dtype=torch.float32
    )
    idx += b1_size

    controller.w2 = torch.tensor(
        vec[idx:idx+w2_size].reshape(controller.w2.shape),
        device=device, dtype=torch.float32
    )
    idx += w2_size

    controller.b2 = torch.tensor(
        vec[idx:idx+b2_size],
        device=device, dtype=torch.float32
    )

    return controller


def make_fitness_function(x_test: torch.Tensor, y_true: torch.Tensor):
    """Create fitness function for CMA-ES (minimization)."""
    def fitness(vec: np.ndarray) -> float:
        controller = vector_to_controller(vec)
        with torch.no_grad():
            y_pred = controller.forward(x_test, track=False)
            mse = torch.mean((y_pred - y_true) ** 2).item()
        return mse  # CMA-ES minimizes, so return MSE directly
    return fitness


def run_cmaes(
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_evals: int = 10000,
    sigma0: float = 0.5,
    popsize: int = None,
    verbose: bool = True,
) -> tuple[SineController, list[dict]]:
    """
    Run CMA-ES optimization.

    Args:
        x_test: Test inputs
        y_true: True sine values
        max_evals: Maximum function evaluations
        sigma0: Initial step size (mutation strength)
        popsize: Population size (None = auto)
        verbose: Print progress

    Returns:
        best_controller: The best network found
        history: List of dicts with generation, mse, k_ratio
    """
    # Initialize from random controller
    init_controller = SineController(device=DEVICE)
    x0 = params_to_vector(init_controller)
    n_params = len(x0)

    if verbose:
        print(f"  Parameters: {n_params}")

    # Create fitness function
    fitness_fn = make_fitness_function(x_test, y_true)

    # CMA-ES options
    opts = {
        'maxfevals': max_evals,
        'verb_disp': 0,  # Quiet mode
        'verb_log': 0,
    }
    if popsize is not None:
        opts['popsize'] = popsize

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    history = []
    gen = 0

    while not es.stop():
        # Get candidate solutions
        solutions = es.ask()

        # Evaluate fitness
        fitnesses = [fitness_fn(s) for s in solutions]

        # Update CMA-ES
        es.tell(solutions, fitnesses)

        # Log progress
        if gen % 10 == 0 or es.stop():
            best_vec = es.result.xbest
            best_mse = es.result.fbest

            # Get saturation
            best_ctrl = vector_to_controller(best_vec)
            _ = best_ctrl.forward(x_test, track=True)  # Populate activations
            k = best_ctrl.get_k()
            k_ratio = k / cfg.HIDDEN_SIZE

            history.append({
                'generation': gen,
                'evaluations': es.result.evaluations,
                'mse': best_mse,
                'k': k,
                'k_ratio': k_ratio,
                'sigma': es.sigma,
            })

            if verbose:
                print(f"  Gen {gen:4d} | Evals {es.result.evaluations:5d} | "
                      f"MSE={best_mse:.6f} | k={k}/{cfg.HIDDEN_SIZE} | sigma={es.sigma:.4f}")

        gen += 1

    # Get final best
    best_controller = vector_to_controller(es.result.xbest)
    _ = best_controller.forward(x_test, track=True)

    if verbose:
        print(f"  -> Stopped: {es.stop()}")

    return best_controller, history


def run_experiment(
    num_runs: int = 3,
    max_evals: int = 10000,
    sigma0: float = 0.5,
    popsize: int = None,
    noise_dims: int = None,
    hidden_size: int = None,
    seed: int = 42,
    output_dir: str = None,
):
    """Run CMA-ES with multiple runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if noise_dims is not None:
        cfg.NOISE_SIGNAL_SIZE = noise_dims
        cfg.EXPANSION_SIZE = cfg.TRUE_SIGNAL_SIZE + noise_dims
    if hidden_size is not None:
        cfg.HIDDEN_SIZE = hidden_size

    print("=" * 70)
    print("CMA-ES EXPERIMENT")
    print("=" * 70)
    print(f"  Runs:            {num_runs}")
    print(f"  Max evaluations: {max_evals}")
    print(f"  Sigma0:          {sigma0}")
    print(f"  Population:      {popsize or 'auto'}")
    print(f"  True signals:    {cfg.TRUE_SIGNAL_SIZE}")
    print(f"  Noise signals:   {cfg.NOISE_SIGNAL_SIZE}")
    print(f"  Hidden size:     {cfg.HIDDEN_SIZE}")
    print(f"  Compression:     {cfg.EXPANSION_SIZE}:{cfg.HIDDEN_SIZE}")
    print(f"  Device:          {DEVICE}")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, cfg.NUM_TEST_POINTS, device=DEVICE)
    y_true = torch.sin(x_test)

    best_ever = None
    best_ever_mse = float('inf')
    all_results = []

    start_time = time.time()

    for run in range(num_runs):
        print(f"\n[Run {run + 1}/{num_runs}]")

        # Different seed per run
        np.random.seed(seed + run)
        torch.manual_seed(seed + run)

        best, history = run_cmaes(
            x_test, y_true,
            max_evals=max_evals,
            sigma0=sigma0,
            popsize=popsize,
            verbose=True,
        )

        # Final evaluation
        with torch.no_grad():
            y_pred = best.forward(x_test, track=True)
            mse = torch.mean((y_pred - y_true) ** 2).item()
        k = best.get_k()
        k_ratio = k / cfg.HIDDEN_SIZE

        all_results.append({
            'run': run,
            'final_mse': mse,
            'final_k': k,
            'final_k_ratio': k_ratio,
            'history': history,
        })

        if mse < best_ever_mse:
            best_ever = best
            best_ever_mse = mse
            print(f"  *** New best! MSE={mse:.6f}, k={k}")

    elapsed = time.time() - start_time

    # Final stats
    final_k = best_ever.get_k()
    final_k_ratio = final_k / cfg.HIDDEN_SIZE

    all_mses = [r['final_mse'] for r in all_results]
    all_k_ratios = [r['final_k_ratio'] for r in all_results]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Best MSE:        {best_ever_mse:.6f}")
    print(f"  Best k:          {final_k}/{cfg.HIDDEN_SIZE} ({final_k_ratio:.1%})")
    print(f"  Mean MSE:        {np.mean(all_mses):.6f} ± {np.std(all_mses):.6f}")
    print(f"  Mean k_ratio:    {np.mean(all_k_ratios):.1%} ± {np.std(all_k_ratios):.1%}")
    print(f"  Time:            {elapsed:.1f}s")
    print("=" * 70)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        summary = {
            'method': 'cma-es',
            'num_runs': num_runs,
            'max_evals': max_evals,
            'sigma0': sigma0,
            'popsize': popsize,
            'true_signal_size': cfg.TRUE_SIGNAL_SIZE,
            'noise_signal_size': cfg.NOISE_SIGNAL_SIZE,
            'hidden_size': cfg.HIDDEN_SIZE,
            'compression_ratio': cfg.EXPANSION_SIZE / cfg.HIDDEN_SIZE,
            'seed': seed,
            'best_mse': best_ever_mse,
            'best_k': final_k,
            'best_k_ratio': final_k_ratio,
            'mean_mse': float(np.mean(all_mses)),
            'std_mse': float(np.std(all_mses)),
            'mean_k_ratio': float(np.mean(all_k_ratios)),
            'std_k_ratio': float(np.std(all_k_ratios)),
            'elapsed_seconds': elapsed,
            'all_results': all_results,
        }

        with open(os.path.join(output_dir, 'cmaes_results.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return best_ever, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES for sine approximation")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--evals", type=int, default=10000, help="Max function evaluations")
    parser.add_argument("--sigma", type=float, default=0.5, help="Initial step size")
    parser.add_argument("--popsize", type=int, default=None, help="Population size")
    parser.add_argument("--noise", type=int, default=None, help="Override noise dimensions")
    parser.add_argument("--hidden", type=int, default=None, help="Override hidden size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"results/cmaes_{timestamp}"

    run_experiment(
        num_runs=args.runs,
        max_evals=args.evals,
        sigma0=args.sigma,
        popsize=args.popsize,
        noise_dims=args.noise,
        hidden_size=args.hidden,
        seed=args.seed,
        output_dir=args.output,
    )
