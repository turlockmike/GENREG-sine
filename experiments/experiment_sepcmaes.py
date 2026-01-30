"""
Experiment 2: Separable CMA-ES

Standard CMA-ES failed due to high dimensionality (2065 params).
Separable CMA-ES uses diagonal covariance matrix, scaling O(n) instead of O(n²).

This should work much better for our high-dimensional problem.
"""

import torch
import numpy as np
import cma
import json
import time
from pathlib import Path

import legacy.sine_config as sine_config as cfg
from legacy.sine_controller import SineController


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def params_to_vector(controller: SineController) -> np.ndarray:
    """Flatten controller weights to 1D numpy array."""
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

    w1_size = controller.w1.numel()
    b1_size = controller.b1.numel()
    w2_size = controller.w2.numel()

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
        vec[idx:],
        device=device, dtype=torch.float32
    )

    return controller


def run_experiment():
    """Compare standard CMA-ES vs Separable CMA-ES."""
    print("=" * 70)
    print("EXPERIMENT 2: SEPARABLE CMA-ES")
    print("=" * 70)
    print("Hypothesis: sep-CMA-ES scales better to high dimensions (2065 params)")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    def fitness(vec):
        controller = vector_to_controller(vec)
        with torch.no_grad():
            pred = controller.forward(x_test, track=False)
            mse = torch.mean((pred - y_true) ** 2).item()
        return mse

    init_controller = SineController(device=DEVICE)
    x0 = params_to_vector(init_controller)
    n_params = len(x0)

    print(f"\nParameters: {n_params}")
    print(f"Max evaluations: 20000")

    results = {}

    # Run Separable CMA-ES
    for variant, opts in [
        ("sep-CMA-ES", {'CMA_diagonal': True, 'maxfevals': 20000, 'verb_disp': 0}),
        ("VD-CMA", {'CMA_diagonal': 2, 'maxfevals': 20000, 'verb_disp': 0}),  # VD-CMA variant
    ]:
        print(f"\n[{variant}]")

        np.random.seed(42)
        start_time = time.time()

        es = cma.CMAEvolutionStrategy(x0, 0.5, opts)

        gen = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [fitness(s) for s in solutions]
            es.tell(solutions, fitnesses)

            if gen % 50 == 0:
                best_mse = es.result.fbest
                print(f"  Gen {gen:4d} | Evals {es.result.evaluations:5d} | MSE={best_mse:.6f} | sigma={es.sigma:.4f}")
            gen += 1

        elapsed = time.time() - start_time

        # Final evaluation
        best_controller = vector_to_controller(es.result.xbest)
        _ = best_controller.forward(x_test, track=True)
        final_mse = es.result.fbest
        final_k = best_controller.get_k()

        print(f"  -> Final: MSE={final_mse:.6f}, k={final_k}/{cfg.HIDDEN_SIZE}, time={elapsed:.1f}s")

        results[variant] = {
            'mse': final_mse,
            'k': final_k,
            'k_ratio': final_k / cfg.HIDDEN_SIZE,
            'time': elapsed,
            'evaluations': es.result.evaluations,
            'final_sigma': es.sigma,
        }

    # Compare with standard CMA-ES result (from previous experiment)
    results['standard-CMA-ES'] = {
        'mse': 0.067,  # From previous run
        'k': 8,
        'k_ratio': 1.0,
        'time': 100.7,
        'evaluations': 20000,
        'note': 'From previous experiment'
    }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<20} | {'MSE':>10} | {'k':>5} | {'Time':>8} | {'Sigma':>8}")
    print("-" * 70)

    for name, r in results.items():
        sigma = r.get('final_sigma', 'N/A')
        sigma_str = f"{sigma:.4f}" if isinstance(sigma, float) else sigma
        print(f"{name:<20} | {r['mse']:>10.6f} | {r['k']:>5} | {r['time']:>7.1f}s | {sigma_str:>8}")

    # Determine winner
    sep_mse = results['sep-CMA-ES']['mse']
    std_mse = results['standard-CMA-ES']['mse']

    if sep_mse < std_mse:
        improvement = (std_mse - sep_mse) / std_mse * 100
        print(f"\n  >> sep-CMA-ES improves over standard by {improvement:.1f}%")
    else:
        print(f"\n  >> No improvement from separable variant")

    # Save results
    output_dir = Path("results/sepcmaes_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
