"""
High-Dimensional Scaling with GSA (v2 - using usen library)

Demonstrates the simplified experiment structure using the usen package.
Compare to highdim_gsa.py (300+ lines) vs this (~80 lines).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime

from usen import SparseNet, train_gsa, train_sa, highdim_problem, mse, saturation, selection_stats


def main():
    print("=" * 60)
    print("HIGH-DIM SCALING (v2 - using usen library)")
    print("=" * 60)

    # Config
    n_features = 1000
    n_true = 10
    n_samples = 500
    H, K = 8, 4
    n_trials = 2

    print(f"\nConfig: {n_features} features, {n_true} true, H={H}, K={K}")

    results = {'gsa': [], 'sa': []}

    for trial in range(n_trials):
        seed = trial * 1000
        print(f"\n--- Trial {trial+1}/{n_trials} (seed={seed}) ---")

        # Generate problem
        X, y, true_features = highdim_problem(n_samples, n_features, n_true, seed=42)

        # GSA
        print("\nGSA:")
        net_gsa = SparseNet(n_features, H, K)
        best_gsa, _ = train_gsa(
            net_gsa, X, y,
            generations=50,  # Short for demo
            pop_size=30,
            seed=seed,
            verbose=True
        )
        gsa_mse = mse(best_gsa, X, y)
        gsa_stats = selection_stats(best_gsa, true_features)
        print(f"  Final: MSE={gsa_mse:.4f}, True={gsa_stats['true_features_found']}/{n_true}")
        results['gsa'].append({'mse': gsa_mse, 'true_found': gsa_stats['true_features_found']})

        # SA
        print("\nSA:")
        net_sa = SparseNet(n_features, H, K)
        best_sa, _ = train_sa(
            net_sa, X, y,
            max_steps=15000,  # Short for demo
            seed=seed,
            verbose=True
        )
        sa_mse = mse(best_sa, X, y)
        sa_stats = selection_stats(best_sa, true_features)
        print(f"  Final: MSE={sa_mse:.4f}, True={sa_stats['true_features_found']}/{n_true}")
        results['sa'].append({'mse': sa_mse, 'true_found': sa_stats['true_features_found']})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for method in ['gsa', 'sa']:
        mses = [r['mse'] for r in results[method]]
        found = [r['true_found'] for r in results[method]]
        print(f"{method.upper()}: MSE={np.mean(mses):.4f}, True={np.mean(found):.1f}/{n_true}")

    print("\nNote: This is a shortened demo. Run highdim_gsa.py for full results.")


if __name__ == "__main__":
    main()
