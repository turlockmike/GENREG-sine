"""
Demo: Dashboard Integration

Simple sine experiment that outputs to results/live/ for dashboard display.
Demonstrates the convention for live experiment monitoring.

Usage:
    # Terminal 1: Start dashboard
    uv run python dashboard.py

    # Terminal 2: Run this experiment
    uv run python experiments/demo_dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
from core.models import UltraSparseController
from core.training import train_gsa


def run_demo():
    """Run a simple sine regression to demo the dashboard."""

    # Config
    hidden = 8
    k = 4
    generations = 200
    population = 20

    # Output paths (dashboard convention)
    live_dir = Path(__file__).parent.parent / "results" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    name = f"sine_H{hidden}_K{k}"
    csv_path = live_dir / f"{name}.csv"
    json_path = live_dir / f"{name}.json"

    # Write config JSON
    config = {
        "hidden": hidden,
        "k": k,
        "population": population,
        "generations": generations,
        "dataset": "sine",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")
    }
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Generate sine data
    np.random.seed(42)
    x = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
    y = np.sin(x)

    # Expand input (simple: just repeat with noise for 16 dims)
    X = np.hstack([x] + [x + np.random.randn(*x.shape) * 0.1 for _ in range(15)])

    X_train = torch.tensor(X[:160], dtype=torch.float32)
    X_test = torch.tensor(X[160:], dtype=torch.float32)
    y_train = torch.tensor(y[:160], dtype=torch.float32)
    y_test = torch.tensor(y[160:], dtype=torch.float32)

    print(f"\n{'='*50}")
    print(f"Dashboard Demo: Sine Regression")
    print(f"{'='*50}")
    print(f"Config: H={hidden}, K={k}, Pop={population}, Gens={generations}")
    print(f"Output: {csv_path}")
    print(f"Dashboard: http://localhost:8050")
    print(f"{'='*50}\n")

    # Open CSV for streaming writes
    start_time = time.time()

    with open(csv_path, 'w') as f:
        f.write("gen,test_accuracy,best_fitness,mean_fitness,temperature,elapsed_s\n")

        def callback(gen, best, fitness, test_acc):
            """Called every generation to write to CSV."""
            elapsed = time.time() - start_time

            # For regression, test_acc is actually negative MSE
            # Convert to a 0-1 "accuracy" for display (1 - normalized MSE)
            with torch.no_grad():
                pred = best.forward(X_test)
                mse = torch.mean((pred - y_test) ** 2).item()
            accuracy = max(0, 1 - mse)  # Simple conversion

            f.write(f"{gen},{accuracy:.6f},{fitness:.6f},{fitness:.6f},0.0,{elapsed:.1f}\n")
            f.flush()

            if gen % 20 == 0:
                print(f"Gen {gen:3d}: MSE={mse:.6f}, Acc={accuracy:.1%}")

        # Run GSA with callback
        best, results, history = train_gsa(
            controller_factory=lambda: UltraSparseController(16, hidden, 1, k),
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test,
            population_size=population,
            generations=generations,
            verbose=False,
            callback=callback
        )

    # Final results
    with torch.no_grad():
        pred = best.forward(X_test)
        final_mse = torch.mean((pred - y_test) ** 2).item()

    print(f"\n{'='*50}")
    print(f"COMPLETE")
    print(f"Final MSE: {final_mse:.6f}")
    print(f"Parameters: {best.num_parameters()}")
    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"{'='*50}\n")

    # Update config with completion status
    config["status"] = "complete"
    config["final_mse"] = final_mse
    config["params"] = best.num_parameters()
    with open(json_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    run_demo()
