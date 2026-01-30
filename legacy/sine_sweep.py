"""
GENREG Saturation Sweep Test - Extended Version
Addresses methodological critiques with proper controls and baselines.

New Tests Added:
1. Threshold Sensitivity: Multiple saturation thresholds (0.90-0.99)
2. Noise vs Dimensionality: 512 clean vs 512 noisy controls
3. Threshold Effect: Granular noise sweep to find exact threshold
4. Task Diversity: XOR, step, sawtooth, chaotic functions
5. Gradient Baselines: Adam/SGD backprop comparison
6. Seed Variance: Multiple seeds to check genetic drift
"""

import os
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable
from copy import deepcopy

from . import sine_config as cfg
from sine_population import SinePopulation


# =============================================================================
# SWEEP CONFIGURATIONS
# =============================================================================

@dataclass
class SweepConfig:
    """Configuration for a single sweep test."""
    test_id: str
    true_signal: int
    noise_dims: int
    hidden_size: int
    hypothesis: str
    task: str = "sine"  # sine, xor, step, sawtooth, chaotic
    saturation_thresholds: List[float] = field(default_factory=lambda: [0.95])
    seeds: List[int] = field(default_factory=lambda: [42])

    @property
    def total_input(self) -> int:
        return self.true_signal + self.noise_dims

    @property
    def compression_ratio(self) -> int:
        return self.total_input // self.hidden_size if self.hidden_size > 0 else 0


# =============================================================================
# ORIGINAL SWEEP CONFIGS
# =============================================================================

# Core Variable: Noise Ratio (Primary Sweep)
NOISE_SWEEP = [
    SweepConfig("N0", 16, 0, 8, "k ≈ 0-1, baseline"),
    SweepConfig("N1", 16, 16, 8, "k low, mild pressure"),
    SweepConfig("N2", 16, 48, 8, "k rising"),
    SweepConfig("N3", 16, 112, 8, "k moderate-high"),
    SweepConfig("N4", 16, 240, 8, "k high (baseline)"),
    SweepConfig("N5", 16, 496, 8, "k ≈ 8/8, stress test"),
]

# Secondary Variable: Hidden Size
HIDDEN_SWEEP = [
    SweepConfig("H1", 16, 240, 4, "k = 4/4, forced full"),
    SweepConfig("H2", 16, 240, 8, "k ≈ 8/8 (baseline)"),
    SweepConfig("H3", 16, 240, 16, "k < 16? more capacity"),
    SweepConfig("H4", 16, 240, 32, "k << 32? excess capacity"),
]

# Control: No Noise, Varying Compression
CONTROL_SWEEP = [
    SweepConfig("C1", 16, 0, 8, "k ≈ 0"),
    SweepConfig("C2", 64, 0, 8, "k ≈ 0 (still no noise)"),
    SweepConfig("C3", 256, 0, 8, "k ≈ 0 (compression alone)"),
]

# =============================================================================
# NEW CRITIQUE-ADDRESSING CONFIGS
# =============================================================================

# CRITIQUE 1: Threshold Sensitivity Analysis
# Test if k-ratio is stable across different saturation thresholds
THRESHOLD_SENSITIVITY = [
    SweepConfig(
        "T1", 16, 240, 8,
        "Threshold sensitivity: test k stability across 0.90-0.99",
        saturation_thresholds=[0.90, 0.92, 0.95, 0.97, 0.99]
    ),
]

# CRITIQUE 2: Noise vs Dimensionality Control
# Isolate whether saturation comes from noise or just high dimensionality
DIMENSIONALITY_CONTROL = [
    SweepConfig("D1", 512, 0, 8, "512 CLEAN signals - proves compression alone"),
    SweepConfig("D2", 16, 496, 8, "16+496 NOISY signals - same dims as D1"),
    SweepConfig("D3", 256, 0, 8, "256 CLEAN signals - intermediate"),
    SweepConfig("D4", 16, 240, 8, "16+240 NOISY signals - same dims as D3"),
]

# CRITIQUE 3: Threshold Effect Investigation
# Granular sweep between 48 and 112 to find exact noise threshold
THRESHOLD_EFFECT = [
    SweepConfig("TE1", 16, 48, 8, "Threshold hunt: 48 noise"),
    SweepConfig("TE2", 16, 64, 8, "Threshold hunt: 64 noise"),
    SweepConfig("TE3", 16, 80, 8, "Threshold hunt: 80 noise"),
    SweepConfig("TE4", 16, 96, 8, "Threshold hunt: 96 noise"),
    SweepConfig("TE5", 16, 112, 8, "Threshold hunt: 112 noise"),
    SweepConfig("TE6", 16, 128, 8, "Threshold hunt: 128 noise"),
]

# CRITIQUE 4: Task Diversity
# Test non-sine tasks to prove universal application
TASK_DIVERSITY = [
    SweepConfig("TASK_SINE", 16, 240, 8, "Baseline sine task", task="sine"),
    SweepConfig("TASK_XOR", 16, 240, 8, "Discrete XOR task", task="xor"),
    SweepConfig("TASK_STEP", 16, 240, 8, "Discontinuous step function", task="step"),
    SweepConfig("TASK_SAWTOOTH", 16, 240, 8, "Non-smooth sawtooth", task="sawtooth"),
    SweepConfig("TASK_CHAOTIC", 16, 240, 8, "Chaotic logistic map", task="chaotic"),
]

# CRITIQUE 6: Seed Variance / Genetic Drift
# Multiple seeds to prove k-ratio is not random drift
SEED_VARIANCE = [
    SweepConfig(
        "S1", 16, 240, 8,
        "Seed variance: prove k converges across seeds",
        seeds=[42, 123, 456, 789, 1337]
    ),
]

# All configurations
ALL_CONFIGS = NOISE_SWEEP + HIDDEN_SWEEP + CONTROL_SWEEP
CRITIQUE_CONFIGS = (THRESHOLD_SENSITIVITY + DIMENSIONALITY_CONTROL +
                    THRESHOLD_EFFECT + TASK_DIVERSITY + SEED_VARIANCE)

# Priority groups
PRIORITY_1 = ["N0", "N2", "N4"]
PRIORITY_2 = ["H1", "H3"]
PRIORITY_3 = ["C2", "C3"]
PRIORITY_4 = ["N5", "H4"]


# =============================================================================
# TERMINATION CONDITIONS
# =============================================================================

TERMINATION = {
    'target_mse': 0.0008,
    'max_generations': 2000,
    'stagnation_window': 150,
    'stagnation_threshold': 0.0001,
}


# =============================================================================
# TASK GENERATORS (CRITIQUE 4)
# =============================================================================

def generate_task_data(task: str, n_points: int = 100, device: str = "cuda"):
    """Generate input/output data for different tasks."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if task == "sine":
        x = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
        y = np.sin(x)

    elif task == "xor":
        # Continuous XOR: sign(sin(x)) * sign(cos(x))
        x = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
        y = np.sign(np.sin(x)) * np.sign(np.cos(x))

    elif task == "step":
        # Discontinuous step function
        x = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
        y = np.where(x < -np.pi, -1.0,
                     np.where(x < 0, 0.0,
                              np.where(x < np.pi, 0.5, 1.0)))

    elif task == "sawtooth":
        # Non-smooth sawtooth wave
        x = np.linspace(-2 * np.pi, 2 * np.pi, n_points)
        y = 2 * (x / (2 * np.pi) - np.floor(x / (2 * np.pi) + 0.5))

    elif task == "chaotic":
        # Logistic map chaos: x_{n+1} = r * x_n * (1 - x_n)
        x = np.linspace(0.01, 0.99, n_points)
        r = 3.9  # Chaotic regime
        y = r * x * (1 - x)
        # Normalize to [-1, 1]
        y = 2 * y - 1

    else:
        raise ValueError(f"Unknown task: {task}")

    x_tensor = torch.tensor(x, device=device, dtype=torch.float32)
    y_tensor = torch.tensor(y, device=device, dtype=torch.float32)

    return x, y, x_tensor, y_tensor


# =============================================================================
# GRADIENT BASELINE (CRITIQUE 5)
# =============================================================================

class BackpropMLP(nn.Module):
    """Standard MLP for gradient-based baseline comparison."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return torch.tanh(self.fc2(h))

    def get_saturation_stats(self, x, threshold: float = 0.95):
        """Get saturation statistics for hidden layer."""
        with torch.no_grad():
            h = torch.tanh(self.fc1(x))
            saturated = (h.abs() > threshold).float()
            k = (saturated.mean(dim=0) > 0.5).sum().item()
            return {
                'k': int(k),
                'n': self.fc1.out_features,
                'k_ratio': k / self.fc1.out_features,
                'per_neuron': saturated.mean(dim=0).cpu().numpy().tolist(),
            }


def run_backprop_baseline(
    config: SweepConfig,
    x_np: np.ndarray,
    y_np: np.ndarray,
    device: str = "cuda",
    max_epochs: int = 5000,
    lr: float = 0.01,
    patience: int = 200,
) -> Dict:
    """
    Run gradient-based baseline (Adam optimizer) for comparison.
    Returns final MSE and saturation stats.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Prepare data
    x_tensor = torch.tensor(x_np, device=device, dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(y_np, device=device, dtype=torch.float32).unsqueeze(-1)

    # Expand input (simplified version - just add noise dimensions)
    if config.noise_dims > 0:
        noise = torch.randn(len(x_np), config.noise_dims, device=device) * 0.5
        # Create true signal features
        true_feats = torch.zeros(len(x_np), config.true_signal, device=device)
        x_norm = x_tensor / (2 * np.pi)
        true_feats[:, 0:1] = x_norm
        if config.true_signal > 1:
            true_feats[:, 1:2] = x_norm ** 2
        if config.true_signal > 2:
            true_feats[:, 2:3] = x_norm ** 3
        # Fourier features
        for i, freq in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
            if 3 + i * 2 < config.true_signal:
                true_feats[:, 3 + i * 2] = torch.sin(freq * x_tensor.squeeze())
            if 4 + i * 2 < config.true_signal:
                true_feats[:, 4 + i * 2] = torch.cos(freq * x_tensor.squeeze())
        x_expanded = torch.cat([true_feats, noise], dim=-1)
    else:
        # Just true signal features
        true_feats = torch.zeros(len(x_np), config.true_signal, device=device)
        x_norm = x_tensor / (2 * np.pi)
        true_feats[:, 0:1] = x_norm
        if config.true_signal > 1:
            true_feats[:, 1:2] = x_norm ** 2
        if config.true_signal > 2:
            true_feats[:, 2:3] = x_norm ** 3
        for i, freq in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
            if 3 + i * 2 < config.true_signal:
                true_feats[:, 3 + i * 2] = torch.sin(freq * x_tensor.squeeze())
            if 4 + i * 2 < config.true_signal:
                true_feats[:, 4 + i * 2] = torch.cos(freq * x_tensor.squeeze())
        x_expanded = true_feats

    # Create model
    model = BackpropMLP(config.total_input, config.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    best_mse = float('inf')
    patience_counter = 0
    mse_history = []

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = model(x_expanded)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

        mse = loss.item()
        mse_history.append(mse)

        if mse < best_mse - 0.0001:
            best_mse = mse
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        if mse < TERMINATION['target_mse']:
            break

    elapsed = time.perf_counter() - start_time

    # Get final saturation stats
    sat_stats = model.get_saturation_stats(x_expanded)

    return {
        'method': 'backprop_adam',
        'final_mse': best_mse,
        'epochs': epoch + 1,
        'elapsed_seconds': elapsed,
        'k_mlp': sat_stats['k'],
        'k_ratio': sat_stats['k_ratio'],
        'per_neuron_saturation': sat_stats['per_neuron'],
        'mse_history': mse_history[::10],  # Subsample for storage
    }


# =============================================================================
# SWEEP RUNNER
# =============================================================================

class SweepRunner:
    """Runs a single sweep configuration and tracks metrics."""

    def __init__(
        self,
        config: SweepConfig,
        population_size: int = 40,
        device: str = "cuda",
        run_baseline: bool = False,
    ):
        self.config = config
        self.population_size = population_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.run_baseline = run_baseline

        # Metrics tracking
        self.trajectory: List[Dict] = []
        self.saturation_history: List[np.ndarray] = []
        self.threshold_results: Dict[float, Dict] = {}  # For threshold sensitivity

        # Convergence tracking
        self.gen_to_mse_01: Optional[int] = None
        self.gen_to_mse_001: Optional[int] = None
        self.gen_to_mse_0008: Optional[int] = None

        # Stagnation tracking
        self.mse_history: List[float] = []

    def _apply_config(self, threshold: float = 0.95):
        """Apply sweep config to global config."""
        cfg.TRUE_SIGNAL_SIZE = self.config.true_signal
        cfg.NOISE_SIGNAL_SIZE = self.config.noise_dims
        cfg.EXPANSION_SIZE = self.config.total_input
        cfg.HIDDEN_SIZE = self.config.hidden_size
        cfg.POPULATION_SIZE = self.population_size
        cfg.INPUT_EXPANSION = True
        cfg.SATURATION_THRESHOLD = threshold

    def _check_stagnation(self) -> bool:
        """Check if training has stagnated."""
        if len(self.mse_history) < TERMINATION['stagnation_window']:
            return False

        window = self.mse_history[-TERMINATION['stagnation_window']:]
        improvement = window[0] - window[-1]
        return improvement < TERMINATION['stagnation_threshold']

    def _record_metrics(self, population, gen: int, threshold: float = 0.95):
        """Record per-generation metrics."""
        stats = population.get_stats()
        best = population.get_best_genome()

        # Per-neuron saturation
        per_neuron = best.mlp_saturation_stats.get('per_neuron', [0] * self.config.hidden_size)

        record = {
            'generation': gen,
            'mse_best': stats['mse_best'],
            'mse_median': stats['mse_median'],
            'k_mlp': stats['k_mlp_best'],
            'k_proteins': stats['k_proteins_best'],
            'k_total': stats['k_total_best'],
            'k_mlp_ratio': stats['k_mlp_best'] / self.config.hidden_size if self.config.hidden_size > 0 else 0,
            'fitness_best': stats['fitness_best'],
            'fitness_median': stats['fitness_median'],
            'age_max': stats['age_max'],
            'threshold': threshold,
        }

        # Add per-neuron saturation
        for i, sat in enumerate(per_neuron):
            record[f'n{i}_sat'] = sat

        self.trajectory.append(record)
        self.saturation_history.append(np.array(per_neuron))
        self.mse_history.append(stats['mse_best'])

        # Track convergence milestones
        if self.gen_to_mse_01 is None and stats['mse_best'] < 0.01:
            self.gen_to_mse_01 = gen
        if self.gen_to_mse_001 is None and stats['mse_best'] < 0.001:
            self.gen_to_mse_001 = gen
        if self.gen_to_mse_0008 is None and stats['mse_best'] < 0.0008:
            self.gen_to_mse_0008 = gen

    def _run_single_seed(
        self,
        seed: int,
        threshold: float,
        log_every: int,
        x_np: np.ndarray,
        y_np: np.ndarray,
    ) -> Dict:
        """Run training for a single seed and threshold combination."""
        import random

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Apply configuration
        self._apply_config(threshold=threshold)

        # Reload modules that cache config values
        import importlib
        from . import sine_controller
        from . import sine_population
        from . import sine_genome
        importlib.reload(sine_controller)
        importlib.reload(sine_population)
        importlib.reload(sine_genome)

        from sine_population import SinePopulation as ReloadedPopulation

        # Initialize population with custom task data
        population = ReloadedPopulation(size=self.population_size, device=self.device)

        # Override test points if not sine task
        if self.config.task != "sine":
            population.x_values = x_np
            population.y_values = y_np
            population.x_tensor = torch.tensor(x_np, device=population.device, dtype=torch.float32)
            population.y_tensor = torch.tensor(y_np, device=population.device, dtype=torch.float32)

        # Reset tracking
        self.trajectory = []
        self.saturation_history = []
        self.mse_history = []
        self.gen_to_mse_01 = None
        self.gen_to_mse_001 = None
        self.gen_to_mse_0008 = None

        start_time = time.perf_counter()
        termination_reason = "max_generations"

        for gen in range(TERMINATION['max_generations']):
            population.evaluate()
            self._record_metrics(population, gen, threshold)

            if gen % log_every == 0:
                rec = self.trajectory[-1]
                print(f"  {gen:5d} | {rec['mse_best']:.6f} | {rec['k_mlp']:2d}/{self.config.hidden_size} | {rec['k_mlp_ratio']:.2f}")

            if rec['mse_best'] < TERMINATION['target_mse']:
                termination_reason = "target_reached"
                break

            if self._check_stagnation():
                termination_reason = "stagnation"
                break

            population.evolve()

        elapsed = time.perf_counter() - start_time

        # Final evaluation
        population.evaluate()
        self._record_metrics(population, population.generation, threshold)

        final_rec = self.trajectory[-1]

        return {
            'seed': seed,
            'threshold': threshold,
            'final_mse': final_rec['mse_best'],
            'final_k_mlp': final_rec['k_mlp'],
            'final_k_ratio': final_rec['k_mlp_ratio'],
            'total_generations': population.generation,
            'termination_reason': termination_reason,
            'elapsed_seconds': elapsed,
            'trajectory': deepcopy(self.trajectory),
            'saturation_heatmap': np.array(self.saturation_history),
        }

    def run(self, log_every: int = 50) -> Dict:
        """Run the sweep test and return results."""
        print(f"\n{'='*60}")
        print(f"Running: {self.config.test_id}")
        print(f"  Task:        {self.config.task}")
        print(f"  True Signal: {self.config.true_signal}")
        print(f"  Noise Dims:  {self.config.noise_dims}")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  Compression: {self.config.compression_ratio}:1")
        print(f"  Thresholds:  {self.config.saturation_thresholds}")
        print(f"  Seeds:       {self.config.seeds}")
        print(f"  Hypothesis:  {self.config.hypothesis}")
        print(f"{'='*60}")

        # Generate task data
        x_np, y_np, x_tensor, y_tensor = generate_task_data(
            self.config.task, n_points=cfg.NUM_TEST_POINTS, device=self.device
        )

        all_runs = []
        seed_results = {}
        threshold_results = {}

        # Run for each seed and threshold combination
        for seed in self.config.seeds:
            seed_results[seed] = {}
            for threshold in self.config.saturation_thresholds:
                print(f"\n  Seed={seed}, Threshold={threshold}")
                print(f"  GEN   | MSE        | k_MLP | k_ratio")
                print(f"  {'-'*40}")

                result = self._run_single_seed(seed, threshold, log_every, x_np, y_np)
                all_runs.append(result)
                seed_results[seed][threshold] = result

                if threshold not in threshold_results:
                    threshold_results[threshold] = []
                threshold_results[threshold].append(result)

                print(f"  -> MSE={result['final_mse']:.6f}, k={result['final_k_mlp']}, gens={result['total_generations']}")

        # Run backprop baseline if requested
        baseline_result = None
        if self.run_baseline:
            print(f"\n  Running Backprop Baseline (Adam)...")
            baseline_result = run_backprop_baseline(self.config, x_np, y_np, device=self.device)
            print(f"  -> Backprop MSE={baseline_result['final_mse']:.6f}, k={baseline_result['k_mlp']}, epochs={baseline_result['epochs']}")

        # Aggregate results
        primary_threshold = 0.95
        primary_runs = [r for r in all_runs if r['threshold'] == primary_threshold]
        if not primary_runs:
            primary_runs = all_runs

        # Compute statistics across seeds
        final_mses = [r['final_mse'] for r in primary_runs]
        final_ks = [r['final_k_mlp'] for r in primary_runs]
        final_k_ratios = [r['final_k_ratio'] for r in primary_runs]

        summary = {
            "test_id": self.config.test_id,
            "task": self.config.task,
            "true_signal": self.config.true_signal,
            "noise_dims": self.config.noise_dims,
            "hidden_size": self.config.hidden_size,
            "compression_ratio": self.config.compression_ratio,
            "population_size": self.population_size,

            # Primary results (mean across seeds at threshold=0.95)
            "final_mse_mean": float(np.mean(final_mses)),
            "final_mse_std": float(np.std(final_mses)),
            "final_k_mlp_mean": float(np.mean(final_ks)),
            "final_k_mlp_std": float(np.std(final_ks)),
            "final_k_ratio_mean": float(np.mean(final_k_ratios)),
            "final_k_ratio_std": float(np.std(final_k_ratios)),

            # Seed variance analysis (CRITIQUE 6)
            "n_seeds": len(self.config.seeds),
            "k_ratio_variance": float(np.var(final_k_ratios)) if len(final_k_ratios) > 1 else 0,
            "k_ratio_range": float(max(final_k_ratios) - min(final_k_ratios)) if len(final_k_ratios) > 1 else 0,

            # Threshold sensitivity (CRITIQUE 1)
            "threshold_sensitivity": {},

            "hypothesis": self.config.hypothesis,
        }

        # Threshold sensitivity analysis
        for thresh, runs in threshold_results.items():
            k_ratios = [r['final_k_ratio'] for r in runs]
            summary["threshold_sensitivity"][str(thresh)] = {
                "k_ratio_mean": float(np.mean(k_ratios)),
                "k_ratio_std": float(np.std(k_ratios)) if len(k_ratios) > 1 else 0,
            }

        # Check threshold stability (CRITIQUE 1)
        if len(threshold_results) > 1:
            thresh_k_means = [v['k_ratio_mean'] for v in summary["threshold_sensitivity"].values()]
            summary["threshold_k_range"] = float(max(thresh_k_means) - min(thresh_k_means))
            summary["threshold_stable"] = summary["threshold_k_range"] < 0.2  # <20% variation = stable

        # Backprop comparison (CRITIQUE 5)
        if baseline_result:
            summary["backprop_baseline"] = {
                "mse": baseline_result['final_mse'],
                "k_mlp": baseline_result['k_mlp'],
                "k_ratio": baseline_result['k_ratio'],
                "epochs": baseline_result['epochs'],
            }
            summary["genreg_vs_backprop_mse"] = summary["final_mse_mean"] - baseline_result['final_mse']
            summary["genreg_better"] = summary["final_mse_mean"] < baseline_result['final_mse']

        # Theory validation
        if self.config.noise_dims == 0:
            summary["theory_prediction"] = "low_k"
            summary["theory_validated"] = summary["final_k_ratio_mean"] < 0.3
        elif self.config.noise_dims >= 200:
            summary["theory_prediction"] = "high_k"
            summary["theory_validated"] = summary["final_k_ratio_mean"] > 0.5
        else:
            summary["theory_prediction"] = "moderate_k"
            summary["theory_validated"] = 0.2 < summary["final_k_ratio_mean"] < 0.9

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.config.test_id}")
        print(f"{'='*60}")
        print(f"  MSE:     {summary['final_mse_mean']:.6f} ± {summary['final_mse_std']:.6f}")
        print(f"  k_mlp:   {summary['final_k_mlp_mean']:.1f} ± {summary['final_k_mlp_std']:.1f}")
        print(f"  k_ratio: {summary['final_k_ratio_mean']:.1%} ± {summary['final_k_ratio_std']:.1%}")

        if len(self.config.seeds) > 1:
            print(f"\n  Seed Variance (CRITIQUE 6):")
            print(f"    k_ratio range: {summary['k_ratio_range']:.1%}")
            print(f"    Consistent: {'YES' if summary['k_ratio_range'] < 0.2 else 'NO'}")

        if len(threshold_results) > 1:
            print(f"\n  Threshold Sensitivity (CRITIQUE 1):")
            for thresh, stats in summary["threshold_sensitivity"].items():
                print(f"    {thresh}: k_ratio = {stats['k_ratio_mean']:.1%}")
            print(f"    Stable: {'YES' if summary.get('threshold_stable', False) else 'NO'}")

        if baseline_result:
            print(f"\n  Backprop Comparison (CRITIQUE 5):")
            print(f"    GENREG MSE: {summary['final_mse_mean']:.6f}")
            print(f"    Backprop MSE: {baseline_result['final_mse']:.6f}")
            print(f"    GENREG better: {'YES' if summary['genreg_better'] else 'NO'}")

        print(f"\n  Theory: {summary['theory_prediction']} -> {'VALIDATED' if summary['theory_validated'] else 'FAILED'}")

        return {
            'summary': summary,
            'all_runs': all_runs,
            'baseline': baseline_result,
        }


# =============================================================================
# MAIN SWEEP FUNCTION
# =============================================================================

def run_sweep(
    configs: List[SweepConfig] = None,
    population_size: int = 40,
    output_dir: str = None,
    log_every: int = 50,
    run_baselines: bool = False,
    include_critiques: bool = False,
):
    """
    Run sweep across multiple configurations.

    Args:
        configs: List of SweepConfig to test (default: all original)
        population_size: Population size for each run
        output_dir: Output directory (default: results/sweep_timestamp)
        log_every: Log frequency
        run_baselines: Whether to run backprop baselines for comparison
        include_critiques: Whether to include critique-addressing configs
    """
    if configs is None:
        configs = ALL_CONFIGS
        if include_critiques:
            configs = configs + CRITIQUE_CONFIGS

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/sweep_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"GENREG SATURATION SWEEP - EXTENDED")
    print(f"{'#'*60}")
    print(f"Output: {output_dir}")
    print(f"Configs: {len(configs)}")
    print(f"Population: {population_size}")
    print(f"Run baselines: {run_baselines}")
    print(f"Max generations: {TERMINATION['max_generations']}")
    print(f"Target MSE: {TERMINATION['target_mse']}")

    all_summaries = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]")

        # Run this configuration
        runner = SweepRunner(
            config,
            population_size=population_size,
            run_baseline=run_baselines,
        )
        results = runner.run(log_every=log_every)

        # Save results for this config
        config_dir = output_dir / config.test_id
        config_dir.mkdir(exist_ok=True)

        # Save config
        config_dict = asdict(config)
        with open(config_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save summary
        with open(config_dir / "final_summary.json", "w") as f:
            json.dump(results['summary'], f, indent=2)

        # Save all runs data
        for run in results['all_runs']:
            run_id = f"seed{run['seed']}_thresh{run['threshold']}"
            run_dir = config_dir / run_id
            run_dir.mkdir(exist_ok=True)

            # Save trajectory CSV
            if run['trajectory']:
                with open(run_dir / "trajectory.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=run['trajectory'][0].keys())
                    writer.writeheader()
                    writer.writerows(run['trajectory'])

            # Save saturation heatmap
            np.save(run_dir / "saturation_heatmap.npy", run['saturation_heatmap'])

        # Save baseline if present
        if results['baseline']:
            with open(config_dir / "backprop_baseline.json", "w") as f:
                # Convert numpy arrays to lists for JSON
                baseline = results['baseline'].copy()
                baseline['per_neuron_saturation'] = list(baseline['per_neuron_saturation'])
                json.dump(baseline, f, indent=2)

        all_summaries.append(results['summary'])

    # Save combined summary
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print final summary table
    print(f"\n{'='*100}")
    print("SWEEP SUMMARY")
    print(f"{'='*100}")
    print(f"{'ID':<12} {'Task':<8} {'Noise':>6} {'Hidden':>6} | {'MSE':>12} {'k':>6} {'k%':>8} | {'Seeds':>5} {'Valid':>6}")
    print("-" * 100)

    for s in all_summaries:
        validated = "YES" if s['theory_validated'] else "NO"
        task = s.get('task', 'sine')[:7]
        print(f"{s['test_id']:<12} {task:<8} {s['noise_dims']:>6} {s['hidden_size']:>6} | "
              f"{s['final_mse_mean']:>12.6f} {s['final_k_mlp_mean']:>6.1f} {s['final_k_ratio_mean']:>7.1%} | "
              f"{s['n_seeds']:>5} {validated:>6}")

    # Summary of critique tests
    print(f"\n{'='*100}")
    print("CRITIQUE ANALYSIS")
    print(f"{'='*100}")

    # Threshold sensitivity
    thresh_tests = [s for s in all_summaries if 'threshold_stable' in s]
    if thresh_tests:
        print("\n1. THRESHOLD SENSITIVITY (0.90-0.99):")
        for s in thresh_tests:
            stable = "STABLE" if s['threshold_stable'] else "UNSTABLE"
            print(f"   {s['test_id']}: k_range={s['threshold_k_range']:.1%} -> {stable}")

    # Dimensionality control
    dim_tests = [s for s in all_summaries if s['test_id'].startswith('D')]
    if dim_tests:
        print("\n2. NOISE vs DIMENSIONALITY:")
        for s in dim_tests:
            print(f"   {s['test_id']}: signal={s['true_signal']}, noise={s['noise_dims']} -> k={s['final_k_ratio_mean']:.1%}")

    # Seed variance
    seed_tests = [s for s in all_summaries if s['n_seeds'] > 1]
    if seed_tests:
        print("\n6. SEED VARIANCE (genetic drift check):")
        for s in seed_tests:
            consistent = "CONSISTENT" if s['k_ratio_range'] < 0.2 else "DRIFT DETECTED"
            print(f"   {s['test_id']}: k_range={s['k_ratio_range']:.1%} -> {consistent}")

    # Backprop comparison
    backprop_tests = [s for s in all_summaries if 'backprop_baseline' in s]
    if backprop_tests:
        print("\n5. BACKPROP COMPARISON:")
        for s in backprop_tests:
            winner = "GENREG" if s['genreg_better'] else "BACKPROP"
            print(f"   {s['test_id']}: GENREG={s['final_mse_mean']:.6f} vs Backprop={s['backprop_baseline']['mse']:.6f} -> {winner}")

    print(f"\nResults saved to: {output_dir}")

    return all_summaries


def run_critique_sweep(population_size: int = 40, **kwargs):
    """Run only the critique-addressing tests."""
    print("\nRunning CRITIQUE-ADDRESSING tests only...")
    return run_sweep(CRITIQUE_CONFIGS, population_size=population_size, run_baselines=True, **kwargs)


def run_full_sweep(population_size: int = 40, **kwargs):
    """Run all tests including critique-addressing configs."""
    print("\nRunning FULL sweep (original + critique tests)...")
    return run_sweep(
        ALL_CONFIGS + CRITIQUE_CONFIGS,
        population_size=population_size,
        run_baselines=True,
        **kwargs
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GENREG Saturation Sweep Test - Extended")
    parser.add_argument("--priority", type=int, choices=[1, 2, 3, 4],
                        help="Run specific priority group (1-4)")
    parser.add_argument("--configs", nargs="+",
                        help="Specific config IDs to run (e.g., N0 N2 N4 D1 D2)")
    parser.add_argument("--critiques", action="store_true",
                        help="Run critique-addressing tests only")
    parser.add_argument("--full", action="store_true",
                        help="Run full sweep including all critique tests")
    parser.add_argument("--population", type=int, default=40,
                        help="Population size (default: 40)")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log frequency (default: 50)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--baselines", action="store_true",
                        help="Run backprop baselines for comparison")

    args = parser.parse_args()

    if args.critiques:
        run_critique_sweep(
            population_size=args.population,
            output_dir=args.output,
            log_every=args.log_every,
        )
    elif args.full:
        run_full_sweep(
            population_size=args.population,
            output_dir=args.output,
            log_every=args.log_every,
        )
    elif args.configs:
        # Run specific configs
        all_available = ALL_CONFIGS + CRITIQUE_CONFIGS
        configs = [c for c in all_available if c.test_id in args.configs]
        if not configs:
            print(f"No matching configs found for: {args.configs}")
            print(f"Available: {[c.test_id for c in all_available]}")
            exit(1)
        run_sweep(
            configs,
            population_size=args.population,
            output_dir=args.output,
            log_every=args.log_every,
            run_baselines=args.baselines,
        )
    elif args.priority:
        # Run priority group
        priority_map = {1: PRIORITY_1, 2: PRIORITY_2, 3: PRIORITY_3, 4: PRIORITY_4}
        test_ids = priority_map[args.priority]
        configs = [c for c in ALL_CONFIGS if c.test_id in test_ids]
        run_sweep(
            configs,
            population_size=args.population,
            output_dir=args.output,
            log_every=args.log_every,
            run_baselines=args.baselines,
        )
    else:
        # Run original configs only
        run_sweep(
            population_size=args.population,
            output_dir=args.output,
            log_every=args.log_every,
            run_baselines=args.baselines,
        )
