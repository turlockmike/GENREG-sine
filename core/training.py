"""
Training Module

Shared training functions for gradient-free optimization:
- Simulated Annealing (SA)
- Hill Climbing
- Genetic Algorithm helpers

All training functions report metrics at regular intervals.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple, List, Dict, Any
from .metrics import compute_metrics, Metrics


def train_sa(
    controller,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_steps: int = 15000,
    t_initial: float = 0.01,
    t_final: float = 0.00001,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    verbose: bool = True,
    report_interval: int = 5000,
) -> Tuple[Any, Metrics, List[Dict]]:
    """
    Train using Simulated Annealing.

    Args:
        controller: Network with forward(), mutate(), clone() methods
        x_test: Test inputs
        y_true: True outputs
        max_steps: Number of optimization steps
        t_initial: Initial temperature
        t_final: Final temperature
        mutation_rate: Probability of mutating each weight
        mutation_scale: Standard deviation of mutation noise
        verbose: Print progress
        report_interval: Steps between progress reports

    Returns:
        best: Best controller found
        final_metrics: Final metrics for best controller
        history: List of metrics at each report interval
    """
    current = controller
    with torch.no_grad():
        pred = current.forward(x_test)
        current_mse = torch.mean((pred - y_true) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    decay = (t_final / t_initial) ** (1.0 / max_steps)
    history = []

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        # Mutate
        mutant = current.clone()
        mutant.mutate(rate=mutation_rate, scale=mutation_scale)

        # Evaluate
        with torch.no_grad():
            pred = mutant.forward(x_test)
            mutant_mse = torch.mean((pred - y_true) ** 2).item()

        # SA acceptance
        delta = current_mse - mutant_mse  # positive if mutant is better
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse

            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        # Report progress
        if step % report_interval == 0:
            metrics = compute_metrics(best, x_test, y_true)
            history.append({
                'step': step,
                **metrics.to_dict()
            })

            if verbose:
                print(f"    Step {step}: {metrics}")

    # Final metrics
    final_metrics = compute_metrics(best, x_test, y_true)

    return best, final_metrics, history


def train_hillclimb(
    controller,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_steps: int = 5000,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    patience: int = 500,
    verbose: bool = True,
    report_interval: int = 1000,
) -> Tuple[Any, Metrics, List[Dict]]:
    """
    Train using Hill Climbing with early stopping.

    Args:
        controller: Network with forward(), mutate(), clone() methods
        x_test: Test inputs
        y_true: True outputs
        max_steps: Maximum optimization steps
        mutation_rate: Probability of mutating each weight
        mutation_scale: Standard deviation of mutation noise
        patience: Steps without improvement before stopping
        verbose: Print progress
        report_interval: Steps between progress reports

    Returns:
        best: Best controller found
        final_metrics: Final metrics for best controller
        history: List of metrics at each report interval
    """
    current = controller
    with torch.no_grad():
        pred = current.forward(x_test)
        current_mse = torch.mean((pred - y_true) ** 2).item()

    best = current.clone()
    best_mse = current_mse
    steps_without_improvement = 0
    history = []

    for step in range(max_steps):
        # Mutate
        mutant = current.clone()
        mutant.mutate(rate=mutation_rate, scale=mutation_scale)

        # Evaluate
        with torch.no_grad():
            pred = mutant.forward(x_test)
            mutant_mse = torch.mean((pred - y_true) ** 2).item()

        # Accept only if better
        if mutant_mse < current_mse:
            current = mutant
            current_mse = mutant_mse
            steps_without_improvement = 0

            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse
        else:
            steps_without_improvement += 1

        # Early stopping
        if steps_without_improvement >= patience:
            if verbose:
                print(f"    Early stopping at step {step} (no improvement for {patience} steps)")
            break

        # Report progress
        if step % report_interval == 0:
            metrics = compute_metrics(best, x_test, y_true)
            history.append({
                'step': step,
                **metrics.to_dict()
            })

            if verbose:
                print(f"    Step {step}: {metrics}")

    # Final metrics
    final_metrics = compute_metrics(best, x_test, y_true)

    return best, final_metrics, history


def train_ga(
    population: List,
    x_test: torch.Tensor,
    y_true: torch.Tensor,
    max_generations: int = 1000,
    elite_fraction: float = 0.2,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
    verbose: bool = True,
    report_interval: int = 100,
) -> Tuple[Any, Metrics, List[Dict]]:
    """
    Train using a simple Genetic Algorithm.

    Args:
        population: List of controllers (initial population)
        x_test: Test inputs
        y_true: True outputs
        max_generations: Number of generations
        elite_fraction: Fraction of population to keep as elites
        mutation_rate: Probability of mutating each weight
        mutation_scale: Standard deviation of mutation noise
        verbose: Print progress
        report_interval: Generations between progress reports

    Returns:
        best: Best controller found
        final_metrics: Final metrics for best controller
        history: List of metrics at each report interval
    """
    pop_size = len(population)
    elite_count = max(1, int(pop_size * elite_fraction))
    history = []

    best = None
    best_mse = float('inf')

    for gen in range(max_generations):
        # Evaluate fitness
        fitness_scores = []
        for controller in population:
            with torch.no_grad():
                pred = controller.forward(x_test)
                mse = torch.mean((pred - y_true) ** 2).item()
            fitness_scores.append(-mse)  # Higher fitness = lower MSE

            if mse < best_mse:
                best = controller.clone()
                best_mse = mse

        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_pop = [population[i] for i in sorted_indices]

        # Report progress
        if gen % report_interval == 0:
            metrics = compute_metrics(best, x_test, y_true)
            history.append({
                'generation': gen,
                **metrics.to_dict()
            })

            if verbose:
                print(f"    Gen {gen}: {metrics}")

        # Selection: keep elites
        elites = sorted_pop[:elite_count]

        # Create new population
        new_population = list(elites)  # Keep elites unchanged

        while len(new_population) < pop_size:
            # Select parent from elites
            parent = elites[np.random.randint(elite_count)]
            child = parent.clone()
            child.mutate(rate=mutation_rate, scale=mutation_scale)
            new_population.append(child)

        population = new_population

    # Final metrics
    final_metrics = compute_metrics(best, x_test, y_true)

    return best, final_metrics, history
