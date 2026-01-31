"""
Training Module

Shared training functions for gradient-free optimization:
- Simulated Annealing (SA)
- Hill Climbing
- Genetic Algorithm (GA)
- Genetic Simulated Annealing (GSA) - population + SA refinement

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


def _roulette_select(population: List[Tuple[Any, float]], n_select: int) -> List[Any]:
    """Roulette wheel selection based on fitness scores."""
    fitnesses = np.array([f for _, f in population])
    min_fitness = fitnesses.min()
    shifted = fitnesses - min_fitness + 1e-8
    probs = shifted / shifted.sum()
    indices = np.random.choice(len(population), size=n_select, p=probs, replace=True)
    return [population[i][0].clone() for i in indices]


def _run_sa_refinement(
    controller,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    temperature: float,
    n_steps: int,
    mutation_rate: float,
    mutation_scale: float,
    index_swap_rate: float,
) -> Tuple[Any, float]:
    """Run SA refinement steps on a single controller."""
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(x_train)
        current_fitness = -torch.mean((pred - y_train) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        # Use the controller's mutate method with appropriate args
        if hasattr(mutant, 'mutate'):
            try:
                mutant.mutate(
                    weight_rate=mutation_rate,
                    weight_scale=mutation_scale,
                    index_swap_rate=index_swap_rate
                )
            except TypeError:
                # Fallback for controllers with different mutate signature
                mutant.mutate(rate=mutation_rate, scale=mutation_scale)

        with torch.no_grad():
            pred = mutant.forward(x_train)
            mutant_fitness = -torch.mean((pred - y_train) ** 2).item()

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def train_gsa(
    controller_factory: Callable[[], Any],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    population_size: int = 50,
    generations: int = 300,
    seed_fraction: float = 0.05,
    sa_steps_per_gen: int = 20,
    t_initial: float = 0.1,
    t_final: float = 0.0001,
    mutation_rate: float = 0.15,
    mutation_scale: float = 0.15,
    index_swap_rate: float = 0.1,
    verbose: bool = True,
    report_interval: int = 20,
    callback: Optional[Callable[[int, Any, float, float], None]] = None,
) -> Tuple[Any, Dict, List[Dict]]:
    """
    Train using Genetic Simulated Annealing (GSA).

    Combines population-based selection (GA) with SA refinement per member.
    Based on Du et al. (2018) "A Genetic Simulated Annealing Algorithm".

    Args:
        controller_factory: Callable that creates a new controller instance
        x_train: Training inputs
        y_train: Training targets (one-hot encoded for classification)
        x_test: Test inputs
        y_test: Test targets (class labels for classification)
        population_size: Number of individuals in population
        generations: Number of generations to run
        seed_fraction: Fraction of top performers kept as seeds (default 5%)
        sa_steps_per_gen: SA refinement steps per member per generation
        t_initial: Initial SA temperature
        t_final: Final SA temperature
        mutation_rate: Probability of mutating each weight
        mutation_scale: Standard deviation of weight mutations
        index_swap_rate: Probability of swapping input indices (for sparse nets)
        verbose: Print progress
        report_interval: Generations between progress reports
        callback: Optional callback(gen, best_controller, best_fitness, test_acc)

    Returns:
        best: Best controller found
        final_results: Dict with final metrics
        history: List of per-generation metrics

    Example:
        from core.training import train_gsa
        from core.models import UltraSparseController

        best, results, history = train_gsa(
            controller_factory=lambda: UltraSparseController(64, 32, 10, 4),
            x_train=X_train, y_train=y_onehot,
            x_test=X_test, y_test=y_test,
            generations=1000,
            verbose=True
        )
        print(f"Test accuracy: {results['test_accuracy']:.1%}")
    """
    # Initialize population
    population = []
    for _ in range(population_size):
        controller = controller_factory()
        with torch.no_grad():
            pred = controller.forward(x_train)
            fitness = -torch.mean((pred - y_train) ** 2).item()
        population.append((controller, fitness))

    # Temperature schedule
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    # Track best
    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    history = []

    for gen in range(generations):
        # Sort by fitness (descending)
        population.sort(key=lambda x: x[1], reverse=True)

        # Natural Selection
        n_seeds = max(1, int(seed_fraction * population_size))
        n_roulette = population_size - n_seeds

        # Seeds: top performers with SA refinement
        seeds = []
        for c, _ in population[:n_seeds]:
            improved, new_fitness = _run_sa_refinement(
                c, x_train, y_train, temperature, sa_steps_per_gen,
                mutation_rate, mutation_scale, index_swap_rate
            )
            seeds.append((improved, new_fitness))

        # Roulette selection for rest
        roulette_controllers = _roulette_select(population, n_roulette)

        new_population = seeds.copy()
        for controller in roulette_controllers:
            improved, new_fitness = _run_sa_refinement(
                controller, x_train, y_train, temperature, sa_steps_per_gen,
                mutation_rate, mutation_scale, index_swap_rate
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
        mean_fitness = np.mean([f for _, f in population])

        # Evaluate test accuracy
        with torch.no_grad():
            preds = best_controller.forward(x_test)
            if preds.dim() > 1 and preds.size(1) > 1:
                # Classification: argmax
                test_accuracy = (preds.argmax(dim=1) == y_test).float().mean().item()
            else:
                # Regression: use MSE
                test_accuracy = -torch.mean((preds.squeeze() - y_test) ** 2).item()

        # Record history
        history.append({
            'generation': gen,
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'test_accuracy': test_accuracy,
            'temperature': temperature,
        })

        # Callback
        if callback is not None:
            callback(gen, best_controller, best_fitness, test_accuracy)

        # Report progress
        if verbose and (gen % report_interval == 0 or gen == generations - 1):
            print(f"Gen {gen:4d}: Acc={test_accuracy:.1%}, "
                  f"Fitness={best_fitness:.4f}, T={temperature:.6f}")

    # Final evaluation
    with torch.no_grad():
        preds = best_controller.forward(x_test)
        if preds.dim() > 1 and preds.size(1) > 1:
            final_test_acc = (preds.argmax(dim=1) == y_test).float().mean().item()
        else:
            final_test_acc = -torch.mean((preds.squeeze() - y_test) ** 2).item()

        train_preds = best_controller.forward(x_train)
        if train_preds.dim() > 1 and train_preds.size(1) > 1:
            train_acc = None  # Can't compute without original labels
        else:
            train_acc = -torch.mean((train_preds.squeeze() - y_train) ** 2).item()

    params = best_controller.num_parameters() if hasattr(best_controller, 'num_parameters') else None

    final_results = {
        'test_accuracy': final_test_acc,
        'train_accuracy': train_acc,
        'params': params,
        'generations': generations,
        'population_size': population_size,
    }

    return best_controller, final_results, history
