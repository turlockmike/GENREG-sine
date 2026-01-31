"""
Experiment: MetabolicFitness - Efficiency-Aware Fitness Function

Question: Does adding an efficiency penalty to fitness improve the accuracy/efficiency
tradeoff compared to pure accuracy-based fitness?

Hypothesis: Biological neurons have metabolic costs, creating pressure for efficient
circuits. Adding a connection cost term will:
1. Prevent networks from becoming unnecessarily dense
2. Find better accuracy/efficiency tradeoffs
3. Be essential for architectures where sparsity isn't enforced (like RetinalNet)

Fitness functions tested:
    1. baseline:    fitness = -MSE (current approach)
    2. linear:      fitness = -MSE - λ * num_connections
    3. log:         fitness = -MSE - λ * log(num_connections)
    4. ratio:       fitness = accuracy / (1 + λ * num_connections)

Baseline: minimal_mut (95.0%) - pop=100, idx=0.02, wt=0.02, H=32, K=4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
import argparse
import json
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MetabolicSparseController:
    """
    UltraSparse controller with metabolic cost tracking.
    Same architecture as UltraSparse but fitness includes efficiency term.
    """

    def __init__(self, input_size, hidden_size, output_size, inputs_per_neuron):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k = inputs_per_neuron

        # Layer 1: input -> hidden (sparse)
        self.indices = np.random.randint(0, input_size, (hidden_size, inputs_per_neuron))
        self.w1 = np.random.randn(hidden_size, inputs_per_neuron).astype(np.float32) * 0.5
        self.b1 = np.zeros(hidden_size, dtype=np.float32)

        # Output layer (dense)
        self.w2 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        gathered = x[:, self.indices]
        pre_act = np.einsum('bnk,nk->bn', gathered, self.w1) + self.b1
        h = np.tanh(pre_act)
        out = np.tanh(h @ self.w2.T + self.b2)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02):
        # Weight mutations
        mask = np.random.random(self.w1.shape) < weight_rate
        self.w1 += mask * np.random.randn(*self.w1.shape).astype(np.float32) * weight_scale
        mask = np.random.random(self.b1.shape) < weight_rate
        self.b1 += mask * np.random.randn(*self.b1.shape).astype(np.float32) * weight_scale
        mask = np.random.random(self.w2.shape) < weight_rate
        self.w2 += mask * np.random.randn(*self.w2.shape).astype(np.float32) * weight_scale
        mask = np.random.random(self.b2.shape) < weight_rate
        self.b2 += mask * np.random.randn(*self.b2.shape).astype(np.float32) * weight_scale

        # Index mutations
        for h in range(self.hidden_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k)
                self.indices[h, idx] = np.random.randint(0, self.input_size)

    def clone(self):
        new = MetabolicSparseController(self.input_size, self.hidden_size, self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.b1 = self.b1.copy()
        new.w2 = self.w2.copy()
        new.b2 = self.b2.copy()
        return new

    def num_parameters(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size

    def num_connections(self):
        """Number of sparse connections (for metabolic cost)."""
        return self.hidden_size * self.k


def compute_fitness(mse, num_connections, fitness_type, lambda_val, accuracy=None):
    """
    Compute fitness with different efficiency penalties.

    Args:
        mse: Mean squared error (lower is better)
        num_connections: Number of active connections
        fitness_type: 'baseline', 'linear', 'log', 'ratio'
        lambda_val: Coefficient for efficiency penalty
        accuracy: Classification accuracy (for ratio fitness)
    """
    if fitness_type == 'baseline':
        # Original: pure accuracy
        return -mse

    elif fitness_type == 'linear':
        # Linear penalty: fitness = -MSE - λ * connections
        return -mse - lambda_val * num_connections

    elif fitness_type == 'log':
        # Log penalty: fitness = -MSE - λ * log(connections)
        return -mse - lambda_val * np.log(num_connections + 1)

    elif fitness_type == 'ratio':
        # Ratio: fitness = accuracy / (1 + λ * connections)
        # Note: accuracy should be in [0, 1]
        if accuracy is None:
            accuracy = 1.0 / (1.0 + mse)  # Approximate accuracy from MSE
        return accuracy / (1.0 + lambda_val * num_connections)

    else:
        raise ValueError(f"Unknown fitness type: {fitness_type}")


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with metabolic fitness function."""
    start_time = time.time()

    H = config['H']
    K = config['K']
    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']
    index_swap_rate = config['index_swap_rate']
    weight_rate = config['weight_rate']
    weight_scale = config['weight_scale']
    fitness_type = config['fitness_type']
    lambda_val = config['lambda']

    # Initialize CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,mse,num_connections\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        controller = MetabolicSparseController(64, H, 10, K)
        with torch.no_grad():
            pred = controller.forward(X_train)
            mse = torch.mean((pred - y_onehot) ** 2).item()
            fitness = compute_fitness(mse, controller.num_connections(),
                                      fitness_type, lambda_val)
        population.append((controller, fitness, mse))

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]
    best_mse = max(population, key=lambda x: x[1])[2]

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        n_seeds = max(1, int(seed_fraction * pop_size))

        fitnesses = np.array([f for _, f, _ in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        # Seeds get SA refinement
        for c, _, _ in population[:n_seeds]:
            improved, new_fitness, new_mse = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps,
                weight_rate, weight_scale, index_swap_rate,
                fitness_type, lambda_val
            )
            new_population.append((improved, new_fitness, new_mse))

        # Roulette selection for rest
        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness, new_mse = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps,
                weight_rate, weight_scale, index_swap_rate,
                fitness_type, lambda_val
            )
            new_population.append((improved, new_fitness, new_mse))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]
            best_mse = gen_best[2]

        temperature *= decay

        # Calculate test accuracy
        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        elapsed = time.time() - start_time
        num_connections = best_controller.num_connections()

        # Write to CSV
        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{best_mse},{num_connections}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, mse={best_mse:.4f}, connections={num_connections}",
                  flush=True)

    # Final accuracy
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller.num_parameters(), time.time() - start_time, best_mse


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps,
              weight_rate, weight_scale, index_swap_rate,
              fitness_type, lambda_val):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_mse = torch.mean((pred - y_onehot) ** 2).item()
        current_fitness = compute_fitness(current_mse, current.num_connections(),
                                          fitness_type, lambda_val)

    best = current.clone()
    best_fitness = current_fitness
    best_mse = current_mse

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(weight_rate, weight_scale, index_swap_rate)

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_mse = torch.mean((pred - y_onehot) ** 2).item()
            mutant_fitness = compute_fitness(mutant_mse, mutant.num_connections(),
                                             fitness_type, lambda_val)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            current_mse = mutant_mse
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
                best_mse = current_mse

    return best, best_fitness, best_mse


def load_data():
    data = load_digits()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    return X_train, y_onehot, X_test, y_test


# Base config (matches minimal_mut winner)
BASE_CONFIG = {
    'H': 32,
    'K': 4,
    'pop_size': 100,
    'generations': 1000,
    'sa_steps': 20,
    'seed_fraction': 0.05,
    'index_swap_rate': 0.02,
    'weight_rate': 0.02,
    'weight_scale': 0.15,
    'seed': 42,
}

# Configurations to test
CONFIGS = {
    # Baseline: no efficiency penalty (current best)
    'baseline': {**BASE_CONFIG, 'fitness_type': 'baseline', 'lambda': 0.0},

    # Linear penalty at different λ values
    'linear_0001': {**BASE_CONFIG, 'fitness_type': 'linear', 'lambda': 0.0001},
    'linear_0005': {**BASE_CONFIG, 'fitness_type': 'linear', 'lambda': 0.0005},
    'linear_001': {**BASE_CONFIG, 'fitness_type': 'linear', 'lambda': 0.001},
    'linear_005': {**BASE_CONFIG, 'fitness_type': 'linear', 'lambda': 0.005},

    # Log penalty at different λ values
    'log_001': {**BASE_CONFIG, 'fitness_type': 'log', 'lambda': 0.001},
    'log_005': {**BASE_CONFIG, 'fitness_type': 'log', 'lambda': 0.005},
    'log_01': {**BASE_CONFIG, 'fitness_type': 'log', 'lambda': 0.01},

    # Ratio fitness at different λ values
    'ratio_0001': {**BASE_CONFIG, 'fitness_type': 'ratio', 'lambda': 0.0001},
    'ratio_001': {**BASE_CONFIG, 'fitness_type': 'ratio', 'lambda': 0.001},
    'ratio_01': {**BASE_CONFIG, 'fitness_type': 'ratio', 'lambda': 0.01},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline',
                        help='Config name: baseline, linear_001, log_01, etc.')
    parser.add_argument('--list', action='store_true', help='List all configs')
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name in sorted(CONFIGS.keys()):
            cfg = CONFIGS[name]
            print(f"  {name}: fitness_type={cfg['fitness_type']}, λ={cfg['lambda']}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        print("Use --list to see available configs")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"METABOLIC FITNESS EXPERIMENT: {config_name}")
    print("=" * 60)
    print(f"Fitness type: {config['fitness_type']}, λ={config['lambda']}")

    X_train, y_onehot, X_test, y_test = load_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"metabolic_{config_name}.csv"

    # Save config metadata
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    accuracy, params, elapsed, final_mse = run_gsa(
        X_train, y_onehot, X_test, y_test, config, csv_path
    )

    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Final MSE: {final_mse:.4f}")
    print(f"Parameters: {params}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
