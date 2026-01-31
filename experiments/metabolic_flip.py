"""
Experiment: MetabolicFlip - Combining Flip Mutations with Metabolic Fitness

Question: Does adding metabolic (efficiency) pressure help flip-based architectures
converge to sparse solutions and improve accuracy?

Previous findings:
- RetinalNet without metabolic fitness: masks converged to 40-50% density (too dense)
- MetabolicFitness on fixed-K: no effect (connections are fixed)
- flip_rate=0.01 worked best for RetinalNet

This experiment tests:
1. FlipSparse: Single-layer with flip mutations (like UltraSparse but flip instead of swap)
2. RetinalNet: Sensor bottleneck with dual flip masks
Both with metabolic fitness at various λ values.

Architectures:
    FlipSparse:  Input (64) --[flip mask]--> Hidden (32) --[dense]--> Output (10)
    RetinalNet:  Input (64) --[flip mask1]--> Sensor (S) --[flip mask2]--> Hidden (32) --[dense]--> Output (10)

Fitness: -MSE - λ * num_active_connections

Baseline: UltraSparse (95.0%) - fixed K=4, index swaps
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


class FlipSparse:
    """
    Single-layer sparse network with flip mutations (not index swaps).

    Like UltraSparse but:
    - Uses binary mask instead of index list
    - Connections can flip on/off
    - Number of active connections emerges through evolution
    """

    def __init__(self, input_size, hidden_size, output_size, initial_density=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer 1: Input -> Hidden (sparse, evolvable mask)
        self.mask = (np.random.random((hidden_size, input_size)) < initial_density).astype(np.float32)
        self.w1 = np.random.randn(hidden_size, input_size).astype(np.float32) * 0.5
        self.b1 = np.zeros(hidden_size, dtype=np.float32)

        # Layer 2: Hidden -> Output (dense)
        self.w2 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        masked_w1 = self.w1 * self.mask
        h = np.tanh(x @ masked_w1.T + self.b1)
        out = np.tanh(h @ self.w2.T + self.b2)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, flip_rate=0.01):
        # Weight mutations (only for active connections)
        weight_mask = np.random.random(self.w1.shape) < weight_rate
        self.w1 += weight_mask * self.mask * np.random.randn(*self.w1.shape).astype(np.float32) * weight_scale

        mask = np.random.random(self.w2.shape) < weight_rate
        self.w2 += mask * np.random.randn(*self.w2.shape).astype(np.float32) * weight_scale

        for b in [self.b1, self.b2]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale

        # Mask flip mutations
        flip = np.random.random(self.mask.shape) < flip_rate
        self.mask = np.where(flip, 1.0 - self.mask, self.mask)

    def clone(self):
        new = FlipSparse(self.input_size, self.hidden_size, self.output_size, initial_density=0.0)
        new.mask = self.mask.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        return new

    def num_active_connections(self):
        return int(self.mask.sum())

    def density(self):
        return self.mask.sum() / self.mask.size


class RetinalNet:
    """Sensor bottleneck with dual evolvable masks."""

    def __init__(self, input_size, sensor_size, hidden_size, output_size, initial_density=0.1):
        self.input_size = input_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer 1: Input -> Sensor (sparse)
        self.mask1 = (np.random.random((sensor_size, input_size)) < initial_density).astype(np.float32)
        self.w1 = np.random.randn(sensor_size, input_size).astype(np.float32) * 0.5
        self.b1 = np.zeros(sensor_size, dtype=np.float32)

        # Layer 2: Sensor -> Hidden (sparse)
        self.mask2 = (np.random.random((hidden_size, sensor_size)) < initial_density).astype(np.float32)
        self.w2 = np.random.randn(hidden_size, sensor_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # Layer 3: Hidden -> Output (dense)
        self.w3 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        sensor = np.tanh(x @ (self.w1 * self.mask1).T + self.b1)
        hidden = np.tanh(sensor @ (self.w2 * self.mask2).T + self.b2)
        out = np.tanh(hidden @ self.w3.T + self.b3)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, flip_rate=0.01):
        # Weight mutations
        for w, mask in [(self.w1, self.mask1), (self.w2, self.mask2)]:
            weight_mask = np.random.random(w.shape) < weight_rate
            w += weight_mask * mask * np.random.randn(*w.shape).astype(np.float32) * weight_scale

        mask = np.random.random(self.w3.shape) < weight_rate
        self.w3 += mask * np.random.randn(*self.w3.shape).astype(np.float32) * weight_scale

        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale

        # Mask flip mutations
        flip1 = np.random.random(self.mask1.shape) < flip_rate
        self.mask1 = np.where(flip1, 1.0 - self.mask1, self.mask1)
        flip2 = np.random.random(self.mask2.shape) < flip_rate
        self.mask2 = np.where(flip2, 1.0 - self.mask2, self.mask2)

    def clone(self):
        new = RetinalNet(self.input_size, self.sensor_size, self.hidden_size,
                         self.output_size, initial_density=0.0)
        new.mask1 = self.mask1.copy()
        new.mask2 = self.mask2.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.w3 = self.w3.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        new.b3 = self.b3.copy()
        return new

    def num_active_connections(self):
        return int(self.mask1.sum() + self.mask2.sum())

    def density(self):
        total = self.mask1.size + self.mask2.size
        active = self.mask1.sum() + self.mask2.sum()
        return active / total


class UltraSparse:
    """Baseline: Fixed-K with index swaps (current best)."""

    def __init__(self, input_size, hidden_size, output_size, k):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k = k

        self.indices = np.random.randint(0, input_size, (hidden_size, k))
        self.w1 = np.random.randn(hidden_size, k).astype(np.float32) * 0.5
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        gathered = x[:, self.indices]
        h = np.tanh(np.einsum('bnk,nk->bn', gathered, self.w1) + self.b1)
        out = np.tanh(h @ self.w2.T + self.b2)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02):
        for w in [self.w1, self.w2]:
            mask = np.random.random(w.shape) < weight_rate
            w += mask * np.random.randn(*w.shape).astype(np.float32) * weight_scale
        for b in [self.b1, self.b2]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale
        for h in range(self.hidden_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k)
                self.indices[h, idx] = np.random.randint(0, self.input_size)

    def clone(self):
        new = UltraSparse(self.input_size, self.hidden_size, self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        return new

    def num_active_connections(self):
        return self.hidden_size * self.k

    def density(self):
        return self.k / self.input_size


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with metabolic fitness."""
    start_time = time.time()

    arch = config['arch']
    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']
    weight_rate = config['weight_rate']
    weight_scale = config['weight_scale']
    lambda_val = config['lambda']

    # Initialize CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,mse,active_connections,density\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))

        if arch == 'baseline':
            controller = UltraSparse(64, config['H'], 10, config['K'])
        elif arch == 'flipsparse':
            controller = FlipSparse(64, config['H'], 10, config['initial_density'])
        elif arch == 'retinal':
            controller = RetinalNet(64, config['S'], config['H'], 10, config['initial_density'])

        with torch.no_grad():
            pred = controller.forward(X_train)
            mse = torch.mean((pred - y_onehot) ** 2).item()
            connections = controller.num_active_connections()
            fitness = -mse - lambda_val * connections
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

        for c, _, _ in population[:n_seeds]:
            improved, new_fitness, new_mse = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps, config, lambda_val
            )
            new_population.append((improved, new_fitness, new_mse))

        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness, new_mse = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps, config, lambda_val
            )
            new_population.append((improved, new_fitness, new_mse))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]
            best_mse = gen_best[2]

        temperature *= decay

        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        connections = best_controller.num_active_connections()
        density = best_controller.density()
        elapsed = time.time() - start_time

        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{best_mse},{connections},{density:.3f}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, mse={best_mse:.4f}, conn={connections}, density={density:.1%}",
                  flush=True)

    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller, time.time() - start_time


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps, config, lambda_val):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_mse = torch.mean((pred - y_onehot) ** 2).item()
        current_fitness = -current_mse - lambda_val * current.num_active_connections()

    best = current.clone()
    best_fitness = current_fitness
    best_mse = current_mse

    for _ in range(n_steps):
        mutant = current.clone()

        if config['arch'] == 'baseline':
            mutant.mutate(config['weight_rate'], config['weight_scale'], config['index_swap_rate'])
        else:
            mutant.mutate(config['weight_rate'], config['weight_scale'], config['flip_rate'])

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_mse = torch.mean((pred - y_onehot) ** 2).item()
            mutant_fitness = -mutant_mse - lambda_val * mutant.num_active_connections()

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


# Base settings
BASE = {
    'pop_size': 100, 'generations': 1000, 'sa_steps': 20, 'seed_fraction': 0.05,
    'weight_rate': 0.02, 'weight_scale': 0.15, 'seed': 42,
}

CONFIGS = {
    # Baseline: Current best (UltraSparse with index swaps)
    'baseline': {
        **BASE, 'arch': 'baseline', 'H': 32, 'K': 4,
        'index_swap_rate': 0.02, 'lambda': 0.0,
    },

    # FlipSparse: Single layer with flip mutations, no metabolic penalty
    'flip_none': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0,
    },
    # FlipSparse with metabolic penalty at various λ
    'flip_λ0001': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0001,
    },
    'flip_λ0005': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0005,
    },
    'flip_λ001': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.001,
    },
    'flip_λ002': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.002,
    },

    # RetinalNet: Sensor bottleneck, no metabolic penalty
    'retinal_none': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0,
    },
    # RetinalNet with metabolic penalty at various λ
    'retinal_λ0001': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0001,
    },
    'retinal_λ0005': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.0005,
    },
    'retinal_λ001': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.001,
    },
    'retinal_λ002': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.1, 'lambda': 0.002,
    },

    # Higher initial density variants (start at 20% instead of 10%)
    'flip_dense_λ001': {
        **BASE, 'arch': 'flipsparse', 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.2, 'lambda': 0.001,
    },
    'retinal_dense_λ001': {
        **BASE, 'arch': 'retinal', 'S': 16, 'H': 32,
        'flip_rate': 0.01, 'initial_density': 0.2, 'lambda': 0.001,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='baseline')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name in sorted(CONFIGS.keys()):
            cfg = CONFIGS[name]
            print(f"  {name}: arch={cfg['arch']}, λ={cfg['lambda']}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"METABOLIC FLIP EXPERIMENT: {config_name}")
    print("=" * 60)
    print(f"Architecture: {config['arch']}, λ={config['lambda']}")

    X_train, y_onehot, X_test, y_test = load_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"mflip_{config_name}.csv"
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    accuracy, controller, elapsed = run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path)

    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Active connections: {controller.num_active_connections()}")
    print(f"Density: {controller.density():.1%}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
