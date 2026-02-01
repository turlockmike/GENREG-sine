"""
Experiment: Digits with Many Seeds

Hypothesis: More seeds with smaller population is better than fewer seeds with larger population.

Test: pop=50, 10 seeds for both fixed K=4 and variableK on digits.
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
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FixedK:
    """Fixed-K sparse controller."""

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
        new = FixedK(self.input_size, self.hidden_size, self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        return new

    def num_connections(self):
        return self.hidden_size * self.k

    def info(self):
        return {'connections': self.num_connections(), 'k': self.k, 'type': 'fixed'}


class VariableK:
    """Per-neuron variable K with index swaps."""

    def __init__(self, input_size, hidden_size, output_size, initial_k=4, max_k=16):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_k = max_k

        self.k = np.full(hidden_size, initial_k, dtype=np.int32)
        self.indices = np.zeros((hidden_size, max_k), dtype=np.int32)
        self.w1 = np.zeros((hidden_size, max_k), dtype=np.float32)
        for h in range(hidden_size):
            self.indices[h, :initial_k] = np.random.randint(0, input_size, initial_k)
            self.w1[h, :initial_k] = np.random.randn(initial_k).astype(np.float32) * 0.5

        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        batch_size = x.shape[0]
        h = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        for i in range(self.hidden_size):
            ki = self.k[i]
            if ki > 0:
                gathered = x[:, self.indices[i, :ki]]
                h[:, i] = gathered @ self.w1[i, :ki] + self.b1[i]
        h = np.tanh(h)
        out = np.tanh(h @ self.w2.T + self.b2)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02,
               grow_rate=0.01, shrink_rate=0.01):
        for h in range(self.hidden_size):
            ki = self.k[h]
            if ki > 0:
                mask = np.random.random(ki) < weight_rate
                self.w1[h, :ki] += mask * np.random.randn(ki).astype(np.float32) * weight_scale
        mask = np.random.random(self.w2.shape) < weight_rate
        self.w2 += mask * np.random.randn(*self.w2.shape).astype(np.float32) * weight_scale
        for b in [self.b1, self.b2]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale

        for h in range(self.hidden_size):
            ki = self.k[h]
            if ki > 0 and np.random.random() < index_swap_rate:
                idx = np.random.randint(0, ki)
                self.indices[h, idx] = np.random.randint(0, self.input_size)

        for h in range(self.hidden_size):
            if np.random.random() < shrink_rate and self.k[h] > 1:
                self.k[h] -= 1
            elif np.random.random() < grow_rate and self.k[h] < self.max_k:
                new_idx = self.k[h]
                self.indices[h, new_idx] = np.random.randint(0, self.input_size)
                self.w1[h, new_idx] = np.float32(np.random.randn() * 0.5)
                self.k[h] += 1

    def clone(self):
        new = VariableK(self.input_size, self.hidden_size, self.output_size,
                        initial_k=1, max_k=self.max_k)
        new.k = self.k.copy()
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        return new

    def num_connections(self):
        return int(self.k.sum())

    def info(self):
        return {
            'connections': self.num_connections(),
            'mean_k': float(self.k.mean()),
            'min_k': int(self.k.min()),
            'max_k': int(self.k.max()),
            'type': 'variable'
        }


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA."""
    start_time = time.time()

    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']
    input_size = X_train.shape[1]
    output_size = y_onehot.shape[1]

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,connections,mean_k\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        if config['arch'] == 'variablek':
            controller = VariableK(input_size, config['H'], output_size,
                                   config['initial_k'], config.get('max_k', 16))
        else:
            controller = FixedK(input_size, config['H'], output_size, config['K'])

        with torch.no_grad():
            pred = controller.forward(X_train)
            fitness = -torch.mean((pred - y_onehot) ** 2).item()
        population.append((controller, fitness))

    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)
        n_seeds = max(1, int(seed_fraction * pop_size))

        fitnesses = np.array([f for _, f in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, sa_steps, config)
            new_population.append((improved, new_fitness))

        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, sa_steps, config)
            new_population.append((improved, new_fitness))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        temperature *= decay

        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        info = best_controller.info()
        elapsed = time.time() - start_time

        if csv_path:
            mean_k = info.get('mean_k', info.get('k', 0))
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{info['connections']},{mean_k}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, conn={info['connections']}, time={elapsed:.0f}s", flush=True)

    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller, time.time() - start_time


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps, config):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_fitness = -torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        if config['arch'] == 'variablek':
            mutant.mutate(config['weight_rate'], config['weight_scale'],
                          config['index_swap_rate'], config['grow_rate'], config['shrink_rate'])
        else:
            mutant.mutate(config['weight_rate'], config['weight_scale'], config['index_swap_rate'])

        with torch.no_grad():
            pred = mutant.forward(X_train)
            mutant_fitness = -torch.mean((pred - y_onehot) ** 2).item()

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def load_digits_data():
    """Load digits dataset."""
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


# Settings: pop=50, 1000 generations, 10 seeds each
BASE = {
    'H': 32, 'pop_size': 50, 'generations': 1000, 'sa_steps': 20, 'seed_fraction': 0.05,
    'weight_rate': 0.02, 'weight_scale': 0.15, 'index_swap_rate': 0.02,
    'grow_rate': 0.01, 'shrink_rate': 0.01, 'initial_k': 4, 'max_k': 16, 'K': 4,
}

CONFIGS = {}

# Generate configs: 2 K types × 10 seeds = 20 configs
for k_type in ['fixed', 'variable']:
    for seed in range(10):
        name = f"digits_{k_type}_seed{seed}"
        CONFIGS[name] = {
            **BASE,
            'arch': 'variablek' if k_type == 'variable' else 'fixedk',
            'seed': seed * 100,
            'description': f"digits, {'variableK init=4' if k_type == 'variable' else 'fixed K=4'}, seed={seed}",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='digits_fixed_seed0')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name in sorted(CONFIGS.keys()):
            cfg = CONFIGS[name]
            print(f"  {name}: {cfg.get('description', '')}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"DIGITS SEEDS: {config_name}")
    print("=" * 60)
    print(f"Description: {config.get('description', '-')}")
    print(f"Population: {config['pop_size']}, Generations: {config['generations']}")

    X_train, y_onehot, X_test, y_test = load_digits_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"digits_seeds_{config_name}.csv"
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    accuracy, controller, elapsed = run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path)

    info = controller.info()
    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Info: {info}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
