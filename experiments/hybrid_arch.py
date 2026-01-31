"""
Experiment: Hybrid Architectures - Combining Fixed-K with Sensor Bottleneck

Question: Can we combine the proven fixed-K constraint with sensor bottleneck architecture
to get the best of both worlds?

Architectures tested:
1. Baseline: UltraSparse (Input --[K=4, swaps]--> Hidden --[dense]--> Output)
2. SensorK: Both layers fixed-K with swaps (Input --[K, swaps]--> Sensor --[K, swaps]--> Hidden)
3. SensorDense: Fixed-K input, dense routing (Input --[K, swaps]--> Sensor --[dense]--> Hidden)
4. SensorFlip: Fixed-K input, flip routing (Input --[K, swaps]--> Sensor --[flip]--> Hidden)
5. VariableK: Per-neuron variable K with swaps (Input --[K[i], swaps]--> Hidden)

Baseline: UltraSparse 94.7% - fixed K=4, index swaps
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


class UltraSparse:
    """Baseline: Fixed-K with index swaps (proven best)."""

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

    def num_connections(self):
        return self.hidden_size * self.k

    def info(self):
        return {'connections': self.num_connections(), 'k': self.k}


class SensorK:
    """
    Option 1: Sensor bottleneck with fixed-K on BOTH layers.
    Input --[K1 per sensor, swaps]--> Sensor --[K2 per hidden, swaps]--> Hidden --[dense]--> Output
    """

    def __init__(self, input_size, sensor_size, hidden_size, output_size, k1, k2):
        self.input_size = input_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k1 = k1
        self.k2 = k2

        # Input -> Sensor (sparse, K1 per sensor neuron)
        self.indices1 = np.random.randint(0, input_size, (sensor_size, k1))
        self.w1 = np.random.randn(sensor_size, k1).astype(np.float32) * 0.5
        self.b1 = np.zeros(sensor_size, dtype=np.float32)

        # Sensor -> Hidden (sparse, K2 per hidden neuron)
        self.indices2 = np.random.randint(0, sensor_size, (hidden_size, k2))
        self.w2 = np.random.randn(hidden_size, k2).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # Hidden -> Output (dense)
        self.w3 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        # Input -> Sensor
        gathered1 = x[:, self.indices1]
        sensor = np.tanh(np.einsum('bnk,nk->bn', gathered1, self.w1) + self.b1)
        # Sensor -> Hidden
        gathered2 = sensor[:, self.indices2]
        hidden = np.tanh(np.einsum('bnk,nk->bn', gathered2, self.w2) + self.b2)
        # Hidden -> Output
        out = np.tanh(hidden @ self.w3.T + self.b3)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02):
        for w in [self.w1, self.w2, self.w3]:
            mask = np.random.random(w.shape) < weight_rate
            w += mask * np.random.randn(*w.shape).astype(np.float32) * weight_scale
        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale
        # Index swaps for layer 1
        for s in range(self.sensor_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k1)
                self.indices1[s, idx] = np.random.randint(0, self.input_size)
        # Index swaps for layer 2
        for h in range(self.hidden_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k2)
                self.indices2[h, idx] = np.random.randint(0, self.sensor_size)

    def clone(self):
        new = SensorK(self.input_size, self.sensor_size, self.hidden_size,
                      self.output_size, self.k1, self.k2)
        new.indices1 = self.indices1.copy()
        new.indices2 = self.indices2.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.w3 = self.w3.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        new.b3 = self.b3.copy()
        return new

    def num_connections(self):
        return self.sensor_size * self.k1 + self.hidden_size * self.k2

    def info(self):
        return {'connections': self.num_connections(), 'k1': self.k1, 'k2': self.k2,
                'sensor': self.sensor_size}


class SensorDense:
    """
    Option 2: Fixed-K input selection, dense sensor routing.
    Input --[K per sensor, swaps]--> Sensor --[dense]--> Hidden --[dense]--> Output
    """

    def __init__(self, input_size, sensor_size, hidden_size, output_size, k):
        self.input_size = input_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k = k

        # Input -> Sensor (sparse, K per sensor neuron)
        self.indices = np.random.randint(0, input_size, (sensor_size, k))
        self.w1 = np.random.randn(sensor_size, k).astype(np.float32) * 0.5
        self.b1 = np.zeros(sensor_size, dtype=np.float32)

        # Sensor -> Hidden (dense)
        self.w2 = np.random.randn(hidden_size, sensor_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # Hidden -> Output (dense)
        self.w3 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        gathered = x[:, self.indices]
        sensor = np.tanh(np.einsum('bnk,nk->bn', gathered, self.w1) + self.b1)
        hidden = np.tanh(sensor @ self.w2.T + self.b2)
        out = np.tanh(hidden @ self.w3.T + self.b3)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02):
        mask = np.random.random(self.w1.shape) < weight_rate
        self.w1 += mask * np.random.randn(*self.w1.shape).astype(np.float32) * weight_scale
        for w in [self.w2, self.w3]:
            mask = np.random.random(w.shape) < weight_rate
            w += mask * np.random.randn(*w.shape).astype(np.float32) * weight_scale
        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale
        for s in range(self.sensor_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k)
                self.indices[s, idx] = np.random.randint(0, self.input_size)

    def clone(self):
        new = SensorDense(self.input_size, self.sensor_size, self.hidden_size,
                          self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.w3 = self.w3.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        new.b3 = self.b3.copy()
        return new

    def num_connections(self):
        return self.sensor_size * self.k + self.hidden_size * self.sensor_size

    def info(self):
        return {'connections': self.num_connections(), 'k': self.k, 'sensor': self.sensor_size}


class SensorFlip:
    """
    Option 3: Fixed-K input selection, flip mask routing.
    Input --[K per sensor, swaps]--> Sensor --[flip mask]--> Hidden --[dense]--> Output
    """

    def __init__(self, input_size, sensor_size, hidden_size, output_size, k, mask_density=0.3):
        self.input_size = input_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.k = k

        # Input -> Sensor (sparse, K per sensor neuron, swaps)
        self.indices = np.random.randint(0, input_size, (sensor_size, k))
        self.w1 = np.random.randn(sensor_size, k).astype(np.float32) * 0.5
        self.b1 = np.zeros(sensor_size, dtype=np.float32)

        # Sensor -> Hidden (flip mask)
        self.mask = (np.random.random((hidden_size, sensor_size)) < mask_density).astype(np.float32)
        self.w2 = np.random.randn(hidden_size, sensor_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # Hidden -> Output (dense)
        self.w3 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        gathered = x[:, self.indices]
        sensor = np.tanh(np.einsum('bnk,nk->bn', gathered, self.w1) + self.b1)
        hidden = np.tanh(sensor @ (self.w2 * self.mask).T + self.b2)
        out = np.tanh(hidden @ self.w3.T + self.b3)
        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, index_swap_rate=0.02, flip_rate=0.01):
        mask = np.random.random(self.w1.shape) < weight_rate
        self.w1 += mask * np.random.randn(*self.w1.shape).astype(np.float32) * weight_scale
        # W2 only active connections
        weight_mask = np.random.random(self.w2.shape) < weight_rate
        self.w2 += weight_mask * self.mask * np.random.randn(*self.w2.shape).astype(np.float32) * weight_scale
        mask = np.random.random(self.w3.shape) < weight_rate
        self.w3 += mask * np.random.randn(*self.w3.shape).astype(np.float32) * weight_scale
        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale
        # Index swaps
        for s in range(self.sensor_size):
            if np.random.random() < index_swap_rate:
                idx = np.random.randint(0, self.k)
                self.indices[s, idx] = np.random.randint(0, self.input_size)
        # Flip mask
        flip = np.random.random(self.mask.shape) < flip_rate
        self.mask = np.where(flip, 1.0 - self.mask, self.mask)

    def clone(self):
        new = SensorFlip(self.input_size, self.sensor_size, self.hidden_size,
                         self.output_size, self.k, mask_density=0.0)
        new.indices = self.indices.copy()
        new.mask = self.mask.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.w3 = self.w3.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        new.b3 = self.b3.copy()
        return new

    def num_connections(self):
        return self.sensor_size * self.k + int(self.mask.sum())

    def info(self):
        return {'connections': self.num_connections(), 'k': self.k, 'sensor': self.sensor_size,
                'mask_density': self.mask.sum() / self.mask.size}


class VariableK:
    """
    Option 4: Per-neuron variable K with index swaps.
    Each hidden neuron has its own K[i] that can grow/shrink.
    """

    def __init__(self, input_size, hidden_size, output_size, initial_k=4, max_k=16):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_k = max_k

        # Each neuron has variable number of inputs
        self.k = np.full(hidden_size, initial_k, dtype=np.int32)
        # Indices stored as padded array (max_k per neuron)
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
        # Weight mutations
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

        # Index swaps
        for h in range(self.hidden_size):
            ki = self.k[h]
            if ki > 0 and np.random.random() < index_swap_rate:
                idx = np.random.randint(0, ki)
                self.indices[h, idx] = np.random.randint(0, self.input_size)

        # Grow/shrink K
        for h in range(self.hidden_size):
            if np.random.random() < shrink_rate and self.k[h] > 1:
                # Remove last connection
                self.k[h] -= 1
            elif np.random.random() < grow_rate and self.k[h] < self.max_k:
                # Add new connection
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
        return {'connections': self.num_connections(), 'mean_k': self.k.mean(),
                'min_k': self.k.min(), 'max_k': self.k.max()}


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with specified architecture."""
    start_time = time.time()

    arch = config['arch']
    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']

    # Initialize CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,connections,extra_info\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        controller = create_controller(config)
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
            with open(csv_path, 'a') as f:
                extra = str(info).replace(',', ';')
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{info['connections']},{extra}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, conn={info['connections']}, {info}", flush=True)

    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller, time.time() - start_time


def create_controller(config):
    """Create controller based on architecture type."""
    arch = config['arch']
    if arch == 'baseline':
        return UltraSparse(64, config['H'], 10, config['K'])
    elif arch == 'sensork':
        return SensorK(64, config['S'], config['H'], 10, config['K1'], config['K2'])
    elif arch == 'sensordense':
        return SensorDense(64, config['S'], config['H'], 10, config['K'])
    elif arch == 'sensorflip':
        return SensorFlip(64, config['S'], config['H'], 10, config['K'], config.get('mask_density', 0.3))
    elif arch == 'variablek':
        return VariableK(64, config['H'], 10, config['initial_k'], config.get('max_k', 16))
    else:
        raise ValueError(f"Unknown arch: {arch}")


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps, config):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_fitness = -torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutate_controller(mutant, config)

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


def mutate_controller(controller, config):
    """Mutate controller based on architecture type."""
    arch = config['arch']
    wr = config.get('weight_rate', 0.02)
    ws = config.get('weight_scale', 0.15)
    isr = config.get('index_swap_rate', 0.02)

    if arch == 'baseline':
        controller.mutate(wr, ws, isr)
    elif arch == 'sensork':
        controller.mutate(wr, ws, isr)
    elif arch == 'sensordense':
        controller.mutate(wr, ws, isr)
    elif arch == 'sensorflip':
        controller.mutate(wr, ws, isr, config.get('flip_rate', 0.01))
    elif arch == 'variablek':
        controller.mutate(wr, ws, isr, config.get('grow_rate', 0.01), config.get('shrink_rate', 0.01))


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
    'weight_rate': 0.02, 'weight_scale': 0.15, 'index_swap_rate': 0.02, 'seed': 42,
}

CONFIGS = {
    # Baseline: UltraSparse (proven best)
    'baseline': {
        **BASE, 'arch': 'baseline', 'H': 32, 'K': 4,
        'description': 'UltraSparse baseline (proven 95%)',
    },

    # Option 1: SensorK - both layers fixed-K with swaps
    'sensork_S8': {
        **BASE, 'arch': 'sensork', 'S': 8, 'H': 32, 'K1': 4, 'K2': 4,
        'description': 'Sensor bottleneck, both layers K=4 swaps',
    },
    'sensork_S16': {
        **BASE, 'arch': 'sensork', 'S': 16, 'H': 32, 'K1': 4, 'K2': 4,
        'description': 'Sensor bottleneck S=16, both K=4',
    },
    'sensork_S16_K8': {
        **BASE, 'arch': 'sensork', 'S': 16, 'H': 32, 'K1': 8, 'K2': 4,
        'description': 'Sensor K1=8 (more input coverage), K2=4',
    },

    # Option 2: SensorDense - fixed-K input, dense routing
    'sensordense_S8': {
        **BASE, 'arch': 'sensordense', 'S': 8, 'H': 32, 'K': 4,
        'description': 'Fixed-K input selection, dense sensor->hidden',
    },
    'sensordense_S16': {
        **BASE, 'arch': 'sensordense', 'S': 16, 'H': 32, 'K': 4,
        'description': 'Fixed-K input S=16, dense routing',
    },

    # Option 3: SensorFlip - fixed-K input, flip mask routing
    'sensorflip_S16': {
        **BASE, 'arch': 'sensorflip', 'S': 16, 'H': 32, 'K': 4,
        'flip_rate': 0.01, 'mask_density': 0.3,
        'description': 'Fixed-K input, flip mask sensor->hidden',
    },
    'sensorflip_S16_dense': {
        **BASE, 'arch': 'sensorflip', 'S': 16, 'H': 32, 'K': 4,
        'flip_rate': 0.01, 'mask_density': 0.5,
        'description': 'Fixed-K input, denser flip mask (50%)',
    },

    # Option 4: VariableK - per-neuron variable K with swaps
    'variablek_init4': {
        **BASE, 'arch': 'variablek', 'H': 32, 'initial_k': 4, 'max_k': 16,
        'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Variable K per neuron, starts at 4',
    },
    'variablek_init2': {
        **BASE, 'arch': 'variablek', 'H': 32, 'initial_k': 2, 'max_k': 16,
        'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Variable K per neuron, starts at 2',
    },
    'variablek_init8': {
        **BASE, 'arch': 'variablek', 'H': 32, 'initial_k': 8, 'max_k': 16,
        'grow_rate': 0.01, 'shrink_rate': 0.02,
        'description': 'Variable K starts at 8, bias toward shrinking',
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
            print(f"  {name}: {cfg.get('description', cfg['arch'])}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"HYBRID ARCH EXPERIMENT: {config_name}")
    print("=" * 60)
    print(f"Architecture: {config['arch']}")
    print(f"Description: {config.get('description', '-')}")

    X_train, y_onehot, X_test, y_test = load_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"hybrid_{config_name}.csv"
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
    print(f"Connections: {info['connections']}")
    print(f"Info: {info}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
