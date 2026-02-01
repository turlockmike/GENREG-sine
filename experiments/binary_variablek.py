"""
Experiment: Binary Classification with VariableK

Question: Does optimal K scale with log₂(classes)?

Hypothesis: If K≈4 is optimal for 10 classes (log₂(10)≈3.3), then for binary
classification (2 classes, log₂(2)=1), optimal K should be ~1-2.

Test: Run VariableK on binary classification problems, see what K converges to.

Binary problems tested:
1. Synthetic Parity - requires XOR-like combination of inputs
2. Synthetic Interaction - label depends on product of feature pairs
3. Breast Cancer with noise (harder)

Key insight: To test K hypothesis, we need problems where optimal K is KNOWN by construction.
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

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
            'k_distribution': np.bincount(self.k, minlength=self.max_k+1).tolist()
        }


class FixedK:
    """Fixed-K baseline for comparison."""

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
        return {'connections': self.num_connections(), 'k': self.k}


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
            f.write("gen,test_accuracy,best_fitness,elapsed_s,connections,mean_k,min_k,max_k\n")

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
            min_k = info.get('min_k', info.get('k', 0))
            max_k = info.get('max_k', info.get('k', 0))
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{info['connections']},{mean_k},{min_k},{max_k}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, conn={info['connections']}, info={info}", flush=True)

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


def load_synthetic_interaction(n_samples=2000, n_features=64, n_informative=4, noise=0.1):
    """
    Synthetic problem: y = sign(x0*x1 + x2*x3 + noise)

    Requires K>=2 to solve because individual features are not predictive.
    Only specific PAIRS of features matter.

    Args:
        n_features: Total features (most are noise)
        n_informative: Features used in label (must be even, pairs multiplied)
        noise: Label noise level (higher = harder)
    """
    np.random.seed(42)

    # Generate random features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Label depends on product of pairs of informative features
    # y = sign(x0*x1 + x2*x3 + ... + noise)
    signal = np.zeros(n_samples)
    for i in range(0, n_informative, 2):
        signal += X[:, i] * X[:, i+1]

    # Add noise and threshold
    signal += np.random.randn(n_samples) * noise
    y = (signal > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    y_onehot = torch.zeros(len(y_train), 2)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    return X_train, y_onehot, X_test, y_test, n_features, 2


def load_synthetic_threshold(n_samples=2000, n_features=64, n_informative=4, threshold=0.5):
    """
    Synthetic problem: y = 1 if sum(x0..x_n_inf) > threshold

    Can be solved with K=1 if we pick the right features, but more features = more robust.
    Tests whether evolution finds minimal sufficient K.
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Label depends on sum of first n_informative features
    signal = X[:, :n_informative].sum(axis=1)
    y = (signal > threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    y_onehot = torch.zeros(len(y_train), 2)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    return X_train, y_onehot, X_test, y_test, n_features, 2


def load_synthetic_xor(n_samples=2000, n_features=64, noise=0.1):
    """
    Synthetic XOR-like problem: y = sign((x0>0) XOR (x1>0))

    Classic non-linear problem that requires K>=2.
    Individual features have zero correlation with label.
    """
    np.random.seed(42)

    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # XOR of signs of first two features
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)

    # Add noise by flipping some labels
    n_flip = int(n_samples * noise)
    flip_idx = np.random.choice(n_samples, n_flip, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    y_onehot = torch.zeros(len(y_train), 2)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    return X_train, y_onehot, X_test, y_test, n_features, 2


def load_breast_cancer_data():
    """Load breast cancer dataset (30 features, 2 classes)."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    y_onehot = torch.zeros(len(y_train), 2)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8

    return X_train, y_onehot, X_test, y_test, 30, 2


BASE = {
    'pop_size': 100, 'generations': 1000, 'sa_steps': 20, 'seed_fraction': 0.05,
    'weight_rate': 0.02, 'weight_scale': 0.15, 'index_swap_rate': 0.02, 'seed': 42,
}

CONFIGS = {
    # Synthetic Interaction: y = sign(x0*x1 + x2*x3) - REQUIRES K>=2
    'interaction_variablek_init4': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'variablek', 'H': 16,
        'initial_k': 4, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Interaction (needs K>=2), variableK init=4',
    },
    'interaction_variablek_init2': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'variablek', 'H': 16,
        'initial_k': 2, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Interaction (needs K>=2), variableK init=2',
    },
    'interaction_variablek_init1': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'variablek', 'H': 16,
        'initial_k': 1, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Interaction (needs K>=2), variableK init=1',
    },
    'interaction_fixedk1': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'fixedk', 'H': 16, 'K': 1,
        'description': 'Interaction, fixed K=1 (should fail)',
    },
    'interaction_fixedk2': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'fixedk', 'H': 16, 'K': 2,
        'description': 'Interaction, fixed K=2 (minimum sufficient)',
    },
    'interaction_fixedk4': {
        **BASE, 'dataset': 'interaction', 'n_informative': 4, 'noise': 0.3,
        'arch': 'fixedk', 'H': 16, 'K': 4,
        'description': 'Interaction, fixed K=4',
    },

    # Synthetic XOR: requires non-linear combination of 2 features
    'xor_variablek_init4': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'variablek', 'H': 16,
        'initial_k': 4, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'XOR (needs K>=2), variableK init=4',
    },
    'xor_variablek_init2': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'variablek', 'H': 16,
        'initial_k': 2, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'XOR (needs K>=2), variableK init=2',
    },
    'xor_variablek_init1': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'variablek', 'H': 16,
        'initial_k': 1, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'XOR (needs K>=2), variableK init=1',
    },
    'xor_fixedk1': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'fixedk', 'H': 16, 'K': 1,
        'description': 'XOR, fixed K=1 (should fail ~50%)',
    },
    'xor_fixedk2': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'fixedk', 'H': 16, 'K': 2,
        'description': 'XOR, fixed K=2 (minimum sufficient)',
    },
    'xor_fixedk4': {
        **BASE, 'dataset': 'xor', 'noise': 0.1,
        'arch': 'fixedk', 'H': 16, 'K': 4,
        'description': 'XOR, fixed K=4',
    },

    # Synthetic Threshold: y = sum(x0..x3) > 0 - CAN work with K=1
    'threshold_variablek_init4': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'variablek', 'H': 16,
        'initial_k': 4, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Threshold (K=1 OK), variableK init=4',
    },
    'threshold_variablek_init2': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'variablek', 'H': 16,
        'initial_k': 2, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Threshold (K=1 OK), variableK init=2',
    },
    'threshold_variablek_init1': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'variablek', 'H': 16,
        'initial_k': 1, 'max_k': 16, 'grow_rate': 0.01, 'shrink_rate': 0.01,
        'description': 'Threshold (K=1 OK), variableK init=1',
    },
    'threshold_fixedk1': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'fixedk', 'H': 16, 'K': 1,
        'description': 'Threshold, fixed K=1',
    },
    'threshold_fixedk2': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'fixedk', 'H': 16, 'K': 2,
        'description': 'Threshold, fixed K=2',
    },
    'threshold_fixedk4': {
        **BASE, 'dataset': 'threshold', 'n_informative': 4,
        'arch': 'fixedk', 'H': 16, 'K': 4,
        'description': 'Threshold, fixed K=4',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='digits01_variablek_init2')
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
    print(f"BINARY VARIABLEK EXPERIMENT: {config_name}")
    print("=" * 60)
    print(f"Description: {config.get('description', '-')}")

    # Load data
    dataset = config['dataset']
    if dataset == 'interaction':
        X_train, y_onehot, X_test, y_test, input_size, num_classes = load_synthetic_interaction(
            n_informative=config.get('n_informative', 4),
            noise=config.get('noise', 0.3)
        )
    elif dataset == 'xor':
        X_train, y_onehot, X_test, y_test, input_size, num_classes = load_synthetic_xor(
            noise=config.get('noise', 0.1)
        )
    elif dataset == 'threshold':
        X_train, y_onehot, X_test, y_test, input_size, num_classes = load_synthetic_threshold(
            n_informative=config.get('n_informative', 4)
        )
    else:
        X_train, y_onehot, X_test, y_test, input_size, num_classes = load_breast_cancer_data()

    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Input size: {input_size}, Classes: {num_classes}")
    print(f"Expected optimal K (log2): {np.log2(num_classes):.2f}")

    csv_path = Path("results/live") / f"binary_{config_name}.csv"
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, 'input_size': input_size,
                   'num_classes': num_classes, 'expected_k': np.log2(num_classes), **config}, f, indent=2)

    accuracy, controller, elapsed = run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path)

    info = controller.info()
    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Final K info: {info}")
    print(f"Expected K (log2 classes): {np.log2(num_classes):.2f}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
