"""
Experiment: GSA with Data Augmentation

Question: Does data augmentation improve GSA accuracy on digits?

Hypothesis: Augmentation (rotation ±10°, shift ±1px) should:
1. Improve generalization (reduce overfitting to training set)
2. Add implicit regularization through data diversity
3. Potentially help escape local minima via stochastic fitness

Test configurations:
- Baseline: No augmentation
- Static augmentation: Pre-generate 2x, 5x augmented training set
- Online augmentation: Random augmentation during fitness evaluation
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

from core.augmentation import augment_digits, augment_batch


class SparseController:
    """Fixed-K sparse controller (proven best architecture)."""

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
        new = SparseController(self.input_size, self.hidden_size, self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()
        return new

    def num_params(self):
        return self.hidden_size * self.k + self.hidden_size + self.output_size * self.hidden_size + self.output_size


def compute_fitness(controller, X, y_onehot, online_aug=False, rotation_range=10, shift_range=1):
    """Compute fitness, optionally with online augmentation."""
    if online_aug:
        # Apply random augmentation to training data each evaluation
        X_eval = augment_batch(X.numpy() if isinstance(X, torch.Tensor) else X,
                               rotation_range=rotation_range, shift_range=shift_range)
        X_eval = torch.from_numpy(X_eval.astype(np.float32))
    else:
        X_eval = X

    with torch.no_grad():
        pred = controller.forward(X_eval)
        fitness = -torch.mean((pred - y_onehot) ** 2).item()
    return fitness


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with optional augmentation."""
    start_time = time.time()

    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']
    online_aug = config.get('online_aug', False)
    rotation_range = config.get('rotation_range', 10)
    shift_range = config.get('shift_range', 1)

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,train_size\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        controller = SparseController(X_train.shape[1], config['H'], y_onehot.shape[1], config['K'])
        fitness = compute_fitness(controller, X_train, y_onehot, online_aug, rotation_range, shift_range)
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

        # Seeds (elite)
        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(c, X_train, y_onehot, temperature, sa_steps, config)
            new_population.append((improved, new_fitness))

        # Rest via roulette selection
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

        # Evaluate on clean test set (no augmentation)
        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        elapsed = time.time() - start_time

        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},{len(X_train)}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, fitness={best_fitness:.4f}, time={elapsed:.0f}s", flush=True)

    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller, time.time() - start_time


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps, config):
    """Run SA steps with optional online augmentation."""
    online_aug = config.get('online_aug', False)
    rotation_range = config.get('rotation_range', 10)
    shift_range = config.get('shift_range', 1)

    current = controller.clone()
    current_fitness = compute_fitness(current, X_train, y_onehot, online_aug, rotation_range, shift_range)

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(config['weight_rate'], config['weight_scale'], config['index_swap_rate'])

        mutant_fitness = compute_fitness(mutant, X_train, y_onehot, online_aug, rotation_range, shift_range)

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def load_digits_data(augment_factor=1, rotation_range=10, shift_range=1, seed=42):
    """Load digits with optional static augmentation."""
    data = load_digits()
    X, y = data.data, data.target

    # Split first, then augment only training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Static augmentation (pre-generate augmented samples)
    if augment_factor > 1:
        X_train, y_train = augment_digits(
            X_train, y_train,
            rotation_range=rotation_range,
            shift_range=shift_range,
            n_augmented=augment_factor - 1,  # -1 because original is included
            seed=seed
        )

    # Normalize after augmentation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # One-hot encode
    y_onehot = torch.zeros(len(y_train), 10)
    y_onehot.scatter_(1, torch.tensor(y_train).unsqueeze(1), 1)
    y_onehot = y_onehot * 1.6 - 0.8  # Scale to [-0.8, 0.8] for tanh

    return X_train, y_onehot, X_test, y_test


# Best known config from ablation (95% accuracy)
BASE = {
    'H': 32, 'K': 4,
    'pop_size': 100, 'generations': 1000,
    'sa_steps': 20, 'seed_fraction': 0.05,
    'weight_rate': 0.02, 'weight_scale': 0.15, 'index_swap_rate': 0.02,
    'rotation_range': 10, 'shift_range': 1,
}

CONFIGS = {
    # Baseline - no augmentation
    'baseline': {
        **BASE, 'augment_factor': 1, 'online_aug': False, 'seed': 42,
        'rotation_range': 0, 'shift_range': 0,
        'description': 'No augmentation (baseline)',
    },

    # Rotation only (online)
    'rot_only': {
        **BASE, 'augment_factor': 1, 'online_aug': True, 'seed': 42,
        'rotation_range': 10, 'shift_range': 0,
        'description': 'Online ±10° rotation only',
    },

    # Shift only (online)
    'shift_only': {
        **BASE, 'augment_factor': 1, 'online_aug': True, 'seed': 42,
        'rotation_range': 0, 'shift_range': 1,
        'description': 'Online ±1px shift only',
    },

    # Both rotation + shift (online)
    'rot_shift': {
        **BASE, 'augment_factor': 1, 'online_aug': True, 'seed': 42,
        'rotation_range': 10, 'shift_range': 1,
        'description': 'Online ±10° rotation + ±1px shift',
    },

    # Static augmentation - pre-generate augmented data
    'static_2x': {
        **BASE, 'augment_factor': 2, 'online_aug': False, 'seed': 42,
        'description': 'Static 2x augmentation',
    },
    'static_5x': {
        **BASE, 'augment_factor': 5, 'online_aug': False, 'seed': 42,
        'description': 'Static 5x augmentation',
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
            print(f"  {name}: {cfg.get('description', '')}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"GSA AUGMENTED: {config_name}")
    print("=" * 60)
    print(f"Description: {config.get('description', '-')}")
    print(f"Augment factor: {config['augment_factor']}x, Online: {config.get('online_aug', False)}")
    print(f"Rotation: ±{config['rotation_range']}°, Shift: ±{config['shift_range']}px")

    # Load data with optional static augmentation
    X_train, y_onehot, X_test, y_test = load_digits_data(
        augment_factor=config['augment_factor'],
        rotation_range=config['rotation_range'],
        shift_range=config['shift_range'],
        seed=config.get('seed', 42)
    )
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"gsa_aug_{config_name}.csv"
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    accuracy, controller, elapsed = run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path)

    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Parameters: {controller.num_params()}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
