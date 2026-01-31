"""
Experiment: RetinalNet - Sensory Bottleneck with Dual Evolvable Masks

Question: Can a sensory bottleneck layer with evolvable binary masks (flip on/off)
achieve better feature selection than fixed-K sparse connectivity?

Hypothesis: By allowing connections to flip on/off (rather than swap), the network
will converge to an optimal sparsity level naturally through selection pressure.
The dual-mask design (input→sensor AND sensor→hidden) mimics biological retinal
processing where information is filtered before reaching cortex.

Architecture:
    Input (64) --[mask1]--> Sensor (S nodes) --[mask2]--> Hidden (32) --[dense]--> Output (10)

    - mask1: Binary mask (64 x S), connections can flip on/off
    - mask2: Binary mask (S x 32), connections can flip on/off
    - Sparsity emerges through evolution, not enforced

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


class RetinalNet:
    """
    Sensory bottleneck architecture with dual evolvable binary masks.

    Architecture:
        Input (64) --[mask1]--> Sensor (S) --[mask2]--> Hidden (H) --[dense]--> Output (10)

    Key difference from UltraSparse:
    - Masks are binary (on/off), not index lists
    - Number of active connections emerges through evolution
    - Both input→sensor AND sensor→hidden have evolvable masks
    """

    def __init__(self, input_size, sensor_size, hidden_size, output_size,
                 initial_density=0.2):
        self.input_size = input_size
        self.sensor_size = sensor_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Layer 1: Input -> Sensor (sparse, evolvable mask)
        self.mask1 = (np.random.random((sensor_size, input_size)) < initial_density).astype(np.float32)
        self.w1 = np.random.randn(sensor_size, input_size).astype(np.float32) * 0.5
        self.b1 = np.zeros(sensor_size, dtype=np.float32)

        # Layer 2: Sensor -> Hidden (sparse, evolvable mask)
        self.mask2 = (np.random.random((hidden_size, sensor_size)) < initial_density).astype(np.float32)
        self.w2 = np.random.randn(hidden_size, sensor_size).astype(np.float32) * 0.5
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # Layer 3: Hidden -> Output (dense, fixed)
        self.w3 = np.random.randn(output_size, hidden_size).astype(np.float32) * 0.5
        self.b3 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # Input -> Sensor (masked)
        masked_w1 = self.w1 * self.mask1
        sensor = np.tanh(x @ masked_w1.T + self.b1)

        # Sensor -> Hidden (masked)
        masked_w2 = self.w2 * self.mask2
        hidden = np.tanh(sensor @ masked_w2.T + self.b2)

        # Hidden -> Output (dense)
        out = np.tanh(hidden @ self.w3.T + self.b3)

        return torch.from_numpy(out)

    def mutate(self, weight_rate=0.02, weight_scale=0.15, flip_rate=0.02):
        """
        Mutate weights and flip mask connections on/off.

        flip_rate: Probability of flipping each connection (on→off or off→on)
        """
        # Weight mutations (only for active connections)
        for w, mask in [(self.w1, self.mask1), (self.w2, self.mask2)]:
            weight_mask = np.random.random(w.shape) < weight_rate
            w += weight_mask * mask * np.random.randn(*w.shape).astype(np.float32) * weight_scale

        # Dense layer weight mutations
        mask = np.random.random(self.w3.shape) < weight_rate
        self.w3 += mask * np.random.randn(*self.w3.shape).astype(np.float32) * weight_scale

        # Bias mutations
        for b in [self.b1, self.b2, self.b3]:
            mask = np.random.random(b.shape) < weight_rate
            b += mask * np.random.randn(*b.shape).astype(np.float32) * weight_scale

        # Mask flip mutations (on→off or off→on)
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
        """Count active (non-masked) connections."""
        return int(self.mask1.sum() + self.mask2.sum())

    def num_parameters(self):
        """Total parameters (weights + biases for active connections + dense layer)."""
        active1 = int(self.mask1.sum())
        active2 = int(self.mask2.sum())
        dense = self.w3.size + self.b3.size
        return active1 + self.b1.size + active2 + self.b2.size + dense

    def sparsity_info(self):
        """Return sparsity statistics."""
        total1 = self.mask1.size
        total2 = self.mask2.size
        active1 = int(self.mask1.sum())
        active2 = int(self.mask2.sum())
        return {
            'mask1_density': active1 / total1,
            'mask2_density': active2 / total2,
            'active_connections': active1 + active2,
            'total_possible': total1 + total2,
        }


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with RetinalNet architecture."""
    start_time = time.time()

    S = config['S']  # Sensor size
    H = config['H']  # Hidden size
    pop_size = config['pop_size']
    generations = config['generations']
    sa_steps = config['sa_steps']
    seed_fraction = config['seed_fraction']
    flip_rate = config['flip_rate']
    weight_rate = config['weight_rate']
    weight_scale = config['weight_scale']
    initial_density = config['initial_density']

    # Initialize CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s,active_connections,mask1_density,mask2_density\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        controller = RetinalNet(64, S, H, 10, initial_density=initial_density)
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

        # Seeds get SA refinement
        for c, _ in population[:n_seeds]:
            improved, new_fitness = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps,
                weight_rate, weight_scale, flip_rate
            )
            new_population.append((improved, new_fitness))

        # Roulette selection for rest
        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps,
                weight_rate, weight_scale, flip_rate
            )
            new_population.append((improved, new_fitness))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        temperature *= decay

        # Calculate accuracy and sparsity
        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        sparsity = best_controller.sparsity_info()
        elapsed = time.time() - start_time

        # Write to CSV
        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f},"
                        f"{sparsity['active_connections']},{sparsity['mask1_density']:.3f},"
                        f"{sparsity['mask2_density']:.3f}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}, connections={sparsity['active_connections']}, "
                  f"density1={sparsity['mask1_density']:.2f}, density2={sparsity['mask2_density']:.2f}",
                  flush=True)

    # Final accuracy
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller, time.time() - start_time


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps,
              weight_rate, weight_scale, flip_rate):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_fitness = -torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(weight_rate, weight_scale, flip_rate)

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


# Configurations to test
CONFIGS = {
    # Baseline: Current best (for comparison)
    'baseline': {
        'type': 'baseline',  # Flag to use UltraSparse instead
        'H': 32, 'K': 4, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'index_swap_rate': 0.02, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'seed': 42,
    },

    # RetinalNet variants - vary sensor size
    'retinal_S8': {
        'S': 8, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.02, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.2, 'seed': 42,
    },
    'retinal_S16': {
        'S': 16, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.02, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.2, 'seed': 42,
    },
    'retinal_S32': {
        'S': 32, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.02, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.2, 'seed': 42,
    },

    # Vary initial density
    'retinal_dense_start': {
        'S': 16, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.02, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.5, 'seed': 42,
    },

    # Vary flip rate
    'retinal_flip01': {
        'S': 16, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.01, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.2, 'seed': 42,
    },
    'retinal_flip05': {
        'S': 16, 'H': 32, 'pop_size': 100, 'generations': 1000,
        'sa_steps': 20, 'seed_fraction': 0.05,
        'flip_rate': 0.05, 'weight_rate': 0.02, 'weight_scale': 0.15,
        'initial_density': 0.2, 'seed': 42,
    },
}


# Import UltraSparse for baseline comparison
from gsa_ablation_suite import UltraSparseController, run_gsa as run_gsa_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='retinal_S16',
                        help='Config name: baseline, retinal_S8, retinal_S16, etc.')
    parser.add_argument('--list', action='store_true', help='List all configs')
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        for name in sorted(CONFIGS.keys()):
            print(f"  {name}")
        return

    if args.config not in CONFIGS:
        print(f"Unknown config: {args.config}")
        print("Use --list to see available configs")
        return

    config = CONFIGS[args.config]
    config_name = args.config

    print("=" * 60)
    print(f"RETINAL NET EXPERIMENT: {config_name}")
    print("=" * 60)

    X_train, y_onehot, X_test, y_test = load_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"retinal_{config_name}.csv"

    # Save config metadata
    json_path = csv_path.with_suffix('.json')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    if config.get('type') == 'baseline':
        # Run baseline UltraSparse
        print("Running BASELINE (UltraSparse)...")
        accuracy, params, elapsed = run_gsa_baseline(
            X_train, y_onehot, X_test, y_test, config, csv_path
        )
        print(f"\nRESULT: {accuracy:.1%}, {params} params, {elapsed:.0f}s")
    else:
        # Run RetinalNet
        print(f"Config: S={config['S']}, H={config['H']}, pop={config['pop_size']}, "
              f"flip_rate={config['flip_rate']}, initial_density={config['initial_density']}")

        accuracy, controller, elapsed = run_gsa(
            X_train, y_onehot, X_test, y_test, config, csv_path
        )

        sparsity = controller.sparsity_info()
        print()
        print("=" * 60)
        print(f"RESULT: {config_name}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Active connections: {sparsity['active_connections']}")
        print(f"Mask1 density: {sparsity['mask1_density']:.1%}")
        print(f"Mask2 density: {sparsity['mask2_density']:.1%}")
        print(f"Parameters: {controller.num_parameters()}")
        print(f"Time: {elapsed:.0f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
