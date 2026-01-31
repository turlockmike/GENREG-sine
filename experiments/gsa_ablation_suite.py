"""
Experiment: Comprehensive GSA Ablation Suite

Question: Which GSA hyperparameter matters most for breaking the ~85% plateau on digits?

Control: L=1, H=32, K=4, pop=50, sa_steps=20, index_swap=0.1, weight_rate=0.15, 1000 gens

Variables tested (one at a time):
- K (sparsity): 2, 4*, 8, 16
- H (width): 16, 32*, 64
- index_swap_rate: 0.0, 0.05, 0.1*, 0.2, 0.3
- weight_rate: 0.05, 0.15*, 0.25, 0.35
- population_size: 25, 50*, 100
- sa_steps_per_gen: 5, 10, 20*, 40
- seed_fraction: 0.01, 0.05*, 0.1, 0.2

* = control value
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
import argparse
import warnings
import json
warnings.filterwarnings('ignore', category=RuntimeWarning)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class UltraSparseController:
    """Single-layer ultra-sparse network for ablation."""

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

    def mutate(self, weight_rate=0.15, weight_scale=0.15, index_swap_rate=0.1):
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
        new = UltraSparseController(self.input_size, self.hidden_size, self.output_size, self.k)
        new.indices = self.indices.copy()
        new.w1 = self.w1.copy()
        new.b1 = self.b1.copy()
        new.w2 = self.w2.copy()
        new.b2 = self.b2.copy()
        return new

    def num_parameters(self):
        return self.w1.size + self.b1.size + self.w2.size + self.b2.size


def run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path=None):
    """Run GSA with given config."""
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

    # Initialize CSV
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("gen,test_accuracy,best_fitness,elapsed_s\n")

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(i * 10 + config.get('seed', 0))
        controller = UltraSparseController(64, H, 10, K)
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
                weight_rate, weight_scale, index_swap_rate
            )
            new_population.append((improved, new_fitness))

        # Roulette selection for rest
        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_fitness = _sa_steps(
                c, X_train, y_onehot, temperature, sa_steps,
                weight_rate, weight_scale, index_swap_rate
            )
            new_population.append((improved, new_fitness))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        temperature *= decay

        # Calculate accuracy
        with torch.no_grad():
            preds = best_controller.forward(X_test).argmax(dim=1)
            acc = (preds == y_test).float().mean().item()

        elapsed = time.time() - start_time

        # Write to CSV
        if csv_path:
            with open(csv_path, 'a') as f:
                f.write(f"{gen},{acc},{best_fitness},{elapsed:.1f}\n")
                f.flush()

        if gen % 100 == 0:
            print(f"    Gen {gen}: {acc:.1%}", flush=True)

    # Final accuracy
    with torch.no_grad():
        preds = best_controller.forward(X_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy, best_controller.num_parameters(), time.time() - start_time


def _sa_steps(controller, X_train, y_onehot, temperature, n_steps,
              weight_rate, weight_scale, index_swap_rate):
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_train)
        current_fitness = -torch.mean((pred - y_onehot) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(weight_rate, weight_scale, index_swap_rate)

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


# Control configuration
CONTROL = {
    'H': 32,
    'K': 4,
    'pop_size': 50,
    'generations': 1000,
    'sa_steps': 20,
    'seed_fraction': 0.05,
    'index_swap_rate': 0.1,
    'weight_rate': 0.15,
    'weight_scale': 0.15,
    'seed': 42,
}

# All ablation configs - each changes ONE variable from control
ABLATIONS = {
    # Sparsity (K)
    'K2': {**CONTROL, 'K': 2},
    'K8': {**CONTROL, 'K': 8},
    'K16': {**CONTROL, 'K': 16},

    # Width (H)
    'H16': {**CONTROL, 'H': 16},
    'H64': {**CONTROL, 'H': 64},

    # Index swap rate (key mutation parameter!)
    'idx0.0': {**CONTROL, 'index_swap_rate': 0.0},  # No index mutation
    'idx0.05': {**CONTROL, 'index_swap_rate': 0.05},
    'idx0.2': {**CONTROL, 'index_swap_rate': 0.2},
    'idx0.3': {**CONTROL, 'index_swap_rate': 0.3},

    # Weight mutation rate
    'wt0.05': {**CONTROL, 'weight_rate': 0.05},
    'wt0.25': {**CONTROL, 'weight_rate': 0.25},
    'wt0.35': {**CONTROL, 'weight_rate': 0.35},

    # Population size
    'pop25': {**CONTROL, 'pop_size': 25},
    'pop100': {**CONTROL, 'pop_size': 100},

    # SA steps per generation
    'sa5': {**CONTROL, 'sa_steps': 5},
    'sa10': {**CONTROL, 'sa_steps': 10},
    'sa40': {**CONTROL, 'sa_steps': 40},

    # Seed fraction (elitism)
    'elite0.01': {**CONTROL, 'seed_fraction': 0.01},
    'elite0.1': {**CONTROL, 'seed_fraction': 0.1},
    'elite0.2': {**CONTROL, 'seed_fraction': 0.2},

    # === COMBO EXPERIMENTS (based on ablation winners) ===

    # Tier 1: Combine top winners
    'combo_top2': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.05},
    'combo_top3': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.05, 'weight_rate': 0.05},
    'combo_all': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.05, 'weight_rate': 0.05, 'sa_steps': 5},

    # Tier 2: Push population further
    'pop150': {**CONTROL, 'pop_size': 150, 'index_swap_rate': 0.05, 'weight_rate': 0.05},
    'pop200': {**CONTROL, 'pop_size': 200, 'index_swap_rate': 0.05, 'weight_rate': 0.05},

    # Tier 3: Explore "less mutation" theme
    'minimal_mut': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.02, 'weight_rate': 0.02},
    'idx_only': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.05, 'weight_rate': 0.0},

    # Tier 4: Elitism interactions
    'elite_combo': {**CONTROL, 'pop_size': 100, 'index_swap_rate': 0.05, 'weight_rate': 0.05, 'seed_fraction': 0.2},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='control',
                        help='Config name: control, K2, K8, H16, idx0.2, etc.')
    parser.add_argument('--list', action='store_true', help='List all configs')
    args = parser.parse_args()

    if args.list:
        print("Available configs:")
        print("  control (baseline)")
        for name in sorted(ABLATIONS.keys()):
            cfg = ABLATIONS[name]
            diff = {k: v for k, v in cfg.items() if v != CONTROL[k]}
            print(f"  {name}: {diff}")
        return

    # Get config
    if args.config == 'control':
        config = CONTROL.copy()
        config_name = 'control'
    elif args.config in ABLATIONS:
        config = ABLATIONS[args.config]
        config_name = args.config
    else:
        print(f"Unknown config: {args.config}")
        print("Use --list to see available configs")
        return

    print("=" * 60)
    print(f"GSA ABLATION: {config_name}")
    print("=" * 60)

    # Show what's different from control
    if config_name != 'control':
        diff = {k: v for k, v in config.items() if v != CONTROL[k]}
        print(f"Changed from control: {diff}")
    print(f"Config: H={config['H']}, K={config['K']}, pop={config['pop_size']}, "
          f"sa_steps={config['sa_steps']}, idx_swap={config['index_swap_rate']}, "
          f"wt_rate={config['weight_rate']}")
    print()

    X_train, y_onehot, X_test, y_test = load_data()
    print(f"Data: {len(X_train)} train, {len(X_test)} test")

    csv_path = Path("results/live") / f"ablation_{config_name}.csv"

    # Save config metadata
    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({'config_name': config_name, **config}, f, indent=2)

    accuracy, params, elapsed = run_gsa(X_train, y_onehot, X_test, y_test, config, csv_path)

    print()
    print("=" * 60)
    print(f"RESULT: {config_name}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Params: {params}")
    print(f"Time: {elapsed:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
