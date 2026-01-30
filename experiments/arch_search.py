"""
Architecture Search for GENREG on Digits

Searches over: H, K, L, index_swap_rate, weight_rate
Uses evolutionary search with early stopping and parallel evaluation.

Progressive logging - results written as they complete.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, asdict
from typing import List, Tuple
import os

# Global data (loaded once, shared via fork)
X_TRAIN = None
X_TEST = None
Y_TRAIN = None
Y_TEST = None
Y_ONEHOT = None


def init_data():
    """Load data into globals."""
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, Y_ONEHOT

    data = load_digits()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_TRAIN = torch.tensor(X_train, dtype=torch.float32)
    X_TEST = torch.tensor(X_test, dtype=torch.float32)
    Y_TRAIN = torch.tensor(y_train, dtype=torch.long)
    Y_TEST = torch.tensor(y_test, dtype=torch.long)

    Y_ONEHOT = torch.zeros(len(Y_TRAIN), 10)
    Y_ONEHOT.scatter_(1, Y_TRAIN.unsqueeze(1), 1)
    Y_ONEHOT = Y_ONEHOT * 1.6 - 0.8


@dataclass
class ArchConfig:
    """Architecture configuration."""
    H: int              # Hidden size per layer
    K: int              # Inputs per neuron
    L: int              # Number of hidden layers
    index_swap_rate: float
    weight_rate: float  # Also used as weight_scale

    def to_tuple(self):
        return (self.H, self.K, self.L, self.index_swap_rate, self.weight_rate)


class DeepSparseController:
    """Multi-layer ultra-sparse network."""

    def __init__(self, input_size, H, K, L, output_size=10):
        self.input_size = input_size
        self.H = H
        self.K = K
        self.L = L
        self.output_size = output_size

        # Build layers
        self.layers = []
        prev_size = input_size
        for _ in range(L):
            layer = {
                'indices': np.random.randint(0, prev_size, (H, K)),
                'weights': np.random.randn(H, K).astype(np.float32) * 0.5,
                'bias': np.zeros(H, dtype=np.float32)
            }
            self.layers.append(layer)
            prev_size = H

        # Output layer
        self.out_weights = np.random.randn(output_size, H).astype(np.float32) * 0.5
        self.out_bias = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        h = x
        for layer in self.layers:
            gathered = h[:, layer['indices']]
            pre_act = np.einsum('bnk,nk->bn', gathered, layer['weights']) + layer['bias']
            h = np.tanh(pre_act)

        out = h @ self.out_weights.T + self.out_bias
        return torch.from_numpy(np.tanh(out))

    def mutate(self, weight_rate, index_swap_rate):
        scale = weight_rate  # Coupled

        for layer_idx, layer in enumerate(self.layers):
            # Weight mutations
            mask = np.random.random(layer['weights'].shape) < weight_rate
            layer['weights'] += mask * np.random.randn(*layer['weights'].shape).astype(np.float32) * scale

            mask = np.random.random(layer['bias'].shape) < weight_rate
            layer['bias'] += mask * np.random.randn(*layer['bias'].shape).astype(np.float32) * scale

            # Index mutations
            max_idx = self.input_size if layer_idx == 0 else self.H
            for h in range(self.H):
                if np.random.random() < index_swap_rate:
                    idx = np.random.randint(0, self.K)
                    layer['indices'][h, idx] = np.random.randint(0, max_idx)

        # Output mutations
        mask = np.random.random(self.out_weights.shape) < weight_rate
        self.out_weights += mask * np.random.randn(*self.out_weights.shape).astype(np.float32) * scale

        mask = np.random.random(self.out_bias.shape) < weight_rate
        self.out_bias += mask * np.random.randn(*self.out_bias.shape).astype(np.float32) * scale

    def clone(self):
        new = DeepSparseController(self.input_size, self.H, self.K, self.L, self.output_size)
        for i, layer in enumerate(self.layers):
            new.layers[i] = {
                'indices': layer['indices'].copy(),
                'weights': layer['weights'].copy(),
                'bias': layer['bias'].copy()
            }
        new.out_weights = self.out_weights.copy()
        new.out_bias = self.out_bias.copy()
        return new

    def num_parameters(self):
        count = sum(l['weights'].size + l['bias'].size for l in self.layers)
        count += self.out_weights.size + self.out_bias.size
        return count


def evaluate_config(args) -> dict:
    """Evaluate a single architecture config with GSA."""
    config, generations, pop_size, seed = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize population
    population = []
    for i in range(pop_size):
        np.random.seed(seed * 1000 + i)
        controller = DeepSparseController(
            input_size=64, H=config.H, K=config.K, L=config.L
        )
        with torch.no_grad():
            pred = controller.forward(X_TRAIN)
            fitness = -torch.mean((pred - Y_ONEHOT) ** 2).item()
        population.append((controller, fitness))

    # GSA training
    t_initial, t_final = 0.1, 0.0001
    decay = (t_final / t_initial) ** (1.0 / generations)
    temperature = t_initial

    best_controller = max(population, key=lambda x: x[1])[0].clone()
    best_fitness = max(population, key=lambda x: x[1])[1]

    checkpoints = []  # Track accuracy over time

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        n_seeds = max(1, int(0.05 * pop_size))

        fitnesses = np.array([f for _, f in population])
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()

        new_population = []

        # Seeds
        for c, _ in population[:n_seeds]:
            improved, new_f = _sa_steps(c, config.weight_rate, config.index_swap_rate, temperature, 20)
            new_population.append((improved, new_f))

        # Roulette
        indices = np.random.choice(len(population), size=pop_size-n_seeds, p=probs, replace=True)
        for idx in indices:
            c = population[idx][0].clone()
            improved, new_f = _sa_steps(c, config.weight_rate, config.index_swap_rate, temperature, 20)
            new_population.append((improved, new_f))

        population = new_population

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_controller = gen_best[0].clone()
            best_fitness = gen_best[1]

        temperature *= decay

        # Checkpoint accuracy
        if gen in [25, 50, 100, 150, 199]:
            with torch.no_grad():
                preds = best_controller.forward(X_TEST).argmax(dim=1)
                acc = (preds == Y_TEST).float().mean().item()
            checkpoints.append((gen, acc))

    # Final accuracy
    with torch.no_grad():
        preds = best_controller.forward(X_TEST).argmax(dim=1)
        final_acc = (preds == Y_TEST).float().mean().item()

        train_preds = best_controller.forward(X_TRAIN).argmax(dim=1)
        train_acc = (train_preds == Y_TRAIN).float().mean().item()

    return {
        'config': asdict(config),
        'test_accuracy': final_acc,
        'train_accuracy': train_acc,
        'params': best_controller.num_parameters(),
        'checkpoints': checkpoints,
        'seed': seed
    }


def _sa_steps(controller, weight_rate, index_swap_rate, temperature, n_steps):
    """Run SA steps on a controller."""
    current = controller.clone()
    with torch.no_grad():
        pred = current.forward(X_TRAIN)
        current_fitness = -torch.mean((pred - Y_ONEHOT) ** 2).item()

    best = current.clone()
    best_fitness = current_fitness

    for _ in range(n_steps):
        mutant = current.clone()
        mutant.mutate(weight_rate, index_swap_rate)

        with torch.no_grad():
            pred = mutant.forward(X_TRAIN)
            mutant_fitness = -torch.mean((pred - Y_ONEHOT) ** 2).item()

        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness

    return best, best_fitness


def run_search():
    """Run architecture search."""

    # Search space
    H_values = [32, 64, 128]
    K_values = [8, 16, 32]
    L_values = [1, 2, 3]
    index_swap_rates = [0.05, 0.1, 0.2]
    weight_rates = [0.1, 0.15, 0.25]

    # Generate all configs
    all_configs = []
    for H in H_values:
        for K in K_values:
            if K > 64:  # K can't exceed input size
                continue
            for L in L_values:
                for isr in index_swap_rates:
                    for wr in weight_rates:
                        all_configs.append(ArchConfig(H, K, L, isr, wr))

    print("=" * 70)
    print("GENREG ARCHITECTURE SEARCH")
    print("=" * 70)
    print(f"\nSearch space: {len(all_configs)} configurations")
    print(f"  H: {H_values}")
    print(f"  K: {K_values}")
    print(f"  L: {L_values}")
    print(f"  index_swap_rate: {index_swap_rates}")
    print(f"  weight_rate: {weight_rates}")

    # Initialize data
    init_data()
    print(f"\nData: {len(X_TRAIN)} train, {len(X_TEST)} test")

    # Setup logging
    log_dir = Path(__file__).parent.parent / "results" / "arch_search"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"search_{timestamp}.jsonl"
    summary_file = log_dir / f"summary_{timestamp}.txt"

    print(f"\nLogs: {log_file}")
    print(f"Summary: {summary_file}")

    # Phase 1: Quick evaluation of all configs (50 generations)
    print("\n" + "=" * 70)
    print("PHASE 1: Quick evaluation (50 generations each)")
    print("=" * 70)

    n_workers = min(cpu_count() - 2, 12)
    print(f"Using {n_workers} parallel workers")

    # Sample configs for phase 1 (random subset if too many)
    phase1_configs = all_configs
    if len(phase1_configs) > 60:
        np.random.seed(42)
        phase1_configs = list(np.random.choice(all_configs, size=60, replace=False))

    tasks = [(cfg, 50, 30, i) for i, cfg in enumerate(phase1_configs)]  # 50 gens, pop=30

    print(f"Evaluating {len(tasks)} configs...")
    start_time = time.time()

    results = []
    with Pool(n_workers, initializer=init_data) as pool:
        for i, result in enumerate(pool.imap_unordered(evaluate_config, tasks)):
            results.append(result)

            # Progressive logging
            with open(log_file, 'a') as f:
                f.write(json.dumps(result) + '\n')

            # Progress update
            cfg = result['config']
            print(f"  [{i+1}/{len(tasks)}] H={cfg['H']}, K={cfg['K']}, L={cfg['L']}, "
                  f"isr={cfg['index_swap_rate']}, wr={cfg['weight_rate']} "
                  f"→ {result['test_accuracy']:.1%}")

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete in {phase1_time/60:.1f} min")

    # Find top configs
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    top_configs = results[:10]

    print("\nTop 10 from Phase 1:")
    for i, r in enumerate(top_configs):
        cfg = r['config']
        print(f"  {i+1}. {r['test_accuracy']:.1%} - H={cfg['H']}, K={cfg['K']}, L={cfg['L']}, "
              f"isr={cfg['index_swap_rate']}, wr={cfg['weight_rate']}")

    # Phase 2: Full training of top configs (200 generations)
    print("\n" + "=" * 70)
    print("PHASE 2: Full training of top configs (200 generations)")
    print("=" * 70)

    phase2_configs = [ArchConfig(**r['config']) for r in top_configs[:5]]
    tasks = [(cfg, 200, 50, i + 1000) for i, cfg in enumerate(phase2_configs)]  # 200 gens, pop=50

    phase2_results = []
    with Pool(n_workers, initializer=init_data) as pool:
        for i, result in enumerate(pool.imap_unordered(evaluate_config, tasks)):
            phase2_results.append(result)

            with open(log_file, 'a') as f:
                result['phase'] = 2
                f.write(json.dumps(result) + '\n')

            cfg = result['config']
            print(f"  [{i+1}/{len(tasks)}] H={cfg['H']}, K={cfg['K']}, L={cfg['L']} "
                  f"→ {result['test_accuracy']:.1%}")

    # Final summary
    phase2_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    best = phase2_results[0]

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nTop 5 configurations (full training):")
    for i, r in enumerate(phase2_results):
        cfg = r['config']
        print(f"  {i+1}. {r['test_accuracy']:.1%} ({r['params']} params) - "
              f"H={cfg['H']}, K={cfg['K']}, L={cfg['L']}, "
              f"isr={cfg['index_swap_rate']}, wr={cfg['weight_rate']}")

    print(f"\n🏆 BEST: {best['test_accuracy']:.1%} accuracy")
    print(f"   Config: H={best['config']['H']}, K={best['config']['K']}, L={best['config']['L']}")
    print(f"   Rates: index_swap={best['config']['index_swap_rate']}, weight={best['config']['weight_rate']}")
    print(f"   Params: {best['params']}")

    # Write summary
    with open(summary_file, 'w') as f:
        f.write("GENREG Architecture Search Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best accuracy: {best['test_accuracy']:.1%}\n")
        f.write(f"Config: {best['config']}\n")
        f.write(f"Params: {best['params']}\n\n")
        f.write("Top 5:\n")
        for i, r in enumerate(phase2_results):
            f.write(f"  {i+1}. {r['test_accuracy']:.1%} - {r['config']}\n")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results saved to: {log_dir}")

    return phase2_results


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)
    run_search()
