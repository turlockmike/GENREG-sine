"""
Experiment: Full Training Method Comparison
Compares GSA vs Single-chain SA vs Original GA vs Backprop

Metrics:
- Accuracy (MSE)
- Efficiency (parameter count)
- Saturation (% neurons |activation| > 0.95)
- Weight magnitudes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import time
from datetime import datetime

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "full_method_comparison"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


class SparseNet:
    """Sparse network for sine regression."""

    def __init__(self, input_dim=16, H=8, K=2):
        self.input_dim = input_dim
        self.H = H
        self.K = K

        self.indices = np.array([
            np.random.choice(input_dim, K, replace=False) for _ in range(H)
        ])
        self.W1 = np.random.randn(H, K).astype(np.float32) * 0.5
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = np.random.randn(1, H).astype(np.float32) * 0.5
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x):
        """Returns (hidden_activations, output)"""
        h = np.zeros((len(x), self.H), dtype=np.float32)
        for i in range(self.H):
            h[:, i] = x[:, self.indices[i]] @ self.W1[i] + self.b1[i]
        h_act = np.tanh(h)
        out = np.tanh(h_act @ self.W2.T + self.b2).flatten()
        return h_act, out

    def predict(self, x):
        _, out = self.forward(x)
        return out

    def mse(self, x, y):
        pred = self.predict(x)
        return np.mean((pred - y) ** 2)

    def saturation(self, x, threshold=0.95):
        h, _ = self.forward(x)
        return (np.abs(h) > threshold).mean()

    def clone(self):
        new = SparseNet(self.input_dim, self.H, self.K)
        new.indices = self.indices.copy()
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new

    def mutate(self, weight_rate=0.2, weight_std=0.1, index_rate=0.15):
        # Weight mutations
        for i in range(self.H):
            if np.random.random() < weight_rate:
                j = np.random.randint(self.K)
                self.W1[i, j] += np.random.randn() * weight_std
            if np.random.random() < weight_rate * 0.5:
                self.b1[i] += np.random.randn() * weight_std * 0.5

        if np.random.random() < weight_rate:
            j = np.random.randint(self.H)
            self.W2[0, j] += np.random.randn() * weight_std

        # Index swaps
        for i in range(self.H):
            if np.random.random() < index_rate:
                j = np.random.randint(self.K)
                available = list(set(range(self.input_dim)) - set(self.indices[i]))
                if available:
                    self.indices[i, j] = np.random.choice(available)
                    self.W1[i, j] = np.random.randn() * 0.3

    def weight_stats(self):
        return {
            'W1_max_abs': float(np.abs(self.W1).max()),
            'W2_max_abs': float(np.abs(self.W2).max()),
        }

    def num_params(self):
        return self.H * self.K + self.H + self.H + 1


def generate_sine_data(n_samples=500, n_features=16, seed=42):
    """Generate sine data with harmonic features."""
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, n_samples).astype(np.float32)
    y = np.sin(x)

    X = np.column_stack([
        np.sin((i+1) * x) if i % 2 == 0 else np.cos((i//2+1) * x)
        for i in range(n_features)
    ]).astype(np.float32)

    return X, y


# =============================================================================
# TRAINING METHODS
# =============================================================================

def train_single_sa(X, y, H=8, K=2, max_steps=15000, seed=42):
    """Single-chain Simulated Annealing."""
    np.random.seed(seed)

    model = SparseNet(X.shape[1], H, K)
    best = model.clone()
    best_mse = model.mse(X, y)

    temp = 1.0
    decay = (0.001 / 1.0) ** (1.0 / max_steps)

    for step in range(max_steps):
        candidate = model.clone()
        candidate.mutate()

        curr_mse = model.mse(X, y)
        new_mse = candidate.mse(X, y)

        delta = new_mse - curr_mse
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            model = candidate
            if new_mse < best_mse:
                best = candidate.clone()
                best_mse = new_mse

        temp *= decay

        if step % 3000 == 0 or step == max_steps - 1:
            sat = best.saturation(X)
            log(f"    Step {step:5d}: MSE={best_mse:.6f}, Sat={sat:.1%}")

    return best


def train_gsa(X, y, H=8, K=2, generations=300, pop_size=50, sa_steps=20, seed=42):
    """Genetic Simulated Annealing (population-based SA)."""
    np.random.seed(seed)

    def fitness(net):
        return -net.mse(X, y)

    # Initialize population
    pop = []
    base = SparseNet(X.shape[1], H, K)
    pop.append((base.clone(), fitness(base)))

    for _ in range(pop_size - 1):
        net = base.clone()
        net.mutate(weight_rate=0.5, weight_std=0.3, index_rate=0.3)
        pop.append((net, fitness(net)))

    best = pop[0][0].clone()
    best_f = fitness(best)

    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    for gen in range(generations):
        pop.sort(key=lambda x: x[1], reverse=True)

        if pop[0][1] > best_f:
            best = pop[0][0].clone()
            best_f = pop[0][1]

        # Elite
        n_elite = max(1, pop_size // 20)
        new_pop = [(c.clone(), f) for c, f in pop[:n_elite]]

        # Roulette selection + SA
        probs = np.array([f for _, f in pop])
        probs = probs - probs.min() + 1e-8
        probs /= probs.sum()

        for _ in range(pop_size - n_elite):
            idx = np.random.choice(len(pop), p=probs)
            c = pop[idx][0].clone()
            curr_f = pop[idx][1]
            best_c, best_inner = c, curr_f

            for _ in range(sa_steps):
                m = c.clone()
                m.mutate()
                f = fitness(m)
                delta = f - curr_f
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    c = m
                    curr_f = f
                    if f > best_inner:
                        best_c, best_inner = m.clone(), f

            new_pop.append((best_c, best_inner))

        pop = new_pop
        temp *= decay

        if gen % 50 == 0 or gen == generations - 1:
            mse = best.mse(X, y)
            sat = best.saturation(X)
            log(f"    Gen {gen:3d}: MSE={mse:.6f}, Sat={sat:.1%}")

    return best


def train_original_ga(X, y, H=8, K=2, generations=300, pop_size=50, seed=42):
    """
    Original GA from sine_population.py - Tiered Evolution.

    Tiers:
    - Elite (5%): survive unchanged
    - Survivors (10%): pass through
    - Clone+Mutate (70%): mutated clones of elite
    - Random (15%): fresh random genomes
    """
    np.random.seed(seed)

    ELITE_PCT = 0.05
    SURVIVE_PCT = 0.10
    CLONE_MUTATE_PCT = 0.70

    def fitness(net):
        return -net.mse(X, y)

    # Initialize population
    pop = []
    for _ in range(pop_size):
        net = SparseNet(X.shape[1], H, K)
        net.mutate(weight_rate=0.5, weight_std=0.3, index_rate=0.3)
        pop.append((net, fitness(net)))

    best = pop[0][0].clone()
    best_f = fitness(best)

    for gen in range(generations):
        # Sort by fitness (descending)
        pop.sort(key=lambda x: x[1], reverse=True)

        if pop[0][1] > best_f:
            best = pop[0][0].clone()
            best_f = pop[0][1]

        # Calculate tier boundaries
        n_elite = max(1, int(pop_size * ELITE_PCT))
        n_survive = int(pop_size * SURVIVE_PCT)
        n_clone_mutate = int(pop_size * CLONE_MUTATE_PCT)
        n_random = pop_size - n_elite - n_survive - n_clone_mutate

        new_pop = []

        # 1. Elite survive unchanged
        for i in range(n_elite):
            new_pop.append((pop[i][0].clone(), pop[i][1]))

        # 2. Survivors pass through
        for i in range(n_survive):
            new_pop.append((pop[n_elite + i][0].clone(), pop[n_elite + i][1]))

        # 3. Clone + mutate from elite
        for i in range(n_clone_mutate):
            parent = pop[i % n_elite][0]
            child = parent.clone()
            child.mutate()
            new_pop.append((child, fitness(child)))

        # 4. Fresh random genomes
        for _ in range(n_random):
            net = SparseNet(X.shape[1], H, K)
            new_pop.append((net, fitness(net)))

        pop = new_pop

        if gen % 50 == 0 or gen == generations - 1:
            mse = best.mse(X, y)
            sat = best.saturation(X)
            log(f"    Gen {gen:3d}: MSE={mse:.6f}, Sat={sat:.1%}")

    return best


def train_backprop(X, y, H=8, K=2, epochs=5000, lr=0.01, seed=42):
    """Train with backpropagation using PyTorch."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create sparse network structure
    base = SparseNet(X.shape[1], H, K)
    indices = base.indices.copy()

    # Build PyTorch model with same sparsity pattern
    class SparseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.indices = indices
            self.W1 = nn.ParameterList([
                nn.Parameter(torch.randn(K) * 0.5) for _ in range(H)
            ])
            self.b1 = nn.Parameter(torch.zeros(H))
            self.W2 = nn.Parameter(torch.randn(1, H) * 0.5)
            self.b2 = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            h = torch.zeros(x.shape[0], H)
            for i in range(H):
                h[:, i] = x[:, self.indices[i]] @ self.W1[i] + self.b1[i]
            h = torch.tanh(h)
            out = torch.tanh(h @ self.W2.T + self.b2)
            return out.squeeze(), h

    model = SparseMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, h = model(X_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            sat = (h.abs() > 0.95).float().mean().item()
            log(f"    Epoch {epoch:5d}: MSE={loss.item():.6f}, Sat={sat:.1%}")

    # Convert back to numpy SparseNet for consistent interface
    result = SparseNet(X.shape[1], H, K)
    result.indices = indices
    for i in range(H):
        result.W1[i] = model.W1[i].detach().numpy()
    result.b1 = model.b1.detach().numpy()
    result.W2 = model.W2.detach().numpy()
    result.b2 = model.b2.detach().numpy()

    return result


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 70)
    log("FULL TRAINING METHOD COMPARISON")
    log("GSA vs Single-chain SA vs Original GA vs Backprop")
    log("=" * 70)

    # Generate data
    X, y = generate_sine_data(n_samples=500, n_features=16)
    log(f"\nData: {len(X)} samples, {X.shape[1]} features")
    log(f"Target: sin(x), range [{y.min():.2f}, {y.max():.2f}]")

    H, K = 8, 2
    log(f"\nArchitecture: H={H}, K={K} ({H*K + H + H + 1} params)")

    n_trials = 3
    log(f"Trials per method: {n_trials}")

    methods = {
        'single_sa': lambda s: train_single_sa(X, y, H, K, max_steps=15000, seed=s),
        'gsa': lambda s: train_gsa(X, y, H, K, generations=300, pop_size=50, sa_steps=20, seed=s),
        'original_ga': lambda s: train_original_ga(X, y, H, K, generations=300, pop_size=50, seed=s),
        'backprop': lambda s: train_backprop(X, y, H, K, epochs=5000, lr=0.01, seed=s),
    }

    all_results = {name: [] for name in methods}

    for trial in range(n_trials):
        seed = trial * 1000
        log(f"\n{'='*70}")
        log(f"TRIAL {trial+1}/{n_trials} (seed={seed})")
        log(f"{'='*70}")

        for i, (name, train_fn) in enumerate(methods.items()):
            log(f"\n{i+1}. {name}:")

            start = time.time()
            model = train_fn(seed)
            train_time = time.time() - start

            mse = model.mse(X, y)
            sat = model.saturation(X)
            ws = model.weight_stats()
            n_params = model.num_params()

            log(f"   → MSE={mse:.6f}, Sat={sat:.1%}, Params={n_params}, Time={train_time:.1f}s")

            all_results[name].append({
                'mse': float(mse),
                'saturation': float(sat),
                'n_params': int(n_params),
                'train_time': float(train_time),
                'W1_max': float(ws['W1_max_abs']),
            })

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    log(f"\n{'Method':<15} {'MSE (mean)':>12} {'MSE (best)':>12} {'Saturation':>12} {'Params':>10} {'Train Time':>12}")
    log("-" * 75)

    summary = {}
    for name in methods:
        results = all_results[name]
        mse_mean = np.mean([r['mse'] for r in results])
        mse_best = np.min([r['mse'] for r in results])
        sat_mean = np.mean([r['saturation'] for r in results])
        n_params = results[0]['n_params']  # Same for all trials
        time_mean = np.mean([r['train_time'] for r in results])
        w1_mean = np.mean([r['W1_max'] for r in results])

        log(f"{name:<15} {mse_mean:>12.6f} {mse_best:>12.6f} {sat_mean:>11.1%} {n_params:>10} {time_mean:>11.1f}s")

        summary[name] = {
            'mse_mean': float(mse_mean),
            'mse_best': float(mse_best),
            'saturation_mean': float(sat_mean),
            'n_params': int(n_params),
            'train_time': float(time_mean),
            'W1_max': float(w1_mean),
        }

    # Analysis
    log(f"\n{'='*70}")
    log("ANALYSIS")
    log(f"{'='*70}")

    log(f"\nAccuracy ranking:")
    for i, (name, s) in enumerate(sorted(summary.items(), key=lambda x: x[1]['mse_mean'])):
        log(f"  {i+1}. {name}: MSE={s['mse_mean']:.6f}")

    log(f"\nSaturation comparison:")
    for name, s in summary.items():
        log(f"  {name}: {s['saturation_mean']:.1%} (W1_max={s['W1_max']:.2f})")

    log(f"\nNetwork size: {summary['gsa']['n_params']} params (all methods use same architecture)")

    # Key insights
    log(f"\n{'='*70}")
    log("KEY INSIGHTS")
    log(f"{'='*70}")

    gsa_mse = summary['gsa']['mse_mean']
    sa_mse = summary['single_sa']['mse_mean']
    ga_mse = summary['original_ga']['mse_mean']
    bp_mse = summary['backprop']['mse_mean']

    log(f"\n1. GSA vs Single SA: {sa_mse/gsa_mse:.1f}x better accuracy")
    log(f"2. GSA vs Original GA: {ga_mse/gsa_mse:.1f}x better accuracy")
    log(f"3. GSA vs Backprop: {gsa_mse/bp_mse:.1f}x MSE ratio")

    gsa_sat = summary['gsa']['saturation_mean']
    sa_sat = summary['single_sa']['saturation_mean']
    ga_sat = summary['original_ga']['saturation_mean']

    log(f"\nSaturation:")
    log(f"  Single SA: {sa_sat:.1%}")
    log(f"  Original GA: {ga_sat:.1%}")
    log(f"  GSA: {gsa_sat:.1%}")
    log(f"  Backprop: {summary['backprop']['saturation_mean']:.1%}")

    if gsa_sat < 0.1 and sa_sat > 0.2:
        log(f"\n*** SATURATION IS TRAINING-METHOD SPECIFIC ***")
        log(f"GSA prevents the saturation that Single SA produces!")

    if ga_sat > 0.2:
        log(f"\n*** Original GA also produces saturation ({ga_sat:.1%}) ***")
        log(f"Population diversity alone doesn't prevent saturation - GSA's SA component helps")
    elif ga_sat < 0.1:
        log(f"\n*** Original GA has low saturation ({ga_sat:.1%}) ***")
        log(f"Random genome injection may help prevent saturation")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"results_{timestamp}.json", 'w') as f:
        json.dump({
            'summary': summary,
            'all_results': all_results,
            'config': {'H': H, 'K': K, 'n_trials': n_trials}
        }, f, indent=2)

    log(f"\nResults saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
