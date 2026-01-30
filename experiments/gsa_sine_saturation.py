"""
Experiment: GSA on Sine Problem - Saturation Check

Tests whether GSA training on sine produces saturation like original SA did.
This helps us understand if saturation is:
1. Problem-specific (sine naturally produces it)
2. Training-method-specific (SA vs GSA difference)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from datetime import datetime

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "gsa_sine_saturation"
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

    def per_neuron_saturation(self, x, threshold=0.95):
        h, _ = self.forward(x)
        return (np.abs(h) > threshold).mean(axis=0)

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
            'W1_mean_abs': float(np.abs(self.W1).mean()),
            'W1_max_abs': float(np.abs(self.W1).max()),
            'W2_mean_abs': float(np.abs(self.W2).mean()),
            'W2_max_abs': float(np.abs(self.W2).max()),
        }

    def num_params(self):
        return self.H * self.K + self.H + self.H + 1


def generate_sine_data(n_samples=500, n_features=16, seed=42):
    """Generate sine data with harmonic features."""
    np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, n_samples).astype(np.float32)
    y = np.sin(x)

    # Create feature matrix: sin(kx), cos(kx) for various k
    X = np.column_stack([
        np.sin((i+1) * x) if i % 2 == 0 else np.cos((i//2+1) * x)
        for i in range(n_features)
    ]).astype(np.float32)

    return X, y


def train_gsa(X, y, H=8, K=2, generations=300, pop_size=50, sa_steps=20, seed=42):
    """Train with GSA on sine."""
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

    history = []

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

        # Log progress
        if gen % 50 == 0 or gen == generations - 1:
            mse = best.mse(X, y)
            sat = best.saturation(X)
            ws = best.weight_stats()
            history.append({
                'gen': gen,
                'mse': mse,
                'saturation': sat,
                'weight_stats': ws
            })
            log(f"  Gen {gen:3d}: MSE={mse:.6f}, Sat={sat:.1%}, W1_max={ws['W1_max_abs']:.2f}")

    return best, history


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 60)
    log("GSA ON SINE: SATURATION CHECK")
    log("=" * 60)

    # Generate data
    X, y = generate_sine_data(n_samples=500, n_features=16)
    log(f"\nData: {len(X)} samples, {X.shape[1]} features")
    log(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Test different configs
    configs = [
        {'H': 8, 'K': 2, 'name': 'Original (H=8, K=2)'},
        {'H': 8, 'K': 4, 'name': 'More connections (H=8, K=4)'},
        {'H': 32, 'K': 4, 'name': 'Larger (H=32, K=4)'},
    ]

    all_results = []

    for config in configs:
        log(f"\n{'='*60}")
        log(f"Config: {config['name']}")
        log(f"{'='*60}")

        model, history = train_gsa(
            X, y,
            H=config['H'],
            K=config['K'],
            generations=300,
            pop_size=50,
            sa_steps=20,
            seed=42
        )

        final_mse = model.mse(X, y)
        final_sat = model.saturation(X)
        ws = model.weight_stats()
        per_neuron = model.per_neuron_saturation(X)

        log(f"\nFinal Results:")
        log(f"  MSE: {final_mse:.6f}")
        log(f"  Saturation: {final_sat:.1%}")
        log(f"  W1 max abs: {ws['W1_max_abs']:.3f}")
        log(f"  W2 max abs: {ws['W2_max_abs']:.3f}")
        log(f"  Per-neuron saturation: {[f'{s:.0%}' for s in per_neuron]}")

        all_results.append({
            'config': config,
            'final_mse': float(final_mse),
            'final_saturation': float(final_sat),
            'weight_stats': ws,
            'per_neuron_saturation': per_neuron.tolist(),
            'history': history
        })

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    log(f"\n{'Config':<30} {'MSE':>12} {'Saturation':>12} {'W1_max':>10}")
    log("-" * 60)
    for r in all_results:
        log(f"{r['config']['name']:<30} {r['final_mse']:>12.6f} {r['final_saturation']:>11.1%} {r['weight_stats']['W1_max_abs']:>10.2f}")

    log("\n" + "=" * 60)
    log("ANALYSIS")
    log("=" * 60)
    log("\nKey Question: Does GSA on sine produce saturation?")

    avg_sat = np.mean([r['final_saturation'] for r in all_results])
    if avg_sat > 0.5:
        log(f"\nYES - Average saturation {avg_sat:.1%} (>50%)")
        log("Saturation is PROBLEM-SPECIFIC (sine naturally produces it)")
    elif avg_sat > 0.1:
        log(f"\nPARTIAL - Average saturation {avg_sat:.1%} (10-50%)")
        log("Some saturation occurs but not dominant")
    else:
        log(f"\nNO - Average saturation {avg_sat:.1%} (<10%)")
        log("GSA may be preventing saturation that original SA produced")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"results_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    log(f"\nResults saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
