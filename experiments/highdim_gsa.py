"""
Experiment 25: High-Dimensional Scaling with GSA

Problem: 1000 features, only 10 true (1% signal density)
Question: Does GSA improve on single SA for high-dim feature selection?

Key Findings:
- GSA and SA perform similarly (~0.077 MSE) on high-dim
- Both are 8.3x better than backprop (0.64 MSE)
- Both use 163x fewer params (49 vs 8017)
- Feature selection works: 7/10 true features found consistently

Conclusion: Evolvable indices are the key mechanism, not GSA vs SA.
Both methods crush backprop on high-dimensional problems.

References:
- Results: results/highdim_gsa/
- Log: docs/experiments_log.md (Experiment 25)
- Related: experiments/highdim_scaling.py (original SA version)
- Uses: usen.SparseNet, usen.train_gsa
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import time
from datetime import datetime
from collections import Counter

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "highdim_gsa"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


class SparseNet:
    """Sparse network for high-dim regression."""

    def __init__(self, input_dim, H=8, K=4):
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

    def clone(self):
        new = SparseNet(self.input_dim, self.H, self.K)
        new.indices = self.indices.copy()
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new

    def mutate(self, weight_rate=0.2, weight_std=0.1, index_rate=0.1):
        for i in range(self.H):
            if np.random.random() < weight_rate:
                j = np.random.randint(self.K)
                self.W1[i, j] += np.random.randn() * weight_std
            if np.random.random() < weight_rate * 0.5:
                self.b1[i] += np.random.randn() * weight_std * 0.5

        if np.random.random() < weight_rate:
            j = np.random.randint(self.H)
            self.W2[0, j] += np.random.randn() * weight_std

        for i in range(self.H):
            if np.random.random() < index_rate:
                j = np.random.randint(self.K)
                available = list(set(range(self.input_dim)) - set(self.indices[i]))
                if available:
                    self.indices[i, j] = np.random.choice(available)
                    self.W1[i, j] = np.random.randn() * 0.3

    def get_selected_indices(self):
        return list(set(self.indices.flatten()))

    def num_params(self):
        return self.H * self.K + self.H + self.H + 1


class DenseNet:
    """Dense network for comparison."""

    def __init__(self, input_dim, H=8):
        self.input_dim = input_dim
        self.H = H
        self.W1 = np.random.randn(H, input_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros(H, dtype=np.float32)
        self.W2 = np.random.randn(1, H).astype(np.float32) * 0.5
        self.b2 = np.zeros(1, dtype=np.float32)

    def predict(self, x):
        h = np.tanh(x @ self.W1.T + self.b1)
        out = np.tanh(h @ self.W2.T + self.b2).flatten()
        return out

    def mse(self, x, y):
        pred = self.predict(x)
        return np.mean((pred - y) ** 2)

    def num_params(self):
        return self.H * self.input_dim + self.H + self.H + 1


def generate_highdim_data(n_samples, n_features=1000, n_true=10, seed=42):
    """Generate high-dimensional regression data."""
    np.random.seed(seed)

    x = np.random.rand(n_samples, n_features).astype(np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    # Linear terms (features 0-4)
    weights = [10, 8, 6, 4, 2]
    for i, w in enumerate(weights):
        if i < n_true:
            y += w * x[:, i]

    # Nonlinear terms (features 5-9)
    if n_true > 5:
        y += 5 * np.sin(np.pi * x[:, 5] * x[:, 6])
    if n_true > 7:
        y += 3 * (x[:, 7] - 0.5) ** 2
    if n_true > 8:
        y += 2 * np.abs(x[:, 8] - 0.5)
    if n_true > 9:
        y += np.cos(2 * np.pi * x[:, 9])

    # Noise
    y += np.random.randn(n_samples).astype(np.float32) * 0.5

    # Normalize for tanh output
    y = (y - y.mean()) / (y.std() + 1e-8) * 0.8

    return x, y, list(range(n_true))


def train_gsa(X, y, true_features, H=8, K=4, generations=300, pop_size=50, sa_steps=20, seed=42):
    """Train with GSA (Genetic Simulated Annealing)."""
    np.random.seed(seed)

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
                m.mutate(index_rate=0.1)  # Higher index rate for 1000 features
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
            selected = best.get_selected_indices()
            true_found = len(set(selected) & set(true_features))
            log(f"  Gen {gen:3d}: MSE={mse:.4f}, True={true_found}/{len(true_features)}")

    return best


def train_single_sa(X, y, true_features, H=8, K=4, max_steps=100000, seed=42):
    """Train with single-chain SA (original method)."""
    np.random.seed(seed)

    model = SparseNet(X.shape[1], H, K)
    best = model.clone()
    best_mse = model.mse(X, y)

    temp = 0.05
    decay = (0.000001 / 0.05) ** (1.0 / max_steps)

    for step in range(max_steps):
        candidate = model.clone()
        candidate.mutate(index_rate=0.1)

        curr_mse = model.mse(X, y)
        new_mse = candidate.mse(X, y)

        delta = new_mse - curr_mse
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            model = candidate
            if new_mse < best_mse:
                best = candidate.clone()
                best_mse = new_mse

        temp *= decay

        if step % 20000 == 0 or step == max_steps - 1:
            selected = best.get_selected_indices()
            true_found = len(set(selected) & set(true_features))
            log(f"  Step {step:5d}: MSE={best_mse:.4f}, True={true_found}/{len(true_features)}")

    return best


def train_backprop(X, y, H=8, epochs=10000, seed=42):
    """Train dense network with backprop (PyTorch)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class DenseMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, H)
            self.fc2 = nn.Linear(H, 1)

        def forward(self, x):
            h = torch.tanh(self.fc1(x))
            return torch.tanh(self.fc2(h)).squeeze()

    model = DenseMLP(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_mse:
            best_mse = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 2000 == 0 or epoch == epochs - 1:
            log(f"  Epoch {epoch:5d}: MSE={loss.item():.4f}")

    model.load_state_dict(best_state)

    # Return params count
    n_params = H * X.shape[1] + H + H + 1
    return best_mse, n_params


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 70)
    log("HIGH-DIMENSIONAL SCALING: GSA vs SA vs Backprop")
    log("=" * 70)

    # Config
    n_features = 1000
    n_true = 10
    n_samples = 500
    H, K = 8, 4
    n_trials = 3

    log(f"\nConfig: {n_features} features, {n_true} true ({100*n_true/n_features:.1f}% signal)")
    log(f"Samples: {n_samples}")
    log(f"Architecture: H={H}, K={K}")

    sparse_params = H * K + H + H + 1
    dense_params = H * n_features + H + H + 1
    log(f"Sparse params: {sparse_params}")
    log(f"Dense params: {dense_params} ({dense_params//sparse_params}x more)")

    # Results
    results = {
        'gsa': [],
        'single_sa': [],
        'backprop': []
    }

    for trial in range(n_trials):
        seed = trial * 1000
        log(f"\n{'='*70}")
        log(f"TRIAL {trial+1}/{n_trials} (seed={seed})")
        log(f"{'='*70}")

        X, y, true_features = generate_highdim_data(n_samples, n_features, n_true, seed=42)
        log(f"True features: {true_features}")

        # 1. GSA
        log(f"\n1. GSA (300 gens, pop=50):")
        start = time.time()
        model_gsa = train_gsa(X, y, true_features, H, K, generations=300, pop_size=50, seed=seed)
        gsa_time = time.time() - start
        gsa_mse = model_gsa.mse(X, y)
        gsa_selected = model_gsa.get_selected_indices()
        gsa_true = len(set(gsa_selected) & set(true_features))
        log(f"   → MSE={gsa_mse:.4f}, True={gsa_true}/{n_true}, Time={gsa_time:.1f}s")
        results['gsa'].append({
            'mse': float(gsa_mse),
            'true_found': gsa_true,
            'selected': gsa_selected,
            'time': gsa_time
        })

        # 2. Single SA
        log(f"\n2. Single SA (100k steps):")
        start = time.time()
        model_sa = train_single_sa(X, y, true_features, H, K, max_steps=100000, seed=seed)
        sa_time = time.time() - start
        sa_mse = model_sa.mse(X, y)
        sa_selected = model_sa.get_selected_indices()
        sa_true = len(set(sa_selected) & set(true_features))
        log(f"   → MSE={sa_mse:.4f}, True={sa_true}/{n_true}, Time={sa_time:.1f}s")
        results['single_sa'].append({
            'mse': float(sa_mse),
            'true_found': sa_true,
            'selected': sa_selected,
            'time': sa_time
        })

        # 3. Backprop
        log(f"\n3. Dense Backprop (10k epochs):")
        start = time.time()
        bp_mse, bp_params = train_backprop(X, y, H, epochs=10000, seed=seed)
        bp_time = time.time() - start
        log(f"   → MSE={bp_mse:.4f}, Params={bp_params}, Time={bp_time:.1f}s")
        results['backprop'].append({
            'mse': float(bp_mse),
            'params': bp_params,
            'time': bp_time
        })

    # Summary
    log(f"\n{'='*70}")
    log("SUMMARY")
    log(f"{'='*70}")

    log(f"\n{'Method':<15} {'MSE (mean)':>12} {'MSE (best)':>12} {'True Found':>12} {'Params':>10}")
    log("-" * 65)

    for name in ['gsa', 'single_sa', 'backprop']:
        mses = [r['mse'] for r in results[name]]
        mean_mse = np.mean(mses)
        best_mse = np.min(mses)

        if name == 'backprop':
            params = results[name][0]['params']
            true_str = "N/A"
        else:
            params = sparse_params
            true_found = np.mean([r['true_found'] for r in results[name]])
            true_str = f"{true_found:.1f}/{n_true}"

        log(f"{name:<15} {mean_mse:>12.4f} {best_mse:>12.4f} {true_str:>12} {params:>10}")

    # Analysis
    log(f"\n{'='*70}")
    log("ANALYSIS")
    log(f"{'='*70}")

    gsa_mean = np.mean([r['mse'] for r in results['gsa']])
    sa_mean = np.mean([r['mse'] for r in results['single_sa']])
    bp_mean = np.mean([r['mse'] for r in results['backprop']])

    log(f"\nGSA vs Single SA: {sa_mean/gsa_mean:.1f}x better accuracy")
    log(f"GSA vs Backprop: {bp_mean/gsa_mean:.1f}x better accuracy")
    log(f"Single SA vs Backprop: {bp_mean/sa_mean:.1f}x better accuracy")

    log(f"\nParameter efficiency:")
    log(f"  Sparse: {sparse_params} params")
    log(f"  Dense: {dense_params} params ({dense_params//sparse_params}x more)")

    if gsa_mean < bp_mean:
        log(f"\n*** GSA WINS ON BOTH ACCURACY AND EFFICIENCY ***")
        log(f"GSA: {gsa_mean:.4f} MSE with {sparse_params} params")
        log(f"Backprop: {bp_mean:.4f} MSE with {dense_params} params")

    # Feature selection
    log(f"\n{'='*70}")
    log("FEATURE SELECTION")
    log(f"{'='*70}")

    for name in ['gsa', 'single_sa']:
        all_selected = []
        for r in results[name]:
            all_selected.extend(r['selected'])
        freq = Counter(all_selected)

        log(f"\n{name} - Top selected features:")
        for idx, count in freq.most_common(15):
            is_true = "TRUE" if idx < n_true else ""
            log(f"  Feature {idx}: {count}/{n_trials} trials {is_true}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else x)

    log(f"\nResults saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
