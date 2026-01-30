"""
Experiment: Training Method Comparison on Sine

Compares GSA vs Single-Chain SA vs Backprop on the original sine problem.

Key Questions:
1. Does GSA achieve similar accuracy to single-chain SA?
2. Does single-chain SA produce high saturation while GSA doesn't?
3. How does backprop compare on the same sparse architecture?
4. What are the efficiency tradeoffs (params, compute, accuracy)?

This could reveal GSA as a significant improvement if it achieves
good accuracy WITHOUT the saturation/brittleness of single-chain SA.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from datetime import datetime

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "training_method_comparison"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


# =============================================================================
# DATA
# =============================================================================

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

    return X, y, x


# =============================================================================
# SPARSE NETWORK (NumPy - for SA and GSA)
# =============================================================================

class SparseNetNumpy:
    """Sparse network for evolutionary training."""

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
        return np.mean((self.predict(x) - y) ** 2)

    def saturation(self, x, threshold=0.95):
        h, _ = self.forward(x)
        return (np.abs(h) > threshold).mean()

    def clone(self):
        new = SparseNetNumpy(self.input_dim, self.H, self.K)
        new.indices = self.indices.copy()
        new.W1 = self.W1.copy()
        new.b1 = self.b1.copy()
        new.W2 = self.W2.copy()
        new.b2 = self.b2.copy()
        return new

    def mutate(self, weight_rate=0.2, weight_std=0.1, index_rate=0.15):
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

    def num_params(self):
        return self.H * self.K + self.H + self.H + 1

    def weight_stats(self):
        return {
            'W1_mean': float(np.abs(self.W1).mean()),
            'W1_max': float(np.abs(self.W1).max()),
            'W2_mean': float(np.abs(self.W2).mean()),
            'W2_max': float(np.abs(self.W2).max()),
        }


# =============================================================================
# SPARSE NETWORK (PyTorch - for Backprop)
# =============================================================================

class SparseNetTorch(nn.Module):
    """Sparse network for backprop training."""

    def __init__(self, input_dim=16, H=8, K=2, indices=None):
        super().__init__()
        self.input_dim = input_dim
        self.H = H
        self.K = K

        # Fixed sparse indices (not learnable)
        if indices is None:
            indices = np.array([
                np.random.choice(input_dim, K, replace=False) for _ in range(H)
            ])
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))

        # Learnable weights
        self.W1 = nn.Parameter(torch.randn(H, K) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(H))
        self.W2 = nn.Parameter(torch.randn(1, H) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.shape[0]
        h = torch.zeros(batch_size, self.H, device=x.device)
        for i in range(self.H):
            selected = x[:, self.indices[i]]
            h[:, i] = (selected * self.W1[i]).sum(dim=1) + self.b1[i]
        h_act = torch.tanh(h)
        out = torch.tanh(h_act @ self.W2.T + self.b2).squeeze(-1)
        return h_act, out

    def predict(self, x):
        _, out = self.forward(x)
        return out

    def saturation(self, x, threshold=0.95):
        with torch.no_grad():
            h, _ = self.forward(x)
            return (h.abs() > threshold).float().mean().item()

    def num_params(self):
        return self.H * self.K + self.H + self.H + 1


# =============================================================================
# TRAINING METHODS
# =============================================================================

def train_single_sa(X, y, H=8, K=2, steps=15000, seed=42, verbose=True):
    """Single-chain Simulated Annealing."""
    np.random.seed(seed)

    model = SparseNetNumpy(X.shape[1], H, K)
    best = model.clone()
    best_mse = model.mse(X, y)

    temp = 1.0
    decay = (0.001 / 1.0) ** (1.0 / steps)

    history = []
    start_time = time.time()

    for step in range(steps):
        mutant = model.clone()
        mutant.mutate()
        mutant_mse = mutant.mse(X, y)

        delta = mutant_mse - model.mse(X, y)
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            model = mutant
            if mutant_mse < best_mse:
                best = mutant.clone()
                best_mse = mutant_mse

        temp *= decay

        if step % 3000 == 0 or step == steps - 1:
            sat = best.saturation(X)
            history.append({'step': step, 'mse': best_mse, 'saturation': sat})
            if verbose:
                log(f"    Step {step:5d}: MSE={best_mse:.6f}, Sat={sat:.1%}")

    elapsed = time.time() - start_time
    return best, history, elapsed


def train_gsa(X, y, H=8, K=2, generations=300, pop_size=50, sa_steps=20, seed=42, verbose=True):
    """Genetic Simulated Annealing (population-based)."""
    np.random.seed(seed)

    def fitness(net):
        return -net.mse(X, y)

    # Initialize population
    base = SparseNetNumpy(X.shape[1], H, K)
    pop = [(base.clone(), fitness(base))]
    for _ in range(pop_size - 1):
        net = base.clone()
        net.mutate(weight_rate=0.5, weight_std=0.3, index_rate=0.3)
        pop.append((net, fitness(net)))

    best = pop[0][0].clone()
    best_f = fitness(best)

    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    history = []
    start_time = time.time()

    for gen in range(generations):
        pop.sort(key=lambda x: x[1], reverse=True)

        if pop[0][1] > best_f:
            best = pop[0][0].clone()
            best_f = pop[0][1]

        n_elite = max(1, pop_size // 20)
        new_pop = [(c.clone(), f) for c, f in pop[:n_elite]]

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
            history.append({'gen': gen, 'mse': mse, 'saturation': sat})
            if verbose:
                log(f"    Gen {gen:3d}: MSE={mse:.6f}, Sat={sat:.1%}")

    elapsed = time.time() - start_time
    return best, history, elapsed


def train_backprop(X, y, H=8, K=2, epochs=5000, lr=0.01, seed=42, verbose=True):
    """Backpropagation with Adam optimizer."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = SparseNetTorch(X.shape[1], H, K)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = []
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()
        _, pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            mse = loss.item()
            sat = model.saturation(X_t)
            history.append({'epoch': epoch, 'mse': mse, 'saturation': sat})
            if verbose:
                log(f"    Epoch {epoch:5d}: MSE={mse:.6f}, Sat={sat:.1%}")

    elapsed = time.time() - start_time

    # Convert to numpy model for consistent interface
    result_model = SparseNetNumpy(X.shape[1], H, K)
    result_model.indices = model.indices.cpu().numpy()
    result_model.W1 = model.W1.detach().cpu().numpy()
    result_model.b1 = model.b1.detach().cpu().numpy()
    result_model.W2 = model.W2.detach().cpu().numpy()
    result_model.b2 = model.b2.detach().cpu().numpy()

    return result_model, history, elapsed


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 70)
    log("TRAINING METHOD COMPARISON: GSA vs SA vs Backprop")
    log("=" * 70)

    # Generate data
    X, y, x_vals = generate_sine_data(n_samples=500, n_features=16)
    log(f"\nData: {len(X)} samples, {X.shape[1]} features")
    log(f"Target: sin(x), range [{y.min():.2f}, {y.max():.2f}]")

    # Test configuration
    H, K = 8, 2
    n_trials = 3
    log(f"\nArchitecture: H={H}, K={K} ({H*K + H + H + 1} params)")
    log(f"Trials per method: {n_trials}")

    results = {
        'single_sa': [],
        'gsa': [],
        'backprop': []
    }

    # Run trials
    for trial in range(n_trials):
        seed = trial * 1000
        log(f"\n{'='*70}")
        log(f"TRIAL {trial + 1}/{n_trials} (seed={seed})")
        log(f"{'='*70}")

        # Single-chain SA
        log(f"\n1. Single-Chain SA (15000 steps):")
        model_sa, hist_sa, time_sa = train_single_sa(X, y, H, K, steps=15000, seed=seed)
        results['single_sa'].append({
            'mse': model_sa.mse(X, y),
            'saturation': model_sa.saturation(X),
            'time': time_sa,
            'weights': model_sa.weight_stats(),
            'history': hist_sa
        })
        log(f"   → MSE={model_sa.mse(X, y):.6f}, Sat={model_sa.saturation(X):.1%}, Time={time_sa:.1f}s")

        # GSA
        log(f"\n2. GSA (300 gens, pop=50, 20 SA steps/member):")
        model_gsa, hist_gsa, time_gsa = train_gsa(X, y, H, K, generations=300, pop_size=50, sa_steps=20, seed=seed)
        results['gsa'].append({
            'mse': model_gsa.mse(X, y),
            'saturation': model_gsa.saturation(X),
            'time': time_gsa,
            'weights': model_gsa.weight_stats(),
            'history': hist_gsa
        })
        log(f"   → MSE={model_gsa.mse(X, y):.6f}, Sat={model_gsa.saturation(X):.1%}, Time={time_gsa:.1f}s")

        # Backprop
        log(f"\n3. Backprop (5000 epochs, Adam lr=0.01):")
        model_bp, hist_bp, time_bp = train_backprop(X, y, H, K, epochs=5000, lr=0.01, seed=seed)
        results['backprop'].append({
            'mse': model_bp.mse(X, y),
            'saturation': model_bp.saturation(X),
            'time': time_bp,
            'weights': model_bp.weight_stats(),
            'history': hist_bp
        })
        log(f"   → MSE={model_bp.mse(X, y):.6f}, Sat={model_bp.saturation(X):.1%}, Time={time_bp:.1f}s")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\n{'Method':<15} {'MSE (mean)':>15} {'MSE (best)':>15} {'Saturation':>12} {'Time':>10}")
    log("-" * 70)

    for method in ['single_sa', 'gsa', 'backprop']:
        mses = [r['mse'] for r in results[method]]
        sats = [r['saturation'] for r in results[method]]
        times = [r['time'] for r in results[method]]
        log(f"{method:<15} {np.mean(mses):>15.6f} {np.min(mses):>15.6f} {np.mean(sats):>11.1%} {np.mean(times):>9.1f}s")

    # Analysis
    log("\n" + "=" * 70)
    log("ANALYSIS")
    log("=" * 70)

    sa_mse = np.mean([r['mse'] for r in results['single_sa']])
    gsa_mse = np.mean([r['mse'] for r in results['gsa']])
    bp_mse = np.mean([r['mse'] for r in results['backprop']])

    sa_sat = np.mean([r['saturation'] for r in results['single_sa']])
    gsa_sat = np.mean([r['saturation'] for r in results['gsa']])
    bp_sat = np.mean([r['saturation'] for r in results['backprop']])

    log(f"\nAccuracy comparison:")
    log(f"  Best MSE: {'Backprop' if bp_mse < min(sa_mse, gsa_mse) else 'GSA' if gsa_mse < sa_mse else 'Single SA'}")
    log(f"  GSA vs SA: {gsa_mse/sa_mse:.2f}x MSE ratio")
    log(f"  GSA vs Backprop: {gsa_mse/bp_mse:.2f}x MSE ratio")

    log(f"\nSaturation comparison:")
    log(f"  Single SA: {sa_sat:.1%}")
    log(f"  GSA: {gsa_sat:.1%}")
    log(f"  Backprop: {bp_sat:.1%}")

    if sa_sat > 0.5 and gsa_sat < 0.1:
        log(f"\n*** KEY FINDING ***")
        log(f"Single SA produces HIGH saturation ({sa_sat:.0%})")
        log(f"GSA produces LOW saturation ({gsa_sat:.0%})")
        log(f"This confirms saturation is TRAINING-METHOD dependent!")

    if gsa_mse < sa_mse * 1.5 and gsa_sat < sa_sat * 0.5:
        log(f"\n*** GSA ADVANTAGE ***")
        log(f"GSA achieves similar accuracy with much lower saturation")
        log(f"This could mean more robust, smoother networks!")

    # Weight analysis
    log(f"\nWeight magnitude comparison:")
    for method in ['single_sa', 'gsa', 'backprop']:
        w1_max = np.mean([r['weights']['W1_max'] for r in results[method]])
        log(f"  {method}: W1_max={w1_max:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"results_{timestamp}.json", 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        json.dump(convert({
            'config': {'H': H, 'K': K, 'n_trials': n_trials},
            'results': results,
            'summary': {
                'single_sa': {'mean_mse': sa_mse, 'mean_sat': sa_sat},
                'gsa': {'mean_mse': gsa_mse, 'mean_sat': gsa_sat},
                'backprop': {'mean_mse': bp_mse, 'mean_sat': bp_sat}
            }
        }), f, indent=2)

    log(f"\nResults saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
