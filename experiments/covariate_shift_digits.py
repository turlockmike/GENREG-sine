"""
Experiment: Covariate Shift on Digits - GENREG vs SGD

Tests whether GENREG (saturated neurons) shows step-function degradation
under covariate shift on a HARDER problem where saturation actually occurs.

Previous experiment on Gaussian clusters failed because GENREG only reached
2.6% saturation. On digits, we've seen 60-90% saturation, so this should
properly test the hypothesis.

Shift types for image data:
- Noise: Add Gaussian noise to pixels
- Brightness: Scale pixel intensities
- Contrast: Increase/decrease contrast around mean
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "covariate_shift_digits"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    """Print and write to log file with flush."""
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


# =============================================================================
# DATA
# =============================================================================

def load_digits_data(seed=42):
    """Load and preprocess sklearn digits dataset."""
    data = load_digits()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test


def apply_shift(X: np.ndarray, shift_type: str, magnitude: float, seed=None) -> np.ndarray:
    """Apply distribution shift to digit images.

    Args:
        X: Data array (n_samples, 64) - flattened 8x8 images
        shift_type: One of 'noise', 'brightness', 'contrast', 'blur'
        magnitude: Shift amount (interpretation depends on type)
        seed: Random seed for reproducible noise

    Returns:
        Shifted data array
    """
    if seed is not None:
        np.random.seed(seed)

    X_shifted = X.copy()

    if shift_type == "noise":
        # Add Gaussian noise
        noise = np.random.randn(*X.shape).astype(np.float32) * magnitude
        X_shifted = X_shifted + noise

    elif shift_type == "brightness":
        # Shift all pixel values (additive)
        X_shifted = X_shifted + magnitude

    elif shift_type == "contrast":
        # Scale around mean (multiplicative)
        # magnitude > 0: increase contrast, < 0: decrease
        scale = 1.0 + magnitude
        mean = X_shifted.mean(axis=1, keepdims=True)
        X_shifted = mean + scale * (X_shifted - mean)

    elif shift_type == "dropout":
        # Randomly zero out pixels (simulates occlusion)
        mask = np.random.random(X.shape) > magnitude
        X_shifted = X_shifted * mask.astype(np.float32)

    elif shift_type == "blur":
        # Simple blur by averaging with neighbors (reshape to 8x8)
        # magnitude controls blur strength (0-1)
        X_2d = X_shifted.reshape(-1, 8, 8)
        blurred = np.zeros_like(X_2d)
        for i in range(8):
            for j in range(8):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 8 and 0 <= nj < 8:
                            neighbors.append(X_2d[:, ni, nj])
                blurred[:, i, j] = np.mean(neighbors, axis=0)
        X_shifted = (1 - magnitude) * X_shifted + magnitude * blurred.reshape(-1, 64)

    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

    return X_shifted.astype(np.float32)


# =============================================================================
# MODELS
# =============================================================================

class SGDNet(nn.Module):
    """Standard neural network trained with SGD/Adam."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.hidden(x))
        return self.output(h)

    def forward_with_activations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out, h

    def get_saturation_stats(self, x: torch.Tensor, threshold: float = 0.95) -> Dict:
        with torch.no_grad():
            _, h = self.forward_with_activations(x)
            saturated = (h.abs() > threshold).float()
            return {
                'mean_saturation': saturated.mean().item(),
                'per_neuron_saturation': saturated.mean(dim=0).cpu().numpy(),
                'gate_config': (saturated.mean(dim=0) > 0.5).cpu().numpy().astype(int)
            }


class GENREGNet:
    """GENREG sparse network with K inputs per hidden neuron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, K: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.K = K

        # Sparse hidden layer
        self.hidden_indices = np.array([
            np.random.choice(input_dim, size=K, replace=False)
            for _ in range(hidden_dim)
        ])
        self.hidden_weights = np.random.randn(hidden_dim, K).astype(np.float32) * 0.5
        self.hidden_bias = np.zeros(hidden_dim, dtype=np.float32)

        # Dense output layer
        self.output_weights = np.random.randn(output_dim, hidden_dim).astype(np.float32) * 0.5
        self.output_bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        for i in range(self.hidden_dim):
            selected = x[:, self.hidden_indices[i]]
            hidden[:, i] = selected @ self.hidden_weights[i] + self.hidden_bias[i]
        hidden = np.tanh(hidden)
        return hidden @ self.output_weights.T + self.output_bias

    def forward_with_activations(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = x.shape[0]
        hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        for i in range(self.hidden_dim):
            selected = x[:, self.hidden_indices[i]]
            hidden[:, i] = selected @ self.hidden_weights[i] + self.hidden_bias[i]
        hidden_act = np.tanh(hidden)
        out = hidden_act @ self.output_weights.T + self.output_bias
        return out, hidden_act

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_saturation_stats(self, x: np.ndarray, threshold: float = 0.95) -> Dict:
        _, h = self.forward_with_activations(x)
        saturated = (np.abs(h) > threshold).astype(np.float32)
        return {
            'mean_saturation': saturated.mean(),
            'per_neuron_saturation': saturated.mean(axis=0),
            'gate_config': (saturated.mean(axis=0) > 0.5).astype(int)
        }

    def clone(self) -> 'GENREGNet':
        new = GENREGNet(self.input_dim, self.hidden_dim, self.output_dim, self.K)
        new.hidden_indices = self.hidden_indices.copy()
        new.hidden_weights = self.hidden_weights.copy()
        new.hidden_bias = self.hidden_bias.copy()
        new.output_weights = self.output_weights.copy()
        new.output_bias = self.output_bias.copy()
        return new

    def mutate(self, weight_rate=0.2, weight_std=0.1, index_swap_rate=0.15):
        for i in range(self.hidden_dim):
            if np.random.random() < weight_rate:
                idx = np.random.randint(self.K)
                self.hidden_weights[i, idx] += np.random.randn() * weight_std
            if np.random.random() < weight_rate * 0.5:
                self.hidden_bias[i] += np.random.randn() * weight_std * 0.5

        if np.random.random() < weight_rate:
            i = np.random.randint(self.output_dim)
            j = np.random.randint(self.hidden_dim)
            self.output_weights[i, j] += np.random.randn() * weight_std

        for i in range(self.hidden_dim):
            if np.random.random() < index_swap_rate:
                old_idx = np.random.randint(self.K)
                available = list(set(range(self.input_dim)) - set(self.hidden_indices[i]))
                if available:
                    self.hidden_indices[i, old_idx] = np.random.choice(available)
                    self.hidden_weights[i, old_idx] = np.random.randn() * 0.3

    def num_params(self) -> int:
        return (self.hidden_dim * self.K + self.hidden_dim +
                self.output_dim * self.hidden_dim + self.output_dim)


# =============================================================================
# TRAINING
# =============================================================================

def train_sgd(model, X_train, y_train, X_val, y_val, epochs=500, lr=0.01,
              batch_size=64, target_acc=0.90, verbose=True):
    """Train SGD network."""
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_train_t[batch_idx]), y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()
            train_acc = (model(X_train_t).argmax(dim=1) == y_train_t).float().mean().item()

        if epoch % 50 == 0 and verbose:
            log(f"  SGD Epoch {epoch}: train={train_acc:.1%}, val={val_acc:.1%}")

        if val_acc >= target_acc:
            if verbose:
                log(f"  SGD reached {target_acc:.0%} at epoch {epoch}")
            break

    return {'final_train_acc': train_acc, 'final_val_acc': val_acc, 'epochs': epoch + 1}


def train_genreg(model, X_train, y_train, X_val, y_val, generations=300,
                 pop_size=50, sa_steps=20, target_acc=0.85, verbose=True):
    """Train GENREG with GSA."""
    n_classes = 10
    y_onehot = np.zeros((len(y_train), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_train)), y_train] = 1
    y_onehot = y_onehot * 1.6 - 0.8

    def fitness(net):
        preds = np.tanh(net.forward(X_train))
        return -np.mean((preds - y_onehot) ** 2)

    def accuracy(net, X, y):
        return (net.predict(X) == y).mean()

    # Initialize population
    population = [(model.clone(), fitness(model))]
    for _ in range(pop_size - 1):
        ind = model.clone()
        ind.mutate(weight_rate=0.5, weight_std=0.3, index_swap_rate=0.3)
        population.append((ind, fitness(ind)))

    best_model = model.clone()
    best_fitness = fitness(best_model)

    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    for gen in range(generations):
        population.sort(key=lambda x: x[1], reverse=True)

        n_elite = max(1, pop_size // 20)
        new_pop = [(ind.clone(), f) for ind, f in population[:n_elite]]

        fitnesses = np.array([f for _, f in population])
        probs = fitnesses - fitnesses.min() + 1e-8
        probs = probs / probs.sum()

        for _ in range(pop_size - n_elite):
            idx = np.random.choice(len(population), p=probs)
            c = population[idx][0].clone()
            current_f = population[idx][1]
            best_c, best_inner_f = c, current_f

            for _ in range(sa_steps):
                mutant = c.clone()
                mutant.mutate()
                f = fitness(mutant)
                delta = f - current_f
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    c = mutant
                    current_f = f
                    if f > best_inner_f:
                        best_c, best_inner_f = mutant.clone(), f

            new_pop.append((best_c, best_inner_f))

        population = new_pop

        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_model = gen_best[0].clone()
            best_fitness = gen_best[1]

        temp *= decay

        val_acc = accuracy(best_model, X_val, y_val)
        train_acc = accuracy(best_model, X_train, y_train)

        if gen % 50 == 0 and verbose:
            sat = best_model.get_saturation_stats(X_train)['mean_saturation']
            log(f"  GENREG Gen {gen}: train={train_acc:.1%}, val={val_acc:.1%}, sat={sat:.1%}")

        if val_acc >= target_acc:
            if verbose:
                log(f"  GENREG reached {target_acc:.0%} at gen {gen}")
            break

    return {
        'model': best_model,
        'final_train_acc': accuracy(best_model, X_train, y_train),
        'final_val_acc': accuracy(best_model, X_val, y_val),
        'generations': gen + 1
    }


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class EvalResult:
    shift_magnitude: float
    accuracy: float
    mean_confidence: float
    mean_saturation: float
    gate_config: np.ndarray


def evaluate_under_shift(model, X_test, y_test, shift_type, max_mag, n_steps, seed=42):
    """Evaluate model under increasing shift."""
    results = []
    magnitudes = np.linspace(0, max_mag, n_steps)

    for mag in magnitudes:
        X_shifted = apply_shift(X_test, shift_type, mag, seed=seed)

        if isinstance(model, SGDNet):
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_shifted)
                logits = model(X_t)
                probs = torch.softmax(logits, dim=1).numpy()
                preds = logits.argmax(dim=1).numpy()
                sat_stats = model.get_saturation_stats(X_t)
        else:
            probs = model.predict_proba(X_shifted)
            preds = probs.argmax(axis=1)
            sat_stats = model.get_saturation_stats(X_shifted)

        results.append(EvalResult(
            shift_magnitude=mag,
            accuracy=(preds == y_test).mean(),
            mean_confidence=probs.max(axis=1).mean(),
            mean_saturation=sat_stats['mean_saturation'],
            gate_config=sat_stats['gate_config']
        ))

    return results


def compute_degradation_metrics(results):
    """Compute degradation curve metrics."""
    accs = np.array([r.accuracy for r in results])
    mags = np.array([r.shift_magnitude for r in results])

    derivatives = np.gradient(accs, mags) if len(accs) > 1 else np.array([0])
    auc = np.trapezoid(accs, mags)

    return {
        'auc': auc,
        'max_derivative': derivatives.min(),
        'mean_derivative': derivatives.mean(),
        'final_accuracy': accs[-1],
        'accuracy_drop': accs[0] - accs[-1]
    }


def track_gate_changes(results):
    """Track GENREG gate configuration changes."""
    configs = np.array([r.gate_config for r in results])
    changes = np.sum(configs[1:] != configs[:-1], axis=1)

    return {
        'total_gate_changes': int(changes.sum()),
        'changes_per_step': changes.tolist(),
        'n_flip_points': int((changes >= 2).sum())
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(sgd_results, genreg_results, shift_type, save_path):
    """Plot degradation comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mags = [r.shift_magnitude for r in sgd_results]

    # Accuracy
    ax = axes[0, 0]
    ax.plot(mags, [r.accuracy for r in sgd_results], 'b-o', label='SGD', lw=2)
    ax.plot(mags, [r.accuracy for r in genreg_results], 'r-s', label='GENREG', lw=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Confidence
    ax = axes[0, 1]
    ax.plot(mags, [r.mean_confidence for r in sgd_results], 'b-o', label='SGD', lw=2)
    ax.plot(mags, [r.mean_confidence for r in genreg_results], 'r-s', label='GENREG', lw=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Mean Confidence')
    ax.set_title('Confidence Under Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GENREG Saturation
    ax = axes[1, 0]
    ax.plot(mags, [r.mean_saturation for r in genreg_results], 'r-s', lw=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Mean Saturation')
    ax.set_title('GENREG Saturation Under Shift')
    ax.grid(True, alpha=0.3)

    # Derivative
    ax = axes[1, 1]
    sgd_acc = np.array([r.accuracy for r in sgd_results])
    genreg_acc = np.array([r.accuracy for r in genreg_results])
    if len(mags) > 1:
        ax.plot(mags, np.gradient(sgd_acc, mags), 'b-o', label='SGD', lw=2)
        ax.plot(mags, np.gradient(genreg_acc, mags), 'r-s', label='GENREG', lw=2)
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Accuracy Derivative')
    ax.set_title('Degradation Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

@dataclass
class Config:
    hidden_dim: int = 32
    K: int = 4
    sgd_epochs: int = 500
    sgd_target: float = 0.95
    genreg_generations: int = 300
    genreg_pop_size: int = 50
    genreg_target: float = 0.85
    shift_types: List[str] = field(default_factory=lambda: ['noise', 'brightness', 'contrast'])
    shift_max: float = 2.0
    shift_steps: int = 25
    seed: int = 42


def main(config: Config):
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 70)
    log("COVARIATE SHIFT ON DIGITS: GENREG vs SGD")
    log("=" * 70)
    log(f"\nConfig: H={config.hidden_dim}, K={config.K}")
    log(f"This is a HARDER problem - expecting higher saturation in GENREG")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load data
    log("\n1. Loading digits dataset...")
    X_train, X_test, y_train, y_test = load_digits_data(config.seed)
    log(f"   Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")

    # Train SGD
    log("\n2. Training SGD model...")
    sgd_model = SGDNet(64, config.hidden_dim, 10)
    sgd_results = train_sgd(sgd_model, X_train, y_train, X_test, y_test,
                            epochs=config.sgd_epochs, target_acc=config.sgd_target)
    sgd_sat = sgd_model.get_saturation_stats(torch.tensor(X_train))
    log(f"   Final: train={sgd_results['final_train_acc']:.1%}, val={sgd_results['final_val_acc']:.1%}")
    log(f"   Saturation: {sgd_sat['mean_saturation']:.1%}")

    # Train GENREG
    log(f"\n3. Training GENREG model (H={config.hidden_dim}, K={config.K})...")
    genreg_model = GENREGNet(64, config.hidden_dim, 10, K=config.K)
    log(f"   Architecture: {genreg_model.num_params()} params")
    genreg_results = train_genreg(genreg_model, X_train, y_train, X_test, y_test,
                                   generations=config.genreg_generations,
                                   pop_size=config.genreg_pop_size,
                                   target_acc=config.genreg_target)
    genreg_model = genreg_results['model']
    genreg_sat = genreg_model.get_saturation_stats(X_train)
    log(f"   Final: train={genreg_results['final_train_acc']:.1%}, val={genreg_results['final_val_acc']:.1%}")
    log(f"   Saturation: {genreg_sat['mean_saturation']:.1%}")

    # Evaluate under shift
    all_results = {}

    for shift_type in config.shift_types:
        log(f"\n4. Evaluating under {shift_type} shift...")

        sgd_shift = evaluate_under_shift(sgd_model, X_test, y_test, shift_type,
                                         config.shift_max, config.shift_steps)
        genreg_shift = evaluate_under_shift(genreg_model, X_test, y_test, shift_type,
                                            config.shift_max, config.shift_steps)

        sgd_metrics = compute_degradation_metrics(sgd_shift)
        genreg_metrics = compute_degradation_metrics(genreg_shift)
        gate_changes = track_gate_changes(genreg_shift)

        log(f"   SGD: AUC={sgd_metrics['auc']:.2f}, max_deriv={sgd_metrics['max_derivative']:.3f}")
        log(f"   GENREG: AUC={genreg_metrics['auc']:.2f}, max_deriv={genreg_metrics['max_derivative']:.3f}")
        log(f"   GENREG gate changes: {gate_changes['total_gate_changes']}, flip points: {gate_changes['n_flip_points']}")

        plot_results(sgd_shift, genreg_shift, shift_type, LOG_DIR / f"degradation_{shift_type}.png")

        all_results[shift_type] = {
            'sgd_metrics': sgd_metrics,
            'genreg_metrics': genreg_metrics,
            'gate_changes': gate_changes
        }

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"\nSGD saturation: {sgd_sat['mean_saturation']:.1%}")
    log(f"GENREG saturation: {genreg_sat['mean_saturation']:.1%}")

    log(f"\n{'Shift':<12} {'SGD AUC':>10} {'GENREG AUC':>12} {'SGD MaxDer':>12} {'GEN MaxDer':>12} {'Gate Flips':>12}")
    log("-" * 70)
    for shift_type in config.shift_types:
        r = all_results[shift_type]
        log(f"{shift_type:<12} {r['sgd_metrics']['auc']:>10.2f} {r['genreg_metrics']['auc']:>12.2f} "
            f"{r['sgd_metrics']['max_derivative']:>12.3f} {r['genreg_metrics']['max_derivative']:>12.3f} "
            f"{r['gate_changes']['total_gate_changes']:>12}")

    log("\n" + "=" * 70)
    log("HYPOTHESIS TEST")
    log("=" * 70)
    log("\nExpected (if hypothesis true):")
    log("  - GENREG max_derivative MORE NEGATIVE than SGD (steeper drops)")
    log("  - GENREG should have gate flip points")
    log("\nObserved:")
    for shift_type in config.shift_types:
        r = all_results[shift_type]
        sgd_d = r['sgd_metrics']['max_derivative']
        gen_d = r['genreg_metrics']['max_derivative']
        steeper = "GENREG" if gen_d < sgd_d else "SGD"
        log(f"  {shift_type}: {steeper} degrades more steeply, {r['gate_changes']['total_gate_changes']} gate flips")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"results_{timestamp}.json", 'w') as f:
        json.dump({
            'config': {'hidden_dim': config.hidden_dim, 'K': config.K},
            'training': {
                'sgd_acc': sgd_results['final_val_acc'],
                'sgd_saturation': sgd_sat['mean_saturation'],
                'genreg_acc': genreg_results['final_val_acc'],
                'genreg_saturation': genreg_sat['mean_saturation']
            },
            'shift_results': {k: {
                'sgd_auc': v['sgd_metrics']['auc'],
                'genreg_auc': v['genreg_metrics']['auc'],
                'sgd_max_deriv': v['sgd_metrics']['max_derivative'],
                'genreg_max_deriv': v['genreg_metrics']['max_derivative'],
                'gate_changes': v['gate_changes']['total_gate_changes']
            } for k, v in all_results.items()}
        }, f, indent=2)

    log(f"\nResults saved to: {LOG_DIR}")


if __name__ == "__main__":
    config = Config(
        hidden_dim=32,
        K=4,
        genreg_generations=300,
        genreg_target=0.80,  # Lower target since digits is harder
        shift_types=['noise', 'brightness', 'contrast'],
        shift_max=2.0,
        shift_steps=25,
    )
    main(config)
