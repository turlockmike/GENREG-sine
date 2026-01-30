"""
Experiment: Covariate Shift Robustness - GENREG vs SGD

Tests whether GENREG (saturated neurons) shows step-function degradation
under covariate shift vs smooth degradation typical of SGD-trained nets.

Hypothesis: Binary/saturated neurons flip suddenly when inputs cross
decision boundaries, causing abrupt accuracy drops. SGD's continuous
activations should degrade more gradually.

Design:
1. Synthetic 2D Gaussian clusters (4 classes, controllable shift)
2. Train GENREG and SGD to comparable accuracy (~95%+)
3. Incrementally shift test distribution
4. Track: accuracy, confidence calibration, saturation, gate configs
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
from typing import List, Tuple, Dict, Optional

# Progressive logging
LOG_DIR = Path(__file__).parent.parent / "results" / "covariate_shift"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "progress.log"


def log(msg):
    """Print and write to log file with flush."""
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} | {msg}\n")


# =============================================================================
# DATA GENERATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for synthetic Gaussian cluster data."""
    n_classes: int = 4
    n_samples_per_class: int = 250
    n_dims: int = 16  # Higher dim so K=4 sparse connectivity makes sense
    cluster_std: float = 0.5
    seed: int = 42

    # Cluster centers will be generated in n_dims space
    # Using corners of hypercube for n_classes
    base_centers: np.ndarray = None  # Generated dynamically based on n_dims

    cluster_separation: float = 2.0  # Multiplier for base centers


def generate_gaussian_clusters(config: DataConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian cluster data for classification in n_dims space."""
    np.random.seed(config.seed)

    # Generate cluster centers at corners of hypercube
    # For 4 classes in n_dims, use first 2 dimensions to separate
    centers = np.zeros((config.n_classes, config.n_dims), dtype=np.float32)
    if config.n_classes == 4:
        # Quadrant pattern in first 2 dims, rest zeros
        centers[0, :2] = [-1, -1]  # Class 0
        centers[1, :2] = [-1,  1]  # Class 1
        centers[2, :2] = [ 1, -1]  # Class 2
        centers[3, :2] = [ 1,  1]  # Class 3
    else:
        # Random centers for other class counts
        centers = np.random.randn(config.n_classes, config.n_dims).astype(np.float32)

    centers *= config.cluster_separation

    X_list, y_list = [], []
    for class_idx in range(config.n_classes):
        # Generate samples around cluster center
        samples = np.random.randn(config.n_samples_per_class, config.n_dims).astype(np.float32) * config.cluster_std
        samples += centers[class_idx]

        X_list.append(samples)
        y_list.append(np.full(config.n_samples_per_class, class_idx))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


@dataclass
class ShiftConfig:
    """Configuration for distribution shift."""
    shift_type: str = "translation"  # translation, rotation, scaling
    max_magnitude: float = 2.0       # Maximum shift amount
    n_steps: int = 20                # Number of evaluation points


def apply_shift(X: np.ndarray, shift_type: str, magnitude: float) -> np.ndarray:
    """Apply distribution shift to data.

    Args:
        X: Data array (n_samples, n_dims)
        shift_type: One of 'translation', 'rotation', 'scaling'
        magnitude: Shift amount (interpretation depends on type)

    Returns:
        Shifted data array
    """
    X_shifted = X.copy()

    if shift_type == "translation":
        # Shift all points in first 2 dimensions (where class structure is)
        X_shifted[:, 0] += magnitude
        X_shifted[:, 1] += magnitude * 0.5  # Diagonal shift

    elif shift_type == "rotation":
        # Rotate first 2 dimensions around origin by magnitude radians
        # (these are the dimensions with class structure)
        cos_m, sin_m = np.cos(magnitude), np.sin(magnitude)
        x0, x1 = X_shifted[:, 0].copy(), X_shifted[:, 1].copy()
        X_shifted[:, 0] = cos_m * x0 - sin_m * x1
        X_shifted[:, 1] = sin_m * x0 + cos_m * x1

    elif shift_type == "scaling":
        # Scale from origin (1.0 = no change, >1 = expand, <1 = contract)
        scale_factor = 1.0 + magnitude
        X_shifted *= scale_factor

    elif shift_type == "noise":
        # Add Gaussian noise with increasing std
        X_shifted += np.random.randn(*X.shape).astype(np.float32) * magnitude

    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

    return X_shifted


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
        """Forward pass that also returns hidden activations."""
        h = torch.tanh(self.hidden(x))
        out = self.output(h)
        return out, h

    def get_saturation_stats(self, x: torch.Tensor, threshold: float = 0.95) -> Dict:
        """Compute saturation statistics for hidden layer."""
        with torch.no_grad():
            _, h = self.forward_with_activations(x)
            saturated = (h.abs() > threshold).float()

            return {
                'mean_saturation': saturated.mean().item(),
                'per_neuron_saturation': saturated.mean(dim=0).cpu().numpy(),
                'gate_config': (saturated.mean(dim=0) > 0.5).cpu().numpy().astype(int)
            }


class GENREGNet:
    """GENREG-style SPARSE network trained with evolutionary methods.

    Key difference from SGDNet: Each hidden neuron connects to only K inputs,
    not all inputs. This sparse connectivity + evolvable indices is what
    creates selection pressure and tends to produce saturated neurons.

    Architecture:
    - Hidden layer: H neurons, each seeing only K inputs (sparse)
    - Output layer: Dense connection from hidden to output
    - Indices are evolvable (can mutate which inputs each neuron sees)

    This is the architecture that produces binary-like saturated neurons,
    which we hypothesize will show step-function degradation under shift.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, K: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.K = K  # Inputs per hidden neuron (sparsity constraint)

        # Hidden layer: each neuron connects to K random inputs
        self.hidden_indices = np.array([
            np.random.choice(input_dim, size=K, replace=False)
            for _ in range(hidden_dim)
        ])
        self.hidden_weights = np.random.randn(hidden_dim, K).astype(np.float32) * 0.5
        self.hidden_bias = np.zeros(hidden_dim, dtype=np.float32)

        # Output layer: fully connected from hidden
        self.output_weights = np.random.randn(output_dim, hidden_dim).astype(np.float32) * 0.5
        self.output_bias = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with sparse connectivity."""
        batch_size = x.shape[0]

        # Hidden layer with sparse inputs
        hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        for i in range(self.hidden_dim):
            selected = x[:, self.hidden_indices[i]]  # (batch, K)
            hidden[:, i] = selected @ self.hidden_weights[i] + self.hidden_bias[i]
        hidden = np.tanh(hidden)

        # Output layer
        return hidden @ self.output_weights.T + self.output_bias

    def forward_with_activations(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass that also returns hidden activations."""
        batch_size = x.shape[0]

        hidden = np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        for i in range(self.hidden_dim):
            selected = x[:, self.hidden_indices[i]]
            hidden[:, i] = selected @ self.hidden_weights[i] + self.hidden_bias[i]
        hidden_act = np.tanh(hidden)

        out = hidden_act @ self.output_weights.T + self.output_bias
        return out, hidden_act

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        logits = self.forward(x)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def get_saturation_stats(self, x: np.ndarray, threshold: float = 0.95) -> Dict:
        """Compute saturation statistics for hidden layer."""
        _, h = self.forward_with_activations(x)
        saturated = (np.abs(h) > threshold).astype(np.float32)

        return {
            'mean_saturation': saturated.mean(),
            'per_neuron_saturation': saturated.mean(axis=0),
            'gate_config': (saturated.mean(axis=0) > 0.5).astype(int),
            'input_indices': self.hidden_indices.copy()  # Track which inputs are used
        }

    def clone(self) -> 'GENREGNet':
        """Create a deep copy."""
        new = GENREGNet(self.input_dim, self.hidden_dim, self.output_dim, self.K)
        new.hidden_indices = self.hidden_indices.copy()
        new.hidden_weights = self.hidden_weights.copy()
        new.hidden_bias = self.hidden_bias.copy()
        new.output_weights = self.output_weights.copy()
        new.output_bias = self.output_bias.copy()
        return new

    def mutate(self, weight_rate: float = 0.2, weight_std: float = 0.1,
               index_swap_rate: float = 0.15):
        """Apply Gaussian mutation to weights AND potentially swap input indices.

        The index swaps are key - this is how GENREG discovers which inputs matter.
        This evolvability of connections (not just weights) is what differentiates
        GENREG from standard gradient-based sparse training.
        """
        # Weight mutations
        for i in range(self.hidden_dim):
            if np.random.random() < weight_rate:
                idx = np.random.randint(self.K)
                self.hidden_weights[i, idx] += np.random.randn() * weight_std

            if np.random.random() < weight_rate * 0.5:
                self.hidden_bias[i] += np.random.randn() * weight_std * 0.5

        # Output weight mutations
        if np.random.random() < weight_rate:
            i = np.random.randint(self.output_dim)
            j = np.random.randint(self.hidden_dim)
            self.output_weights[i, j] += np.random.randn() * weight_std

        # Index swaps - THE KEY MECHANISM for feature discovery
        # This is what creates selection pressure and tends to produce saturation
        for i in range(self.hidden_dim):
            if np.random.random() < index_swap_rate:
                old_idx = np.random.randint(self.K)
                # Find a new input not currently used by this neuron
                available = list(set(range(self.input_dim)) - set(self.hidden_indices[i]))
                if available:
                    new_input = np.random.choice(available)
                    self.hidden_indices[i, old_idx] = new_input
                    # Reset the weight for the new connection
                    self.hidden_weights[i, old_idx] = np.random.randn() * 0.3

    def num_params(self) -> int:
        """Count parameters (not including indices, which are structural)."""
        hidden_params = self.hidden_dim * self.K + self.hidden_dim  # weights + bias
        output_params = self.output_dim * self.hidden_dim + self.output_dim
        return hidden_params + output_params

    def get_connectivity_info(self) -> Dict:
        """Return info about sparse connectivity for analysis."""
        all_indices = self.hidden_indices.flatten()
        unique_inputs = len(np.unique(all_indices))
        return {
            'K': self.K,
            'total_connections': len(all_indices),
            'unique_inputs_used': unique_inputs,
            'input_coverage': unique_inputs / self.input_dim,
            'indices_per_neuron': self.hidden_indices.tolist()
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_sgd(
    model: SGDNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 500,
    lr: float = 0.01,
    batch_size: int = 64,
    target_acc: float = 0.95,
    verbose: bool = True
) -> Dict:
    """Train SGD network with Adam optimizer."""

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(len(X_train_t))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = perm[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

            train_logits = model(X_train_t)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == y_train_t).float().mean().item()

        if epoch % 50 == 0 and verbose:
            log(f"  SGD Epoch {epoch}: train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

        history.append({
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'loss': total_loss / n_batches
        })

        # Early stopping if target reached
        if val_acc >= target_acc:
            if verbose:
                log(f"  SGD reached target {target_acc:.0%} at epoch {epoch}")
            break

    return {
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'epochs': epoch + 1,
        'history': history
    }


def train_genreg(
    model: GENREGNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    generations: int = 500,
    pop_size: int = 50,
    sa_steps_per_member: int = 20,
    weight_rate: float = 0.2,
    weight_std: float = 0.1,
    index_swap_rate: float = 0.15,
    target_acc: float = 0.95,
    verbose: bool = True
) -> Dict:
    """Train GENREG network with GSA (Genetic Simulated Annealing).

    This matches the training used in extreme_sparsity.py:
    - Population-based with roulette selection
    - SA inner loop (multiple mutation attempts per member)
    - Elitism (top 5% preserved)
    - Temperature cooling for SA acceptance

    The SA inner loop is important - it allows each member to do local
    optimization before selection, which helps find good configurations.
    """

    # Fitness function: negative MSE (for SA acceptance)
    # Using one-hot encoding for smoother gradients
    n_classes = len(np.unique(y_train))
    y_onehot = np.zeros((len(y_train), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_train)), y_train] = 1
    y_onehot = y_onehot * 1.6 - 0.8  # Scale to [-0.8, 0.8] for tanh output

    def fitness(net: GENREGNet) -> float:
        logits = net.forward(X_train)
        preds = np.tanh(logits)  # Match output range
        return -np.mean((preds - y_onehot) ** 2)

    def accuracy(net: GENREGNet, X: np.ndarray, y: np.ndarray) -> float:
        preds = net.predict(X)
        return (preds == y).mean()

    # Initialize population
    population = []
    for _ in range(pop_size):
        ind = model.clone()
        ind.mutate(weight_rate=0.5, weight_std=0.3, index_swap_rate=0.3)
        population.append((ind, fitness(ind)))

    population[0] = (model.clone(), fitness(model))

    best_model = model.clone()
    best_fitness = fitness(best_model)

    # SA temperature schedule
    temp = 0.1
    decay = (0.0001 / 0.1) ** (1.0 / generations)

    history = []

    for gen in range(generations):
        # Sort by fitness (descending - higher is better)
        population.sort(key=lambda x: x[1], reverse=True)

        # Elitism: keep top 5%
        n_elite = max(1, pop_size // 20)
        new_pop = [(ind.clone(), f) for ind, f in population[:n_elite]]

        # Roulette wheel selection probabilities
        fitnesses = np.array([f for _, f in population])
        probs = fitnesses - fitnesses.min() + 1e-8
        probs = probs / probs.sum()

        # Generate rest of population
        for _ in range(pop_size - n_elite):
            # Roulette selection
            idx = np.random.choice(len(population), p=probs)
            c = population[idx][0].clone()
            current_f = population[idx][1]
            best_c, best_inner_f = c, current_f

            # SA inner loop - multiple mutation attempts
            for _ in range(sa_steps_per_member):
                mutant = c.clone()
                mutant.mutate(weight_rate=weight_rate, weight_std=weight_std,
                             index_swap_rate=index_swap_rate)
                f = fitness(mutant)

                # SA acceptance criterion
                delta = f - current_f
                if delta > 0 or np.random.random() < np.exp(delta / temp):
                    c = mutant
                    current_f = f
                    if f > best_inner_f:
                        best_c, best_inner_f = mutant.clone(), f

            new_pop.append((best_c, best_inner_f))

        population = new_pop

        # Track global best
        gen_best = max(population, key=lambda x: x[1])
        if gen_best[1] > best_fitness:
            best_model = gen_best[0].clone()
            best_fitness = gen_best[1]

        # Cool temperature
        temp *= decay

        # Evaluate
        val_acc = accuracy(best_model, X_val, y_val)
        train_acc = accuracy(best_model, X_train, y_train)

        if gen % 50 == 0 and verbose:
            sat_stats = best_model.get_saturation_stats(X_train)
            log(f"  GENREG Gen {gen}: train={train_acc:.1%}, val={val_acc:.1%}, sat={sat_stats['mean_saturation']:.1%}")

        history.append({
            'generation': gen,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_fitness': best_fitness
        })

        # Early stopping
        if val_acc >= target_acc:
            if verbose:
                log(f"  GENREG reached target {target_acc:.0%} at gen {gen}")
            break

    final_val_acc = accuracy(best_model, X_val, y_val)
    final_train_acc = accuracy(best_model, X_train, y_train)

    return {
        'model': best_model,
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'generations': gen + 1,
        'history': history
    }


# =============================================================================
# EVALUATION UNDER SHIFT
# =============================================================================

@dataclass
class EvalResult:
    """Results from evaluating a model under distribution shift."""
    shift_magnitude: float
    accuracy: float
    mean_confidence: float
    confidence_when_correct: float
    confidence_when_wrong: float
    mean_saturation: float
    gate_config: np.ndarray
    predictions: np.ndarray
    confidences: np.ndarray


def evaluate_under_shift(
    model,  # Either SGDNet or GENREGNet
    X_test: np.ndarray,
    y_test: np.ndarray,
    shift_config: ShiftConfig,
) -> List[EvalResult]:
    """Evaluate model across range of distribution shifts."""

    results = []
    magnitudes = np.linspace(0, shift_config.max_magnitude, shift_config.n_steps)

    for mag in magnitudes:
        # Apply shift
        X_shifted = apply_shift(X_test, shift_config.shift_type, mag)

        # Get predictions and confidence
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

        # Compute metrics
        confidences = probs.max(axis=1)
        correct_mask = (preds == y_test)
        accuracy = correct_mask.mean()

        results.append(EvalResult(
            shift_magnitude=mag,
            accuracy=accuracy,
            mean_confidence=confidences.mean(),
            confidence_when_correct=confidences[correct_mask].mean() if correct_mask.any() else 0,
            confidence_when_wrong=confidences[~correct_mask].mean() if (~correct_mask).any() else 0,
            mean_saturation=sat_stats['mean_saturation'],
            gate_config=sat_stats['gate_config'],
            predictions=preds,
            confidences=confidences
        ))

    return results


def compute_degradation_metrics(results: List[EvalResult]) -> Dict:
    """Compute metrics characterizing the degradation curve."""

    accuracies = np.array([r.accuracy for r in results])
    magnitudes = np.array([r.shift_magnitude for r in results])

    # Compute derivative (steepness of degradation)
    if len(accuracies) > 1:
        derivatives = np.gradient(accuracies, magnitudes)
    else:
        derivatives = np.array([0])

    # Find "cliff" points where accuracy drops sharply
    cliff_threshold = -0.1  # 10% drop per unit shift
    cliff_indices = np.where(derivatives < cliff_threshold)[0]

    # Compute area under accuracy curve (higher = more robust)
    auc = np.trapezoid(accuracies, magnitudes)

    # Find shift magnitude where accuracy drops below 50%
    below_50_idx = np.where(accuracies < 0.5)[0]
    shift_to_50 = magnitudes[below_50_idx[0]] if len(below_50_idx) > 0 else magnitudes[-1]

    return {
        'auc': auc,
        'max_derivative': derivatives.min(),  # Most negative = steepest drop
        'mean_derivative': derivatives.mean(),
        'n_cliff_points': len(cliff_indices),
        'shift_to_50_percent': shift_to_50,
        'final_accuracy': accuracies[-1]
    }


def track_gate_changes(results: List[EvalResult]) -> Dict:
    """Track how GENREG gate configurations change across shift levels."""

    configs = np.array([r.gate_config for r in results])

    # Count changes between consecutive shift levels
    changes = np.sum(configs[1:] != configs[:-1], axis=1)

    # Find "flip points" where multiple gates change
    flip_threshold = 2  # At least 2 gates flip
    flip_indices = np.where(changes >= flip_threshold)[0]

    return {
        'total_gate_changes': changes.sum(),
        'changes_per_step': changes.tolist(),
        'n_flip_points': len(flip_indices),
        'flip_magnitudes': [results[i+1].shift_magnitude for i in flip_indices],
        'initial_config': configs[0].tolist(),
        'final_config': configs[-1].tolist()
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_degradation_comparison(
    sgd_results: List[EvalResult],
    genreg_results: List[EvalResult],
    shift_type: str,
    save_path: Path
):
    """Create comparison plots of degradation curves."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    magnitudes = [r.shift_magnitude for r in sgd_results]

    # 1. Accuracy vs Shift
    ax = axes[0, 0]
    ax.plot(magnitudes, [r.accuracy for r in sgd_results], 'b-o', label='SGD', linewidth=2)
    ax.plot(magnitudes, [r.accuracy for r in genreg_results], 'r-s', label='GENREG', linewidth=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Degradation Under Covariate Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Confidence vs Shift
    ax = axes[0, 1]
    ax.plot(magnitudes, [r.mean_confidence for r in sgd_results], 'b-o', label='SGD', linewidth=2)
    ax.plot(magnitudes, [r.mean_confidence for r in genreg_results], 'r-s', label='GENREG', linewidth=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Mean Confidence')
    ax.set_title('Confidence Under Shift')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Saturation vs Shift (GENREG only)
    ax = axes[1, 0]
    ax.plot(magnitudes, [r.mean_saturation for r in genreg_results], 'r-s', linewidth=2)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Mean Saturation')
    ax.set_title('GENREG Saturation Under Shift')
    ax.grid(True, alpha=0.3)

    # 4. Accuracy derivative (degradation steepness)
    ax = axes[1, 1]
    sgd_acc = np.array([r.accuracy for r in sgd_results])
    genreg_acc = np.array([r.accuracy for r in genreg_results])
    mags = np.array(magnitudes)

    if len(mags) > 1:
        sgd_deriv = np.gradient(sgd_acc, mags)
        genreg_deriv = np.gradient(genreg_acc, mags)

        ax.plot(magnitudes, sgd_deriv, 'b-o', label='SGD', linewidth=2)
        ax.plot(magnitudes, genreg_deriv, 'r-s', label='GENREG', linewidth=2)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel(f'Shift Magnitude ({shift_type})')
    ax.set_ylabel('Accuracy Derivative')
    ax.set_title('Degradation Rate (more negative = steeper drop)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_gate_changes(
    results: List[EvalResult],
    save_path: Path
):
    """Visualize GENREG gate configuration changes."""

    configs = np.array([r.gate_config for r in results])
    magnitudes = [r.shift_magnitude for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Gate configuration heatmap
    ax = axes[0]
    im = ax.imshow(configs.T, aspect='auto', cmap='RdYlBu',
                   extent=[magnitudes[0], magnitudes[-1], 0, configs.shape[1]])
    ax.set_xlabel('Shift Magnitude')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Gate Configuration (1=saturated, 0=linear)')
    plt.colorbar(im, ax=ax)

    # 2. Number of gate changes per step
    ax = axes[1]
    changes = np.sum(configs[1:] != configs[:-1], axis=1)
    ax.bar(magnitudes[1:], changes, width=(magnitudes[1]-magnitudes[0])*0.8)
    ax.set_xlabel('Shift Magnitude')
    ax.set_ylabel('Gate Changes')
    ax.set_title('Gate Flips Between Shift Levels')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_and_decision_boundary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model,
    title: str,
    save_path: Path
):
    """Visualize 2D data and decision boundary."""

    fig, ax = plt.subplots(figsize=(8, 8))

    # Create grid for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    # Get predictions on grid
    if isinstance(model, SGDNet):
        model.eval()
        with torch.no_grad():
            Z = model(torch.tensor(grid)).argmax(dim=1).numpy()
    else:
        Z = model.predict(grid)

    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Plot data points
    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                         cmap='viridis', edgecolors='k', s=30, alpha=0.7)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Class')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    # Data
    data_config: DataConfig = field(default_factory=DataConfig)

    # Model
    hidden_dim: int = 32
    K: int = 4  # Inputs per hidden neuron (sparsity) - GENREG only

    # Training
    sgd_epochs: int = 500
    sgd_lr: float = 0.01
    genreg_generations: int = 500
    genreg_pop_size: int = 50
    genreg_sa_steps: int = 20  # SA steps per member
    target_accuracy: float = 0.95

    # Evaluation
    shift_types: List[str] = field(default_factory=lambda: ['translation', 'rotation', 'scaling'])
    shift_max_magnitude: float = 2.0
    shift_n_steps: int = 20

    # Output
    seed: int = 42


def run_experiment(config: ExperimentConfig):
    """Run full covariate shift comparison experiment."""

    # Clear previous log
    if LOG_FILE.exists():
        LOG_FILE.unlink()

    log("=" * 70)
    log("COVARIATE SHIFT EXPERIMENT: GENREG (sparse) vs SGD (dense)")
    log("=" * 70)
    log(f"\nConfig: H={config.hidden_dim}, K={config.K} (GENREG), target_acc={config.target_accuracy:.0%}")
    log(f"GENREG uses sparse connectivity ({config.K} inputs/neuron) + evolvable indices")
    log(f"SGD uses dense connectivity (all inputs) - standard baseline")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Generate data
    log("\n1. Generating synthetic data...")
    X, y = generate_gaussian_clusters(config.data_config)

    # Split into train/val/test
    n = len(X)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    log(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    input_dim = X_train.shape[1]
    output_dim = config.data_config.n_classes

    # Train SGD model
    log("\n2. Training SGD model...")
    sgd_model = SGDNet(input_dim, config.hidden_dim, output_dim)
    sgd_results = train_sgd(
        sgd_model, X_train, y_train, X_val, y_val,
        epochs=config.sgd_epochs,
        lr=config.sgd_lr,
        target_acc=config.target_accuracy
    )
    log(f"   Final: train={sgd_results['final_train_acc']:.1%}, val={sgd_results['final_val_acc']:.1%}")

    # Train GENREG model (sparse architecture)
    log(f"\n3. Training GENREG model (H={config.hidden_dim}, K={config.K})...")
    genreg_model = GENREGNet(input_dim, config.hidden_dim, output_dim, K=config.K)
    log(f"   Architecture: {genreg_model.num_params()} params, {config.K} inputs/neuron")
    genreg_results = train_genreg(
        genreg_model, X_train, y_train, X_val, y_val,
        generations=config.genreg_generations,
        pop_size=config.genreg_pop_size,
        sa_steps_per_member=config.genreg_sa_steps,
        target_acc=config.target_accuracy
    )
    genreg_model = genreg_results['model']
    sat_stats = genreg_model.get_saturation_stats(X_train)
    log(f"   Final: train={genreg_results['final_train_acc']:.1%}, val={genreg_results['final_val_acc']:.1%}")
    log(f"   Saturation: {sat_stats['mean_saturation']:.1%} of neurons saturated")

    # Skip decision boundary plots for high-dimensional data (>2D)
    # They only work for 2D visualization
    if config.data_config.n_dims == 2:
        log("\n4. Saving decision boundary visualizations...")
        plot_data_and_decision_boundary(
            X_train, y_train, sgd_model,
            f"SGD Model (acc={sgd_results['final_val_acc']:.1%})",
            LOG_DIR / "sgd_decision_boundary.png"
        )
        plot_data_and_decision_boundary(
            X_train, y_train, genreg_model,
            f"GENREG Model (acc={genreg_results['final_val_acc']:.1%})",
            LOG_DIR / "genreg_decision_boundary.png"
        )
    else:
        log(f"\n4. Skipping decision boundary visualization (data is {config.data_config.n_dims}D, not 2D)")

    # Evaluate under each shift type
    all_results = {}

    for shift_type in config.shift_types:
        log(f"\n5. Evaluating under {shift_type} shift...")

        shift_config = ShiftConfig(
            shift_type=shift_type,
            max_magnitude=config.shift_max_magnitude,
            n_steps=config.shift_n_steps
        )

        # Evaluate both models
        sgd_shift_results = evaluate_under_shift(sgd_model, X_test, y_test, shift_config)
        genreg_shift_results = evaluate_under_shift(genreg_model, X_test, y_test, shift_config)

        # Compute degradation metrics
        sgd_metrics = compute_degradation_metrics(sgd_shift_results)
        genreg_metrics = compute_degradation_metrics(genreg_shift_results)

        # Track GENREG gate changes
        gate_changes = track_gate_changes(genreg_shift_results)

        log(f"   SGD: AUC={sgd_metrics['auc']:.2f}, max_deriv={sgd_metrics['max_derivative']:.3f}")
        log(f"   GENREG: AUC={genreg_metrics['auc']:.2f}, max_deriv={genreg_metrics['max_derivative']:.3f}")
        log(f"   GENREG gate changes: {gate_changes['total_gate_changes']}, flip points: {gate_changes['n_flip_points']}")

        # Save plots
        plot_degradation_comparison(
            sgd_shift_results, genreg_shift_results, shift_type,
            LOG_DIR / f"degradation_{shift_type}.png"
        )
        plot_gate_changes(genreg_shift_results, LOG_DIR / f"gate_changes_{shift_type}.png")

        all_results[shift_type] = {
            'sgd_metrics': sgd_metrics,
            'genreg_metrics': genreg_metrics,
            'gate_changes': gate_changes,
            'sgd_accuracies': [r.accuracy for r in sgd_shift_results],
            'genreg_accuracies': [r.accuracy for r in genreg_shift_results],
            'magnitudes': [r.shift_magnitude for r in sgd_shift_results]
        }

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log(f"\n{'Shift Type':<15} {'SGD AUC':>10} {'GENREG AUC':>12} {'SGD MaxDeriv':>12} {'GENREG MaxDeriv':>15}")
    log("-" * 70)
    for shift_type in config.shift_types:
        r = all_results[shift_type]
        log(f"{shift_type:<15} {r['sgd_metrics']['auc']:>10.2f} {r['genreg_metrics']['auc']:>12.2f} "
            f"{r['sgd_metrics']['max_derivative']:>12.3f} {r['genreg_metrics']['max_derivative']:>15.3f}")

    log("\nHYPOTHESIS TEST:")
    log("- GENREG (sparse, saturated): Expected to show STEP-FUNCTION degradation")
    log("  → Binary gates flip suddenly when inputs cross decision boundaries")
    log("  → Should see: more negative max_derivative, distinct flip points")
    log("")
    log("- SGD (dense, continuous): Expected to show SMOOTH degradation")
    log("  → Continuous activations shift gradually")
    log("  → Should see: gentler slope, no sudden drops")
    log("")
    log("Key Metrics:")
    log("- max_derivative: More negative = steeper drops (step-like)")
    log("- n_flip_points: GENREG-specific - where gate configs change")

    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = LOG_DIR / f"results_{timestamp}.json"

    # Convert numpy arrays for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    with open(results_file, 'w') as f:
        json.dump(convert_for_json({
            'config': {
                'hidden_dim': config.hidden_dim,
                'K': config.K,
                'target_accuracy': config.target_accuracy,
                'shift_types': config.shift_types,
                'shift_max_magnitude': config.shift_max_magnitude
            },
            'training': {
                'sgd_final_acc': sgd_results['final_val_acc'],
                'sgd_params': sum(p.numel() for p in sgd_model.parameters()),
                'genreg_final_acc': genreg_results['final_val_acc'],
                'genreg_params': genreg_model.num_params(),
                'genreg_saturation': sat_stats['mean_saturation'],
                'genreg_connectivity': genreg_model.get_connectivity_info()
            },
            'shift_results': all_results
        }), f, indent=2)

    log(f"\nResults saved to: {results_file}")
    log(f"Plots saved to: {LOG_DIR}")


if __name__ == "__main__":
    config = ExperimentConfig(
        hidden_dim=32,
        K=4,  # Sparse connectivity - matches optimal from extreme_sparsity
        target_accuracy=0.90,  # Slightly lower target to ensure both reach it
        genreg_generations=300,
        genreg_pop_size=50,
        genreg_sa_steps=20,
        shift_types=['translation', 'rotation'],  # Most interpretable shifts
        shift_max_magnitude=3.0,  # Large enough to see full degradation
        shift_n_steps=25,  # Enough resolution to see step vs smooth
    )
    run_experiment(config)
