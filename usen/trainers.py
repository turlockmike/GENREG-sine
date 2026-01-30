"""
Training algorithms for USEN networks.

- train_gsa: Genetic Simulated Annealing (recommended)
- train_sa: Single-chain Simulated Annealing
- train_backprop: Gradient descent baseline
"""

import numpy as np
from typing import Optional, Callable, List, Tuple
from .networks import SparseNet


def train_gsa(
    net: SparseNet,
    X: np.ndarray,
    y: np.ndarray,
    generations: int = 300,
    pop_size: int = 50,
    sa_steps: int = 20,
    temp_init: float = 0.1,
    temp_final: float = 0.0001,
    weight_rate: float = 0.2,
    weight_std: float = 0.1,
    index_rate: float = 0.1,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, SparseNet, float], None]] = None,
    verbose: bool = False,
) -> Tuple[SparseNet, List[dict]]:
    """
    Train with Genetic Simulated Annealing.

    Combines population-based evolution with local SA refinement.
    This is the recommended training method for USEN.

    Args:
        net: Initial network (will be cloned for population)
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        generations: Number of generations
        pop_size: Population size
        sa_steps: SA steps per member per generation
        temp_init: Initial SA temperature
        temp_final: Final SA temperature
        weight_rate: Weight mutation rate
        weight_std: Weight mutation std
        index_rate: Index swap rate
        seed: Random seed
        callback: Called each generation with (gen, best_net, best_mse)
        verbose: Print progress

    Returns:
        Tuple of (best_network, history)
    """
    if seed is not None:
        np.random.seed(seed)

    def fitness(net: SparseNet) -> float:
        pred = net.predict(X)
        return -np.mean((pred - y) ** 2)

    # Initialize population
    pop = []
    for _ in range(pop_size):
        member = net.clone()
        member.mutate(weight_rate=0.5, weight_std=0.3, index_rate=0.3)
        pop.append((member, fitness(member)))

    best = pop[0][0].clone()
    best_f = fitness(best)

    # Temperature schedule
    temp = temp_init
    decay = (temp_final / temp_init) ** (1.0 / generations)

    history = []

    for gen in range(generations):
        # Sort by fitness (descending)
        pop.sort(key=lambda x: x[1], reverse=True)

        # Update best
        if pop[0][1] > best_f:
            best = pop[0][0].clone()
            best_f = pop[0][1]

        # Elite selection (top 5%)
        n_elite = max(1, pop_size // 20)
        new_pop = [(c.clone(), f) for c, f in pop[:n_elite]]

        # Roulette selection for rest
        probs = np.array([f for _, f in pop])
        probs = probs - probs.min() + 1e-8
        probs /= probs.sum()

        for _ in range(pop_size - n_elite):
            idx = np.random.choice(len(pop), p=probs)
            c = pop[idx][0].clone()
            curr_f = pop[idx][1]
            best_c, best_inner = c, curr_f

            # SA refinement
            for _ in range(sa_steps):
                m = c.clone()
                m.mutate(weight_rate=weight_rate, weight_std=weight_std, index_rate=index_rate)
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

        # Record history
        mse = -best_f
        record = {'gen': gen, 'mse': float(mse)}
        history.append(record)

        if callback:
            callback(gen, best, mse)

        if verbose and (gen % 50 == 0 or gen == generations - 1):
            print(f"  Gen {gen:3d}: MSE={mse:.6f}")

    return best, history


def train_sa(
    net: SparseNet,
    X: np.ndarray,
    y: np.ndarray,
    max_steps: int = 15000,
    temp_init: float = 1.0,
    temp_final: float = 0.001,
    weight_rate: float = 0.2,
    weight_std: float = 0.1,
    index_rate: float = 0.1,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, SparseNet, float], None]] = None,
    verbose: bool = False,
) -> Tuple[SparseNet, List[dict]]:
    """
    Train with single-chain Simulated Annealing.

    Note: GSA is generally preferred. Single SA can produce high saturation.

    Args:
        net: Initial network
        X: Input features
        y: Target values
        max_steps: Number of SA steps
        temp_init: Initial temperature
        temp_final: Final temperature
        weight_rate: Weight mutation rate
        weight_std: Weight mutation std
        index_rate: Index swap rate
        seed: Random seed
        callback: Called periodically with (step, best_net, best_mse)
        verbose: Print progress

    Returns:
        Tuple of (best_network, history)
    """
    if seed is not None:
        np.random.seed(seed)

    current = net.clone()
    current_mse = np.mean((current.predict(X) - y) ** 2)

    best = current.clone()
    best_mse = current_mse

    temp = temp_init
    decay = (temp_final / temp_init) ** (1.0 / max_steps)

    history = []

    for step in range(max_steps):
        candidate = current.clone()
        candidate.mutate(weight_rate=weight_rate, weight_std=weight_std, index_rate=index_rate)

        cand_mse = np.mean((candidate.predict(X) - y) ** 2)
        delta = cand_mse - current_mse

        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current = candidate
            current_mse = cand_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        temp *= decay

        if step % 3000 == 0 or step == max_steps - 1:
            history.append({'step': step, 'mse': float(best_mse)})

            if callback:
                callback(step, best, best_mse)

            if verbose:
                print(f"  Step {step:5d}: MSE={best_mse:.6f}")

    return best, history


def train_backprop(
    X: np.ndarray,
    y: np.ndarray,
    H: int = 8,
    K: Optional[int] = None,
    indices: Optional[np.ndarray] = None,
    epochs: int = 5000,
    lr: float = 0.01,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[SparseNet, List[dict]]:
    """
    Train sparse network with backpropagation (PyTorch).

    Used as a baseline for comparison. Note that backprop cannot
    discover which indices matter - they must be provided or random.

    Args:
        X: Input features
        y: Target values
        H: Hidden size
        K: Inputs per neuron (if None, uses dense network)
        indices: Pre-specified indices (if None, uses random)
        epochs: Training epochs
        lr: Learning rate
        seed: Random seed
        verbose: Print progress

    Returns:
        Tuple of (trained_network, history)
    """
    import torch
    import torch.nn as nn

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    input_dim = X.shape[1]

    # Create index pattern
    if K is None:
        K = input_dim  # Dense

    if indices is None:
        indices = np.array([
            np.random.choice(input_dim, min(K, input_dim), replace=False)
            for _ in range(H)
        ])

    # Build PyTorch model
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

    history = []
    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred, _ = model(X_t)
        loss = ((pred - y_t) ** 2).mean()
        loss.backward()
        optimizer.step()

        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 1000 == 0 or epoch == epochs - 1:
            history.append({'epoch': epoch, 'mse': float(mse)})
            if verbose:
                print(f"  Epoch {epoch:5d}: MSE={mse:.6f}")

    # Convert to SparseNet
    model.load_state_dict(best_state)
    result = SparseNet(input_dim, H, K)
    result.indices = indices
    for i in range(H):
        result.W1[i] = model.W1[i].detach().numpy()
    result.b1 = model.b1.detach().numpy()
    result.W2 = model.W2.detach().numpy()
    result.b2 = model.b2.detach().numpy()

    return result, history
