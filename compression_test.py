"""
Compression comparison: How small can each method go?
Tests backprop vs simulated annealing at different hidden sizes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

import sine_config as cfg
from sine_controller import SineController, expand_input

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackpropMLP(nn.Module):
    """Standard MLP for backprop."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)
    
    def get_saturation(self, x):
        """Check saturation of hidden layer."""
        with torch.no_grad():
            h = torch.tanh(self.fc1(x))
            saturated = (h.abs() > 0.95).float().mean().item()
        return saturated


def run_backprop(hidden_size, noise_dims=240, max_epochs=500, lr=0.01):
    """Run backprop training."""
    input_size = 16 + noise_dims
    
    # Temporarily override config
    old_noise = cfg.NOISE_SIGNAL_SIZE
    old_expansion = cfg.EXPANSION_SIZE
    cfg.NOISE_SIGNAL_SIZE = noise_dims
    cfg.EXPANSION_SIZE = 16 + noise_dims
    
    # Data
    x_raw = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE).unsqueeze(-1)
    x = expand_input(x_raw)
    y = torch.sin(x_raw.squeeze())
    
    # Model
    model = BackpropMLP(input_size, hidden_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    start = time.time()
    best_mse = float('inf')
    patience = 50
    no_improve = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        mse = loss.item()
        if mse < best_mse:
            best_mse = mse
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience or mse < 0.0001:
            break
    
    elapsed = time.time() - start
    saturation = model.get_saturation(x)
    
    # Restore config
    cfg.NOISE_SIGNAL_SIZE = old_noise
    cfg.EXPANSION_SIZE = old_expansion
    
    return best_mse, saturation, elapsed, epoch + 1


def run_annealing(hidden_size, noise_dims=240, max_steps=10000):
    """Run simulated annealing."""
    old_noise = cfg.NOISE_SIGNAL_SIZE
    old_expansion = cfg.EXPANSION_SIZE
    old_hidden = cfg.HIDDEN_SIZE
    
    cfg.NOISE_SIGNAL_SIZE = noise_dims
    cfg.EXPANSION_SIZE = 16 + noise_dims
    cfg.HIDDEN_SIZE = hidden_size
    
    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)
    
    def evaluate(ctrl):
        with torch.no_grad():
            pred = ctrl.forward(x_test, track=True)
            return -torch.mean((pred - y_true) ** 2).item()
    
    # Annealing
    current = SineController(device=DEVICE)
    current_fitness = evaluate(current)
    best = current.clone()
    best_fitness = current_fitness
    
    t_initial, t_final = 0.01, 0.0001
    decay = (t_final / t_initial) ** (1.0 / max_steps)
    
    start = time.time()
    
    for step in range(max_steps):
        temperature = t_initial * (decay ** step)
        
        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)
        mutant_fitness = evaluate(mutant)
        
        delta = mutant_fitness - current_fitness
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_fitness = mutant_fitness
            if current_fitness > best_fitness:
                best = current.clone()
                best_fitness = current_fitness
    
    elapsed = time.time() - start
    k_ratio = best.get_k() / hidden_size
    
    # Restore config
    cfg.NOISE_SIGNAL_SIZE = old_noise
    cfg.EXPANSION_SIZE = old_expansion
    cfg.HIDDEN_SIZE = old_hidden
    
    return -best_fitness, k_ratio, elapsed


def main():
    print("=" * 70)
    print("COMPRESSION TEST: Backprop vs Simulated Annealing")
    print("=" * 70)
    print(f"Noise dimensions: 240")
    print(f"Testing hidden sizes: 8, 6, 4, 3, 2")
    print("=" * 70)
    
    hidden_sizes = [8, 6, 4, 3, 2]
    
    print(f"\n{'Hidden':>8} | {'Backprop MSE':>12} {'k%':>6} {'Time':>6} | {'Annealing MSE':>13} {'k%':>6} {'Time':>6}")
    print("-" * 70)
    
    for h in hidden_sizes:
        bp_mse, bp_sat, bp_time, bp_epochs = run_backprop(h)
        sa_mse, sa_sat, sa_time = run_annealing(h)
        
        print(f"{h:>8} | {bp_mse:>12.6f} {bp_sat*100:>5.0f}% {bp_time:>5.1f}s | {sa_mse:>13.6f} {sa_sat*100:>5.0f}% {sa_time:>5.1f}s")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
