"""
Experiment: True Noise vs Leaky Noise

The original "noise" signals contain information correlated with sin(x):
- -sin(x*1.0) has r=-1.0 (perfect anti-correlation!)
- sign(sin(x*1.05)) has r=+0.87
- Several frequencies near 1.0 have |r|>0.5

This experiment compares:
1. Original "leaky" noise - contains correlated signals
2. True noise - signals guaranteed uncorrelated with sin(x)

True noise options:
- Random values (no structure)
- Functions of x that are orthogonal to sin(x) on [-2π, 2π]
- Frequencies that are NOT near integer multiples of 1.0
"""

import torch
import numpy as np
import json
from pathlib import Path

import sine_config as cfg

DEVICE = torch.device("cpu")


def expand_input_true_noise(x: torch.Tensor, noise_type='random') -> torch.Tensor:
    """
    Expand input with TRUE noise that doesn't correlate with sin(x).

    Args:
        x: Input tensor of shape (batch, 1)
        noise_type: 'random', 'orthogonal', or 'safe_freq'
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    x_norm = x / (2 * np.pi)

    # TRUE SIGNALS (16) - same as original
    true_signal = torch.zeros(batch_size, 16, device=device, dtype=dtype)
    true_signal[:, 0:1] = x_norm
    true_signal[:, 1:2] = x_norm ** 2
    true_signal[:, 2:3] = x_norm ** 3

    freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=device, dtype=dtype)
    freq_x = x * freqs
    true_signal[:, 3:9] = torch.sin(freq_x)
    true_signal[:, 9:15] = torch.cos(freq_x)

    # NOISE SIGNALS (240) - truly uncorrelated
    noise_signal = torch.zeros(batch_size, 240, device=device, dtype=dtype)

    if noise_type == 'random':
        # Pure random noise - no structure, no correlation
        # Use seeded generator for reproducibility
        gen = torch.Generator(device=device)
        gen.manual_seed(99999)
        noise_signal = torch.randn(batch_size, 240, device=device, dtype=dtype, generator=gen) * 0.5

    elif noise_type == 'safe_freq':
        # Use frequencies that are FAR from integers (won't correlate with sin(x))
        # Avoid: 0.5, 1.0, 1.5, 2.0, ... (anything that could beat with sin(x))
        # Use: irrational multiples like π, e, √2, etc.

        col = 0

        # Irrational frequency multipliers - won't resonate with sin(x)
        irrational_bases = [np.pi, np.e, np.sqrt(2), np.sqrt(3), np.sqrt(5),
                          np.pi/np.e, np.e/np.pi, np.sqrt(7), np.sqrt(11)]

        # Sin at irrational frequencies (80 signals)
        for i in range(80):
            base = irrational_bases[i % len(irrational_bases)]
            freq = base * (0.1 + 0.1 * (i // len(irrational_bases)))
            phase = (i * 0.7) % (2 * np.pi)  # Deterministic phase
            noise_signal[:, col] = torch.sin(x.squeeze() * freq + phase)
            col += 1

        # Cos at irrational frequencies (80 signals)
        for i in range(80):
            base = irrational_bases[i % len(irrational_bases)]
            freq = base * (0.1 + 0.1 * (i // len(irrational_bases)))
            phase = (i * 0.3) % (2 * np.pi)
            noise_signal[:, col] = torch.cos(x.squeeze() * freq + phase)
            col += 1

        # Even-power polynomials (orthogonal to sin on symmetric interval) (40 signals)
        for i in range(40):
            power = 4 + 2 * i  # x^4, x^6, x^8, ... (even powers)
            noise_signal[:, col] = x_norm.squeeze() ** power
            col += 1

        # Remaining: structured but uncorrelated
        remaining = 240 - col
        for i in range(remaining):
            # Products of irrational-frequency trig functions
            f1 = irrational_bases[i % len(irrational_bases)]
            f2 = irrational_bases[(i + 3) % len(irrational_bases)]
            noise_signal[:, col] = torch.sin(x.squeeze() * f1) * torch.sin(x.squeeze() * f2)
            col += 1

    elif noise_type == 'orthogonal':
        # Use functions mathematically orthogonal to sin(x) on [-2π, 2π]
        # cos(nx) is orthogonal to sin(x) for n ≠ 1
        # sin(nx) is orthogonal to sin(x) for n ≠ 1

        col = 0

        # Cosines (orthogonal to sin(x) by symmetry) - 60 signals
        for n in range(60):
            freq = 0.1 * (n + 1)  # 0.1, 0.2, 0.3, ... but skip 1.0
            if abs(freq - 1.0) < 0.05:
                freq = 1.07  # Skip near 1.0
            noise_signal[:, col] = torch.cos(x.squeeze() * freq)
            col += 1

        # Even functions (orthogonal to odd sin(x)) - 60 signals
        for n in range(60):
            power = 2 * (n + 1)  # x^2, x^4, x^6, ...
            noise_signal[:, col] = (x_norm.squeeze() ** power) * torch.cos(x.squeeze() * 0.1 * n)
            col += 1

        # High frequency sines (will average to ~0 correlation) - 60 signals
        for n in range(60):
            freq = 10 + n * 0.5  # 10, 10.5, 11, ... (high freq = low correlation)
            noise_signal[:, col] = torch.sin(x.squeeze() * freq)
            col += 1

        # Remaining: random
        remaining = 240 - col
        gen = torch.Generator(device=device)
        gen.manual_seed(88888)
        noise_signal[:, col:] = torch.randn(batch_size, remaining, device=device, dtype=dtype, generator=gen) * 0.5

    return torch.cat([true_signal, noise_signal], dim=-1)


class TrueNoiseController:
    """Controller with configurable noise type."""

    def __init__(self, noise_type='random', device=None):
        self.device = device or DEVICE
        self.noise_type = noise_type

        input_size = 256
        hidden_size = 8

        scale_w1 = np.sqrt(2.0 / input_size)
        scale_w2 = np.sqrt(2.0 / hidden_size)

        self.w1 = torch.randn(hidden_size, input_size, device=self.device) * scale_w1
        self.b1 = torch.zeros(hidden_size, device=self.device)
        self.w2 = torch.randn(1, hidden_size, device=self.device) * scale_w2
        self.b2 = torch.zeros(1, device=self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        # Use true noise expansion
        x = expand_input_true_noise(x, self.noise_type)

        hidden = torch.tanh(torch.nn.functional.linear(x, self.w1, self.b1))
        output = torch.tanh(torch.nn.functional.linear(hidden, self.w2, self.b2))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def mutate(self, rate=0.1, scale=0.1):
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * scale
            param.data += mask.float() * noise

    def clone(self):
        new = TrueNoiseController(noise_type=self.noise_type, device=self.device)
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        return new


def verify_noise_correlation(noise_type):
    """Verify that noise signals don't correlate with sin(x)."""
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 100).unsqueeze(-1)
    y_true = torch.sin(x.squeeze())

    expanded = expand_input_true_noise(x, noise_type)

    # Check correlation of each noise signal
    correlations = []
    for i in range(16, 256):  # Skip true signals
        signal = expanded[:, i]
        corr = np.corrcoef(signal.numpy(), y_true.numpy())[0, 1]
        correlations.append(abs(corr))

    correlations = np.array(correlations)
    return {
        'max_correlation': float(np.nanmax(correlations)),
        'mean_correlation': float(np.nanmean(correlations)),
        'signals_above_0.3': int(np.sum(correlations > 0.3)),
        'signals_above_0.5': int(np.sum(correlations > 0.5)),
        'signals_above_0.7': int(np.sum(correlations > 0.7)),
    }


def train_sa(controller, x_test, y_true, max_steps=15000):
    """Train using simulated annealing."""
    current = controller
    with torch.no_grad():
        pred = current.forward(x_test)
        current_mse = torch.mean((pred - y_true) ** 2).item()

    best = current.clone()
    best_mse = current_mse

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / max_steps)

    for step in range(max_steps):
        temperature = t_initial * (decay ** step)

        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)

        with torch.no_grad():
            pred = mutant.forward(x_test)
            mutant_mse = torch.mean((pred - y_true) ** 2).item()

        # SA acceptance (minimizing MSE)
        delta = current_mse - mutant_mse  # positive if mutant is better
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse

            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        if step % 5000 == 0:
            print(f"    Step {step}: MSE = {best_mse:.6f}")

    return best, best_mse


def run_experiment():
    """Compare leaky noise vs true noise."""
    print("=" * 70)
    print("EXPERIMENT: TRUE NOISE VS LEAKY NOISE")
    print("=" * 70)

    torch.manual_seed(42)
    np.random.seed(42)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    results = {'noise_types': {}}

    # First verify noise correlations
    print("\n" + "=" * 70)
    print("NOISE CORRELATION VERIFICATION")
    print("=" * 70)

    noise_types = ['random', 'safe_freq', 'orthogonal']

    # Also check original noise
    from sine_controller import expand_input
    x_expanded_orig = expand_input(x_test.unsqueeze(-1))
    orig_correlations = []
    for i in range(16, 256):
        signal = x_expanded_orig[:, i]
        corr = np.corrcoef(signal.numpy(), y_true.numpy())[0, 1]
        orig_correlations.append(abs(corr))
    orig_correlations = np.array(orig_correlations)

    print(f"\n{'Noise Type':<15} | {'Max |r|':<10} | {'Mean |r|':<10} | {'|r|>0.3':<8} | {'|r|>0.5':<8} | {'|r|>0.7':<8}")
    print("-" * 75)
    print(f"{'ORIGINAL':<15} | {np.nanmax(orig_correlations):<10.4f} | {np.nanmean(orig_correlations):<10.4f} | {np.sum(orig_correlations > 0.3):<8} | {np.sum(orig_correlations > 0.5):<8} | {np.sum(orig_correlations > 0.7):<8}")

    for noise_type in noise_types:
        stats = verify_noise_correlation(noise_type)
        print(f"{noise_type:<15} | {stats['max_correlation']:<10.4f} | {stats['mean_correlation']:<10.4f} | {stats['signals_above_0.3']:<8} | {stats['signals_above_0.5']:<8} | {stats['signals_above_0.7']:<8}")
        results['noise_types'][noise_type] = {'correlation_stats': stats}

    # Train with each noise type
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON (15k steps SA)")
    print("=" * 70)

    # Original (leaky) noise
    print("\n[Training with ORIGINAL (leaky) noise]")
    torch.manual_seed(42)
    np.random.seed(42)
    from sine_controller import SineController
    orig_controller = SineController(device=DEVICE)

    # Train original
    current = orig_controller
    with torch.no_grad():
        pred = current.forward(x_test)
        current_mse = torch.mean((pred - y_true) ** 2).item()
    best_orig = current.clone()
    best_orig_mse = current_mse

    t_initial, t_final = 0.01, 0.00001
    decay = (t_final / t_initial) ** (1.0 / 15000)

    for step in range(15000):
        temperature = t_initial * (decay ** step)
        mutant = current.clone()
        mutant.mutate(rate=0.1, scale=0.1)
        with torch.no_grad():
            pred = mutant.forward(x_test)
            mutant_mse = torch.mean((pred - y_true) ** 2).item()
        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_orig_mse:
                best_orig = current.clone()
                best_orig_mse = current_mse
        if step % 5000 == 0:
            print(f"    Step {step}: MSE = {best_orig_mse:.6f}")

    results['noise_types']['original'] = {'mse': best_orig_mse}

    # Train with each true noise type
    for noise_type in noise_types:
        print(f"\n[Training with {noise_type.upper()} noise]")
        torch.manual_seed(42)
        np.random.seed(42)

        controller = TrueNoiseController(noise_type=noise_type, device=DEVICE)
        best, best_mse = train_sa(controller, x_test, y_true, max_steps=15000)

        results['noise_types'][noise_type]['mse'] = best_mse

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Noise Type':<15} | {'Final MSE':<12} | {'vs Original':<15}")
    print("-" * 50)

    orig_mse = results['noise_types']['original']['mse']
    print(f"{'ORIGINAL':<15} | {orig_mse:<12.6f} | {'(baseline)':<15}")

    for noise_type in noise_types:
        mse = results['noise_types'][noise_type]['mse']
        ratio = mse / orig_mse
        print(f"{noise_type:<15} | {mse:<12.6f} | {ratio:.2f}x {'worse' if ratio > 1 else 'better'}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if all(results['noise_types'][nt]['mse'] > orig_mse * 1.5 for nt in noise_types):
        print("""
CONFIRMED: The network was exploiting correlated noise signals!

With truly uncorrelated noise, performance degrades significantly.
This proves the original "noise" was leaking information about sin(x).

The network found it easier to:
  - Use -sin(x*1.0) directly (flip the sign)
  - Use sign(sin(x*1.05)) (square wave approximation)

Than to:
  - Learn the proper Fourier reconstruction from true signals
""")
    else:
        print("""
Interesting: Performance with true noise is comparable to original.
The network may be successfully learning to filter noise and use true signals.
""")

    # Save results
    output_dir = Path("results/true_noise_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
