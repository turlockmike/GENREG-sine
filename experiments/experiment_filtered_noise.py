"""
Experiment: Filtered Noise - Removing Correlated Signals

Tests what happens when we remove the correlated "noise" signals that
were leaking information about sin(x).

Remove levels:
- 'none': Keep all noise (original, baseline)
- 'perfect': Remove only -sin(x*1.0) which has r=-1.0
- 'high': Remove all signals with |r| > 0.7
- 'medium': Remove all signals with |r| > 0.3
"""

import torch
import numpy as np
import json
from pathlib import Path

DEVICE = torch.device("cpu")

# Original noise setup (same seeds as sine_controller.py)
_noise_rng = np.random.RandomState(12345)
_noise_freqs = _noise_rng.uniform(0.1, 10.0, size=100)
_noise_phases = _noise_rng.uniform(0, 2 * np.pi, size=100)
_noise_scales = _noise_rng.uniform(0.5, 2.0, size=100)


def expand_input_filtered(x, remove_level='none'):
    """
    Expand input, optionally removing correlated noise signals.

    Args:
        x: Input tensor
        remove_level: 'none', 'perfect', 'high', or 'medium'

    Returns:
        expanded: Expanded tensor
        noise_count: Number of noise signals kept
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype
    x_norm = x / (2 * np.pi)

    # TRUE SIGNALS (16) - always included
    true_signal = torch.zeros(batch_size, 16, device=device, dtype=dtype)
    true_signal[:, 0:1] = x_norm
    true_signal[:, 1:2] = x_norm ** 2
    true_signal[:, 2:3] = x_norm ** 3

    freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=device, dtype=dtype)
    freq_x = x * freqs
    true_signal[:, 3:9] = torch.sin(freq_x)
    true_signal[:, 9:15] = torch.cos(freq_x)

    # Precompute y_true for correlation check
    y_true = torch.sin(x.squeeze())

    # Correlation threshold based on remove level
    if remove_level == 'none':
        threshold = 1.1  # Keep everything
    elif remove_level == 'perfect':
        threshold = 0.99  # Only remove r=-1.0
    elif remove_level == 'high':
        threshold = 0.7
    elif remove_level == 'medium':
        threshold = 0.3
    else:
        threshold = 1.1

    def should_keep(signal):
        """Check if signal correlation is below threshold."""
        if signal.std() < 1e-6:  # Constant signal
            return True
        corr = np.corrcoef(signal.numpy(), y_true.numpy())[0, 1]
        if np.isnan(corr):
            return True
        return abs(corr) < threshold

    noise_signals = []

    # 1. Wrong frequencies (50 signals)
    for i in range(50):
        signal = torch.sin(x.squeeze() * _noise_freqs[i] + _noise_phases[i])
        if should_keep(signal):
            noise_signals.append(signal)

    # 2. Cosines at wrong frequencies (50 signals)
    for i in range(50):
        idx = min(50 + i, 99)
        signal = torch.cos(x.squeeze() * _noise_freqs[idx] + _noise_phases[idx])
        if should_keep(signal):
            noise_signals.append(signal)

    # 3. Inverted signals (20 signals) - THIS IS WHERE -sin(x) LIVES
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        signal = -torch.sin(x.squeeze() * k)
        if should_keep(signal):
            noise_signals.append(signal)
    for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        signal = -torch.cos(x.squeeze() * k)
        if should_keep(signal):
            noise_signals.append(signal)

    # 4. Polynomial garbage (20 signals)
    for power in range(4, 24):
        signal = x_norm.squeeze() ** power
        if should_keep(signal):
            noise_signals.append(signal)

    # 5. Mixed/product terms (20 signals)
    for i in range(20):
        signal = torch.sin(x.squeeze() * _noise_freqs[i]) * torch.cos(x.squeeze() * _noise_freqs[20 + i])
        if should_keep(signal):
            noise_signals.append(signal)

    # 6. Tanh distortions (20 signals)
    for i in range(20):
        signal = torch.tanh(x.squeeze() * _noise_scales[i])
        if should_keep(signal):
            noise_signals.append(signal)

    # 7. Exponential-based (20 signals)
    for i in range(20):
        signal = torch.exp(-torch.abs(x.squeeze() * _noise_scales[min(20 + i, 99)] * 0.3)) - 0.5
        if should_keep(signal):
            noise_signals.append(signal)

    # 8. Square wave (20 signals)
    for i in range(20):
        signal = torch.sign(torch.sin(x.squeeze() * _noise_freqs[min(40 + i, 99)]))
        if should_keep(signal):
            noise_signals.append(signal)

    # 9. Sawtooth (20 signals)
    for i in range(20):
        signal = torch.fmod(x.squeeze() * _noise_freqs[min(60 + i, 99)] + _noise_phases[min(60 + i, 99)], 2.0) - 1.0
        if should_keep(signal):
            noise_signals.append(signal)

    # Stack noise signals
    if noise_signals:
        noise_tensor = torch.stack(noise_signals, dim=1)
    else:
        noise_tensor = torch.zeros(batch_size, 0, device=device, dtype=dtype)

    return torch.cat([true_signal, noise_tensor], dim=-1), len(noise_signals)


class FilteredNoiseController:
    """Controller with configurable noise filtering."""

    def __init__(self, remove_level='none', device=None, input_size=None, noise_count=None):
        self.device = device or DEVICE
        self.remove_level = remove_level

        # Determine input size by doing a test expansion
        if input_size is None:
            x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100).unsqueeze(-1)
            expanded, noise_count = expand_input_filtered(x_test, remove_level)
            self.input_size = expanded.shape[1]
            self.noise_count = noise_count
        else:
            self.input_size = input_size
            self.noise_count = noise_count

        hidden_size = 8
        scale_w1 = np.sqrt(2.0 / self.input_size)
        scale_w2 = np.sqrt(2.0 / hidden_size)

        self.w1 = torch.randn(hidden_size, self.input_size, device=self.device) * scale_w1
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

        expanded, _ = expand_input_filtered(x, self.remove_level)

        hidden = torch.tanh(torch.nn.functional.linear(expanded, self.w1, self.b1))
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
        new = FilteredNoiseController(
            remove_level=self.remove_level,
            device=self.device,
            input_size=self.input_size,
            noise_count=self.noise_count
        )
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        return new

    def num_parameters(self):
        return self.input_size * 8 + 8 + 8 + 1


def train_sa(controller, x_test, y_true, max_steps=15000, verbose=True):
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

        delta = current_mse - mutant_mse
        if delta > 0 or np.random.random() < np.exp(delta / temperature):
            current = mutant
            current_mse = mutant_mse
            if current_mse < best_mse:
                best = current.clone()
                best_mse = current_mse

        if verbose and step % 5000 == 0:
            print(f"    Step {step}: MSE = {best_mse:.6f}")

    return best, best_mse


def run_experiment():
    """Run the filtered noise experiment."""
    print("=" * 70)
    print("EXPERIMENT: Removing Correlated Noise Signals")
    print("=" * 70)
    print("Testing what happens when we remove -sin(x) and other correlated signals")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    levels = ['none', 'perfect', 'high', 'medium']
    level_descriptions = {
        'none': 'Nothing (baseline)',
        'perfect': 'Only -sin(x) (r=1.0)',
        'high': 'All |r| > 0.7 (4 signals)',
        'medium': 'All |r| > 0.3 (15 signals)'
    }

    results = {'levels': {}}

    for level in levels:
        print(f"\n[Remove level: {level.upper()} - {level_descriptions[level]}]")
        torch.manual_seed(42)
        np.random.seed(42)

        controller = FilteredNoiseController(remove_level=level, device=DEVICE)
        print(f"    Input size: {controller.input_size} (16 true + {controller.noise_count} noise)")
        print(f"    Parameters: {controller.num_parameters()}")

        best, best_mse = train_sa(controller, x_test, y_true, max_steps=15000)

        results['levels'][level] = {
            'description': level_descriptions[level],
            'mse': best_mse,
            'input_size': controller.input_size,
            'noise_count': controller.noise_count,
            'params': controller.num_parameters()
        }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Level':<10} | {'Removed':<25} | {'Inputs':<8} | {'MSE':<12} | {'vs Baseline'}")
    print("-" * 80)

    baseline_mse = results['levels']['none']['mse']
    for level in levels:
        r = results['levels'][level]
        ratio = r['mse'] / baseline_mse
        status = "baseline" if level == 'none' else f"{ratio:.2f}x {'worse' if ratio > 1 else 'better'}"
        print(f"{level:<10} | {r['description']:<25} | {r['input_size']:<8} | {r['mse']:<12.6f} | {status}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    perfect_mse = results['levels']['perfect']['mse']
    if perfect_mse > baseline_mse * 1.2:
        print(f"""
Removing just -sin(x) increased MSE by {perfect_mse/baseline_mse:.1f}x

This confirms the network was using -sin(x) as a shortcut!
Without this "cheat code", it must actually learn from the Fourier basis.
""")
    else:
        print(f"""
Removing -sin(x) had minimal impact ({perfect_mse/baseline_mse:.2f}x).
The network found other correlated signals to exploit.
""")

    # Save results
    output_dir = Path("results/filtered_noise_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
