"""
Experiment: Clean Noise - True Signals + Uncorrelated Noise

Tests whether the network can learn to filter out truly uncorrelated noise
when there are no "cheat codes" hidden in the noise.

Configurations:
- 16 true signals only (baseline)
- 16 true + 16 uncorrelated noise
- 16 true + 32 uncorrelated noise
- 16 true + 64 uncorrelated noise
- 16 true + 240 uncorrelated noise (same size as original)
"""

import torch
import numpy as np
import json
from pathlib import Path

DEVICE = torch.device("cpu")


def expand_input_clean(x, noise_count=0, noise_seed=99999):
    """
    Expand input with true signals + truly uncorrelated noise.

    The noise is deterministic (seeded) but uncorrelated with sin(x).
    Uses high-frequency sinusoids that average to ~0 correlation.
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype
    x_norm = x / (2 * np.pi)

    # TRUE SIGNALS (16)
    true_signal = torch.zeros(batch_size, 16, device=device, dtype=dtype)
    true_signal[:, 0:1] = x_norm
    true_signal[:, 1:2] = x_norm ** 2
    true_signal[:, 2:3] = x_norm ** 3

    freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=device, dtype=dtype)
    freq_x = x * freqs
    true_signal[:, 3:9] = torch.sin(freq_x)
    true_signal[:, 9:15] = torch.cos(freq_x)

    if noise_count == 0:
        return true_signal

    # UNCORRELATED NOISE
    # Use high frequencies (10+) and irrational multipliers to ensure no correlation
    rng = np.random.RandomState(noise_seed)

    noise_signal = torch.zeros(batch_size, noise_count, device=device, dtype=dtype)

    for i in range(noise_count):
        # High frequency base (10-50) ensures rapid oscillation = ~0 correlation
        freq = 10 + rng.uniform(0, 40)
        phase = rng.uniform(0, 2 * np.pi)

        # Alternate between sin and cos
        if i % 2 == 0:
            noise_signal[:, i] = torch.sin(x.squeeze() * freq + phase)
        else:
            noise_signal[:, i] = torch.cos(x.squeeze() * freq + phase)

    return torch.cat([true_signal, noise_signal], dim=-1)


def verify_noise_correlation(noise_count):
    """Verify that noise signals don't correlate with sin(x)."""
    x = torch.linspace(-2 * np.pi, 2 * np.pi, 100)
    y_true = torch.sin(x)

    expanded = expand_input_clean(x, noise_count)

    correlations = []
    for i in range(16, 16 + noise_count):
        signal = expanded[:, i]
        corr = np.corrcoef(signal.numpy(), y_true.numpy())[0, 1]
        correlations.append(abs(corr))

    if correlations:
        return {
            'max': max(correlations),
            'mean': np.mean(correlations),
            'above_0.1': sum(1 for c in correlations if c > 0.1)
        }
    return {'max': 0, 'mean': 0, 'above_0.1': 0}


class CleanNoiseController:
    """Controller with true signals + clean uncorrelated noise."""

    def __init__(self, noise_count=0, device=None):
        self.device = device or DEVICE
        self.noise_count = noise_count
        self.input_size = 16 + noise_count

        hidden_size = 8
        scale_w1 = np.sqrt(2.0 / self.input_size)
        scale_w2 = np.sqrt(2.0 / hidden_size)

        self.w1 = torch.randn(hidden_size, self.input_size, device=self.device) * scale_w1
        self.b1 = torch.zeros(hidden_size, device=self.device)
        self.w2 = torch.randn(1, hidden_size, device=self.device) * scale_w2
        self.b2 = torch.zeros(1, device=self.device)
        self.last_hidden = None

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        expanded = expand_input_clean(x, self.noise_count)

        hidden = torch.tanh(torch.nn.functional.linear(expanded, self.w1, self.b1))
        self.last_hidden = hidden.detach()
        output = torch.tanh(torch.nn.functional.linear(hidden, self.w2, self.b2))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def get_saturation(self):
        if self.last_hidden is None:
            return 0.0
        abs_act = self.last_hidden.abs()
        saturated = (abs_act > 0.95).float()
        return saturated.mean().item()

    def mutate(self, rate=0.1, scale=0.1):
        for param in [self.w1, self.b1, self.w2, self.b2]:
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * scale
            param.data += mask.float() * noise

    def clone(self):
        new = CleanNoiseController(noise_count=self.noise_count, device=self.device)
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        return new

    def num_parameters(self):
        return self.input_size * 8 + 8 + 8 + 1

    def analyze_weight_distribution(self):
        """Analyze how weights are distributed between true and noise inputs."""
        w1 = self.w1.abs().cpu().numpy()

        true_weights = w1[:, :16].sum()
        noise_weights = w1[:, 16:].sum() if self.noise_count > 0 else 0
        total = true_weights + noise_weights

        return {
            'true_weight_pct': true_weights / total * 100 if total > 0 else 100,
            'noise_weight_pct': noise_weights / total * 100 if total > 0 else 0,
            'true_weight_avg': w1[:, :16].mean(),
            'noise_weight_avg': w1[:, 16:].mean() if self.noise_count > 0 else 0
        }


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
            _ = best.forward(x_test)
            sat = best.get_saturation()
            print(f"    Step {step}: MSE = {best_mse:.6f}, Saturation = {sat*100:.1f}%")

    return best, best_mse


def run_experiment():
    """Run the clean noise experiment."""
    print("=" * 70)
    print("EXPERIMENT: True Signals + Uncorrelated Noise")
    print("=" * 70)
    print("Testing if network can filter truly uncorrelated noise")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    # Verify noise is uncorrelated
    print("\nNoise correlation verification:")
    for nc in [16, 32, 64, 240]:
        stats = verify_noise_correlation(nc)
        print(f"  {nc} noise signals: max|r|={stats['max']:.4f}, mean|r|={stats['mean']:.4f}")

    # Test configurations
    noise_counts = [0, 16, 32, 64, 240]
    results = {'configs': {}}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for noise_count in noise_counts:
        print(f"\n[16 true + {noise_count} noise = {16 + noise_count} inputs]")
        torch.manual_seed(42)
        np.random.seed(42)

        controller = CleanNoiseController(noise_count=noise_count, device=DEVICE)
        print(f"    Parameters: {controller.num_parameters()}")

        best, best_mse = train_sa(controller, x_test, y_true, max_steps=15000)

        # Final evaluation
        _ = best.forward(x_test)
        saturation = best.get_saturation()
        weights = best.analyze_weight_distribution()

        results['configs'][noise_count] = {
            'input_size': 16 + noise_count,
            'params': controller.num_parameters(),
            'mse': best_mse,
            'saturation': saturation,
            'true_weight_pct': weights['true_weight_pct'],
            'noise_weight_pct': weights['noise_weight_pct']
        }

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Noise':<8} | {'Inputs':<8} | {'Params':<8} | {'MSE':<12} | {'Saturation':<12} | {'True Wt%':<10} | {'Noise Wt%'}")
    print("-" * 90)

    baseline_mse = results['configs'][0]['mse']
    for nc in noise_counts:
        r = results['configs'][nc]
        ratio = r['mse'] / baseline_mse
        print(f"{nc:<8} | {r['input_size']:<8} | {r['params']:<8} | {r['mse']:<12.6f} | {r['saturation']*100:<11.1f}% | {r['true_weight_pct']:<10.1f} | {r['noise_weight_pct']:.1f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if network learned to ignore noise
    nc_240 = results['configs'][240]
    if nc_240['true_weight_pct'] > 20:  # More than proportional share (16/256 = 6.25%)
        print(f"""
The network learned to PARTIALLY filter noise!

With 240 uncorrelated noise signals:
- True signals get {nc_240['true_weight_pct']:.1f}% of weight (vs 6.25% if random)
- Noise signals get {nc_240['noise_weight_pct']:.1f}% of weight
- MSE: {nc_240['mse']:.6f} (vs {baseline_mse:.6f} baseline)

The network is putting {nc_240['true_weight_pct']/6.25:.1f}x more weight on true signals
than random chance would suggest.
""")
    else:
        print(f"""
The network did NOT learn to filter noise effectively.
True signals only get {nc_240['true_weight_pct']:.1f}% of weight.
""")

    # Check saturation trend
    print("Saturation trend:")
    for nc in noise_counts:
        r = results['configs'][nc]
        print(f"  {nc} noise: {r['saturation']*100:.1f}% saturation")

    # Save results
    output_dir = Path("results/clean_noise_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
