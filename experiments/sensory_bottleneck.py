"""
Experiment: Sensory Bottleneck

Architecture:
  Environment (256 signals) → Sensory Layer (N neurons) → Processing Layer (8 neurons) → Output (1)

The bottleneck at the sensory layer forces the network to be selective about
which environmental signals to pay attention to. Evolution should pressure
those neurons to focus on the most relevant signals.

This mimics biology: organisms have limited sensory capacity and must evolve
to detect the most relevant signals from a noisy environment.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json

from legacy.sine_controller import expand_input
from core.metrics import compute_metrics, Metrics, format_metrics_table
from core.training import train_sa

DEVICE = torch.device("cpu")


class SensoryBottleneckController:
    """
    Network with sensory bottleneck architecture.

    Environment (256) → Sensory (N) → Processing (8) → Output (1)
    """

    def __init__(self, sensory_size=2, processing_size=8, device=None):
        self.device = device or DEVICE
        self.sensory_size = sensory_size
        self.processing_size = processing_size
        self.input_size = 256

        # Layer 1: Environment → Sensory (bottleneck)
        scale_w1 = np.sqrt(2.0 / self.input_size)
        self.w1 = torch.randn(sensory_size, self.input_size, device=self.device) * scale_w1
        self.b1 = torch.zeros(sensory_size, device=self.device)

        # Layer 2: Sensory → Processing
        scale_w2 = np.sqrt(2.0 / sensory_size)
        self.w2 = torch.randn(processing_size, sensory_size, device=self.device) * scale_w2
        self.b2 = torch.zeros(processing_size, device=self.device)

        # Layer 3: Processing → Output
        scale_w3 = np.sqrt(2.0 / processing_size)
        self.w3 = torch.randn(1, processing_size, device=self.device) * scale_w3
        self.b3 = torch.zeros(1, device=self.device)

        # For tracking activations
        self.last_hidden = None  # Used by compute_metrics

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)

        original_dim = x.dim()
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)

        # Environment expansion (256 signals)
        env = expand_input(x)

        # Sensory layer (bottleneck)
        sensory = torch.tanh(torch.nn.functional.linear(env, self.w1, self.b1))

        # Processing layer
        processing = torch.tanh(torch.nn.functional.linear(sensory, self.w2, self.b2))
        self.last_hidden = processing.detach()  # For saturation tracking

        # Output
        output = torch.tanh(torch.nn.functional.linear(processing, self.w3, self.b3))

        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)

        return output

    def mutate(self, rate=0.1, scale=0.1):
        for param in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * scale
            param.data += mask.float() * noise

    def clone(self):
        new = SensoryBottleneckController(
            sensory_size=self.sensory_size,
            processing_size=self.processing_size,
            device=self.device
        )
        new.w1 = self.w1.clone()
        new.b1 = self.b1.clone()
        new.w2 = self.w2.clone()
        new.b2 = self.b2.clone()
        new.w3 = self.w3.clone()
        new.b3 = self.b3.clone()
        return new

    def num_parameters(self):
        return (self.input_size * self.sensory_size + self.sensory_size +
                self.sensory_size * self.processing_size + self.processing_size +
                self.processing_size * 1 + 1)

    def analyze_sensory_attention(self):
        """Analyze what each sensory neuron is paying attention to."""
        w1 = self.w1.abs().cpu().numpy()

        results = []
        for i in range(self.sensory_size):
            weights = w1[i]
            true_weight = weights[:16].sum()
            noise_weight = weights[16:].sum()
            total = true_weight + noise_weight

            top_indices = np.argsort(weights)[-5:][::-1]

            results.append({
                'neuron': i,
                'true_weight_pct': true_weight / total * 100,
                'noise_weight_pct': noise_weight / total * 100,
                'top_inputs': top_indices.tolist(),
                'focuses_on_true': any(idx < 16 for idx in top_indices[:3])
            })

        return results


def run_experiment():
    """Run the sensory bottleneck experiment."""
    print("=" * 70)
    print("EXPERIMENT: Sensory Bottleneck Architecture")
    print("=" * 70)
    print("Environment (256) → Sensory (N) → Processing (8) → Output (1)")
    print("=" * 70)

    x_test = torch.linspace(-2 * np.pi, 2 * np.pi, 100, device=DEVICE)
    y_true = torch.sin(x_test)

    sensory_sizes = [1, 2, 4, 8, 16]
    results = {'configs': {}}

    for sensory_size in sensory_sizes:
        print(f"\n[Sensory size: {sensory_size}]")
        torch.manual_seed(42)
        np.random.seed(42)

        controller = SensoryBottleneckController(
            sensory_size=sensory_size,
            processing_size=8,
            device=DEVICE
        )
        print(f"    Architecture: 256 → {sensory_size} → 8 → 1")
        print(f"    Parameters: {controller.num_parameters()}")

        best, final_metrics, history = train_sa(
            controller, x_test, y_true,
            max_steps=15000,
            verbose=True,
            report_interval=5000
        )

        # Analyze attention
        attention = best.analyze_sensory_attention()

        results['configs'][sensory_size] = {
            'metrics': final_metrics.to_dict(),
            'params': controller.num_parameters(),
            'attention': attention
        }

        # Print attention analysis
        print(f"\n    Sensory neuron attention:")
        for a in attention:
            focus = "TRUE" if a['focuses_on_true'] else "NOISE"
            print(f"      Neuron {a['neuron']}: {a['true_weight_pct']:.1f}% true → {focus}")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Sensory':<8} | {'MSE':<12} | {'Energy':<10} | {'Saturation':<12} | {'True Focus'}")
    print("-" * 65)

    for size in sensory_sizes:
        r = results['configs'][size]
        m = r['metrics']
        true_focus = sum(1 for a in r['attention'] if a['focuses_on_true'])
        print(f"{size:<8} | {m['mse']:<12.6f} | {m['energy']:<10.4f} | {m['saturation']*100:<11.1f}% | {true_focus}/{size}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "sensory_bottleneck"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_experiment()
