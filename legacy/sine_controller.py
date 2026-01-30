"""
Sine Test Controller
MLP with input expansion: TRUE signals + NOISY competing signals.

Architecture:
  x → [16 true + 240 noise = 256D] → hidden (8) → output (1)

The 256 → 8 compression (32:1) forces massive saturation.
Network must learn to filter noise and focus on true signal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from . import sine_config as cfg

# Fixed random seed for reproducible noise patterns
_noise_rng = np.random.RandomState(12345)
_noise_freqs = _noise_rng.uniform(0.1, 10.0, size=100)
_noise_phases = _noise_rng.uniform(0, 2 * np.pi, size=100)
_noise_scales = _noise_rng.uniform(0.5, 2.0, size=100)


def expand_input(x: torch.Tensor) -> torch.Tensor:
    """
    Expand scalar x into TRUE signals + NOISY competing signals.

    TRUE SIGNALS (16):
      - Fourier basis that can represent sin(x)
      - x, x², x³, sin(kx), cos(kx) for useful k values

    NOISE SIGNALS (240):
      - Wrong frequencies (won't help predict sin(x))
      - Phase-shifted garbage
      - Random polynomial combinations
      - Inverted/conflicting signals
      - Other functions (tanh, exp, etc.)

    Args:
        x: Input tensor of shape (batch, 1)

    Returns:
        Expanded tensor of shape (batch, 256)
    """
    # Ensure shape is (batch, 1)
    if x.dim() == 1:
        x = x.unsqueeze(-1)

    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Normalize x to [-1, 1] range (input is [-2π, 2π])
    x_norm = x / (2 * np.pi)

    # =========================================================
    # TRUE SIGNALS (16) - useful for predicting sin(x)
    # =========================================================
    # Vectorized: pre-allocate and fill
    true_signal = torch.zeros(batch_size, cfg.TRUE_SIGNAL_SIZE, device=device, dtype=dtype)

    # Raw input and polynomials
    true_signal[:, 0:1] = x_norm
    true_signal[:, 1:2] = x_norm ** 2
    true_signal[:, 2:3] = x_norm ** 3

    # Fourier basis at useful frequencies - vectorized
    freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], device=device, dtype=dtype)
    # x is (batch, 1), freqs is (6,) -> broadcast to (batch, 6)
    freq_x = x * freqs  # (batch, 6)
    true_signal[:, 3:9] = torch.sin(freq_x)
    true_signal[:, 9:15] = torch.cos(freq_x)
    # Remaining columns stay zero (padding)

    # =========================================================
    # NOISE SIGNALS - misleading, conflicting, useless
    # =========================================================
    # Only generate noise if NOISE_SIGNAL_SIZE > 0
    if cfg.NOISE_SIGNAL_SIZE > 0:
        # Pre-allocate noise tensor and fill with vectorized operations
        noise_signal = torch.zeros(batch_size, cfg.NOISE_SIGNAL_SIZE, device=device, dtype=dtype)
        col = 0

        # Convert numpy arrays to tensors once (cached on device)
        noise_freqs_t = torch.tensor(_noise_freqs, device=device, dtype=dtype)
        noise_phases_t = torch.tensor(_noise_phases, device=device, dtype=dtype)
        noise_scales_t = torch.tensor(_noise_scales, device=device, dtype=dtype)

        # Helper to safely fill columns without exceeding NOISE_SIGNAL_SIZE
        def fill_cols(data, n_cols):
            nonlocal col
            actual = min(n_cols, cfg.NOISE_SIGNAL_SIZE - col)
            if actual > 0:
                noise_signal[:, col:col+actual] = data[:, :actual] if data.dim() > 1 else data[:actual]
                col += actual

        # 1. Wrong frequencies (50 signals) - vectorized
        n1 = min(50, cfg.NOISE_SIGNAL_SIZE - col)
        if n1 > 0:
            freqs_1 = noise_freqs_t[:n1]
            phases_1 = noise_phases_t[:n1]
            noise_signal[:, col:col+n1] = torch.sin(x * freqs_1 + phases_1)
            col += n1

        # 2. Cosines at wrong frequencies (50 signals) - vectorized
        n2 = min(50, cfg.NOISE_SIGNAL_SIZE - col)
        if n2 > 0:
            freqs_2 = noise_freqs_t[50:50+n2] if len(noise_freqs_t) >= 50+n2 else noise_freqs_t[:n2]
            phases_2 = noise_phases_t[50:50+n2] if len(noise_phases_t) >= 50+n2 else noise_phases_t[:n2]
            noise_signal[:, col:col+n2] = torch.cos(x * freqs_2 + phases_2)
            col += n2

        # 3. Inverted/negated true-ish signals (20 signals) - vectorized
        inv_freqs = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], device=device, dtype=dtype)
        n3a = min(10, cfg.NOISE_SIGNAL_SIZE - col)
        if n3a > 0:
            noise_signal[:, col:col+n3a] = -torch.sin(x * inv_freqs[:n3a])
            col += n3a
        n3b = min(10, cfg.NOISE_SIGNAL_SIZE - col)
        if n3b > 0:
            noise_signal[:, col:col+n3b] = -torch.cos(x * inv_freqs[:n3b])
            col += n3b

        # 4. Polynomial garbage (20 signals) - vectorized
        n4 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n4 > 0:
            powers = torch.arange(4, 4+n4, device=device, dtype=dtype).unsqueeze(0)
            noise_signal[:, col:col+n4] = x_norm ** powers
            col += n4

        # 5. Mixed/product terms (20 signals) - vectorized
        n5 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n5 > 0:
            freqs_5a = noise_freqs_t[:n5]
            freqs_5b = noise_freqs_t[20:20+n5]
            noise_signal[:, col:col+n5] = torch.sin(x * freqs_5a) * torch.cos(x * freqs_5b)
            col += n5

        # 6. Tanh distortions (20 signals) - vectorized
        n6 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n6 > 0:
            scales_6 = noise_scales_t[:n6]
            noise_signal[:, col:col+n6] = torch.tanh(x * scales_6)
            col += n6

        # 7. Exponential-based noise (20 signals) - vectorized
        n7 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n7 > 0:
            scales_7 = noise_scales_t[20:20+n7] * 0.3
            noise_signal[:, col:col+n7] = torch.exp(-torch.abs(x * scales_7)) - 0.5
            col += n7

        # 8. Square wave approximations (20 signals) - vectorized
        n8 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n8 > 0:
            freqs_8 = noise_freqs_t[40:40+n8]
            noise_signal[:, col:col+n8] = torch.sign(torch.sin(x * freqs_8))
            col += n8

        # 9. Sawtooth-like (20 signals) - vectorized
        n9 = min(20, cfg.NOISE_SIGNAL_SIZE - col)
        if n9 > 0:
            freqs_9 = noise_freqs_t[60:60+n9]
            phases_9 = noise_phases_t[60:60+n9]
            noise_signal[:, col:col+n9] = torch.fmod(x * freqs_9 + phases_9, 2.0) - 1.0
            col += n9

        # Fill remaining with random noise if needed
        remaining = cfg.NOISE_SIGNAL_SIZE - col
        if remaining > 0:
            noise_signal[:, col:] = torch.randn(batch_size, remaining, device=device, dtype=dtype) * 0.5

        # Combine: [TRUE | NOISE]
        expanded = torch.cat([true_signal, noise_signal], dim=-1)
    else:
        # No noise - just return true signal
        expanded = true_signal

    return expanded


class SineController:
    """
    MLP: expanded_input (256) -> hidden (8) -> output (1)

    Input: 16 TRUE signals + 240 NOISE signals = 256 total
    Compression: 256 -> 8 = 32:1 ratio!

    The network must learn to:
    1. Filter out the 240 noisy/misleading signals
    2. Focus on the 16 true signals
    3. Use saturation to "gate off" irrelevant inputs

    Tracks activation saturation for theory validation.
    """

    def __init__(self, device=None):
        if device is None:
            device = cfg.DEVICE
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Input size depends on expansion
        if cfg.INPUT_EXPANSION:
            input_size = cfg.EXPANSION_SIZE
        else:
            input_size = cfg.INPUT_SIZE

        # Xavier-like initialization
        scale_w1 = np.sqrt(2.0 / input_size)
        scale_w2 = np.sqrt(2.0 / cfg.HIDDEN_SIZE)

        # Layer 1: input -> hidden (THIS IS WHERE COMPRESSION HAPPENS)
        self.w1 = torch.randn(
            cfg.HIDDEN_SIZE, input_size,
            device=self.device, dtype=torch.float32
        ) * scale_w1
        self.b1 = torch.zeros(cfg.HIDDEN_SIZE, device=self.device, dtype=torch.float32)

        # Layer 2: hidden -> output
        self.w2 = torch.randn(
            cfg.OUTPUT_SIZE, cfg.HIDDEN_SIZE,
            device=self.device, dtype=torch.float32
        ) * scale_w2
        self.b2 = torch.zeros(cfg.OUTPUT_SIZE, device=self.device, dtype=torch.float32)

        # Activation tracking for saturation analysis
        self.last_hidden_activations = None
        self.activation_history = []

    def forward(self, x, track=True):
        """
        Forward pass with optional activation tracking.
        Handles both single values and batches.

        Args:
            x: Input tensor - shape () or (N,) or (N, 1)
            track: Whether to store activations for saturation analysis

        Returns:
            output: Prediction(s) scaled to [-1, 1]
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        elif x.device != self.device:
            x = x.to(self.device)

        # Track original shape for output
        original_dim = x.dim()

        # Normalize to (batch, 1) shape
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)  # (N,) -> (N, 1)

        # Expand input if configured
        if cfg.INPUT_EXPANSION:
            x = expand_input(x)

        # Layer 1: Tanh activation (THIS IS WHERE SATURATION HAPPENS)
        hidden = torch.tanh(F.linear(x, self.w1, self.b1))

        # Store activations for saturation tracking
        if track:
            self.last_hidden_activations = hidden.detach()
            if cfg.TRACK_ACTIVATIONS:
                self.activation_history.append(
                    hidden.abs().mean(dim=0).cpu().numpy()
                )

        # Layer 2: Tanh to bound output to [-1, 1]
        output = torch.tanh(F.linear(hidden, self.w2, self.b2))

        # Return in appropriate shape
        if original_dim == 0:
            output = output.squeeze()
        elif original_dim == 1:
            output = output.squeeze(-1)  # (N, 1) -> (N,)

        return output

    def get_saturation_per_neuron(self) -> np.ndarray:
        """
        Compute saturation ratio for each hidden neuron.
        """
        if self.last_hidden_activations is None:
            return np.zeros(cfg.HIDDEN_SIZE)

        abs_act = self.last_hidden_activations.abs()
        saturated = (abs_act > cfg.SATURATION_THRESHOLD).float()
        saturation_ratio = saturated.mean(dim=0).cpu().numpy()
        return saturation_ratio

    def get_k(self) -> int:
        """
        Get k = number of saturated neurons.
        """
        saturation = self.get_saturation_per_neuron()
        return int(np.sum(saturation > 0.5))

    def get_saturation_stats(self) -> dict:
        """Get detailed saturation statistics."""
        saturation = self.get_saturation_per_neuron()
        return {
            'k': self.get_k(),
            'n': cfg.HIDDEN_SIZE,
            'k_ratio': self.get_k() / cfg.HIDDEN_SIZE,
            'mean_saturation': float(saturation.mean()),
            'max_saturation': float(saturation.max()),
            'min_saturation': float(saturation.min()),
            'per_neuron': saturation.tolist(),
        }

    def clear_activation_history(self):
        """Clear stored activation history."""
        self.activation_history = []

    def mutate(self, rate=None, scale=None):
        """Mutate weights in-place."""
        rate = rate if rate is not None else cfg.MUTATION_RATE
        scale = scale if scale is not None else cfg.MUTATION_SCALE

        for param in [self.w1, self.b1, self.w2, self.b2]:
            mask = torch.rand_like(param) < rate
            noise = torch.randn_like(param) * scale
            param.data += mask.float() * noise

    def clone(self):
        """Create a deep copy."""
        new_controller = SineController(device=self.device)
        new_controller.w1 = self.w1.clone()
        new_controller.b1 = self.b1.clone()
        new_controller.w2 = self.w2.clone()
        new_controller.b2 = self.b2.clone()
        return new_controller

    def num_parameters(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in [self.w1, self.b1, self.w2, self.b2])

    @staticmethod
    def random(device=None):
        """Create new random controller."""
        return SineController(device=device)
