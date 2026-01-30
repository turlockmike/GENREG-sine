"""
Sine Test Genome
Container for MLP controller + protein cascade working in PARALLEL.

Architecture (matching original GENREG):
  x ─┬─→ Controller (MLP) ─→ prediction
     │
     └─→ Protein Cascade ─→ trust ─→ fitness modifier

Fitness = -MSE * trust_multiplier
"""

import torch
import numpy as np
import sine_config as cfg
from sine_controller import SineController
from sine_proteins import (
    create_sine_protein_cascade,
    run_protein_cascade,
    reset_protein_cascade,
    get_cascade_saturation_stats,
)


class SineGenome:
    """
    A genome contains:
    - controller: MLP neural network (produces predictions)
    - proteins: Cascade that produces trust signals (modifies fitness)

    Both work in PARALLEL on the same input.
    """

    _id_counter = 0

    def __init__(self, controller=None, proteins=None, device=None):
        SineGenome._id_counter += 1
        self.id = SineGenome._id_counter

        if device is None:
            device = cfg.DEVICE
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # Neural network (MLP) - produces predictions
        self.controller = controller or SineController(device=self.device)

        # Protein cascade - produces trust signals
        self.proteins = proteins or (
            create_sine_protein_cascade() if cfg.PROTEINS_ENABLED else []
        )

        # Fitness metrics
        self.mse = float('inf')
        self.trust = 0.0  # Accumulated trust from proteins

        # Saturation stats for MLP
        self.k_mlp = 0
        self.mlp_saturation_stats = {}

        # Saturation stats for proteins
        self.k_proteins = 0
        self.protein_saturation_stats = {}

        # Combined
        self.k_total = 0
        self.n_total = 0

        # Tracking
        self.age = 0
        self.parent_id = None
        self.birth_generation = 0

    def evaluate(self, x_values, y_true):
        """
        Evaluate genome on sine approximation task.

        MLP and proteins run in PARALLEL:
        - MLP produces predictions (BATCHED for speed)
        - Proteins compare predictions to targets and produce trust

        Args:
            x_values: Input points - torch.Tensor (N,) on device
            y_true: True sin(x) values - torch.Tensor (N,) on device
        """
        # Reset proteins for fresh evaluation
        if self.proteins:
            reset_protein_cascade(self.proteins)

        # Reset trust
        self.trust = 0.0

        # BATCHED MLP evaluation - input already on GPU
        with torch.no_grad():
            pred_tensor = self.controller.forward(x_values, track=True)

        # Run protein cascade with batched sampling (still sequential for state)
        if cfg.PROTEINS_ENABLED and self.proteins:
            n_points = x_values.shape[0]
            # Sample fewer points for proteins - use stride indexing
            step = max(1, n_points // 20)  # ~20 protein evaluations
            # Get sampled indices as tensor slice
            indices = torch.arange(0, n_points, step, device=self.device)
            x_sampled = x_values[indices]
            pred_sampled = pred_tensor[indices]
            y_sampled = y_true[indices]

            # Convert sampled points to CPU numpy once for protein cascade
            x_np = x_sampled.cpu().numpy()
            pred_np = pred_sampled.cpu().numpy()
            y_np = y_sampled.cpu().numpy()

            # Run protein cascade on sampled points
            for i in range(len(x_np)):
                trust_delta = run_protein_cascade(
                    self.proteins,
                    x=float(x_np[i]),
                    prediction=float(pred_np[i]),
                    target=float(y_np[i])
                )
                self.trust += trust_delta

        # Compute MSE on GPU
        mse_tensor = ((pred_tensor - y_true) ** 2).mean()
        self.mse = mse_tensor.item()

        # Update saturation stats
        self._update_saturation_stats()

    def get_fitness(self) -> float:
        """
        Fitness = -MSE * trust_multiplier

        Trust from proteins modifies fitness:
        - Positive trust boosts fitness
        - Negative trust penalizes fitness
        """
        # Base fitness (negative MSE, higher is better)
        base_fitness = -self.mse

        # Trust multiplier (like original GENREG)
        trust_multiplier = max(0.1, 1.0 + self.trust * cfg.PROTEIN_TRUST_SCALE)

        return base_fitness * trust_multiplier

    def _update_saturation_stats(self):
        """Update saturation statistics for both MLP and proteins."""
        # MLP saturation
        self.mlp_saturation_stats = self.controller.get_saturation_stats()
        self.k_mlp = self.mlp_saturation_stats['k']

        # Protein saturation
        if cfg.PROTEINS_ENABLED and self.proteins:
            self.protein_saturation_stats = get_cascade_saturation_stats(self.proteins)
            self.k_proteins = self.protein_saturation_stats['k_proteins']
        else:
            self.protein_saturation_stats = {'k_proteins': 0, 'n_proteins': 0}
            self.k_proteins = 0

        # Combined totals
        n_mlp = cfg.HIDDEN_SIZE
        n_proteins = len(self.proteins) if self.proteins else 0

        self.k_total = self.k_mlp + self.k_proteins
        self.n_total = n_mlp + n_proteins

    def get_full_saturation_stats(self) -> dict:
        """Get combined saturation stats for MLP + proteins."""
        return {
            'k_mlp': self.k_mlp,
            'k_proteins': self.k_proteins,
            'k_total': self.k_total,
            'n_mlp': cfg.HIDDEN_SIZE,
            'n_proteins': len(self.proteins) if self.proteins else 0,
            'n_total': self.n_total,
            'mlp_per_neuron': self.mlp_saturation_stats.get('per_neuron', []),
            'protein_per_unit': self.protein_saturation_stats.get('per_protein_saturation', []),
        }

    def mutate(self, rate=None, scale=None):
        """Mutate both controller and proteins."""
        self.controller.mutate(rate=rate, scale=scale)

        if cfg.PROTEINS_ENABLED and self.proteins:
            for protein in self.proteins:
                protein.mutate()

    def clone(self):
        """Create deep copy with new ID."""
        new_proteins = [p.clone() for p in self.proteins] if self.proteins else []

        new_genome = SineGenome(
            controller=self.controller.clone(),
            proteins=new_proteins,
            device=self.device
        )
        new_genome.parent_id = self.id
        return new_genome

    def reset_for_evaluation(self):
        """Reset per-generation metrics."""
        self.mse = float('inf')
        self.trust = 0.0
        self.k_mlp = 0
        self.k_proteins = 0
        self.k_total = 0
        self.mlp_saturation_stats = {}
        self.protein_saturation_stats = {}
        self.controller.clear_activation_history()
        if self.proteins:
            reset_protein_cascade(self.proteins)

    @staticmethod
    def random(device=None):
        """Create new random genome."""
        return SineGenome(device=device)
