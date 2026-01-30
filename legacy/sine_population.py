"""
Sine Test Population
Evolution engine with saturation tracking for MLP + Proteins.
"""

import torch
import numpy as np
from typing import List
from . import sine_config as cfg
from sine_genome import SineGenome


class SinePopulation:
    """
    Manages population through evolutionary cycles.
    Tracks saturation metrics for BOTH MLP neurons AND proteins.
    """

    def __init__(self, size: int = None, device=None):
        self.size = size or cfg.POPULATION_SIZE

        if device is None:
            device = cfg.DEVICE
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.generation = 0

        # Initialize population
        self.genomes: List[SineGenome] = [
            SineGenome.random(device=self.device)
            for _ in range(self.size)
        ]

        for g in self.genomes:
            g.birth_generation = 0

        # Generate test points for sine evaluation
        self._generate_test_points()

        # Statistics tracking
        self.best_fitness_ever = float('-inf')
        self.best_mse_ever = float('inf')

        # Saturation history (key data for theory validation)
        self.saturation_history = []

    def _generate_test_points(self):
        """Generate fixed test points for evaluation - pre-convert to GPU tensors."""
        self.x_values = np.linspace(-2 * np.pi, 2 * np.pi, cfg.NUM_TEST_POINTS)
        self.y_values = np.sin(self.x_values)
        # Pre-convert to GPU tensors for faster evaluation
        self.x_tensor = torch.tensor(self.x_values, device=self.device, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.y_values, device=self.device, dtype=torch.float32)

    def evaluate(self):
        """Evaluate all genomes on sine approximation."""
        for genome in self.genomes:
            genome.reset_for_evaluation()
            genome.evaluate(self.x_tensor, self.y_tensor)

    def evolve(self):
        """Run one generation of evolution."""
        # Sort by fitness (descending) - use numpy argsort for speed on larger populations
        fitnesses = np.array([g.get_fitness() for g in self.genomes], dtype=np.float32)
        sorted_indices = np.argsort(fitnesses)[::-1]  # descending
        self.genomes = [self.genomes[i] for i in sorted_indices]

        # Calculate tier boundaries
        n_elite = int(self.size * cfg.ELITE_PCT)
        n_survive = int(self.size * cfg.SURVIVE_PCT)
        n_clone_mutate = int(self.size * cfg.CLONE_MUTATE_PCT)
        n_random = self.size - n_elite - n_survive - n_clone_mutate

        # Pre-allocate new population list with known size
        new_population = [None] * self.size
        next_gen = self.generation + 1

        # 1. Elite survive unchanged - direct slice assignment
        for i in range(n_elite):
            self.genomes[i].age += 1
            new_population[i] = self.genomes[i]

        # 2. Survivors pass through
        offset = n_elite
        for i in range(n_survive):
            self.genomes[n_elite + i].age += 1
            new_population[offset + i] = self.genomes[n_elite + i]

        # 3. Mutated clones of elite
        offset = n_elite + n_survive
        for i in range(n_clone_mutate):
            parent = self.genomes[i % n_elite]  # Use indices directly
            child = parent.clone()
            child.mutate()
            child.birth_generation = next_gen
            new_population[offset + i] = child

        # 4. Fresh random genomes
        offset = n_elite + n_survive + n_clone_mutate
        for i in range(n_random):
            new_genome = SineGenome.random(device=self.device)
            new_genome.birth_generation = next_gen
            new_population[offset + i] = new_genome

        self.genomes = new_population
        self.generation = next_gen

        # Update best ever stats (best is at index 0 after sort)
        best = self.genomes[0]
        best_fitness = best.get_fitness()
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.best_mse_ever = best.mse

    def record_saturation(self):
        """
        Record saturation statistics for MLP + proteins.
        This is the key data for validating the hybrid computation theory.
        """
        # Collect from all genomes using numpy arrays for vectorized stats
        n_genomes = len(self.genomes)
        k_mlp_arr = np.empty(n_genomes, dtype=np.float32)
        k_protein_arr = np.empty(n_genomes, dtype=np.float32)
        k_total_arr = np.empty(n_genomes, dtype=np.float32)

        for i, g in enumerate(self.genomes):
            k_mlp_arr[i] = g.k_mlp
            k_protein_arr[i] = g.k_proteins
            k_total_arr[i] = g.k_total

        # Best genome stats
        best = self.get_best_genome()
        best_stats = best.get_full_saturation_stats()

        record = {
            'generation': self.generation,
            # Best genome
            'best_k_mlp': best.k_mlp,
            'best_k_proteins': best.k_proteins,
            'best_k_total': best.k_total,
            'best_fitness': best.get_fitness(),
            'best_mse': best.mse,
            # Population stats - vectorized numpy operations
            'mean_k_mlp': float(k_mlp_arr.mean()),
            'mean_k_proteins': float(k_protein_arr.mean()),
            'mean_k_total': float(k_total_arr.mean()),
            'std_k_total': float(k_total_arr.std()),
            # Dimensions
            'n_mlp': best_stats['n_mlp'],
            'n_proteins': best_stats['n_proteins'],
            'n_total': best_stats['n_total'],
            # Per-unit saturation (for heatmaps)
            'best_mlp_per_neuron': best_stats['mlp_per_neuron'],
            'best_protein_per_unit': best_stats['protein_per_unit'],
        }

        self.saturation_history.append(record)
        return record

    def get_stats(self) -> dict:
        """Get population statistics - vectorized."""
        n_genomes = len(self.genomes)

        # Pre-allocate numpy arrays for vectorized operations
        fitnesses = np.empty(n_genomes, dtype=np.float32)
        mses = np.empty(n_genomes, dtype=np.float32)
        ages = np.empty(n_genomes, dtype=np.int32)
        k_mlp = np.empty(n_genomes, dtype=np.int32)
        k_proteins = np.empty(n_genomes, dtype=np.int32)
        k_total = np.empty(n_genomes, dtype=np.int32)

        for i, g in enumerate(self.genomes):
            fitnesses[i] = g.get_fitness()
            mses[i] = g.mse
            ages[i] = g.age
            k_mlp[i] = g.k_mlp
            k_proteins[i] = g.k_proteins
            k_total[i] = g.k_total

        best = self.get_best_genome()

        return {
            'generation': self.generation,
            'fitness_best': float(fitnesses.max()),
            'fitness_median': float(np.median(fitnesses)),
            'mse_best': float(mses.min()),
            'mse_median': float(np.median(mses)),
            'age_max': int(ages.max()),
            'best_ever_fitness': self.best_fitness_ever,
            'best_ever_mse': self.best_mse_ever,
            # Saturation stats
            'k_mlp_best': best.k_mlp,
            'k_proteins_best': best.k_proteins,
            'k_total_best': best.k_total,
            'k_mlp_mean': float(k_mlp.mean()),
            'k_proteins_mean': float(k_proteins.mean()),
            'k_total_mean': float(k_total.mean()),
            'n_mlp': cfg.HIDDEN_SIZE,
            'n_proteins': len(best.proteins) if best.proteins else 0,
            'n_total': best.n_total,
        }

    def get_best_genome(self) -> SineGenome:
        """Return best genome by fitness."""
        return max(self.genomes, key=lambda g: g.get_fitness())

    def get_saturation_trajectory(self) -> dict:
        """
        Get the full saturation trajectory for analysis.
        Returns data needed to validate the theory's predictions.
        """
        if not self.saturation_history:
            return {}

        generations = [r['generation'] for r in self.saturation_history]
        k_mlp = [r['best_k_mlp'] for r in self.saturation_history]
        k_proteins = [r['best_k_proteins'] for r in self.saturation_history]
        k_total = [r['best_k_total'] for r in self.saturation_history]
        fitness = [r['best_fitness'] for r in self.saturation_history]
        mse = [r['best_mse'] for r in self.saturation_history]

        return {
            'generations': generations,
            'k_mlp': k_mlp,
            'k_proteins': k_proteins,
            'k_total': k_total,
            'fitness': fitness,
            'mse': mse,
            'n_mlp': self.saturation_history[-1]['n_mlp'] if self.saturation_history else 0,
            'n_proteins': self.saturation_history[-1]['n_proteins'] if self.saturation_history else 0,
            'n_total': self.saturation_history[-1]['n_total'] if self.saturation_history else 0,
            'full_history': self.saturation_history,
        }

    def save_checkpoint(self, path: str):
        """Save population state."""
        import pickle
        state = {
            'generation': self.generation,
            'genomes': self.genomes,
            'best_fitness_ever': self.best_fitness_ever,
            'best_mse_ever': self.best_mse_ever,
            'saturation_history': self.saturation_history,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_checkpoint(self, path: str):
        """Load population state."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.generation = state['generation']
        self.genomes = state['genomes']
        self.best_fitness_ever = state['best_fitness_ever']
        self.best_mse_ever = state['best_mse_ever']
        self.saturation_history = state.get('saturation_history', [])
