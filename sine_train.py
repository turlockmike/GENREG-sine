"""
Sine Test Training Script
Validates Emergent Hybrid Computation Theory

The theory predicts:
1. k (saturated neurons/proteins) will increase over training
2. Initial phase: low k (exploration)
3. Intermediate phase: rising k (locking in modes)
4. Convergence phase: high k stability

We track:
- k_mlp: saturated MLP neurons
- k_proteins: saturated protein units
- k_total: combined saturation
- Correlation between k and fitness
"""

import os
import torch
import numpy as np
import random
import json
from datetime import datetime
from pathlib import Path

import sine_config as cfg
from sine_population import SinePopulation
import pickle


def save_best_genome(genome, population, path):
    """Save only the best genome (lightweight checkpoint)."""
    state = {
        'genome': genome,
        'generation': population.generation,
        'best_mse': genome.mse,
        'best_fitness': genome.get_fitness(),
        'k_mlp': genome.k_mlp,
        'k_proteins': genome.k_proteins,
        'k_total': genome.k_total,
        'saturation_history': population.saturation_history,
    }
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def train():
    """Main training loop."""
    clear_screen()
    print("=" * 70)
    print("EMERGENT HYBRID COMPUTATION TEST")
    print("GENREG: MLP + Protein Cascade")
    print("=" * 70)

    # Setup
    setup_seed(cfg.SEED)

    # Check device
    if torch.cuda.is_available() and cfg.DEVICE == "cuda":
        device = torch.device("cuda")
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")

    # Create session directory
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"results/session_{session_id}")
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session: {session_dir.name}")

    # Initialize population
    print(f"\nInitializing population of {cfg.POPULATION_SIZE} genomes...")
    population = SinePopulation(device=device)

    # Architecture info
    sample = population.genomes[0]
    n_proteins = len(sample.proteins) if sample.proteins else 0

    print(f"\nArchitecture:")
    print(f"  Proteins: {n_proteins} units in cascade")
    print(f"  MLP Input: {cfg.TRUE_SIGNAL_SIZE} TRUE + {cfg.NOISE_SIGNAL_SIZE} NOISE = {cfg.EXPANSION_SIZE} total")
    print(f"  MLP: {cfg.EXPANSION_SIZE} -> {cfg.HIDDEN_SIZE} -> {cfg.OUTPUT_SIZE}")
    print(f"  Compression: {cfg.EXPANSION_SIZE}:{cfg.HIDDEN_SIZE} = {cfg.EXPANSION_SIZE // cfg.HIDDEN_SIZE}:1")
    print(f"  MLP params: {sample.controller.num_parameters():,}")
    print(f"  Total units (n): {cfg.HIDDEN_SIZE + n_proteins}")

    print(f"\nEvolution tiers:")
    print(f"  Elite:        {cfg.ELITE_PCT*100:.0f}%")
    print(f"  Survivors:    {cfg.SURVIVE_PCT*100:.0f}%")
    print(f"  Clone+Mutate: {cfg.CLONE_MUTATE_PCT*100:.0f}%")
    print(f"  Fresh Random: {cfg.RANDOM_PCT*100:.0f}%")

    print(f"\nTheory Predictions:")
    print(f"  - Initial phase: k should be LOW (broad exploration)")
    print(f"  - Intermediate:  k should RISE (locking in modes)")
    print(f"  - Convergence:   k should be HIGH and STABLE")
    print(f"\nSaturation threshold: |h| > {cfg.SATURATION_THRESHOLD}")

    print("\n" + "=" * 70)
    print("GEN   |   MSE      | k_MLP   | k_PROT  | k_TOTAL")
    print("-" * 70)

    try:
        for gen in range(cfg.GENERATIONS):
            # Evaluate all genomes
            population.evaluate()

            # Record saturation data
            sat_record = population.record_saturation()

            # Get stats
            stats = population.get_stats()

            # Log
            if gen % cfg.LOG_EVERY == 0:
                n_mlp = stats['n_mlp']
                n_prot = stats['n_proteins']
                n_total = stats['n_total']

                k_mlp = stats['k_mlp_best']
                k_prot = stats['k_proteins_best']
                k_total = stats['k_total_best']

                # Visual bars (ASCII compatible)
                mlp_bar = '#' * k_mlp + '.' * (n_mlp - k_mlp)
                prot_bar = '#' * k_prot + '.' * (n_prot - k_prot)

                log_str = (
                    f"{gen:5d} | "
                    f"{stats['mse_best']:.6f} | "
                    f"{k_mlp:2d}/{n_mlp} [{mlp_bar}] | "
                    f"{k_prot:2d}/{n_prot} [{prot_bar}] | "
                    f"{k_total:2d}/{n_total}"
                )
                print(log_str)

            # Evolve
            population.evolve()

            # Checkpoint - save only best genome
            if gen % cfg.CHECKPOINT_EVERY == 0 and gen > 0:
                best = population.get_best_genome()
                ckpt_path = session_dir / "checkpoint_best.pkl"
                save_best_genome(best, population, ckpt_path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")

    finally:
        # Final evaluation
        population.evaluate()
        population.record_saturation()

        # Save final best genome
        best = population.get_best_genome()
        final_path = session_dir / "checkpoint_best.pkl"
        save_best_genome(best, population, final_path)
        print(f"\nFinal checkpoint: {final_path}")

        # Save saturation trajectory as JSON for analysis
        trajectory = population.get_saturation_trajectory()
        trajectory_path = session_dir / "saturation_trajectory.json"
        with open(trajectory_path, 'w') as f:
            json.dump(trajectory, f, indent=2)
        print(f"Saturation data: {trajectory_path}")

        # Final stats
        stats = population.get_stats()
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"  Generations: {population.generation}")
        print(f"  Best MSE: {stats['best_ever_mse']:.6f}")
        print(f"\n  Final Saturation (best genome):")
        print(f"    k_MLP:      {stats['k_mlp_best']} / {stats['n_mlp']}")
        print(f"    k_Proteins: {stats['k_proteins_best']} / {stats['n_proteins']}")
        print(f"    k_Total:    {stats['k_total_best']} / {stats['n_total']}")

        # Theory validation summary
        print("\n" + "-" * 70)
        print("THEORY VALIDATION")
        print("-" * 70)

        if trajectory and 'k_total' in trajectory:
            k_trajectory = trajectory['k_total']
            n_total = trajectory['n_total']

            if len(k_trajectory) >= 3:
                # Split into thirds for phase analysis
                third = len(k_trajectory) // 3
                initial_k = np.mean(k_trajectory[:third])
                middle_k = np.mean(k_trajectory[third:2*third])
                final_k = np.mean(k_trajectory[2*third:])

                print(f"\n  Phase Analysis (n_total = {n_total}):")
                print(f"    Initial  (gen 0-{third-1}):     k_avg = {initial_k:.1f}")
                print(f"    Middle   (gen {third}-{2*third-1}):   k_avg = {middle_k:.1f}")
                print(f"    Final    (gen {2*third}-{len(k_trajectory)-1}): k_avg = {final_k:.1f}")

                # Check if theory holds
                if initial_k < middle_k < final_k:
                    print(f"\n  [OK] SUPPORTS THEORY: k increased monotonically across phases")
                elif initial_k < final_k:
                    print(f"\n  [~] PARTIAL SUPPORT: k increased overall but not monotonically")
                else:
                    print(f"\n  [X] CONTRADICTS THEORY: k did not increase as predicted")

                # Check final saturation level
                if n_total > 0:
                    final_k_ratio = final_k / n_total
                    if final_k_ratio > 0.5:
                        print(f"  [OK] High final saturation ({final_k_ratio*100:.0f}%) suggests hybrid regime")
                    else:
                        print(f"  [?] Low final saturation ({final_k_ratio*100:.0f}%) - may need more generations")

                # MLP vs Protein breakdown
                k_mlp_traj = trajectory['k_mlp']
                k_prot_traj = trajectory['k_proteins']

                mlp_increase = np.mean(k_mlp_traj[2*third:]) - np.mean(k_mlp_traj[:third])
                prot_increase = np.mean(k_prot_traj[2*third:]) - np.mean(k_prot_traj[:third])

                print(f"\n  Component Breakdown:")
                print(f"    MLP neurons:  Δk = {mlp_increase:+.1f}")
                print(f"    Proteins:     Δk = {prot_increase:+.1f}")

        print("\n" + "=" * 70)
        print(f"Run 'python sine_analyze.py {session_dir}' for detailed visualization")
        print("=" * 70)


if __name__ == "__main__":
    train()
