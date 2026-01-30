"""
Sine Test Analysis
Visualize saturation trajectory for MLP + Proteins to validate hybrid computation theory.
"""

import json
import sys
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Text-only analysis available.")


def load_trajectory(session_dir: str) -> dict:
    """Load saturation trajectory from session."""
    path = Path(session_dir) / "saturation_trajectory.json"
    if not path.exists():
        raise FileNotFoundError(f"No trajectory file found at {path}")

    with open(path, 'r') as f:
        return json.load(f)


def text_analysis(trajectory: dict):
    """Print text-based analysis of the trajectory."""
    generations = trajectory['generations']
    k_mlp = trajectory['k_mlp']
    k_proteins = trajectory['k_proteins']
    k_total = trajectory['k_total']
    mse = trajectory['mse']
    n_mlp = trajectory['n_mlp']
    n_proteins = trajectory['n_proteins']
    n_total = trajectory['n_total']

    print("=" * 70)
    print("SATURATION TRAJECTORY ANALYSIS")
    print("GENREG: MLP + Protein Cascade")
    print("=" * 70)

    print(f"\nTotal generations: {len(generations)}")
    print(f"Architecture: {n_proteins} proteins + {n_mlp} MLP neurons = {n_total} total units")

    # k statistics
    print(f"\n{'='*30} MLP NEURONS {'='*30}")
    print(f"  Initial k: {k_mlp[0]}")
    print(f"  Final k:   {k_mlp[-1]}")
    print(f"  Min k:     {min(k_mlp)}")
    print(f"  Max k:     {max(k_mlp)}")
    print(f"  Mean k:    {np.mean(k_mlp):.2f}")

    print(f"\n{'='*30} PROTEINS {'='*33}")
    print(f"  Initial k: {k_proteins[0]}")
    print(f"  Final k:   {k_proteins[-1]}")
    print(f"  Min k:     {min(k_proteins)}")
    print(f"  Max k:     {max(k_proteins)}")
    print(f"  Mean k:    {np.mean(k_proteins):.2f}")

    print(f"\n{'='*30} COMBINED {'='*33}")
    print(f"  Initial k: {k_total[0]}")
    print(f"  Final k:   {k_total[-1]}")
    print(f"  Min k:     {min(k_total)}")
    print(f"  Max k:     {max(k_total)}")
    print(f"  Mean k:    {np.mean(k_total):.2f}")

    # Phase analysis
    third = len(k_total) // 3
    if third > 0:
        initial_phase = np.mean(k_total[:third])
        middle_phase = np.mean(k_total[third:2*third])
        final_phase = np.mean(k_total[2*third:])

        print(f"\n{'='*25} PHASE ANALYSIS {'='*27}")
        print(f"  Initial phase avg k: {initial_phase:.2f}")
        print(f"  Middle phase avg k:  {middle_phase:.2f}")
        print(f"  Final phase avg k:   {final_phase:.2f}")

        # Trend
        if initial_phase < middle_phase < final_phase:
            print(f"\n  → MONOTONIC INCREASE (strongly supports theory)")
        elif initial_phase < final_phase:
            print(f"\n  → OVERALL INCREASE (partially supports theory)")
        else:
            print(f"\n  → NO CLEAR INCREASE (needs investigation)")

        # Component breakdown
        mlp_init = np.mean(k_mlp[:third])
        mlp_final = np.mean(k_mlp[2*third:])
        prot_init = np.mean(k_proteins[:third])
        prot_final = np.mean(k_proteins[2*third:])

        print(f"\n  Component Trajectories:")
        print(f"    MLP:      {mlp_init:.1f} → {mlp_final:.1f} (Δ = {mlp_final - mlp_init:+.1f})")
        print(f"    Proteins: {prot_init:.1f} → {prot_final:.1f} (Δ = {prot_final - prot_init:+.1f})")

    # Fitness correlation
    if len(k_total) > 1:
        correlation = np.corrcoef(k_total, [-m for m in mse])[0, 1]
        print(f"\n{'='*25} CORRELATIONS {'='*28}")
        print(f"  k_total vs fitness: r = {correlation:.3f}")
        if correlation > 0.5:
            print("    → Strong positive (more saturation = better fitness)")
        elif correlation > 0:
            print("    → Weak positive correlation")
        else:
            print("    → Negative or no correlation")

    # ASCII visualization of k over time
    print("\n" + "=" * 70)
    print("k TRAJECTORY (ASCII)")
    print("=" * 70)

    # Sample every Nth generation for display
    sample_rate = max(1, len(generations) // 40)
    sampled_gens = generations[::sample_rate]
    sampled_k_mlp = k_mlp[::sample_rate]
    sampled_k_prot = k_proteins[::sample_rate]

    print(f"{'Gen':>5} | {'MLP':<{n_mlp+4}} | {'Proteins':<{n_proteins+4}}")
    print("-" * 70)

    for gen, km, kp in zip(sampled_gens, sampled_k_mlp, sampled_k_prot):
        mlp_bar = '█' * km + '░' * (n_mlp - km)
        prot_bar = '█' * kp + '░' * (n_proteins - kp)
        print(f"{gen:5d} | [{mlp_bar}] | [{prot_bar}]")

    print("-" * 70)


def plot_analysis(trajectory: dict, save_path: str = None):
    """Create matplotlib visualizations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available for plotting")
        return

    generations = trajectory['generations']
    k_mlp = trajectory['k_mlp']
    k_proteins = trajectory['k_proteins']
    k_total = trajectory['k_total']
    mse = trajectory['mse']
    n_mlp = trajectory['n_mlp']
    n_proteins = trajectory['n_proteins']
    n_total = trajectory['n_total']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Emergent Hybrid Computation: MLP + Proteins', fontsize=14, fontweight='bold')

    # Plot 1: k components over generations
    ax1 = axes[0, 0]
    ax1.plot(generations, k_mlp, 'b-', linewidth=2, label=f'MLP neurons (n={n_mlp})')
    ax1.plot(generations, k_proteins, 'g-', linewidth=2, label=f'Proteins (n={n_proteins})')
    ax1.plot(generations, k_total, 'r--', linewidth=2, label=f'Total (n={n_total})')
    ax1.axhline(y=n_total/2, color='gray', linestyle=':', alpha=0.5, label='50% saturation')
    ax1.fill_between(generations, 0, k_mlp, alpha=0.2, color='blue')
    ax1.fill_between(generations, k_mlp, [km + kp for km, kp in zip(k_mlp, k_proteins)],
                     alpha=0.2, color='green')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('k (saturated units)')
    ax1.set_title('Saturation Emergence by Component')
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, n_total + 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: MSE over generations (log scale)
    ax2 = axes[0, 1]
    ax2.semilogy(generations, mse, 'purple', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('MSE (log scale)')
    ax2.set_title('Fitness Improvement')
    ax2.grid(True, alpha=0.3)

    # Plot 3: k_total vs MSE scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(k_total, mse, c=generations, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('k_total (saturated units)')
    ax3.set_ylabel('MSE')
    ax3.set_title('Saturation vs Fitness Relationship')
    ax3.set_yscale('log')
    plt.colorbar(scatter, ax=ax3, label='Generation')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Stacked area showing MLP vs Protein contribution
    ax4 = axes[1, 1]
    ax4.stackplot(generations, k_mlp, k_proteins,
                  labels=['MLP Saturated', 'Proteins Saturated'],
                  colors=['steelblue', 'forestgreen'], alpha=0.7)
    ax4.axhline(y=n_total, color='red', linestyle='--', label=f'Total units (n={n_total})')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Saturated Units')
    ax4.set_title('Component Contribution to Saturation')
    ax4.legend(loc='upper left')
    ax4.set_ylim(0, n_total + 2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def analyze_phases(trajectory: dict) -> dict:
    """
    Detailed phase analysis to validate theory predictions.
    """
    k_total = np.array(trajectory['k_total'])
    k_mlp = np.array(trajectory['k_mlp'])
    k_proteins = np.array(trajectory['k_proteins'])
    n_total = trajectory['n_total']
    n_mlp = trajectory['n_mlp']
    n_proteins = trajectory['n_proteins']

    # Split into thirds for phase analysis
    third = len(k_total) // 3

    phases = {
        'initial': {
            'range': (0, third),
            'k_total_mean': float(np.mean(k_total[:third])),
            'k_mlp_mean': float(np.mean(k_mlp[:third])),
            'k_proteins_mean': float(np.mean(k_proteins[:third])),
        },
        'intermediate': {
            'range': (third, 2*third),
            'k_total_mean': float(np.mean(k_total[third:2*third])),
            'k_mlp_mean': float(np.mean(k_mlp[third:2*third])),
            'k_proteins_mean': float(np.mean(k_proteins[third:2*third])),
        },
        'convergence': {
            'range': (2*third, len(k_total)),
            'k_total_mean': float(np.mean(k_total[2*third:])),
            'k_mlp_mean': float(np.mean(k_mlp[2*third:])),
            'k_proteins_mean': float(np.mean(k_proteins[2*third:])),
        },
    }

    # Theory predictions
    predictions = {
        'k_total_increases': phases['initial']['k_total_mean'] < phases['convergence']['k_total_mean'],
        'k_mlp_increases': phases['initial']['k_mlp_mean'] < phases['convergence']['k_mlp_mean'],
        'k_proteins_increases': phases['initial']['k_proteins_mean'] < phases['convergence']['k_proteins_mean'],
        'final_saturation_high': phases['convergence']['k_total_mean'] > n_total / 2,
        'monotonic_increase': (phases['initial']['k_total_mean'] <
                               phases['intermediate']['k_total_mean'] <
                               phases['convergence']['k_total_mean']),
    }

    support_count = sum(predictions.values())
    if support_count >= 4:
        theory_support = "STRONG"
    elif support_count >= 3:
        theory_support = "MODERATE"
    elif support_count >= 2:
        theory_support = "WEAK"
    else:
        theory_support = "CONTRADICTED"

    return {
        'phases': phases,
        'predictions': predictions,
        'theory_support': theory_support,
        'n_total': n_total,
        'n_mlp': n_mlp,
        'n_proteins': n_proteins,
    }


def main():
    if len(sys.argv) < 2:
        # Look for most recent session
        results_dir = Path("results")
        if results_dir.exists():
            sessions = sorted(results_dir.glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if sessions:
                session_dir = str(sessions[0])
                print(f"Using most recent session: {session_dir}")
            else:
                print("Usage: python sine_analyze.py <session_dir>")
                print("No sessions found in results/")
                return
        else:
            print("Usage: python sine_analyze.py <session_dir>")
            return
    else:
        session_dir = sys.argv[1]

    try:
        trajectory = load_trajectory(session_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Text analysis (always available)
    text_analysis(trajectory)

    # Phase analysis
    print("\n" + "=" * 70)
    print("DETAILED PHASE ANALYSIS")
    print("=" * 70)

    phase_results = analyze_phases(trajectory)

    print(f"\nPhase Statistics (k values):")
    print(f"  {'Phase':<15} {'k_total':>10} {'k_mlp':>10} {'k_proteins':>12}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    for phase_name, phase_data in phase_results['phases'].items():
        print(f"  {phase_name.capitalize():<15} "
              f"{phase_data['k_total_mean']:>10.2f} "
              f"{phase_data['k_mlp_mean']:>10.2f} "
              f"{phase_data['k_proteins_mean']:>12.2f}")

    print(f"\nTheory Predictions:")
    for pred_name, pred_value in phase_results['predictions'].items():
        status = "✓" if pred_value else "✗"
        print(f"  {status} {pred_name}: {pred_value}")

    print(f"\nOverall Theory Support: {phase_results['theory_support']}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        plot_path = Path(session_dir) / "analysis_plot.png"
        plot_analysis(trajectory, str(plot_path))


if __name__ == "__main__":
    main()
