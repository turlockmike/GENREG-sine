"""
Sine Test Live Visualization
Shows real sine wave vs best genome prediction side by side.
Runs forever, updating when new checkpoints appear.
"""

import time
import numpy as np
from pathlib import Path
import pickle

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    print("ERROR: matplotlib required. Install with: pip install matplotlib")
    exit(1)

from . import sine_config as cfg


def find_latest_session():
    """Find the most recent session directory."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    sessions = sorted(results_dir.glob("session_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def find_latest_checkpoint(session_dir):
    """Find the checkpoint file in a session."""
    if session_dir is None:
        return None

    best_ckpt = session_dir / "checkpoint_best.pkl"
    if best_ckpt.exists():
        return best_ckpt

    return None


def load_checkpoint(path):
    """Load a checkpoint and return the best genome."""
    with open(path, 'rb') as f:
        state = pickle.load(f)

    if 'genome' in state:
        return {
            'genome': state['genome'],
            'generation': state['generation'],
            'best_mse': state.get('best_mse', state['genome'].mse),
            'saturation_history': state.get('saturation_history', []),
        }

    genomes = state['genomes']
    genomes.sort(key=lambda g: g.get_fitness(), reverse=True)
    best = genomes[0]

    return {
        'genome': best,
        'generation': state['generation'],
        'best_mse': state.get('best_mse_ever', best.mse),
        'saturation_history': state.get('saturation_history', []),
    }


def get_predictions(genome, x_values):
    """Get predictions from a genome."""
    import torch

    predictions = []
    with torch.no_grad():
        for x in x_values:
            x_tensor = torch.tensor([x], device=genome.device, dtype=torch.float32)
            pred = genome.controller.forward(x_tensor, track=False)
            predictions.append(pred.item())

    return np.array(predictions)


class LiveVisualizer:
    def __init__(self):
        self.session_dir = None
        self.last_checkpoint_path = None
        self.last_checkpoint_mtime = 0

        # Test points
        self.x_values = np.linspace(-2 * np.pi, 2 * np.pi, 200)
        self.y_true = np.sin(self.x_values)

        # History for plotting
        self.gen_history = []
        self.mse_history = []
        self.k_mlp_history = []
        self.k_prot_history = []

        # Setup figure
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10))

        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main comparison plot (top left, spans 2 cols)
        self.ax_main = self.fig.add_subplot(gs[0, :2])
        self.ax_main.set_title('Sine Wave Approximation', fontsize=12, fontweight='bold')
        self.ax_main.set_xlabel('x')
        self.ax_main.set_ylabel('y')
        self.line_true, = self.ax_main.plot(self.x_values, self.y_true, 'g-', linewidth=2, label='True sin(x)')
        self.line_pred, = self.ax_main.plot(self.x_values, self.y_true, 'c-', linewidth=2, label='Predicted')
        self.ax_main.legend(loc='upper right')
        self.ax_main.set_ylim(-1.5, 1.5)
        self.ax_main.grid(True, alpha=0.3)

        # Error plot (top right)
        self.ax_error = self.fig.add_subplot(gs[0, 2])
        self.ax_error.set_title('Error Distribution', fontsize=12)
        self.ax_error.set_xlabel('x')
        self.ax_error.set_ylabel('Error')
        self.line_error, = self.ax_error.plot(self.x_values, np.zeros_like(self.x_values), 'r-', linewidth=1.5)
        self.ax_error.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        self.ax_error.set_ylim(-1, 1)
        self.ax_error.grid(True, alpha=0.3)

        # MSE history plot (middle left)
        self.ax_mse = self.fig.add_subplot(gs[1, 0])
        self.ax_mse.set_title('MSE Over Time', fontsize=12)
        self.ax_mse.set_xlabel('Generation')
        self.ax_mse.set_ylabel('MSE (log)')
        self.line_mse, = self.ax_mse.plot([], [], 'y-', linewidth=2)
        self.ax_mse.set_yscale('log')
        self.ax_mse.grid(True, alpha=0.3)

        # Saturation history plot (middle center)
        self.ax_sat_hist = self.fig.add_subplot(gs[1, 1])
        self.ax_sat_hist.set_title('Saturation Over Time (k)', fontsize=12)
        self.ax_sat_hist.set_xlabel('Generation')
        self.ax_sat_hist.set_ylabel('Saturated Units (k)')
        self.line_k_mlp, = self.ax_sat_hist.plot([], [], 'b-', linewidth=2, label='MLP')
        self.line_k_prot, = self.ax_sat_hist.plot([], [], 'g-', linewidth=2, label='Proteins')
        self.ax_sat_hist.legend(loc='upper left')
        self.ax_sat_hist.grid(True, alpha=0.3)

        # Architecture info (middle right)
        self.ax_arch = self.fig.add_subplot(gs[1, 2])
        self.ax_arch.set_title('Architecture', fontsize=12)
        self.ax_arch.axis('off')
        self.arch_text = self.ax_arch.text(0.05, 0.95, '', transform=self.ax_arch.transAxes,
                                            fontsize=10, verticalalignment='top',
                                            fontfamily='monospace')

        # Stats (bottom left)
        self.ax_stats = self.fig.add_subplot(gs[2, 0])
        self.ax_stats.set_title('Current Stats', fontsize=12)
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.05, 0.95, 'Waiting...',
                                              transform=self.ax_stats.transAxes,
                                              fontsize=10, verticalalignment='top',
                                              fontfamily='monospace')

        # Saturation bars (bottom center)
        self.ax_sat = self.fig.add_subplot(gs[2, 1])
        self.ax_sat.set_title('Saturation Status', fontsize=12)
        self.ax_sat.axis('off')
        self.sat_text = self.ax_sat.text(0.05, 0.95, '',
                                          transform=self.ax_sat.transAxes,
                                          fontsize=10, verticalalignment='top',
                                          fontfamily='monospace')

        # Theory validation (bottom right)
        self.ax_theory = self.fig.add_subplot(gs[2, 2])
        self.ax_theory.set_title('Theory Validation', fontsize=12)
        self.ax_theory.axis('off')
        self.theory_text = self.ax_theory.text(0.05, 0.95, '',
                                                transform=self.ax_theory.transAxes,
                                                fontsize=10, verticalalignment='top',
                                                fontfamily='monospace')

        plt.tight_layout()

    def update(self, frame):
        """Update the visualization."""
        # Find latest session if not set
        if self.session_dir is None:
            self.session_dir = find_latest_session()
            if self.session_dir is None:
                self.stats_text.set_text('No sessions found.\nRun sine_train.py first.')
                return []

        # Find latest checkpoint
        checkpoint_path = find_latest_checkpoint(self.session_dir)
        if checkpoint_path is None:
            self.stats_text.set_text(f'Session: {self.session_dir.name}\nNo checkpoints yet...')
            return []

        # Check if checkpoint has changed
        current_mtime = checkpoint_path.stat().st_mtime
        if checkpoint_path == self.last_checkpoint_path and current_mtime == self.last_checkpoint_mtime:
            return []

        self.last_checkpoint_path = checkpoint_path
        self.last_checkpoint_mtime = current_mtime

        try:
            # Load checkpoint
            data = load_checkpoint(checkpoint_path)
            genome = data['genome']
            generation = data['generation']
            sat_history = data.get('saturation_history', [])

            # Get predictions
            predictions = get_predictions(genome, self.x_values)

            # Calculate error
            error = predictions - self.y_true
            mse = np.mean(error ** 2)

            # Update history
            self.gen_history.append(generation)
            self.mse_history.append(mse)
            self.k_mlp_history.append(genome.k_mlp)
            self.k_prot_history.append(genome.k_proteins)

            # Update main plot
            self.line_pred.set_ydata(predictions)

            # Update error plot
            self.line_error.set_ydata(error)

            # Update MSE history
            self.line_mse.set_data(self.gen_history, self.mse_history)
            self.ax_mse.relim()
            self.ax_mse.autoscale_view()

            # Update saturation history
            self.line_k_mlp.set_data(self.gen_history, self.k_mlp_history)
            self.line_k_prot.set_data(self.gen_history, self.k_prot_history)
            self.ax_sat_hist.relim()
            self.ax_sat_hist.autoscale_view()

            # Architecture info
            arch_str = f"""INPUT EXPANSION
True signals:  {cfg.TRUE_SIGNAL_SIZE}
Noise signals: {cfg.NOISE_SIGNAL_SIZE}
Total inputs:  {cfg.EXPANSION_SIZE}

MLP NETWORK
{cfg.EXPANSION_SIZE} -> {cfg.HIDDEN_SIZE} -> {cfg.OUTPUT_SIZE}
Compression: {cfg.EXPANSION_SIZE//cfg.HIDDEN_SIZE}:1

PROTEINS
Units: {len(genome.proteins)}"""
            self.arch_text.set_text(arch_str)

            # Update stats
            stats_str = f"""Generation: {generation}

MSE:  {mse:.6f}
RMSE: {np.sqrt(mse):.6f}
Max Error: {np.max(np.abs(error)):.4f}

Trust: {genome.trust:.4f}
Fitness: {genome.get_fitness():.6f}"""
            self.stats_text.set_text(stats_str)

            # Update saturation display
            k_mlp = genome.k_mlp
            k_prot = genome.k_proteins
            n_mlp = cfg.HIDDEN_SIZE
            n_prot = len(genome.proteins) if genome.proteins else 0
            k_total = k_mlp + k_prot
            n_total = n_mlp + n_prot

            mlp_bar = '#' * k_mlp + '.' * (n_mlp - k_mlp)
            prot_bar = '#' * k_prot + '.' * (n_prot - k_prot)

            sat_str = f"""MLP Hidden ({n_mlp} neurons)
k={k_mlp}/{n_mlp} [{mlp_bar}]

Proteins ({n_prot} units)
k={k_prot}/{n_prot} [{prot_bar}]

TOTAL: {k_total}/{n_total}
       ({100*k_total/max(1,n_total):.0f}% saturated)"""
            self.sat_text.set_text(sat_str)

            # Theory validation
            if len(self.k_mlp_history) >= 10:
                early_k = np.mean(self.k_mlp_history[:len(self.k_mlp_history)//3])
                late_k = np.mean(self.k_mlp_history[-len(self.k_mlp_history)//3:])
                k_increasing = late_k > early_k

                theory_str = f"""THEORY PREDICTIONS

1. k should INCREASE
   Early k_avg: {early_k:.1f}
   Late k_avg:  {late_k:.1f}
   {'[OK] INCREASING' if k_increasing else '[X] NOT increasing'}

2. High k = noise filtering
   Current: {k_total}/{n_total}
   {'[OK] HIGH' if k_total/max(1,n_total) > 0.5 else '[?] LOW'}

3. Fitness improves
   {'[OK] MSE dropping' if len(self.mse_history) > 1 and self.mse_history[-1] < self.mse_history[0] else '...'}"""
            else:
                theory_str = "Collecting data...\n(need 10+ updates)"
            self.theory_text.set_text(theory_str)

            # Update title
            self.fig.suptitle(
                f'EMERGENT HYBRID COMPUTATION TEST | Gen {generation} | MSE: {mse:.6f} | k={k_total}/{n_total}',
                fontsize=14, fontweight='bold'
            )

        except Exception as e:
            self.stats_text.set_text(f'Error:\n{e}')

        return []

    def run(self):
        """Run the live visualization forever."""
        print("=" * 60)
        print("SINE TEST LIVE VISUALIZATION")
        print("=" * 60)
        print(f"Architecture: {cfg.TRUE_SIGNAL_SIZE} true + {cfg.NOISE_SIGNAL_SIZE} noise = {cfg.EXPANSION_SIZE} inputs")
        print(f"Compression: {cfg.EXPANSION_SIZE} -> {cfg.HIDDEN_SIZE} ({cfg.EXPANSION_SIZE//cfg.HIDDEN_SIZE}:1)")
        print("=" * 60)
        print("Monitoring for checkpoints...")
        print("Press Ctrl+C to exit")
        print("=" * 60)

        ani = FuncAnimation(self.fig, self.update, interval=2000, blit=False, cache_frame_data=False)
        plt.show()


def main():
    viz = LiveVisualizer()
    viz.run()


if __name__ == "__main__":
    main()
