"""
GENREG Humanoid-v5 Checkpoint Tester
Load and visualize the best genome from a checkpoint.
"""

import pickle
import gymnasium as gym
from pathlib import Path
import genreg_config as cfg


def scan_checkpoints():
    """Scan checkpoints folder and return available sessions"""
    checkpoint_root = Path("checkpoints")
    if not checkpoint_root.exists():
        print("No checkpoints folder found!")
        return []

    sessions = []
    for session_dir in sorted(checkpoint_root.iterdir()):
        if session_dir.is_dir() and session_dir.name.startswith("session_"):
            # Find checkpoint files in this session
            pkl_files = list(session_dir.glob("*.pkl"))
            if pkl_files:
                # Get the latest checkpoint (by name or modification time)
                latest = max(pkl_files, key=lambda p: p.stat().st_mtime)
                sessions.append({
                    'name': session_dir.name,
                    'path': session_dir,
                    'latest_checkpoint': latest,
                    'num_checkpoints': len(pkl_files),
                })

    return sessions


def select_session(sessions):
    """Let user select a session"""
    print("\n" + "=" * 60)
    print("Available Sessions")
    print("=" * 60)

    for i, session in enumerate(sessions):
        print(f"  [{i+1}] {session['name']}")
        print(f"      Latest: {session['latest_checkpoint'].name}")
        print(f"      Checkpoints: {session['num_checkpoints']}")
        print()

    while True:
        try:
            choice = input("Select session number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
            print("Invalid selection, try again.")
        except ValueError:
            print("Please enter a number.")


def load_checkpoint(checkpoint_path):
    """Load population from checkpoint"""
    print(f"\nLoading: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        state = pickle.load(f)

    print(f"  Generation: {state['generation']}")
    print(f"  Genomes: {len(state['genomes'])}")
    print(f"  Best fitness ever: {state['best_fitness_ever']:.2f}")
    print(f"  Best reward ever: {state['best_reward_ever']:.2f}")

    if 'energy_baseline' in state:
        print(f"  Energy baseline: {state['energy_baseline']:.4f}")

    if 'distance_record' in state:
        print(f"  Distance record: {state['distance_record']:.2f}")

    return state


def get_best_genome(state):
    """Get the best genome from loaded state"""
    genomes = state['genomes']
    # Sort by fitness and return best
    best = max(genomes, key=lambda g: g.get_fitness())
    print(f"\nBest genome:")
    print(f"  ID: {best.id}")
    print(f"  Age: {best.age}")
    print(f"  Fitness: {best.get_fitness():.2f}")
    print(f"  Max distance: {best.max_distance:.2f}")
    print(f"  Longest episode: {best.longest_episode} steps")
    return best


def run_visualization(genome, num_episodes=3):
    """Run the genome with rendering enabled"""
    print("\n" + "=" * 60)
    print("Starting Visualization")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")

    # Create environment with rendering
    env = gym.make(cfg.ENV_NAME, render_mode="human")

    try:
        for ep in range(num_episodes):
            observation, info = env.reset()
            genome.episode_reset()

            episode_reward = 0.0
            step = 0
            start_x = info.get('x_position', 0.0)
            max_distance = 0.0

            while True:
                # Get action from genome
                action = genome.get_action(observation)

                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1

                # Track distance
                current_x = info.get('x_position', 0.0)
                distance = current_x - start_x
                max_distance = max(max_distance, distance)

                if terminated or truncated:
                    break

            print(f"Episode {ep+1}: {step} steps, reward: {episode_reward:.2f}, distance: {max_distance:.2f}")

    except KeyboardInterrupt:
        print("\n\nVisualization stopped.")

    finally:
        env.close()


def main():
    print("=" * 60)
    print("GENREG Humanoid-v5 Checkpoint Tester")
    print("=" * 60)

    # Scan for checkpoints
    sessions = scan_checkpoints()

    if not sessions:
        print("No checkpoint sessions found in ./checkpoints/")
        return

    # Let user select
    session = select_session(sessions)
    if session is None:
        print("Exiting.")
        return

    # Load checkpoint
    state = load_checkpoint(session['latest_checkpoint'])

    # Get best genome
    genome = get_best_genome(state)

    # Ask for number of episodes
    try:
        num_eps = input("\nNumber of episodes to run (default 3): ").strip()
        num_eps = int(num_eps) if num_eps else 3
    except ValueError:
        num_eps = 3

    # Run visualization
    run_visualization(genome, num_episodes=num_eps)

    print("\nDone!")


if __name__ == "__main__":
    main()
