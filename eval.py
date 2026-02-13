"""
Evaluation script for trained Ramen Shop Operations RL agent.

Evaluates a trained model on multiple episodes and outputs detailed metrics
including reward breakdown, wait times, completed orders, idle time, etc.
"""

import sys
import os
from pathlib import Path
import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env import RamenShopEnv
from config import SCENARIOS, DEFAULT_SCENARIO, DEFAULT_SEED, ACTION_SPACE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent for Ramen Shop operations"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help="Evaluation scenario",
    )

    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for evaluation",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also evaluate random baseline",
    )

    return parser.parse_args()


def evaluate_model(
    model: PPO,
    env: RamenShopEnv,
    n_episodes: int,
    deterministic: bool = True,
    render: bool = False,
) -> Dict:
    """
    Evaluate model over multiple episodes.

    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Use deterministic actions
        render: Render environment

    Returns:
        Dictionary with aggregated metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_completed = []
    episode_wait_times = []
    episode_idle_times = []
    episode_action_changes = []

    step_data = []  # Detailed step-by-step data

    for ep in range(n_episodes):
        obs, info = env.reset(seed=env.unwrapped._seed + ep)
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            # Collect step data
            step_data.append({
                "episode": ep,
                "step": episode_length,
                "action": ACTION_SPACE[int(action)],
                "reward": reward,
                **info["reward_breakdown"],
                **info["metrics"],
            })

            done = terminated or truncated

        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if "episode" in info:
            ep_info = info["episode"]
            episode_completed.append(ep_info.get("total_completed", 0))
            episode_wait_times.append(ep_info.get("avg_wait_time", 0.0))
            episode_idle_times.append(ep_info.get("total_idle_time", 0.0))
            episode_action_changes.append(ep_info.get("action_changes", 0))

        print(f"Episode {ep+1}/{n_episodes}: "
              f"reward={episode_reward:.2f}, "
              f"completed={episode_completed[-1]}, "
              f"wait_time={episode_wait_times[-1]:.1f}s")

    # Aggregate statistics
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_completed": np.mean(episode_completed),
        "std_completed": np.std(episode_completed),
        "mean_wait_time": np.mean(episode_wait_times),
        "std_wait_time": np.std(episode_wait_times),
        "mean_idle_time": np.mean(episode_idle_times),
        "std_idle_time": np.std(episode_idle_times),
        "mean_action_changes": np.mean(episode_action_changes),
        "std_action_changes": np.std(episode_action_changes),
        "n_episodes": n_episodes,
        "step_data": pd.DataFrame(step_data),
    }

    return results


def evaluate_baseline(
    env: RamenShopEnv,
    n_episodes: int,
) -> Dict:
    """
    Evaluate random baseline policy.

    Args:
        env: Environment
        n_episodes: Number of episodes

    Returns:
        Dictionary with aggregated metrics
    """
    print("\nEvaluating random baseline...")

    episode_rewards = []
    episode_completed = []
    episode_wait_times = []
    episode_idle_times = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=env.unwrapped._seed + ep)
        done = False
        episode_reward = 0.0

        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)

        if "episode" in info:
            ep_info = info["episode"]
            episode_completed.append(ep_info.get("total_completed", 0))
            episode_wait_times.append(ep_info.get("avg_wait_time", 0.0))
            episode_idle_times.append(ep_info.get("total_idle_time", 0.0))

    return {
        "mean_reward": np.mean(episode_rewards),
        "mean_completed": np.mean(episode_completed),
        "mean_wait_time": np.mean(episode_wait_times),
        "mean_idle_time": np.mean(episode_idle_times),
    }


def print_results(results: Dict, baseline: Dict = None):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    print(f"\nPerformance over {results['n_episodes']} episodes:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Completed Orders: {results['mean_completed']:.1f} ± {results['std_completed']:.1f}")
    print(f"  Mean Wait Time: {results['mean_wait_time']:.1f}s ± {results['std_wait_time']:.1f}s")
    print(f"  Mean Idle Time: {results['mean_idle_time']:.1f}s ± {results['std_idle_time']:.1f}s")
    print(f"  Mean Action Changes: {results['mean_action_changes']:.1f} ± {results['std_action_changes']:.1f}")

    if baseline:
        print("\n" + "-"*60)
        print("Baseline Comparison (Random Policy):")
        print("-"*60)
        print(f"  Baseline Reward: {baseline['mean_reward']:.2f}")
        print(f"  Baseline Completed: {baseline['mean_completed']:.1f}")
        print(f"  Baseline Wait Time: {baseline['mean_wait_time']:.1f}s")
        print(f"  Baseline Idle Time: {baseline['mean_idle_time']:.1f}s")

        print("\n  Improvement:")
        reward_improvement = (results['mean_reward'] - baseline['mean_reward']) / abs(baseline['mean_reward']) * 100
        completed_improvement = (results['mean_completed'] - baseline['mean_completed']) / baseline['mean_completed'] * 100
        print(f"    Reward: {reward_improvement:+.1f}%")
        print(f"    Completed Orders: {completed_improvement:+.1f}%")

    print("\n" + "="*60 + "\n")


def save_results(results: Dict, args, output_dir: Path):
    """Save evaluation results to CSV and summary file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(exist_ok=True)

    # Save step-by-step data
    step_data_path = output_dir / f"eval_steps_{args.scenario}_{timestamp}.csv"
    results["step_data"].to_csv(step_data_path, index=False)
    print(f"Step data saved to: {step_data_path}")

    # Save summary
    summary_path = output_dir / f"eval_summary_{args.scenario}_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Scenario: {args.scenario}\n")
        f.write(f"N Episodes: {args.n_episodes}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"  Mean Completed: {results['mean_completed']:.1f} ± {results['std_completed']:.1f}\n")
        f.write(f"  Mean Wait Time: {results['mean_wait_time']:.1f}s ± {results['std_wait_time']:.1f}s\n")
        f.write(f"  Mean Idle Time: {results['mean_idle_time']:.1f}s ± {results['std_idle_time']:.1f}s\n")
        f.write(f"  Mean Action Changes: {results['mean_action_changes']:.1f} ± {results['std_action_changes']:.1f}\n")

    print(f"Summary saved to: {summary_path}")


def main(args):
    """Main evaluation function."""
    print("="*60)
    print("Ramen Shop Operations - Model Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.n_episodes}")
    print("="*60 + "\n")

    # Load model
    print("Loading model...")
    try:
        model = PPO.load(args.model_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return

    # Create environment
    render_mode = "human" if args.render else None
    env = RamenShopEnv(
        scenario=args.scenario,
        seed=args.seed,
        render_mode=render_mode,
    )

    # Evaluate trained model
    print(f"Evaluating on {args.scenario} scenario...\n")
    results = evaluate_model(
        model,
        env,
        args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
    )

    # Evaluate baseline if requested
    baseline_results = None
    if args.baseline:
        baseline_results = evaluate_baseline(env, args.n_episodes)

    # Print results
    print_results(results, baseline_results)

    # Save results
    output_dir = Path(args.output_dir)
    save_results(results, args, output_dir)

    # Cleanup
    env.close()

    print("Evaluation completed!\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
