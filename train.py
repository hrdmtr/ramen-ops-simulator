"""
Training script for Ramen Shop Operations RL agent.

Uses PPO from stable-baselines3 to train an agent that optimizes
staff allocation for throughput and psychological safety.
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from env import RamenShopEnv
from config import SCENARIOS, DEFAULT_SCENARIO, DEFAULT_SEED


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for Ramen Shop operations"
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help="Training scenario",
    )

    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps",
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for models",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for tensorboard logs",
    )

    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluation frequency (in timesteps)",
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency (in timesteps)",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )

    return parser.parse_args()


def make_env(scenario: str, seed: int, rank: int):
    """
    Create a single environment instance.

    Args:
        scenario: Scenario name
        seed: Base random seed
        rank: Environment rank (for multi-env training)

    Returns:
        Callable that creates the environment
    """
    def _init():
        env = RamenShopEnv(scenario=scenario, seed=seed + rank)
        env = Monitor(env)  # Wrap with Monitor for logging
        return env

    return _init


def train(args):
    """
    Train PPO agent.

    Args:
        args: Parsed command line arguments
    """
    print("="*60)
    print("Ramen Shop Operations - RL Training")
    print("="*60)
    print(f"Scenario: {args.scenario}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")

    # Create output directories
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.scenario}_{timestamp}"

    # Create training environments
    print(f"Creating {args.n_envs} training environments...")
    train_env = make_vec_env(
        make_env(args.scenario, args.seed, 0),
        n_envs=args.n_envs,
    )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = RamenShopEnv(
        scenario=args.scenario,
        seed=args.seed + 1000,  # Different seed for eval
    )
    eval_env = Monitor(eval_env)

    # Setup callbacks
    callbacks = []

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / run_name),
        log_path=str(log_dir / run_name),
        eval_freq=args.eval_freq // args.n_envs,  # Adjust for parallel envs
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=args.verbose,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjust for parallel envs
        save_path=str(output_dir / run_name / "checkpoints"),
        name_prefix="ppo_ramen",
        verbose=args.verbose,
    )
    callbacks.append(checkpoint_callback)

    callback = CallbackList(callbacks)

    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=2048 // args.n_envs,  # Adjust for parallel envs
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        verbose=args.verbose,
        tensorboard_log=str(log_dir / run_name),
        seed=args.seed,
    )

    print("\nModel architecture:")
    print(model.policy)
    print()

    # Train
    print("Starting training...\n")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final model
    final_model_path = output_dir / run_name / "final_model"
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save training info
    info_path = output_dir / run_name / "training_info.txt"
    with open(info_path, "w") as f:
        f.write(f"Scenario: {args.scenario}\n")
        f.write(f"Total timesteps: {args.total_timesteps}\n")
        f.write(f"N environments: {args.n_envs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Timestamp: {timestamp}\n")

    print(f"Training info saved to: {info_path}")

    # Cleanup
    train_env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir / run_name}")
    print(f"\nTo evaluate the model:")
    print(f"  python eval.py --model-path {final_model_path}.zip")
    print()


if __name__ == "__main__":
    args = parse_args()
    train(args)
