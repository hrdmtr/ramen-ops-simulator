"""
Gymnasium-compatible RL environment for Ramen Shop Operations.

This environment wraps the simulator and provides a standard interface
for reinforcement learning with stable-baselines3.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple, Any

from sim import RamenShopSimulator
from reward import RewardCalculator, RewardBreakdown
from config import (
    PROCESSES, ACTION_SPACE, EPISODE_STEPS, SCENARIOS, DEFAULT_SCENARIO,
    MAX_ARRIVALS_PER_STEP, MAX_WIP_PER_PROCESS, MAX_IDLE_TIME, MAX_PENDING_TIME,
    DECISION_INTERVAL, DEFAULT_SEED
)


class RamenShopEnv(gym.Env):
    """
    Gymnasium environment for Ramen Shop staff allocation optimization.

    Observation Space (Box):
        - arrivals_last_step: Number of orders that arrived in last 30s (normalized)
        - wip_per_process: WIP count for each process (normalized)
        - bottleneck_wip: Maximum WIP across processes (normalized)
        - float_idle_time: Cumulative idle time for float staff (normalized)
        - current_action: One-hot encoding of current float action
        - pending_action: One-hot encoding of pending action (or zeros if none)
        - pending_time_left: Time until pending action applies (normalized)

    Action Space (Discrete):
        Corresponds to ACTION_SPACE in config

    Reward:
        Multi-objective reward focusing on throughput, wait time, idle reduction,
        and instruction stability (see reward.py)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenario: str = DEFAULT_SCENARIO,
        seed: Optional[int] = DEFAULT_SEED,
        render_mode: Optional[str] = None,
        enable_logging: bool = False,
    ):
        """
        Initialize environment.

        Args:
            scenario: Scenario name from SCENARIOS
            seed: Random seed
            render_mode: Rendering mode (currently only "human" for text output)
            enable_logging: Enable detailed event logging
        """
        super().__init__()

        self.scenario = scenario
        self.scenario_config = SCENARIOS[scenario]
        self.render_mode = render_mode
        self._seed = seed
        self.enable_logging = enable_logging

        # Initialize simulator and reward calculator
        self.sim = RamenShopSimulator(scenario=scenario, seed=seed, enable_logging=enable_logging)
        self.reward_calc = RewardCalculator()

        # Episode tracking
        self.current_step = 0
        self.episode_rewards: list[RewardBreakdown] = []
        self.episode_metrics: list[Dict] = []
        self.last_arrivals = 0

        # Define action space
        self.action_space = spaces.Discrete(len(ACTION_SPACE))

        # Define observation space
        self.observation_space = self._create_observation_space()

        # Previous action for change penalty
        self.previous_action = 0  # DO_NOTHING

    def _create_observation_space(self) -> spaces.Box:
        """
        Create observation space with proper dimensionality.

        Observation vector:
        - arrivals_last_step (1)
        - wip_per_process (len(PROCESSES))
        - bottleneck_wip (1)
        - float_idle_time (1)
        - current_action (len(ACTION_SPACE))
        - pending_action (len(ACTION_SPACE))
        - pending_time_left (1)
        """
        num_features = (
            1  # arrivals
            + len(PROCESSES)  # wip per process
            + 1  # bottleneck
            + 1  # idle time
            + len(ACTION_SPACE)  # current action (one-hot)
            + len(ACTION_SPACE)  # pending action (one-hot)
            + 1  # pending time
        )

        # All features normalized to [0, 1]
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_features,),
            dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation from simulator state.

        Returns:
            Normalized observation vector
        """
        state = self.sim.get_state()

        obs = []

        # 1. Arrivals last step (normalized)
        obs.append(min(self.last_arrivals / MAX_ARRIVALS_PER_STEP, 1.0))

        # 2. WIP per process (normalized)
        for process in PROCESSES:
            wip = state.get(f"{process.id}_wait", 0) + state.get(f"{process.id}_in_progress", 0)
            obs.append(min(wip / MAX_WIP_PER_PROCESS, 1.0))

        # 3. Bottleneck WIP (normalized)
        bottleneck = state.get("bottleneck_wip", 0)
        obs.append(min(bottleneck / MAX_WIP_PER_PROCESS, 1.0))

        # 4. Float idle time (normalized)
        idle_time = state.get("float_idle_time", 0.0)
        obs.append(min(idle_time / MAX_IDLE_TIME, 1.0))

        # 5. Current action (one-hot)
        current_action_name = state.get("current_action", "DO_NOTHING")
        current_action_idx = ACTION_SPACE.index(current_action_name) if current_action_name in ACTION_SPACE else 0
        current_action_onehot = np.zeros(len(ACTION_SPACE))
        current_action_onehot[current_action_idx] = 1.0
        obs.extend(current_action_onehot)

        # 6. Pending action (one-hot or zeros)
        pending_action_name = state.get("pending_action")
        if pending_action_name and pending_action_name in ACTION_SPACE:
            pending_action_idx = ACTION_SPACE.index(pending_action_name)
            pending_action_onehot = np.zeros(len(ACTION_SPACE))
            pending_action_onehot[pending_action_idx] = 1.0
        else:
            pending_action_onehot = np.zeros(len(ACTION_SPACE))
        obs.extend(pending_action_onehot)

        # 7. Pending time left (normalized)
        pending_time = state.get("pending_time_left", 0.0)
        obs.append(min(pending_time / MAX_PENDING_TIME, 1.0))

        return np.array(obs, dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (e.g., scenario selection)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Handle scenario override
        if options and "scenario" in options:
            self.scenario = options["scenario"]
            self.scenario_config = SCENARIOS[self.scenario]

        # Reset simulator
        if seed is not None:
            self._seed = seed
            self.sim = RamenShopSimulator(scenario=self.scenario, seed=seed, enable_logging=self.enable_logging)
        else:
            self.sim.reset()

        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []
        self.last_arrivals = 0
        self.previous_action = 0

        # Get initial observation
        obs = self._get_observation()

        info = {
            "scenario": self.scenario,
            "episode_step": self.current_step,
        }

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action index (from 0 to len(ACTION_SPACE)-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Convert action index to action name
        action_name = ACTION_SPACE[action]

        # Step simulator
        metrics = self.sim.step(
            action=action_name,
            arrival_rate=self.scenario_config["arrival_rate"],
            duration=DECISION_INTERVAL,
        )

        # Track arrivals
        self.last_arrivals = metrics.get("arrivals", 0)

        # Calculate reward
        metrics["action_changed"] = (action != self.previous_action)
        reward_breakdown = self.reward_calc.calculate_from_metrics(metrics)
        reward = reward_breakdown.total

        # Update tracking
        self.episode_rewards.append(reward_breakdown)
        self.episode_metrics.append(metrics)
        self.previous_action = action
        self.current_step += 1

        # Get next observation
        obs = self._get_observation()

        # Check termination
        terminated = self.current_step >= EPISODE_STEPS
        truncated = False

        # Build info dict
        info = {
            "episode_step": self.current_step,
            "metrics": metrics,
            "reward_breakdown": reward_breakdown.to_dict(),
        }

        # Add episode summary on termination
        if terminated:
            from reward import compute_episode_metrics
            episode_summary = compute_episode_metrics(self.episode_rewards)
            sim_summary = self.sim.get_metrics_summary()
            info["episode"] = {**episode_summary, **sim_summary}

            # Save logs if logging is enabled
            if self.enable_logging and self.sim.logger:
                log_files = self.sim.logger.save_to_csv(prefix="episode_log")
                info["log_files"] = log_files
                print(f"\n📁 Logs saved:")
                for log_type, filename in log_files.items():
                    print(f"   {log_type}: {filename}")
                self.sim.logger.print_summary()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render environment state (text-based for now)."""
        if self.render_mode != "human":
            return None

        state = self.sim.get_state()
        output = [
            f"\n{'='*60}",
            f"Step {self.current_step}/{EPISODE_STEPS} | Time: {state['current_time']:.1f}s",
            f"{'='*60}",
            f"WIP Total: {state['wip_total']} | Completed: {state['total_completed']}",
        ]

        for process in PROCESSES:
            wait = state.get(f"{process.id}_wait", 0)
            in_prog = state.get(f"{process.id}_in_progress", 0)
            output.append(f"  {process.id}: wait={wait}, in_progress={in_prog}")

        output.append(f"Float Action: {state['current_action']}")
        if state['pending_action']:
            output.append(f"Pending: {state['pending_action']} (in {state['pending_time_left']:.1f}s)")

        output.append(f"{'='*60}\n")

        print("\n".join(output))
        return "\n".join(output)

    def close(self):
        """Clean up resources."""
        pass


# Register environment with Gymnasium
gym.register(
    id="RamenShop-v0",
    entry_point="env:RamenShopEnv",
    max_episode_steps=EPISODE_STEPS,
)


if __name__ == "__main__":
    # Test environment
    print("Testing RamenShopEnv...\n")

    env = RamenShopEnv(scenario="base", seed=42, render_mode="human")

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Actions: {ACTION_SPACE}\n")

    # Run a few steps with random actions
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs[:5]}... (showing first 5)\n")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step+1}:")
        print(f"  Action: {ACTION_SPACE[action]}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Reward breakdown: {info['reward_breakdown']}")
        print(f"  Metrics: completed={info['metrics']['completed']}, wip={info['metrics']['wip_total']}")

        env.render()

        if terminated or truncated:
            print("Episode finished!")
            if "episode" in info:
                print(f"Episode summary: {info['episode']}")
            break

    env.close()
    print("\n✓ Environment test completed successfully!")
