"""
Reward calculation module for RL training.

Computes multi-objective reward focusing on:
1. Throughput (completed bowls)
2. Customer satisfaction (wait time)
3. Staff psychological safety (reducing idle/cognitive freeze)
4. Operational stability (avoiding frequent instruction changes)
"""

from typing import Dict, Tuple
from dataclasses import dataclass
from config import REWARD_WEIGHTS, WIP_OVERFLOW_THRESHOLD


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""
    completed_bowls: float = 0.0
    wait_time_penalty: float = 0.0
    wip_overflow_penalty: float = 0.0
    idle_penalty: float = 0.0
    change_penalty: float = 0.0
    dish_shortage_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "reward_completed": self.completed_bowls,
            "reward_wait": self.wait_time_penalty,
            "reward_wip_overflow": self.wip_overflow_penalty,
            "reward_idle": self.idle_penalty,
            "reward_change": self.change_penalty,
            "reward_dish_shortage": self.dish_shortage_penalty,
            "reward_total": self.total,
        }


class RewardCalculator:
    """
    Calculates reward based on simulation metrics and RL objectives.

    Objectives:
    - Maximize throughput (completed orders)
    - Minimize wait time (customer satisfaction)
    - Minimize idle time (psychological safety)
    - Avoid WIP overflow (quality control)
    - Penalize frequent action changes (operational stability)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize reward calculator.

        Args:
            weights: Custom reward weights (uses config defaults if None)
        """
        self.weights = weights if weights is not None else REWARD_WEIGHTS.copy()

    def calculate(
        self,
        completed_bowls: int,
        avg_wait_time: float,
        wip_total: int,
        idle_time: float,
        action_changed: bool,
        clean_dishes: int = 30,
        empty_seats: int = 0,
        step_duration: float = 30.0,
    ) -> RewardBreakdown:
        """
        Calculate reward for a single step.

        Args:
            completed_bowls: Number of orders completed this step
            avg_wait_time: Average wait time for orders in system (seconds)
            wip_total: Total work-in-progress (orders)
            idle_time: Float staff idle time this step (seconds)
            action_changed: Whether action changed from previous step
            clean_dishes: Number of clean dishes available
            empty_seats: Number of empty seats (potential customers)
            step_duration: Length of time step (seconds)

        Returns:
            RewardBreakdown with component rewards and total
        """
        breakdown = RewardBreakdown()

        # 1. Completed bowls (positive reward)
        breakdown.completed_bowls = completed_bowls * self.weights["completed_bowls"]

        # 2. Wait time penalty (negative)
        # Only penalize if there are orders in the system
        if wip_total > 0:
            breakdown.wait_time_penalty = avg_wait_time * self.weights["avg_wait_time"]
        else:
            breakdown.wait_time_penalty = 0.0

        # 3. WIP overflow penalty (negative)
        # Penalize for each item over threshold
        overflow = max(0, wip_total - WIP_OVERFLOW_THRESHOLD)
        breakdown.wip_overflow_penalty = overflow * self.weights["wip_overflow"]

        # 4. Idle time penalty (negative)
        # Penalize idle time to encourage proactive helping
        breakdown.idle_penalty = idle_time * self.weights["idle_penalty"]

        # 5. Action change penalty (negative)
        # Discourage frequent changes to avoid staff confusion
        if action_changed:
            breakdown.change_penalty = self.weights["change_penalty"]
        else:
            breakdown.change_penalty = 0.0

        # 6. Dish shortage penalty (negative)
        # Critical penalty when clean dishes are running low
        # Penalize proportionally to shortage severity
        DISH_THRESHOLD = 10  # Minimum desired clean dishes
        if clean_dishes < DISH_THRESHOLD:
            shortage = DISH_THRESHOLD - clean_dishes
            breakdown.dish_shortage_penalty = shortage * self.weights["dish_shortage"]
        else:
            breakdown.dish_shortage_penalty = 0.0

        # Total reward
        breakdown.total = (
            breakdown.completed_bowls
            + breakdown.wait_time_penalty  # Already negative
            + breakdown.wip_overflow_penalty  # Already negative
            + breakdown.idle_penalty  # Already negative
            + breakdown.change_penalty  # Already negative
            + breakdown.dish_shortage_penalty  # Already negative
        )

        return breakdown

    def calculate_from_metrics(
        self,
        current_metrics: Dict,
        previous_state: Dict = None,
    ) -> RewardBreakdown:
        """
        Calculate reward from simulation metrics.

        Args:
            current_metrics: Current step metrics from simulator
            previous_state: Previous state (for computing averages)

        Returns:
            RewardBreakdown
        """
        # Extract metrics
        completed = current_metrics.get("completed", 0)
        wip_total = current_metrics.get("wip_total", 0)
        idle_time_step = current_metrics.get("idle_time_step", 0.0)
        action_changed = current_metrics.get("action_changed", False)
        clean_dishes = current_metrics.get("clean_dishes", 30)
        empty_seats = current_metrics.get("empty_seats", 0)

        # Estimate average wait time
        # For simplicity, use WIP as proxy (more WIP = longer wait)
        # In a real implementation, track actual wait times per order
        avg_wait_time = wip_total * 10.0  # Rough estimate: 10s per item in queue

        return self.calculate(
            completed_bowls=completed,
            avg_wait_time=avg_wait_time,
            wip_total=wip_total,
            idle_time=idle_time_step,
            action_changed=action_changed,
            clean_dishes=clean_dishes,
            empty_seats=empty_seats,
        )

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update reward weights dynamically.

        Args:
            new_weights: Dictionary of weight updates
        """
        for key, value in new_weights.items():
            if key in self.weights:
                self.weights[key] = value
            else:
                raise ValueError(f"Unknown reward weight: {key}")

    def get_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        return self.weights.copy()


def compute_episode_metrics(reward_history: list[RewardBreakdown]) -> Dict[str, float]:
    """
    Compute aggregate metrics over an episode.

    Args:
        reward_history: List of RewardBreakdown objects for each step

    Returns:
        Dictionary with episode-level metrics
    """
    if not reward_history:
        return {
            "episode_reward": 0.0,
            "episode_completed": 0.0,
            "episode_wait_penalty": 0.0,
            "episode_idle_penalty": 0.0,
            "episode_changes": 0.0,
        }

    return {
        "episode_reward": sum(r.total for r in reward_history),
        "episode_completed": sum(r.completed_bowls for r in reward_history),
        "episode_wait_penalty": sum(r.wait_time_penalty for r in reward_history),
        "episode_wip_overflow_penalty": sum(r.wip_overflow_penalty for r in reward_history),
        "episode_idle_penalty": sum(r.idle_penalty for r in reward_history),
        "episode_change_penalty": sum(r.change_penalty for r in reward_history),
        "num_steps": len(reward_history),
    }


if __name__ == "__main__":
    # Test reward calculation
    calc = RewardCalculator()

    print("Testing reward calculator...\n")

    # Test case 1: Good performance
    print("Case 1: High throughput, low wait time")
    r1 = calc.calculate(
        completed_bowls=3,
        avg_wait_time=50.0,
        wip_total=5,
        idle_time=0.0,
        action_changed=False,
    )
    print(f"  Reward: {r1.total:.2f}")
    print(f"  Breakdown: {r1.to_dict()}\n")

    # Test case 2: Bottleneck
    print("Case 2: Bottleneck (high WIP, long wait)")
    r2 = calc.calculate(
        completed_bowls=1,
        avg_wait_time=200.0,
        wip_total=15,
        idle_time=10.0,
        action_changed=False,
    )
    print(f"  Reward: {r2.total:.2f}")
    print(f"  Breakdown: {r2.to_dict()}\n")

    # Test case 3: Frequent changes
    print("Case 3: Frequent action changes")
    r3 = calc.calculate(
        completed_bowls=2,
        avg_wait_time=80.0,
        wip_total=8,
        idle_time=5.0,
        action_changed=True,  # Changed action
    )
    print(f"  Reward: {r3.total:.2f}")
    print(f"  Breakdown: {r3.to_dict()}\n")

    # Test episode metrics
    print("Episode metrics:")
    episode_metrics = compute_episode_metrics([r1, r2, r3])
    for key, value in episode_metrics.items():
        print(f"  {key}: {value:.2f}")
