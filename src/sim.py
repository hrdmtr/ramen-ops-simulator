"""
Discrete Event Simulation (DES) for Ramen Shop Operations.

This module implements a simple yet flexible DES that can accommodate
varying numbers of processes based on real shop observations.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from config import (
    PROCESSES, ProcessConfig, FIXED_STAFF, NUM_FLOAT_STAFF,
    ACTION_SPACE, HELP_EFFECT_MULTIPLIERS, DECISION_INTERVAL
)


class OrderStatus(Enum):
    """Order lifecycle status."""
    WAITING = "waiting"      # Waiting for process to start
    IN_PROGRESS = "in_progress"  # Currently being processed
    COMPLETED = "completed"  # Finished all processes


@dataclass
class Order:
    """Represents a single ramen order."""
    id: int
    arrival_time: float
    current_process_idx: int = 0  # Index in PROCESSES list
    status: OrderStatus = OrderStatus.WAITING
    process_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    wait_times: Dict[str, float] = field(default_factory=dict)  # process_id -> wait_time

    def current_process(self) -> Optional[ProcessConfig]:
        """Get current process config."""
        if self.current_process_idx < len(PROCESSES):
            return PROCESSES[self.current_process_idx]
        return None

    def is_complete(self) -> bool:
        """Check if order has completed all processes."""
        return self.current_process_idx >= len(PROCESSES)


@dataclass
class FloatStaff:
    """Represents a float staff member who can be assigned to different tasks."""
    id: int
    current_action: str = "DO_NOTHING"
    action_start_time: float = 0.0
    idle_time: float = 0.0  # Cumulative idle time


class RamenShopSimulator:
    """
    Discrete Event Simulation for ramen shop operations.

    Simulates:
    - Poisson order arrivals
    - Multi-stage processing (configurable via PROCESSES)
    - Fixed and float staff
    - Action delays and effects
    """

    def __init__(self, scenario: str = "base", seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            scenario: Scenario name from config.SCENARIOS
            seed: Random seed for reproducibility
        """
        self.scenario = scenario
        self.rng = np.random.RandomState(seed)

        # Time tracking
        self.current_time: float = 0.0

        # Orders
        self.order_counter: int = 0
        self.orders: List[Order] = []

        # Staff
        self.float_staff = FloatStaff(id=0)

        # Action tracking (for delay)
        self.pending_action: Optional[str] = None
        self.pending_action_time_left: float = 0.0

        # Metrics
        self.completed_orders: List[Order] = []
        self.total_idle_time: float = 0.0
        self.action_change_count: int = 0
        self.last_action: str = "DO_NOTHING"

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.current_time = 0.0
        self.order_counter = 0
        self.orders = []
        self.float_staff = FloatStaff(id=0)
        self.pending_action = None
        self.pending_action_time_left = 0.0
        self.completed_orders = []
        self.total_idle_time = 0.0
        self.action_change_count = 0
        self.last_action = "DO_NOTHING"

    def set_float_action(self, action: str, delay: float = 10.0) -> None:
        """
        Set float staff action with delay.

        Args:
            action: Action name from ACTION_SPACE
            delay: Time until action takes effect (seconds)
        """
        assert action in ACTION_SPACE, f"Invalid action: {action}"
        self.pending_action = action
        self.pending_action_time_left = delay

    def _apply_pending_action(self) -> None:
        """Apply pending action if delay has elapsed."""
        if self.pending_action is not None:
            new_action = self.pending_action
            if new_action != self.float_staff.current_action:
                self.action_change_count += 1
                self.last_action = self.float_staff.current_action

            self.float_staff.current_action = new_action
            self.float_staff.action_start_time = self.current_time
            self.pending_action = None
            self.pending_action_time_left = 0.0

    def _generate_arrivals(self, duration: float, arrival_rate: float) -> int:
        """Generate Poisson arrivals for the time step."""
        expected = arrival_rate * duration
        num_arrivals = self.rng.poisson(expected)

        for _ in range(num_arrivals):
            # Distribute arrivals uniformly within the time step
            arrival_offset = self.rng.uniform(0, duration)
            order = Order(
                id=self.order_counter,
                arrival_time=self.current_time + arrival_offset
            )
            self.orders.append(order)
            self.order_counter += 1

        return num_arrivals

    def _get_process_capacity(self, process_id: str) -> float:
        """
        Calculate effective processing capacity for a process.

        Takes into account fixed staff and float staff help.

        Returns:
            Capacity multiplier (1.0 = base, higher = faster)
        """
        # Base capacity from fixed staff
        base_capacity = FIXED_STAFF.get(process_id, 0)

        # Check if float staff is helping
        help_action = f"{process_id.upper()}_HELP"
        if self.float_staff.current_action == help_action:
            multiplier = HELP_EFFECT_MULTIPLIERS.get(help_action, 1.0)
            return base_capacity * multiplier

        return float(base_capacity)

    def _process_orders(self, duration: float) -> int:
        """
        Process orders through the pipeline.

        Simplified model:
        - Each process has a mean completion time
        - If capacity > 0, orders are processed
        - Orders advance through processes sequentially

        Returns:
            Number of orders completed in this step
        """
        completed_count = 0

        # Process each stage
        for process_idx, process in enumerate(PROCESSES):
            capacity = self._get_process_capacity(process.id)

            if capacity <= 0:
                continue  # No one working on this process

            # Find orders at this process stage
            orders_at_stage = [
                o for o in self.orders
                if not o.is_complete()
                and o.current_process_idx == process_idx
            ]

            for order in orders_at_stage:
                if order.status == OrderStatus.WAITING:
                    # Start processing
                    order.status = OrderStatus.IN_PROGRESS
                    order.process_start_time = self.current_time

                    # Calculate wait time
                    wait_time = self.current_time - order.arrival_time
                    for prev_idx in range(process_idx):
                        prev_process = PROCESSES[prev_idx]
                        wait_time -= order.wait_times.get(prev_process.id, 0.0)
                    order.wait_times[process.id] = max(0, wait_time)

                elif order.status == OrderStatus.IN_PROGRESS:
                    # Check if processing is complete
                    elapsed = self.current_time - order.process_start_time

                    # Sample processing time (log-normal distribution)
                    mean = process.mean_time
                    std = process.std_time

                    # Adjust by capacity
                    effective_time = mean / capacity

                    if elapsed >= effective_time:
                        # Move to next process
                        order.current_process_idx += 1
                        order.status = OrderStatus.WAITING
                        order.process_start_time = None

                        # Check if completely done
                        if order.is_complete():
                            order.status = OrderStatus.COMPLETED
                            order.completion_time = self.current_time
                            self.completed_orders.append(order)
                            completed_count += 1

        # Remove completed orders from active list
        self.orders = [o for o in self.orders if not o.is_complete()]

        return completed_count

    def _update_idle_time(self, duration: float) -> None:
        """Track idle time for float staff."""
        if self.float_staff.current_action == "DO_NOTHING":
            self.float_staff.idle_time += duration
            self.total_idle_time += duration

    def step(self, action: str, arrival_rate: float, duration: float = DECISION_INTERVAL) -> Dict:
        """
        Advance simulation by one time step.

        Args:
            action: Action to take (will be applied after delay)
            arrival_rate: Order arrival rate (orders per second)
            duration: Time step duration (seconds)

        Returns:
            Dictionary with step metrics
        """
        # Set new action (with delay)
        from config import ACTION_DELAY
        self.set_float_action(action, delay=ACTION_DELAY)

        # Update pending action timer
        if self.pending_action_time_left > 0:
            self.pending_action_time_left -= duration
            if self.pending_action_time_left <= 0:
                self._apply_pending_action()

        # Generate new arrivals
        num_arrivals = self._generate_arrivals(duration, arrival_rate)

        # Process orders
        num_completed = self._process_orders(duration)

        # Update idle time
        self._update_idle_time(duration)

        # Advance time
        self.current_time += duration

        # Collect metrics
        metrics = {
            "time": self.current_time,
            "arrivals": num_arrivals,
            "completed": num_completed,
            "wip_total": len(self.orders),
            "idle_time_step": duration if self.float_staff.current_action == "DO_NOTHING" else 0.0,
            "action_changed": self.action_change_count > 0 and action != self.last_action,
            "current_action": self.float_staff.current_action,
            "pending_action": self.pending_action,
            "pending_time_left": self.pending_action_time_left,
        }

        # Per-process WIP
        for process_idx, process in enumerate(PROCESSES):
            wip = len([
                o for o in self.orders
                if o.current_process_idx == process_idx
            ])
            metrics[f"wip_{process.id}"] = wip

        return metrics

    def get_state(self) -> Dict:
        """
        Get current simulator state for RL observation.

        Returns:
            Dictionary with state information
        """
        state = {
            "current_time": self.current_time,
            "wip_total": len(self.orders),
            "float_idle_time": self.float_staff.idle_time,
            "current_action": self.float_staff.current_action,
            "pending_action": self.pending_action,
            "pending_time_left": self.pending_action_time_left,
            "total_completed": len(self.completed_orders),
        }

        # Per-process WIP breakdown
        for process_idx, process in enumerate(PROCESSES):
            waiting = len([
                o for o in self.orders
                if o.current_process_idx == process_idx
                and o.status == OrderStatus.WAITING
            ])
            in_progress = len([
                o for o in self.orders
                if o.current_process_idx == process_idx
                and o.status == OrderStatus.IN_PROGRESS
            ])
            state[f"{process.id}_wait"] = waiting
            state[f"{process.id}_in_progress"] = in_progress

        # Bottleneck detection (max WIP across processes)
        wip_by_process = [
            len([o for o in self.orders if o.current_process_idx == i])
            for i in range(len(PROCESSES))
        ]
        state["bottleneck_wip"] = max(wip_by_process) if wip_by_process else 0

        return state

    def get_metrics_summary(self) -> Dict:
        """Get summary metrics for the entire episode."""
        if not self.completed_orders:
            avg_wait = 0.0
        else:
            total_wait = sum(
                o.completion_time - o.arrival_time
                for o in self.completed_orders
            )
            avg_wait = total_wait / len(self.completed_orders)

        return {
            "total_completed": len(self.completed_orders),
            "total_wip": len(self.orders),
            "avg_wait_time": avg_wait,
            "total_idle_time": self.total_idle_time,
            "action_changes": self.action_change_count,
        }


if __name__ == "__main__":
    # Quick test
    from config import SCENARIOS, DEFAULT_SCENARIO

    scenario = SCENARIOS[DEFAULT_SCENARIO]
    sim = RamenShopSimulator(scenario=DEFAULT_SCENARIO, seed=42)

    print(f"Testing simulator with scenario: {scenario['name']}")
    print(f"Processes: {[p.id for p in PROCESSES]}")
    print(f"Actions: {ACTION_SPACE}\n")

    # Run a few steps
    for step_num in range(5):
        action = ACTION_SPACE[step_num % len(ACTION_SPACE)]
        metrics = sim.step(
            action=action,
            arrival_rate=scenario["arrival_rate"],
        )
        print(f"Step {step_num}: {metrics}")

    print(f"\nFinal summary: {sim.get_metrics_summary()}")
