"""
Configuration for Ramen Shop Operations Simulator.

This file contains all parameters that can be adjusted based on
real shop observations. Processes can be easily added/removed/modified.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class ProcessConfig:
    """Configuration for a single process step."""
    id: str
    mean_time: float  # seconds (actual work time)
    std_time: float   # seconds
    setup_time: float = 5.0  # seconds (preparation time before work starts)

    def __post_init__(self):
        assert self.mean_time > 0, f"mean_time must be positive: {self.mean_time}"
        assert self.std_time >= 0, f"std_time must be non-negative: {self.std_time}"
        assert self.setup_time >= 0, f"setup_time must be non-negative: {self.setup_time}"


# =============================================================================
# PROCESS DEFINITIONS (easily modifiable based on real shop observations)
# =============================================================================

PROCESSES = [
    ProcessConfig(id="boil", mean_time=90.0, std_time=10.0),       # 麺茹で
    ProcessConfig(id="plate", mean_time=15.0, std_time=2.0),       # 盛り付け (5s setup + 15s work)
    ProcessConfig(id="serve", mean_time=15.0, std_time=2.0),       # 配膳 (5s setup + 15s work)
]

# Future expansion examples (commented out for now):
# ProcessConfig(id="prep", mean_time=60.0, std_time=10.0),         # 下準備
# ProcessConfig(id="toppings", mean_time=25.0, std_time=5.0),      # トッピング
# ProcessConfig(id="quality_check", mean_time=10.0, std_time=2.0), # 品質確認


# =============================================================================
# STAFF CONFIGURATION
# =============================================================================

# Fixed staff assignments (process_id -> number of staff)
# IMPORTANT: Set to 0 for all processes to require explicit instructions
FIXED_STAFF = {
    "boil": 0,    # No dedicated staff - instruction required
    "plate": 0,   # No dedicated staff - instruction required
    "serve": 0,   # No dedicated staff - instruction required
}

# Number of float staff (can be assigned to any task)
NUM_FLOAT_STAFF = 1

# Process capacity limits (maximum concurrent orders being processed)
# Note: Physical capacity is 4 for boiling, but 1 person can only manage 2
PROCESS_CAPACITY_LIMITS = {
    "boil": 4,    # Physical capacity: 4 noodle portions (but 1 person limited to 2)
    "plate": 10,  # No practical limit for plating
    "serve": 10,  # No practical limit for serving
}

# Capacity limits when only 1 person is working (without helper)
PROCESS_CAPACITY_LIMITS_SOLO = {
    "boil": 2,    # 1 person can manage max 2 portions
    "plate": 10,  # No change
    "serve": 10,  # No change
}

# Maximum concurrent dish washing (only 1 person can wash at a time)
MAX_CONCURRENT_DISH_WASHING = 1


# =============================================================================
# ACTIONS (Float staff can perform these tasks)
# =============================================================================

ACTION_SPACE = [
    "DO_NOTHING",      # Stay idle or continue current task
    "BOIL_HELP",       # Help with boiling (麺茹で)
    "PLATE_HELP",      # Help with plating (盛り付け)
    "SERVE_HELP",      # Help with serving (配膳)
    "DISH_WASH",       # Wash dirty dishes (皿洗い)
]

# Action duration (all actions last this long before next decision)
ACTION_DURATION = 30.0  # seconds

# Action application delay (time until action takes effect)
ACTION_DELAY = 10.0  # seconds


# =============================================================================
# SCENARIO CONFIGURATIONS
# =============================================================================

SCENARIOS = {
    "base": {
        "name": "Normal Operations",
        "arrival_rate": 1.0 / 60.0,  # λ = 1 order per 60 seconds
        "duration": 1200.0,           # 20 minutes
    },
    "peak": {
        "name": "Peak Hours",
        "arrival_rate": 1.0 / 30.0,  # λ = 1 order per 30 seconds (2x base)
        "duration": 1200.0,           # 20 minutes
    },
    "low": {
        "name": "Low Traffic",
        "arrival_rate": 1.0 / 120.0,  # λ = 1 order per 120 seconds (0.5x base)
        "duration": 1200.0,            # 20 minutes
    },
}

DEFAULT_SCENARIO = "base"


# =============================================================================
# RESTAURANT CAPACITY PARAMETERS
# =============================================================================

# Total number of seats in the restaurant
TOTAL_SEATS = 10

# Average eating time (seconds) - how long customers stay after receiving food
EATING_TIME_MEAN = 360.0  # 6 minutes (realistic for ramen shop)
EATING_TIME_STD = 60.0    # 1 minute variation

# Dish washing parameters
DISH_WASHING_TIME_PER_DISH = 20.0  # seconds per dish (fixed)
TOTAL_DISHES = 30  # Total number of dishes in the restaurant

# Initial number of clean dishes available
INITIAL_CLEAN_DISHES = 30


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Time step for RL agent decisions
DECISION_INTERVAL = 30.0  # seconds (must equal ACTION_DURATION)

# Episode length
EPISODE_STEPS = 240  # 240 steps × 30s = 120 minutes (extended to ensure dish washing is critical)

# Random seed
DEFAULT_SEED = 42


# =============================================================================
# REWARD WEIGHTS (tunable based on priorities)
# =============================================================================

REWARD_WEIGHTS = {
    "completed_bowls": 10.0,        # +10 per completed bowl
    "avg_wait_time": -0.1,          # -0.1 per second of average wait time
    "wip_overflow": -5.0,           # -5 per item over threshold
    "idle_penalty": -0.2,           # -0.2 per second of idle time
    "change_penalty": -2.0,         # -2 for changing action from previous step
    "dish_shortage": -2.0,          # -2 per dish below threshold (reduced to allow cooking help)
}

# WIP (Work In Progress) threshold for overflow penalty
WIP_OVERFLOW_THRESHOLD = 10  # items


# =============================================================================
# HELPER EFFECT
# =============================================================================

# Helper increases CAPACITY, not speed
# When float staff helps, the capacity limit increases (e.g., boil: 2 -> 4)
# Processing time remains the same


# =============================================================================
# OBSERVATION SPACE NORMALIZATION BOUNDS
# =============================================================================

# Maximum expected values for normalization (can be adjusted based on scenarios)
MAX_ARRIVALS_PER_STEP = 5      # Max orders arriving in 30s
MAX_WIP_PER_PROCESS = 15       # Max items waiting/in-progress per process
MAX_IDLE_TIME = 30.0           # Max idle time tracked (= DECISION_INTERVAL)
MAX_PENDING_TIME = 10.0        # Max pending time (= ACTION_DELAY)
MAX_WAITING_CUSTOMERS = 20     # Max customers waiting for seats
MAX_DISHES = 50                # Max dishes (clean or dirty)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config() -> None:
    """Validate configuration consistency."""
    assert len(PROCESSES) > 0, "At least one process must be defined"
    assert NUM_FLOAT_STAFF >= 1, "At least one float staff required"
    assert ACTION_DURATION > 0, "ACTION_DURATION must be positive"
    assert ACTION_DELAY >= 0, "ACTION_DELAY must be non-negative"
    assert ACTION_DELAY < ACTION_DURATION, "ACTION_DELAY must be less than ACTION_DURATION"
    assert EPISODE_STEPS > 0, "EPISODE_STEPS must be positive"

    # Validate capacity limits are defined
    for process in PROCESSES:
        if process.id not in PROCESS_CAPACITY_LIMITS:
            print(f"Warning: No capacity limit defined for {process.id}")
        if process.id not in PROCESS_CAPACITY_LIMITS_SOLO:
            print(f"Warning: No solo capacity limit defined for {process.id}")

    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print(f"\nProcesses: {[p.id for p in PROCESSES]}")
    print(f"Actions: {ACTION_SPACE}")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
