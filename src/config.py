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
    mean_time: float  # seconds
    std_time: float   # seconds

    def __post_init__(self):
        assert self.mean_time > 0, f"mean_time must be positive: {self.mean_time}"
        assert self.std_time >= 0, f"std_time must be non-negative: {self.std_time}"


# =============================================================================
# PROCESS DEFINITIONS (easily modifiable based on real shop observations)
# =============================================================================

PROCESSES = [
    ProcessConfig(id="boil", mean_time=180.0, std_time=20.0),      # 麺茹で
    ProcessConfig(id="plate", mean_time=30.0, std_time=5.0),       # 盛り付け
    ProcessConfig(id="serve", mean_time=20.0, std_time=3.0),       # 配膳
]

# Future expansion examples (commented out for now):
# ProcessConfig(id="prep", mean_time=60.0, std_time=10.0),         # 下準備
# ProcessConfig(id="toppings", mean_time=25.0, std_time=5.0),      # トッピング
# ProcessConfig(id="quality_check", mean_time=10.0, std_time=2.0), # 品質確認


# =============================================================================
# STAFF CONFIGURATION
# =============================================================================

# Fixed staff assignments (process_id -> number of staff)
FIXED_STAFF = {
    "boil": 1,    # 1 person dedicated to boiling
    "plate": 1,   # 1 person for plating (can also serve if needed)
}

# Number of float staff (can be assigned to any task)
NUM_FLOAT_STAFF = 1


# =============================================================================
# ACTIONS (Float staff can perform these tasks)
# =============================================================================

ACTION_SPACE = [
    "DO_NOTHING",      # Stay idle or continue current task
    "DISH_HELP",       # Help with dishwashing (parallel track)
    "PLATE_HELP",      # Help with plating
    "SERVE_HELP",      # Help with serving
    "RESTOCK_HELP",    # Help with restocking (parallel track)
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
        "arrival_rate": 1.0 / 30.0,  # λ = 1 order per 30 seconds
        "duration": 1200.0,           # 20 minutes
    },
}

DEFAULT_SCENARIO = "base"


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Time step for RL agent decisions
DECISION_INTERVAL = 30.0  # seconds (must equal ACTION_DURATION)

# Episode length
EPISODE_STEPS = 40  # 40 steps × 30s = 20 minutes

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
}

# WIP (Work In Progress) threshold for overflow penalty
WIP_OVERFLOW_THRESHOLD = 10  # items


# =============================================================================
# HELPER EFFECT MULTIPLIERS
# =============================================================================

# When float staff helps a process, how much does it speed up?
# 1.0 = no effect, 2.0 = doubles the throughput
HELP_EFFECT_MULTIPLIERS = {
    "PLATE_HELP": 1.5,   # 50% faster plating when helper joins
    "SERVE_HELP": 1.5,   # 50% faster serving when helper joins
    "DISH_HELP": 1.3,    # 30% faster dishwashing
    "RESTOCK_HELP": 1.3, # 30% faster restocking
}


# =============================================================================
# OBSERVATION SPACE NORMALIZATION BOUNDS
# =============================================================================

# Maximum expected values for normalization (can be adjusted based on scenarios)
MAX_ARRIVALS_PER_STEP = 5      # Max orders arriving in 30s
MAX_WIP_PER_PROCESS = 15       # Max items waiting/in-progress per process
MAX_IDLE_TIME = 30.0           # Max idle time tracked (= DECISION_INTERVAL)
MAX_PENDING_TIME = 10.0        # Max pending time (= ACTION_DELAY)


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

    # Validate all help actions have multipliers
    help_actions = [a for a in ACTION_SPACE if a.endswith("_HELP")]
    for action in help_actions:
        if action not in HELP_EFFECT_MULTIPLIERS:
            print(f"Warning: No multiplier defined for {action}, defaulting to 1.0")

    print("✓ Configuration validated successfully")


if __name__ == "__main__":
    validate_config()
    print(f"\nProcesses: {[p.id for p in PROCESSES]}")
    print(f"Actions: {ACTION_SPACE}")
    print(f"Scenarios: {list(SCENARIOS.keys())}")
