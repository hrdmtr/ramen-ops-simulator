"""
Staff assignment logic for AI instructions.
Determines which staff member should receive which instruction.
"""

from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum


class StaffMember(Enum):
    """Available staff members."""
    A = "Aさん"
    B = "Bさん"
    C = "Cさん"


@dataclass
class StaffStatus:
    """Track status of a staff member."""
    name: str
    current_task: str = "DO_NOTHING"
    task_start_time: float = 0.0
    is_available: bool = True


class StaffAssignmentManager:
    """
    Manages staff assignment for AI instructions.

    Strategies:
    1. Round-robin: Rotate through available staff
    2. Task-based: Assign based on current task
    3. Availability-based: Assign to whoever is free
    """

    def __init__(self, num_staff: int = 3, strategy: str = "round_robin"):
        """
        Initialize staff assignment manager.

        Args:
            num_staff: Number of available staff members (default: 3)
            strategy: Assignment strategy ("round_robin", "availability", "task_based")
        """
        self.num_staff = num_staff
        self.strategy = strategy

        # Initialize staff members
        self.staff = {
            StaffMember.A: StaffStatus(name="Aさん"),
            StaffMember.B: StaffStatus(name="Bさん"),
            StaffMember.C: StaffStatus(name="Cさん"),
        }

        # Round-robin counter
        self.round_robin_index = 0

    def assign_task(self, action_name: str, current_time: float) -> Optional[str]:
        """
        Assign a task to a staff member.

        Args:
            action_name: Task to assign (e.g., "BOIL_HELP", "DISH_WASH")
            current_time: Current simulation time

        Returns:
            Staff member name (e.g., "Aさん") or None if no one available
        """
        if action_name == "DO_NOTHING":
            return None

        if self.strategy == "round_robin":
            return self._assign_round_robin(action_name, current_time)
        elif self.strategy == "availability":
            return self._assign_by_availability(action_name, current_time)
        elif self.strategy == "task_based":
            return self._assign_by_task(action_name, current_time)
        else:
            return self._assign_round_robin(action_name, current_time)

    def _assign_round_robin(self, action_name: str, current_time: float) -> str:
        """Round-robin assignment: rotate through staff."""
        staff_list = list(self.staff.keys())
        selected = staff_list[self.round_robin_index % len(staff_list)]
        self.round_robin_index += 1

        # Update staff status
        self.staff[selected].current_task = action_name
        self.staff[selected].task_start_time = current_time
        self.staff[selected].is_available = False

        return self.staff[selected].name

    def _assign_by_availability(self, action_name: str, current_time: float) -> Optional[str]:
        """Assign to the first available staff member."""
        for member, status in self.staff.items():
            if status.is_available:
                status.current_task = action_name
                status.task_start_time = current_time
                status.is_available = False
                return status.name

        # If no one is available, assign to least recently used
        return self._assign_round_robin(action_name, current_time)

    def _assign_by_task(self, action_name: str, current_time: float) -> str:
        """Assign based on task affinity (e.g., same person for same task)."""
        # Task affinity mapping
        task_affinity = {
            "BOIL_HELP": StaffMember.A,
            "PLATE_HELP": StaffMember.B,
            "SERVE_HELP": StaffMember.B,
            "DISH_WASH": StaffMember.C,
        }

        preferred_member = task_affinity.get(action_name, StaffMember.A)

        # If preferred member is available, assign to them
        if self.staff[preferred_member].is_available:
            self.staff[preferred_member].current_task = action_name
            self.staff[preferred_member].task_start_time = current_time
            self.staff[preferred_member].is_available = False
            return self.staff[preferred_member].name

        # Otherwise, assign to any available
        return self._assign_by_availability(action_name, current_time)

    def complete_task(self, staff_name: str):
        """Mark a task as completed for a staff member."""
        for member, status in self.staff.items():
            if status.name == staff_name:
                status.current_task = "DO_NOTHING"
                status.is_available = True
                break

    def get_staff_status(self) -> Dict[str, Dict]:
        """Get current status of all staff members."""
        return {
            status.name: {
                "current_task": status.current_task,
                "task_start_time": status.task_start_time,
                "is_available": status.is_available,
            }
            for status in self.staff.values()
        }


# Quick test
if __name__ == "__main__":
    print("スタッフ割り当てテスト")
    print("=" * 60)

    manager = StaffAssignmentManager(strategy="round_robin")

    # Simulate task assignments
    tasks = ["BOIL_HELP", "DISH_WASH", "BOIL_HELP", "PLATE_HELP", "SERVE_HELP"]

    for i, task in enumerate(tasks):
        staff = manager.assign_task(task, current_time=i * 30.0)
        print(f"ステップ{i+1}: {task} → {staff}")

    print("\n最終スタッフ状態:")
    for name, status in manager.get_staff_status().items():
        print(f"  {name}: {status['current_task']}")

    print("\n✓ スタッフ割り当てテスト完了")
