"""
Detailed logging module for Ramen Shop Operations.

Provides timestamped logs for:
- Staff instructions (who, what, when)
- Order completions (order ID, completion time)
- Process transitions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class InstructionLog:
    """Log entry for staff instructions."""
    timestamp: float  # Simulation time in seconds
    staff_id: str
    instruction: str
    description: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "time_seconds": self.timestamp,
            "time_formatted": self._format_time(self.timestamp),
            "staff_id": self.staff_id,
            "instruction": self.instruction,
            "description": self.description,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


@dataclass
class OrderCompletionLog:
    """Log entry for order completions."""
    order_id: int
    arrival_time: float
    completion_time: float
    total_time: float
    process_times: Dict[str, float]

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        base_dict = {
            "order_id": self.order_id,
            "arrival_time_seconds": self.arrival_time,
            "arrival_time_formatted": InstructionLog._format_time(self.arrival_time),
            "completion_time_seconds": self.completion_time,
            "completion_time_formatted": InstructionLog._format_time(self.completion_time),
            "total_time_seconds": self.total_time,
        }
        # Add process times
        for process_id, time in self.process_times.items():
            base_dict[f"{process_id}_time"] = time
        return base_dict


@dataclass
class ProcessTransitionLog:
    """Log entry for process transitions."""
    timestamp: float
    order_id: int
    from_process: Optional[str]
    to_process: Optional[str]
    action: str  # "started", "completed", "moved_to_next"

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "time_seconds": self.timestamp,
            "time_formatted": InstructionLog._format_time(self.timestamp),
            "order_id": self.order_id,
            "from_process": from_process if (from_process := self.from_process) else "none",
            "to_process": to_process if (to_process := self.to_process) else "completed",
            "action": self.action,
        }


class SimulationLogger:
    """
    Centralized logger for simulation events.

    Tracks:
    - Staff instructions
    - Order completions
    - Process transitions
    """

    def __init__(self):
        """Initialize logger."""
        self.instruction_logs: List[InstructionLog] = []
        self.completion_logs: List[OrderCompletionLog] = []
        self.transition_logs: List[ProcessTransitionLog] = []

    def log_instruction(
        self,
        timestamp: float,
        staff_id: str,
        instruction: str,
        description: str = ""
    ) -> None:
        """
        Log a staff instruction.

        Args:
            timestamp: Simulation time in seconds
            staff_id: Staff identifier (e.g., "float_0")
            instruction: Instruction code (e.g., "PLATE_HELP")
            description: Human-readable description
        """
        log = InstructionLog(
            timestamp=timestamp,
            staff_id=staff_id,
            instruction=instruction,
            description=description or instruction,
        )
        self.instruction_logs.append(log)

    def log_order_completion(
        self,
        order_id: int,
        arrival_time: float,
        completion_time: float,
        process_times: Dict[str, float],
    ) -> None:
        """
        Log an order completion.

        Args:
            order_id: Order ID
            arrival_time: When order arrived (seconds)
            completion_time: When order completed (seconds)
            process_times: Time spent in each process
        """
        log = OrderCompletionLog(
            order_id=order_id,
            arrival_time=arrival_time,
            completion_time=completion_time,
            total_time=completion_time - arrival_time,
            process_times=process_times,
        )
        self.completion_logs.append(log)

    def log_process_transition(
        self,
        timestamp: float,
        order_id: int,
        from_process: Optional[str],
        to_process: Optional[str],
        action: str,
    ) -> None:
        """
        Log a process transition for an order.

        Args:
            timestamp: Simulation time in seconds
            order_id: Order ID
            from_process: Previous process (None if starting)
            to_process: Next process (None if completing)
            action: Type of transition
        """
        log = ProcessTransitionLog(
            timestamp=timestamp,
            order_id=order_id,
            from_process=from_process,
            to_process=to_process,
            action=action,
        )
        self.transition_logs.append(log)

    def get_instruction_df(self) -> pd.DataFrame:
        """Get instructions as DataFrame."""
        if not self.instruction_logs:
            return pd.DataFrame()
        return pd.DataFrame([log.to_dict() for log in self.instruction_logs])

    def get_completion_df(self) -> pd.DataFrame:
        """Get completions as DataFrame."""
        if not self.completion_logs:
            return pd.DataFrame()
        return pd.DataFrame([log.to_dict() for log in self.completion_logs])

    def get_transition_df(self) -> pd.DataFrame:
        """Get transitions as DataFrame."""
        if not self.transition_logs:
            return pd.DataFrame()
        return pd.DataFrame([log.to_dict() for log in self.transition_logs])

    def save_to_csv(self, prefix: str = "simulation_log") -> Dict[str, str]:
        """
        Save all logs to CSV files.

        Args:
            prefix: Filename prefix

        Returns:
            Dictionary mapping log type to filename
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filenames = {}

        # Save instructions
        instructions_df = self.get_instruction_df()
        if not instructions_df.empty:
            filename = f"{prefix}_instructions_{timestamp}.csv"
            instructions_df.to_csv(filename, index=False)
            filenames["instructions"] = filename

        # Save completions
        completions_df = self.get_completion_df()
        if not completions_df.empty:
            filename = f"{prefix}_completions_{timestamp}.csv"
            completions_df.to_csv(filename, index=False)
            filenames["completions"] = filename

        # Save transitions
        transitions_df = self.get_transition_df()
        if not transitions_df.empty:
            filename = f"{prefix}_transitions_{timestamp}.csv"
            transitions_df.to_csv(filename, index=False)
            filenames["transitions"] = filename

        return filenames

    def print_summary(self) -> None:
        """Print a summary of logged events."""
        print("\n" + "="*60)
        print("SIMULATION LOG SUMMARY")
        print("="*60)

        # Instructions
        print(f"\n📋 Staff Instructions: {len(self.instruction_logs)}")
        if self.instruction_logs:
            print("\nRecent instructions:")
            for log in self.instruction_logs[-5:]:
                print(f"  {log.to_dict()['time_formatted']} | "
                      f"{log.staff_id}: {log.instruction}")

        # Completions
        print(f"\n✅ Completed Orders: {len(self.completion_logs)}")
        if self.completion_logs:
            print("\nRecent completions:")
            for log in self.completion_logs[-5:]:
                d = log.to_dict()
                print(f"  Order #{d['order_id']} completed at "
                      f"{d['completion_time_formatted']} "
                      f"(total: {d['total_time_seconds']:.0f}s)")

        # Transitions
        print(f"\n🔄 Process Transitions: {len(self.transition_logs)}")

        print("\n" + "="*60 + "\n")

    def reset(self) -> None:
        """Clear all logs."""
        self.instruction_logs.clear()
        self.completion_logs.clear()
        self.transition_logs.clear()


if __name__ == "__main__":
    # Test logger
    logger = SimulationLogger()

    # Test instruction logging
    logger.log_instruction(30.0, "float_0", "PLATE_HELP", "Helping with plating")
    logger.log_instruction(60.0, "float_0", "SERVE_HELP", "Helping with serving")

    # Test completion logging
    logger.log_order_completion(
        order_id=1,
        arrival_time=10.0,
        completion_time=150.0,
        process_times={"boil": 90.0, "plate": 30.0, "serve": 20.0}
    )

    # Test transition logging
    logger.log_process_transition(10.0, 1, None, "boil", "started")
    logger.log_process_transition(100.0, 1, "boil", "plate", "moved_to_next")
    logger.log_process_transition(150.0, 1, "serve", None, "completed")

    # Print summary
    logger.print_summary()

    # Show DataFrames
    print("\nInstructions DataFrame:")
    print(logger.get_instruction_df())

    print("\nCompletions DataFrame:")
    print(logger.get_completion_df())

    print("\nTransitions DataFrame:")
    print(logger.get_transition_df())
