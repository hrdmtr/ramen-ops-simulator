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
    ACTION_SPACE, DECISION_INTERVAL,
    TOTAL_SEATS, EATING_TIME_MEAN, EATING_TIME_STD,
    DISH_WASHING_TIME_PER_DISH, TOTAL_DISHES, INITIAL_CLEAN_DISHES,
    PROCESS_CAPACITY_LIMITS, PROCESS_CAPACITY_LIMITS_SOLO, MAX_CONCURRENT_DISH_WASHING
)
from logger import SimulationLogger


class OrderStatus(Enum):
    """Order lifecycle status."""
    WAITING = "waiting"      # Waiting for process to start
    SETUP = "setup"          # In setup phase (preparation before work)
    IN_PROGRESS = "in_progress"  # Currently being processed (actual work)
    COMPLETED = "completed"  # Finished all processes


@dataclass
class Order:
    """Represents a single ramen order."""
    id: int
    arrival_time: float
    current_process_idx: int = 0  # Index in PROCESSES list
    status: OrderStatus = OrderStatus.WAITING
    setup_start_time: Optional[float] = None  # When setup phase started
    process_start_time: Optional[float] = None  # When actual work started
    completion_time: Optional[float] = None
    wait_times: Dict[str, float] = field(default_factory=dict)  # process_id -> wait_time
    process_times: Dict[str, float] = field(default_factory=dict)  # process_id -> actual_processing_time

    def current_process(self) -> Optional[ProcessConfig]:
        """Get current process config."""
        if self.current_process_idx < len(PROCESSES):
            return PROCESSES[self.current_process_idx]
        return None

    def is_complete(self) -> bool:
        """Check if order has completed all processes."""
        return self.current_process_idx >= len(PROCESSES)


@dataclass
class Customer:
    """Represents a customer in the restaurant."""
    id: int
    order_id: int  # Associated order ID
    arrival_time: float
    seat_time: Optional[float] = None  # When they were seated
    served_time: Optional[float] = None  # When food was delivered
    planned_leave_time: Optional[float] = None  # When they plan to leave
    left_time: Optional[float] = None  # When they actually left


@dataclass
class FloatStaff:
    """Represents a float staff member who can be assigned to different tasks."""
    id: int
    current_action: str = "DO_NOTHING"
    action_start_time: float = 0.0
    idle_time: float = 0.0  # Cumulative idle time


@dataclass
class DirtyDish:
    """Represents a dirty dish waiting to be washed."""
    id: int
    created_time: float
    wash_start_time: Optional[float] = None
    washing_completed_at: Optional[float] = None  # When it will be done


class RamenShopSimulator:
    """
    Discrete Event Simulation for ramen shop operations.

    Simulates:
    - Poisson order arrivals
    - Multi-stage processing (configurable via PROCESSES)
    - Fixed and float staff
    - Action delays and effects
    """

    def __init__(self, scenario: str = "base", seed: Optional[int] = None, enable_logging: bool = False, enable_state_log: bool = False):
        """
        Initialize simulator.

        Args:
            scenario: Scenario name from config.SCENARIOS
            seed: Random seed for reproducibility
            enable_logging: Enable detailed logging of events
            enable_state_log: Enable real-time state change logging to console
        """
        self.scenario = scenario
        self.rng = np.random.RandomState(seed)

        # Time tracking
        self.current_time: float = 0.0

        # Orders
        self.order_counter: int = 0
        self.orders: List[Order] = []

        # Customers (restaurant capacity management)
        self.customer_counter: int = 0
        self.waiting_customers: List[Customer] = []  # Waiting for seats
        self.seated_customers: List[Customer] = []   # Eating or waiting for food
        self.completed_customers: List[Customer] = []  # Finished and left

        # Map order_id to customer_id for food delivery
        self.order_to_customer: Dict[int, int] = {}

        # Dishes
        self.clean_dishes: int = INITIAL_CLEAN_DISHES
        self.dirty_dishes: List[DirtyDish] = []
        self.dish_counter: int = 0

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

        # Logging
        self.enable_logging = enable_logging
        self.logger = SimulationLogger() if enable_logging else None
        self.enable_state_log = enable_state_log

    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.current_time = 0.0
        self.order_counter = 0
        self.orders = []
        self.customer_counter = 0
        self.waiting_customers = []
        self.seated_customers = []
        self.completed_customers = []
        self.order_to_customer = {}
        self.clean_dishes = INITIAL_CLEAN_DISHES
        self.dirty_dishes = []
        self.dish_counter = 0
        self.float_staff = FloatStaff(id=0)
        self.pending_action = None
        self.pending_action_time_left = 0.0
        self.completed_orders = []
        self.total_idle_time = 0.0
        self.action_change_count = 0
        self.last_action = "DO_NOTHING"
        if self.logger:
            self.logger.reset()

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

            # Log instruction
            if self.logger:
                self.logger.log_instruction(
                    timestamp=self.current_time,
                    staff_id=f"float_{self.float_staff.id}",
                    instruction=new_action,
                    description=f"Float staff assigned to: {new_action}"
                )

            self.pending_action = None
            self.pending_action_time_left = 0.0

    def _generate_arrivals(self, duration: float, arrival_rate: float) -> int:
        """Generate Poisson arrivals for the time step."""
        expected = arrival_rate * duration
        num_arrivals = self.rng.poisson(expected)

        for _ in range(num_arrivals):
            # Distribute arrivals uniformly within the time step
            arrival_offset = self.rng.uniform(0, duration)
            arrival_time = self.current_time + arrival_offset

            # Create customer
            customer = Customer(
                id=self.customer_counter,
                order_id=self.order_counter,
                arrival_time=arrival_time
            )
            self.customer_counter += 1

            # Check if seats are available
            occupied_seats = len(self.seated_customers)
            if occupied_seats < TOTAL_SEATS:
                # Seat immediately
                customer.seat_time = arrival_time
                self.seated_customers.append(customer)

                # Create order only if clean dishes are available
                if self.clean_dishes > 0:
                    order = Order(
                        id=self.order_counter,
                        arrival_time=arrival_time
                    )
                    self.orders.append(order)
                    self.order_to_customer[self.order_counter] = customer.id
                    # Reserve a clean dish for this order
                    self.clean_dishes -= 1
                    self.order_counter += 1
                    self._log_state_change(f"客来店→着席 (客#{customer.id}, 注文#{order.id})")
                else:
                    # If no clean dishes, customer waits at seat but no order is created
                    self._log_state_change(f"客来店→着席 (客#{customer.id}) ⚠️ 皿不足で注文不可")
            else:
                # Add to waiting queue
                self.waiting_customers.append(customer)
                self._log_state_change(f"客来店→満席で待ち (客#{customer.id})")

        return num_arrivals

    def _get_process_capacity(self, process_id: str) -> float:
        """
        Calculate effective processing capacity for a process.

        IMPORTANT: With FIXED_STAFF = 0, only processes with assigned float staff have capacity!
        - If float staff is assigned to this process: return 1.0
        - Otherwise: return 0.0 (no one working on this process)

        Returns:
            1.0 if someone is working on this process, 0.0 otherwise
        """
        # Check if float staff is assigned to help with this process
        help_action = f"{process_id.upper()}_HELP"
        if self.float_staff.current_action == help_action:
            return 1.0  # Float staff is assigned to this process

        # Check fixed staff assignment (from config)
        fixed_count = FIXED_STAFF.get(process_id, 0)
        if fixed_count > 0:
            return 1.0  # Fixed staff is present

        # No one is working on this process
        return 0.0

    def _process_orders(self, duration: float) -> int:
        """
        Process orders through the pipeline.

        IMPORTANT: Helpers increase CONCURRENT CAPACITY, NOT speed!
        - Without helper: capacity_limit = 2 (solo limit)
        - With helper: capacity_limit = 4 (full physical capacity)
        - Processing time: ALWAYS constant (90s for boil, etc.)

        Returns:
            Number of orders completed in this step
        """
        completed_count = 0

        # Process each stage
        for process_idx, process in enumerate(PROCESSES):
            capacity = self._get_process_capacity(process.id)

            if capacity <= 0:
                continue  # No one working on this process

            # Get capacity limit for this process (depends on whether helper is present)
            help_action = f"{process.id.upper()}_HELP"
            if self.float_staff.current_action == help_action:
                # Helper is present: use full physical capacity
                capacity_limit = PROCESS_CAPACITY_LIMITS.get(process.id, 999)
            else:
                # No helper: use solo capacity limit
                capacity_limit = PROCESS_CAPACITY_LIMITS_SOLO.get(process.id, 999)

            # Find orders at this process stage
            orders_at_stage = [
                o for o in self.orders
                if not o.is_complete()
                and o.current_process_idx == process_idx
            ]

            # Count orders currently in progress (both SETUP and IN_PROGRESS occupy capacity)
            in_progress_count = len([
                o for o in orders_at_stage
                if o.status in (OrderStatus.SETUP, OrderStatus.IN_PROGRESS)
            ])

            for order in orders_at_stage:
                if order.status == OrderStatus.WAITING:
                    # Check if we can start processing (capacity limit)
                    if in_progress_count >= capacity_limit:
                        continue  # Cannot start new order, capacity limit reached

                    # Start setup phase
                    order.status = OrderStatus.SETUP
                    order.setup_start_time = self.current_time
                    in_progress_count += 1  # Increment count for capacity tracking

                    # Calculate wait time
                    wait_time = self.current_time - order.arrival_time
                    for prev_idx in range(process_idx):
                        prev_process = PROCESSES[prev_idx]
                        wait_time -= order.wait_times.get(prev_process.id, 0.0)
                    order.wait_times[process.id] = max(0, wait_time)

                    # Log setup start
                    if self.logger:
                        from_proc = PROCESSES[process_idx - 1].id if process_idx > 0 else None
                        self.logger.log_process_transition(
                            timestamp=self.current_time,
                            order_id=order.id,
                            from_process=from_proc,
                            to_process=process.id,
                            action="setup_started"
                        )

                elif order.status == OrderStatus.SETUP:
                    # Check if setup phase is complete
                    setup_elapsed = self.current_time - order.setup_start_time

                    if setup_elapsed >= process.setup_time:
                        # Move to actual work phase
                        order.status = OrderStatus.IN_PROGRESS
                        order.process_start_time = self.current_time

                        # Log work start
                        if self.logger:
                            self.logger.log_process_transition(
                                timestamp=self.current_time,
                                order_id=order.id,
                                from_process=process.id,
                                to_process=process.id,
                                action="work_started"
                            )

                elif order.status == OrderStatus.IN_PROGRESS:
                    # Check if processing is complete
                    elapsed = self.current_time - order.process_start_time

                    # Processing time is constant (NOT affected by helper)
                    # Helpers increase CAPACITY LIMIT, not speed
                    mean = process.mean_time
                    std = process.std_time
                    effective_time = mean  # Always constant time, regardless of helpers

                    if elapsed >= effective_time:
                        # Record total processing time (setup + work)
                        order.process_times[process.id] = process.setup_time + effective_time

                        # Move to next process
                        next_idx = order.current_process_idx + 1
                        order.current_process_idx = next_idx
                        order.status = OrderStatus.WAITING
                        order.process_start_time = None

                        # Check if completely done
                        if order.is_complete():
                            order.status = OrderStatus.COMPLETED
                            order.completion_time = self.current_time
                            self.completed_orders.append(order)
                            completed_count += 1

                            # Deliver food to customer
                            if order.id in self.order_to_customer:
                                customer_id = self.order_to_customer[order.id]
                                customer = next((c for c in self.seated_customers if c.id == customer_id), None)
                                if customer:
                                    customer.served_time = self.current_time
                                    # Calculate eating duration
                                    eating_duration = max(0, self.rng.normal(EATING_TIME_MEAN, EATING_TIME_STD))
                                    customer.planned_leave_time = self.current_time + eating_duration
                                    wait_time = self.current_time - order.arrival_time
                                    self._log_state_change(f"料理完成→提供 (注文#{order.id}, 客#{customer_id}, 待ち時間{wait_time:.0f}秒)")

                            # Log completion
                            if self.logger:
                                self.logger.log_order_completion(
                                    order_id=order.id,
                                    arrival_time=order.arrival_time,
                                    completion_time=order.completion_time,
                                    process_times=order.wait_times
                                )
                                self.logger.log_process_transition(
                                    timestamp=self.current_time,
                                    order_id=order.id,
                                    from_process=process.id,
                                    to_process=None,
                                    action="completed"
                                )
                        else:
                            # Log transition to next process
                            if self.logger:
                                next_process = PROCESSES[next_idx].id if next_idx < len(PROCESSES) else None
                                self.logger.log_process_transition(
                                    timestamp=self.current_time,
                                    order_id=order.id,
                                    from_process=process.id,
                                    to_process=next_process,
                                    action="moved_to_next"
                                )

        # Remove completed orders from active list
        self.orders = [o for o in self.orders if not o.is_complete()]

        return completed_count

    def _process_customers(self, duration: float) -> None:
        """Process customer lifecycle: eating and leaving."""
        # Check for customers who finished eating
        customers_to_leave = [
            c for c in self.seated_customers
            if c.planned_leave_time is not None and c.planned_leave_time <= self.current_time
        ]

        for customer in customers_to_leave:
            customer.left_time = self.current_time
            self.completed_customers.append(customer)
            self.seated_customers.remove(customer)

            # Create dirty dish
            dirty_dish = DirtyDish(
                id=self.dish_counter,
                created_time=self.current_time
            )
            self.dirty_dishes.append(dirty_dish)
            self.dish_counter += 1

            eating_time = self.current_time - customer.served_time if customer.served_time else 0
            self._log_state_change(f"客退店 (客#{customer.id}, 食事時間{eating_time:.0f}秒)")

        # Seat waiting customers if seats became available
        while self.waiting_customers and len(self.seated_customers) < TOTAL_SEATS:
            customer = self.waiting_customers.pop(0)
            customer.seat_time = self.current_time
            self.seated_customers.append(customer)
            wait_time = self.current_time - customer.arrival_time

            # Create order only if clean dishes are available
            if self.clean_dishes > 0:
                order = Order(
                    id=self.order_counter,
                    arrival_time=customer.arrival_time
                )
                self.orders.append(order)
                customer.order_id = self.order_counter
                self.order_to_customer[self.order_counter] = customer.id
                # Reserve a clean dish for this order
                self.clean_dishes -= 1
                self.order_counter += 1
                self._log_state_change(f"待ち客着席 (客#{customer.id}, 注文#{order.id}, 待ち時間{wait_time:.0f}秒)")
            else:
                # If no clean dishes, customer waits at seat but no order is created
                self._log_state_change(f"待ち客着席 (客#{customer.id}) ⚠️ 皿不足で注文不可")

    def _process_dish_washing(self, duration: float) -> None:
        """
        Process dish washing by float staff when assigned to DISH_WASH.

        Constraint: Only 1 person can wash dishes at a time (MAX_CONCURRENT_DISH_WASHING = 1)
        """
        # Only wash dishes if float staff is assigned to dish washing
        if self.float_staff.current_action != "DISH_WASH":
            return

        if not self.dirty_dishes:
            return  # No dirty dishes to wash

        # Find dishes that are currently being washed
        dishes_being_washed = [d for d in self.dirty_dishes if d.wash_start_time is not None]

        # Start washing new dishes if we have capacity (max 1 concurrent)
        if len(dishes_being_washed) < MAX_CONCURRENT_DISH_WASHING:
            # Start washing the oldest dirty dish
            dish = self.dirty_dishes[0]
            dish.wash_start_time = self.current_time
            dish.washing_completed_at = self.current_time + DISH_WASHING_TIME_PER_DISH

        # Complete washed dishes
        dishes_to_clean = []
        for dish in self.dirty_dishes:
            if dish.washing_completed_at is not None and self.current_time >= dish.washing_completed_at:
                dishes_to_clean.append(dish)

        # Remove cleaned dishes and add to clean pile
        for dish in dishes_to_clean:
            self.dirty_dishes.remove(dish)
            self.clean_dishes += 1
            self._log_state_change(f"皿洗い完了 (皿#{dish.id})")

            # Start washing the next dish immediately if available
            if self.dirty_dishes and self.float_staff.current_action == "DISH_WASH":
                next_dish = self.dirty_dishes[0]
                if next_dish.wash_start_time is None:
                    next_dish.wash_start_time = self.current_time
                    next_dish.washing_completed_at = self.current_time + DISH_WASHING_TIME_PER_DISH
                    self._log_state_change(f"皿洗い開始 (皿#{next_dish.id})")

    def _log_state_change(self, event: str) -> None:
        """Log state changes to console if enabled."""
        if not self.enable_state_log:
            return

        # Format time
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # Count cooking orders by process (distinguish waiting vs in progress)
        boil_wait = len([o for o in self.orders if o.current_process_idx == 0 and o.status == OrderStatus.WAITING])
        boil_prog = len([o for o in self.orders if o.current_process_idx == 0 and o.status == OrderStatus.IN_PROGRESS])
        plate_wait = len([o for o in self.orders if o.current_process_idx == 1 and o.status == OrderStatus.WAITING])
        plate_prog = len([o for o in self.orders if o.current_process_idx == 1 and o.status == OrderStatus.IN_PROGRESS])
        serve_wait = len([o for o in self.orders if o.current_process_idx == 2 and o.status == OrderStatus.WAITING])
        serve_prog = len([o for o in self.orders if o.current_process_idx == 2 and o.status == OrderStatus.IN_PROGRESS])

        # Count customers
        waiting_for_food = len([c for c in self.seated_customers if c.served_time is None])
        eating = len([c for c in self.seated_customers if c.served_time is not None])

        # Count dishes
        washing = len([d for d in self.dirty_dishes if d.wash_start_time is not None])

        print(f"\n[{time_str}] {event}")
        print(f"  席: 空席={TOTAL_SEATS - len(self.seated_customers)}/{TOTAL_SEATS}, "
              f"着席={len(self.seated_customers)}, 待ち={len(self.waiting_customers)}")
        print(f"  客: 料理待ち={waiting_for_food}, 食事中={eating}")
        print(f"  調理: 茹で={boil_prog}/{boil_wait}(作業中/待ち), 盛付け={plate_prog}/{plate_wait}, 配膳={serve_prog}/{serve_wait}")
        print(f"  皿: きれい={self.clean_dishes}, 汚れ={len(self.dirty_dishes)}, 洗浄中={washing}")
        print(f"  完了: {len(self.completed_orders)}杯")

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

        # Process customer lifecycle (eating and leaving)
        self._process_customers(duration)

        # Process dish washing
        self._process_dish_washing(duration)

        # Process orders
        num_completed = self._process_orders(duration)

        # Update idle time
        self._update_idle_time(duration)

        # Advance time
        self.current_time += duration

        # Count customers by status
        customers_waiting_for_food = len([
            c for c in self.seated_customers
            if c.served_time is None
        ])
        customers_eating = len([
            c for c in self.seated_customers
            if c.served_time is not None
        ])

        # Count dishes being washed
        dishes_being_washed = len([
            d for d in self.dirty_dishes
            if d.wash_start_time is not None
        ])

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

            # Restaurant capacity state
            "waiting_customers": len(self.waiting_customers),
            "seated_customers": len(self.seated_customers),
            "customers_eating": customers_eating,
            "customers_waiting_for_food": customers_waiting_for_food,
            "empty_seats": TOTAL_SEATS - len(self.seated_customers),

            # Dish state
            "clean_dishes": self.clean_dishes,
            "dirty_dishes": len(self.dirty_dishes),
            "dishes_being_washed": dishes_being_washed,
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
        # Count customers by status
        customers_waiting_for_food = len([
            c for c in self.seated_customers
            if c.served_time is None
        ])
        customers_eating = len([
            c for c in self.seated_customers
            if c.served_time is not None
        ])

        # Count dishes being washed
        dishes_being_washed = len([
            d for d in self.dirty_dishes
            if d.wash_start_time is not None
        ])

        state = {
            "current_time": self.current_time,
            "wip_total": len(self.orders),
            "float_idle_time": self.float_staff.idle_time,
            "current_action": self.float_staff.current_action,
            "pending_action": self.pending_action,
            "pending_time_left": self.pending_action_time_left,
            "total_completed": len(self.completed_orders),

            # Restaurant capacity state
            "waiting_customers": len(self.waiting_customers),
            "seated_customers": len(self.seated_customers),
            "customers_eating": customers_eating,
            "customers_waiting_for_food": customers_waiting_for_food,
            "empty_seats": TOTAL_SEATS - len(self.seated_customers),
            "total_seats": TOTAL_SEATS,

            # Dish state
            "clean_dishes": self.clean_dishes,
            "dirty_dishes": len(self.dirty_dishes),
            "dishes_being_washed": dishes_being_washed,
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

    def get_process_time_statistics(self) -> Dict:
        """
        Calculate statistics for each process stage.

        Returns:
            Dictionary with process-level statistics including:
            - mean: Average processing time
            - std: Standard deviation
            - min: Minimum processing time
            - max: Maximum processing time
            - count: Number of orders processed
        """
        stats = {}

        for process in PROCESSES:
            process_id = process.id
            times = [
                order.process_times[process_id]
                for order in self.completed_orders
                if process_id in order.process_times
            ]

            if times:
                stats[process_id] = {
                    "mean": np.mean(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "count": len(times),
                    "config_mean": process.mean_time,  # Configured mean for comparison
                    "config_std": process.std_time,
                }
            else:
                stats[process_id] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                    "config_mean": process.mean_time,
                    "config_std": process.std_time,
                }

        return stats


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
