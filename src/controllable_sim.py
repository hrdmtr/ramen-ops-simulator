"""
Controllable Simulator for Manual Testing and Verification.

Extends RamenShopSimulator to allow manual customer addition and
provides callbacks for order events.
"""

from typing import Optional, Callable, List
from dataclasses import dataclass
from sim import RamenShopSimulator, Customer, Order
from config import TOTAL_SEATS, INITIAL_CLEAN_DISHES


@dataclass
class PendingOrder:
    """Represents a customer who will place an order after a delay."""
    customer_id: int
    order_time: float  # When the order will be placed


class ControllableSimulator(RamenShopSimulator):
    """
    Extended simulator that supports:
    - Manual customer addition via add_manual_customer()
    - 5-second delay before order is placed (simulating ticket machine)
    - Callback when order is confirmed (for immediate boil instruction)
    """

    def __init__(
        self,
        scenario: str = "base",
        seed: Optional[int] = None,
        enable_logging: bool = False,
        enable_state_log: bool = False,
        on_order_confirmed: Optional[Callable[[int, float], None]] = None
    ):
        """
        Initialize controllable simulator.

        Args:
            scenario: Scenario name
            seed: Random seed
            enable_logging: Enable detailed logging
            enable_state_log: Enable state change logging
            on_order_confirmed: Callback function(order_id, current_time) when order is placed
        """
        super().__init__(scenario, seed, enable_logging, enable_state_log)
        self.pending_orders: List[PendingOrder] = []
        self.on_order_confirmed = on_order_confirmed

    def reset(self) -> None:
        """Reset simulator including manual customer tracking."""
        super().reset()
        self.pending_orders = []

    def add_manual_customer(self) -> dict:
        """
        Manually add a customer to the restaurant.

        Returns:
            Dictionary with customer info and status
        """
        # Create customer
        customer = Customer(
            id=self.customer_counter,
            order_id=-1,  # Will be assigned when order is placed
            arrival_time=self.current_time
        )
        self.customer_counter += 1

        # Check if seats are available
        occupied_seats = len(self.seated_customers)
        if occupied_seats < TOTAL_SEATS:
            # Seat immediately
            customer.seat_time = self.current_time
            self.seated_customers.append(customer)

            # Schedule order for 5 seconds later
            pending = PendingOrder(
                customer_id=customer.id,
                order_time=self.current_time + 5.0
            )
            self.pending_orders.append(pending)

            self._log_state_change(
                f"手動客追加→着席 (客#{customer.id}) - 5秒後に注文予定"
            )

            return {
                "success": True,
                "customer_id": customer.id,
                "status": "seated",
                "order_time": pending.order_time,
                "message": f"Customer #{customer.id} seated. Order will be placed in 5 seconds."
            }
        else:
            # Add to waiting queue
            self.waiting_customers.append(customer)
            self._log_state_change(f"手動客追加→満席で待ち (客#{customer.id})")

            return {
                "success": True,
                "customer_id": customer.id,
                "status": "waiting",
                "order_time": None,
                "message": f"Customer #{customer.id} is waiting. Will be seated when available."
            }

    def _process_pending_orders(self) -> None:
        """Process pending orders that are ready to be placed."""
        orders_to_place = [
            po for po in self.pending_orders
            if po.order_time <= self.current_time
        ]

        for pending in orders_to_place:
            # Find customer
            customer = next(
                (c for c in self.seated_customers if c.id == pending.customer_id),
                None
            )

            if customer is None:
                # Customer may have left already or still waiting
                self.pending_orders.remove(pending)
                continue

            # Create order only if clean dishes are available
            if self.clean_dishes > 0:
                order = Order(
                    id=self.order_counter,
                    arrival_time=customer.arrival_time
                )
                self.orders.append(order)
                customer.order_id = self.order_counter
                self.order_to_customer[self.order_counter] = customer.id

                # Reserve a clean dish
                self.clean_dishes -= 1
                order_id = self.order_counter
                self.order_counter += 1

                self._log_state_change(
                    f"注文確定 (客#{customer.id}, 注文#{order_id}) → 即座に茹で指示"
                )

                # Trigger callback for immediate boil instruction
                # Pass current state to avoid lock issues
                if self.on_order_confirmed:
                    state = self.get_state()
                    self.on_order_confirmed(order_id, self.current_time, state)
            else:
                # No clean dishes available
                self._log_state_change(
                    f"注文不可 (客#{customer.id}) ⚠️ 皿不足"
                )

            # Remove from pending list
            self.pending_orders.remove(pending)

    def step(self, action: str, arrival_rate: float, duration: float = None) -> dict:
        """
        Advance simulation by one time step.

        Overrides parent to include pending order processing.
        """
        if duration is None:
            from config import DECISION_INTERVAL
            duration = DECISION_INTERVAL

        # Process pending orders first
        self._process_pending_orders()

        # Call parent step
        return super().step(action, arrival_rate, duration)

    def get_state_for_ui(self) -> dict:
        """
        Get state formatted for UI display.

        Returns comprehensive state including pending orders.
        """
        base_state = self.get_state()

        # Add pending order information
        base_state["pending_orders"] = len(self.pending_orders)
        base_state["next_order_time"] = (
            min(po.order_time for po in self.pending_orders)
            if self.pending_orders else None
        )

        # Format time
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        base_state["time_formatted"] = f"{minutes:02d}:{seconds:02d}"

        return base_state


if __name__ == "__main__":
    # Test manual customer addition
    def on_order_callback(order_id: int, time: float, state: dict):
        print(f"🔔 ORDER CONFIRMED: #{order_id} at t={time:.1f}s → Issue BOIL instruction!")
        print(f"   State: boil_wait={state.get('boil_wait', 0)}, boil_in_progress={state.get('boil_in_progress', 0)}")

    sim = ControllableSimulator(
        scenario="base",
        seed=42,
        enable_state_log=True,
        on_order_confirmed=on_order_callback
    )

    print("=== Testing Manual Customer Addition ===\n")

    # Add 3 customers manually
    for i in range(3):
        result = sim.add_manual_customer()
        print(f"\n{result['message']}\n")

    # Simulate time passing
    print("\n=== Advancing time by 10 seconds ===\n")
    for t in range(10):
        sim.step(action="DO_NOTHING", arrival_rate=0.0, duration=1.0)

    print(f"\n=== Final State ===")
    print(f"Orders created: {len(sim.orders) + len(sim.completed_orders)}")
    print(f"Customers seated: {len(sim.seated_customers)}")
    print(f"Pending orders: {len(sim.pending_orders)}")
