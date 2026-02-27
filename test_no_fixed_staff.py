"""
Test script to verify that processes don't progress without explicit instructions.

With FIXED_STAFF = 0, all processes should require explicit float staff assignment.
"""

import sys
sys.path.insert(0, 'src')

from sim import RamenShopSimulator

def test_no_fixed_staff():
    """Test that orders stay in queue without instructions."""

    print("=" * 60)
    print("Testing No-Fixed-Staff Configuration")
    print("=" * 60)
    print()

    sim = RamenShopSimulator(scenario="base", seed=42, enable_state_log=True)

    print("Initial state:")
    print(f"  Float staff action: {sim.float_staff.current_action}")
    print()

    # Step 1: Create some orders
    print("Step 1: Advancing 60 seconds with DO_NOTHING action")
    print("        (Some orders should arrive)")
    print()

    for i in range(6):
        sim.step(action="DO_NOTHING", arrival_rate=1.0/60.0, duration=10.0)
        state = sim.get_state()
        print(f"  t={sim.current_time:.0f}s - boil_wait={state['boil_wait']}, "
              f"boil_in_progress={state['boil_in_progress']}, "
              f"completed={state['total_completed']}")

    print()
    print("Expected behavior: Orders should be WAITING in boil queue, NOT in progress")
    print("                  (because no staff is assigned to boiling)")
    print()

    # Step 2: Assign float staff to help with boiling
    print("Step 2: Assigning float staff to BOIL_HELP")
    print()

    for i in range(6):
        sim.step(action="BOIL_HELP", arrival_rate=1.0/60.0, duration=10.0)
        state = sim.get_state()
        print(f"  t={sim.current_time:.0f}s - boil_wait={state['boil_wait']}, "
              f"boil_in_progress={state['boil_in_progress']}, "
              f"completed={state['total_completed']}")

    print()
    print("Expected behavior: Orders should now START PROCESSING (move to in_progress)")
    print()

    # Step 3: Check if plate queue builds up
    print("Step 3: Continuing with BOIL_HELP (plate queue should build up)")
    print()

    for i in range(6):
        sim.step(action="BOIL_HELP", arrival_rate=1.0/60.0, duration=10.0)
        state = sim.get_state()
        print(f"  t={sim.current_time:.0f}s - plate_wait={state['plate_wait']}, "
              f"plate_in_progress={state['plate_in_progress']}, "
              f"completed={state['total_completed']}")

    print()
    print("Expected behavior: Completed boiling should wait in PLATE queue")
    print("                  (no one working on plating)")
    print()

    # Step 4: Assign float staff to plating
    print("Step 4: Switching float staff to PLATE_HELP")
    print()

    for i in range(6):
        sim.step(action="PLATE_HELP", arrival_rate=1.0/60.0, duration=10.0)
        state = sim.get_state()
        print(f"  t={sim.current_time:.0f}s - plate_wait={state['plate_wait']}, "
              f"plate_in_progress={state['plate_in_progress']}, "
              f"serve_wait={state['serve_wait']}, "
              f"completed={state['total_completed']}")

    print()
    print("Expected behavior: Plate queue should start processing")
    print("                  Completed plating should wait in SERVE queue")
    print()

    # Step 5: Assign float staff to serving
    print("Step 5: Switching float staff to SERVE_HELP")
    print()

    for i in range(6):
        sim.step(action="SERVE_HELP", arrival_rate=1.0/60.0, duration=10.0)
        state = sim.get_state()
        print(f"  t={sim.current_time:.0f}s - serve_wait={state['serve_wait']}, "
              f"serve_in_progress={state['serve_in_progress']}, "
              f"completed={state['total_completed']}")

    print()
    print("Expected behavior: Serve queue should process and complete orders")
    print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total orders completed: {len(sim.completed_orders)}")
    print()
    print("✅ SUCCESS: If orders only progressed when staff was assigned!")
    print("❌ FAILURE: If orders progressed without staff assignment")
    print("=" * 60)


if __name__ == "__main__":
    test_no_fixed_staff()
