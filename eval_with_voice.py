"""
Evaluate trained model with voice announcements.
AI instructions are announced via text-to-speech.
"""

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import time

from src.sim import RamenShopSimulator
from src.voice import VoiceOutput
from src.staff_assignment import StaffAssignmentManager
from src.config import (
    SCENARIOS, DEFAULT_SCENARIO, PROCESSES, ACTION_SPACE,
    MAX_ARRIVALS_PER_STEP, MAX_WIP_PER_PROCESS, MAX_IDLE_TIME,
    MAX_PENDING_TIME, MAX_WAITING_CUSTOMERS, MAX_DISHES, DECISION_INTERVAL,
    TOTAL_SEATS
)


def build_observation(state, last_arrivals=0):
    """Build observation vector matching RamenShopEnv._get_observation()."""
    obs = []

    # 1. Arrivals last step (normalized)
    obs.append(min(last_arrivals / MAX_ARRIVALS_PER_STEP, 1.0))

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

    # 8. Restaurant/customer state (normalized)
    obs.append(min(state.get("waiting_customers", 0) / MAX_WAITING_CUSTOMERS, 1.0))
    obs.append(min(state.get("seated_customers", 0) / TOTAL_SEATS, 1.0))
    obs.append(min(state.get("customers_eating", 0) / TOTAL_SEATS, 1.0))
    obs.append(min(state.get("customers_waiting_for_food", 0) / TOTAL_SEATS, 1.0))
    obs.append(min(state.get("empty_seats", 0) / TOTAL_SEATS, 1.0))

    # 9. Dish state (normalized)
    obs.append(min(state.get("clean_dishes", 0) / MAX_DISHES, 1.0))
    obs.append(min(state.get("dirty_dishes", 0) / MAX_DISHES, 1.0))
    obs.append(min(state.get("dishes_being_washed", 0) / MAX_DISHES, 1.0))

    return np.array(obs, dtype=np.float32)


def format_time(seconds):
    """Format time as HH:MM:SS"""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def run_with_voice(model_path: Path, scenario: str, duration_minutes: int, voice_enabled: bool = True, realtime: bool = False, speech_rate: int = 200, time_scale: float = 1.0):
    """Run simulation with voice announcements."""

    print("=" * 120)
    print("音声案内付きシミュレーション")
    print("=" * 120)
    print(f"モデル: {model_path}")
    print(f"シナリオ: {scenario}")
    print(f"営業時間: {duration_minutes}分")
    print(f"音声案内: {'有効' if voice_enabled else '無効'}")
    if voice_enabled:
        speed_multiplier = speech_rate / 200
        print(f"音声速度: {speech_rate} wpm (約{speed_multiplier:.1f}倍速)")
    if realtime:
        print(f"リアルタイム実行: 有効 (実時間の{time_scale:.1f}倍速)")
        print(f"  → 30秒ステップを{DECISION_INTERVAL / time_scale:.1f}秒で実行")
    else:
        print(f"リアルタイム実行: 無効 (最高速)")
    print("=" * 120)
    print()

    # Initialize voice output
    voice = VoiceOutput(enabled=voice_enabled, voice_type="system", speech_rate=speech_rate)

    # Initialize staff assignment manager
    staff_manager = StaffAssignmentManager(strategy="round_robin")

    # Load model
    model = PPO.load(model_path)

    # Initialize simulator
    scenario_config = SCENARIOS[scenario]
    sim = RamenShopSimulator(
        scenario=scenario,
        seed=42
    )

    # Calculate steps
    total_steps = int(duration_minutes * 60 / DECISION_INTERVAL)

    last_arrivals = 0
    action_counts = {action: 0 for action in ACTION_SPACE}
    voice_announcement_count = 0

    print("【営業開始】")
    if voice_enabled:
        print("  🔊 音声: 営業を開始します")
        voice.speak("営業を開始します", blocking=True)
        voice_announcement_count += 1
        print(f"  ✓ 音声再生完了 (累計: {voice_announcement_count}回)")
    print()

    for step in range(total_steps):
        # Get current state
        state = sim.get_state()

        # Build observation
        obs = build_observation(state, last_arrivals)

        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        action_name = ACTION_SPACE[action]
        action_counts[action_name] += 1

        # Announce action change
        if action_name != state.get("current_action"):
            # Assign to a staff member
            staff_name = staff_manager.assign_task(action_name, sim.current_time)

            if staff_name:
                print(f"[{format_time(sim.current_time)}] AI指示: {staff_name} → {action_name}")
                # Use blocking mode to prevent voice overlap
                voice.announce_instruction(action_name, staff_name=staff_name, blocking=True, state=state)
                voice_announcement_count += 1
                print(f"  ✓ 音声再生完了 (累計: {voice_announcement_count}回)")
            else:
                print(f"[{format_time(sim.current_time)}] AI指示: {action_name}")
                voice.announce_instruction(action_name, blocking=True, state=state)
                voice_announcement_count += 1
                print(f"  ✓ 音声再生完了 (累計: {voice_announcement_count}回)")

        # Take step
        reward_breakdown = sim.step(action_name, arrival_rate=scenario_config["arrival_rate"])

        # Track arrivals for next observation
        last_arrivals = len([o for o in sim.orders if o.arrival_time > sim.current_time - DECISION_INTERVAL])

        # Print periodic status
        if (step + 1) % 2 == 0:  # Every minute
            print(f"  [{format_time(sim.current_time)}] 完成: {len(sim.completed_orders)}杯 | "
                  f"待客: {len(sim.waiting_customers)}人 | "
                  f"在席: {len(sim.seated_customers)}人 | "
                  f"皿: {state.get('clean_dishes')}枚")

        # Realtime mode: wait for scaled time to pass
        if realtime:
            # time_scale = 10 means 30 seconds of simulation time in 3 seconds of real time
            time.sleep(DECISION_INTERVAL / time_scale)

    # Final summary
    print()
    print("=" * 120)
    print("【営業終了】")
    if voice_enabled:
        print("  🔊 音声: 営業を終了します")
        voice.speak("営業を終了します", blocking=True)
        voice_announcement_count += 1
        print(f"  ✓ 音声再生完了 (累計: {voice_announcement_count}回)")
    print("=" * 120)

    completed_orders = len(sim.completed_orders)
    departed_customers = len(sim.completed_customers)

    wait_times = [
        order.completion_time - order.arrival_time
        for order in sim.completed_orders
        if order.completion_time is not None
    ]
    avg_wait = np.mean(wait_times) if wait_times else 0

    print(f"📊 完成料理数: {completed_orders}杯 ({completed_orders / (duration_minutes / 60):.1f}杯/時間)")
    print(f"👥 退店客数: {departed_customers}人")
    print(f"⏱️  平均待ち時間: {avg_wait:.1f}秒 ({avg_wait / 60:.1f}分)")
    if voice_enabled:
        print(f"🔊 音声案内回数: {voice_announcement_count}回")

    # Action distribution
    print()
    print("🎯 指示の分布:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100.0 * count / total_steps
            print(f"   {action:>15}: {count:4d}回 ({pct:5.1f}%)")

    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Simulation with voice announcements")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO,
        choices=list(SCENARIOS.keys()),
        help=f"Scenario to run (default: {DEFAULT_SCENARIO})"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Simulation duration in minutes (default: 10 minutes)"
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice output"
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run in real-time with time scaling"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Time scale for realtime mode (default: 1.0, 10x=10.0). 10x means 30s simulated in 3s real time."
    )
    parser.add_argument(
        "--speech-rate",
        type=int,
        default=200,
        help="Speech rate in words per minute (default: 200, 2x=400, 3x=600, 5x=1000, 10x=2000)"
    )

    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}")
        return

    run_with_voice(
        args.model_path,
        args.scenario,
        args.duration,
        voice_enabled=not args.no_voice,
        realtime=args.realtime,
        speech_rate=args.speech_rate,
        time_scale=args.time_scale
    )


if __name__ == "__main__":
    main()
