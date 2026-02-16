"""
Web Server for Manual Simulator Control.

Provides REST API and WebSocket interface for browser-based
simulation control and visualization.
"""

import sys
import os
import threading
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllable_sim import ControllableSimulator
from voice import VoiceOutput
from config import SCENARIOS, DECISION_INTERVAL

# Initialize voice output
voice_output = VoiceOutput(enabled=True, voice_type="system", speech_rate=200)


# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global simulator instance
simulator = None
simulator_lock = threading.Lock()
simulation_running = False
simulation_thread = None

# Track last instruction time to avoid spamming
last_instruction_time = {
    "boil": 0.0,
    "plate": 0.0,
    "serve": 0.0,
    "dish": 0.0
}
INSTRUCTION_COOLDOWN = 10.0  # seconds between similar instructions


def on_order_confirmed(order_id: int, current_time: float, state: dict):
    """
    Callback when order is confirmed.
    Issues immediate boil instruction via voice.

    Args:
        order_id: Order ID
        current_time: Current simulation time
        state: Current simulator state (passed to avoid lock issues)
    """
    print(f"[ORDER CONFIRMED] #{order_id} at t={current_time:.1f}s", flush=True)

    # Get boil queue info from passed state
    boil_waiting = state.get("boil_wait", 0)
    boil_in_progress = state.get("boil_in_progress", 0)

    # Calculate quantity (2-4 items)
    quantity = max(2, min(4, boil_waiting + 1))

    # Issue voice instruction
    instruction_text = f"{quantity}玉茹でてください"
    print(f"[VOICE] 🔊 {instruction_text}", flush=True)

    try:
        voice_output.speak(instruction_text, blocking=False)
    except Exception as e:
        print(f"[VOICE ERROR] {e}", flush=True)

    # Broadcast event to all clients
    socketio.emit('order_confirmed', {
        'order_id': order_id,
        'time': current_time,
        'instruction': instruction_text
    })


def initialize_simulator(scenario: str = "base", seed: int = 42):
    """Initialize the simulator instance."""
    global simulator
    with simulator_lock:
        simulator = ControllableSimulator(
            scenario=scenario,
            seed=seed,
            enable_logging=False,
            enable_state_log=True,
            on_order_confirmed=on_order_confirmed
        )
    print(f"[INIT] Simulator initialized with scenario: {scenario}")


def check_and_issue_instructions(state: dict, current_time: float):
    """
    Monitor queues and issue voice instructions when needed.

    Args:
        state: Current simulator state
        current_time: Current simulation time
    """
    global last_instruction_time

    # Check plate queue (盛り付け)
    plate_wait = state.get("plate_wait", 0)
    plate_in_progress = state.get("plate_in_progress", 0)
    if plate_wait >= 1 and (current_time - last_instruction_time["plate"]) >= INSTRUCTION_COOLDOWN:
        quantity = max(2, min(4, plate_wait))
        instruction_text = f"{quantity}皿盛り付けしてください"
        print(f"[VOICE] 🔊 {instruction_text}", flush=True)
        try:
            voice_output.speak(instruction_text, blocking=False)
        except Exception as e:
            print(f"[VOICE ERROR] {e}", flush=True)
        last_instruction_time["plate"] = current_time

        socketio.emit('instruction_issued', {
            'type': 'plate',
            'time': current_time,
            'instruction': instruction_text
        })

    # Check serve queue (配膳)
    serve_wait = state.get("serve_wait", 0)
    serve_in_progress = state.get("serve_in_progress", 0)
    if serve_wait >= 1 and (current_time - last_instruction_time["serve"]) >= INSTRUCTION_COOLDOWN:
        quantity = max(2, min(4, serve_wait))
        instruction_text = f"{quantity}皿配膳してください"
        print(f"[VOICE] 🔊 {instruction_text}", flush=True)
        try:
            voice_output.speak(instruction_text, blocking=False)
        except Exception as e:
            print(f"[VOICE ERROR] {e}", flush=True)
        last_instruction_time["serve"] = current_time

        socketio.emit('instruction_issued', {
            'type': 'serve',
            'time': current_time,
            'instruction': instruction_text
        })

    # Check dirty dishes (皿洗い)
    dirty_dishes = state.get("dirty_dishes", 0)
    clean_dishes = state.get("clean_dishes", 0)
    if (dirty_dishes >= 10 or clean_dishes <= 5) and (current_time - last_instruction_time["dish"]) >= INSTRUCTION_COOLDOWN:
        if dirty_dishes >= 10:
            instruction_text = f"皿洗いをお願いします。汚れた皿が{dirty_dishes}枚あります"
        else:
            instruction_text = f"皿洗いをお願いします。綺麗な皿が残り{clean_dishes}枚です"
        print(f"[VOICE] 🔊 {instruction_text}", flush=True)
        try:
            voice_output.speak(instruction_text, blocking=False)
        except Exception as e:
            print(f"[VOICE ERROR] {e}", flush=True)
        last_instruction_time["dish"] = current_time

        socketio.emit('instruction_issued', {
            'type': 'dish',
            'time': current_time,
            'instruction': instruction_text
        })


def simulation_loop():
    """Background thread that advances the simulation."""
    global simulation_running
    print("[LOOP] Simulation loop started", flush=True)

    while simulation_running:
        with simulator_lock:
            if simulator is not None:
                # Advance simulation by 1 second
                arrival_rate = SCENARIOS["base"]["arrival_rate"]
                metrics = simulator.step(
                    action="DO_NOTHING",
                    arrival_rate=0.0,  # No automatic arrivals, only manual
                    duration=1.0
                )

                # Get current state
                state = simulator.get_state_for_ui()
                current_time = simulator.current_time

        # Check and issue instructions based on queue states
        check_and_issue_instructions(state, current_time)

        # Broadcast state to all connected clients
        socketio.emit('state_update', state)

        # Sleep for 1 second (real-time simulation)
        time.sleep(1.0)

    print("[LOOP] Simulation loop stopped", flush=True)


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize or reset the simulator."""
    data = request.get_json() or {}
    scenario = data.get('scenario', 'base')
    seed = data.get('seed', 42)

    initialize_simulator(scenario, seed)

    return jsonify({
        'success': True,
        'message': f'Simulator initialized with scenario: {scenario}'
    })


@app.route('/api/start', methods=['POST'])
def api_start():
    """Start the simulation loop."""
    global simulation_running, simulation_thread

    if simulation_running:
        return jsonify({
            'success': False,
            'message': 'Simulation is already running'
        }), 400

    print("[API] Starting simulation...", flush=True)
    simulation_running = True
    simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
    simulation_thread.start()
    print(f"[API] Thread started: {simulation_thread.is_alive()}", flush=True)

    return jsonify({
        'success': True,
        'message': 'Simulation started'
    })


@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop the simulation loop."""
    global simulation_running

    if not simulation_running:
        return jsonify({
            'success': False,
            'message': 'Simulation is not running'
        }), 400

    simulation_running = False

    return jsonify({
        'success': True,
        'message': 'Simulation stopped'
    })


@app.route('/api/add_customer', methods=['POST'])
def api_add_customer():
    """Manually add a customer to the restaurant."""
    if simulator is None:
        return jsonify({
            'success': False,
            'message': 'Simulator not initialized'
        }), 400

    with simulator_lock:
        result = simulator.add_manual_customer()

    # Broadcast updated state
    with simulator_lock:
        state = simulator.get_state_for_ui()
    socketio.emit('state_update', state)

    return jsonify(result)


@app.route('/api/state', methods=['GET'])
def api_get_state():
    """Get current simulator state."""
    if simulator is None:
        return jsonify({
            'success': False,
            'message': 'Simulator not initialized'
        }), 400

    with simulator_lock:
        state = simulator.get_state_for_ui()

    return jsonify(state)


@app.route('/api/metrics', methods=['GET'])
def api_get_metrics():
    """Get simulator metrics summary."""
    if simulator is None:
        return jsonify({
            'success': False,
            'message': 'Simulator not initialized'
        }), 400

    with simulator_lock:
        metrics = simulator.get_metrics_summary()

    return jsonify(metrics)


@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    print('[WS] Client connected')
    emit('connected', {'message': 'Connected to simulator server'})

    # Send current state if simulator is initialized
    if simulator is not None:
        with simulator_lock:
            state = simulator.get_state_for_ui()
        emit('state_update', state)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    print('[WS] Client disconnected')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Ramen Shop Simulator - Web Interface")
    print("="*60)
    print("\n🌐 Starting server at http://localhost:8080")
    print("\n📝 Usage:")
    print("  1. Open http://localhost:8080 in your browser")
    print("  2. Click 'Initialize' to start the simulator")
    print("  3. Click 'Start Simulation' to begin")
    print("  4. Click '来店' button to add customers manually")
    print("\n🔊 Voice instructions will be played when orders are confirmed")
    print("\n" + "="*60 + "\n")

    # Initialize simulator on startup
    initialize_simulator()

    # Start Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=8080, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
