// WebSocket connection
const socket = io('http://localhost:8080');

// State
let simulationRunning = false;

// DOM Elements
const initBtn = document.getElementById('initBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const addCustomerBtn = document.getElementById('addCustomerBtn');
const eventLog = document.getElementById('eventLog');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();

    // Initial state: disable buttons until initialized
    startBtn.disabled = true;
    stopBtn.disabled = true;
    addCustomerBtn.disabled = true;

    logEvent('システム起動しました。まず「初期化」ボタンを押してください。', 'success');
});

// Event Listeners
function setupEventListeners() {
    initBtn.addEventListener('click', initializeSimulator);
    startBtn.addEventListener('click', startSimulation);
    stopBtn.addEventListener('click', stopSimulation);
    addCustomerBtn.addEventListener('click', addCustomer);

    // WebSocket events
    socket.on('connect', handleConnect);
    socket.on('disconnect', handleDisconnect);
    socket.on('state_update', handleStateUpdate);
    socket.on('order_confirmed', handleOrderConfirmed);
}

// API Calls
async function initializeSimulator() {
    initBtn.disabled = true;
    initBtn.textContent = '初期化中...';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch('/api/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario: 'base', seed: 42 }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const data = await response.json();
        logEvent('✅ ' + data.message, 'success');

        // Enable start button
        startBtn.disabled = false;
    } catch (error) {
        if (error.name === 'AbortError') {
            logEvent(`❌ 初期化がタイムアウトしました`, 'error');
        } else {
            logEvent(`❌ 初期化エラー: ${error.message}`, 'error');
        }
    } finally {
        initBtn.textContent = '初期化';
        initBtn.disabled = false;
    }
}

async function startSimulation() {
    startBtn.disabled = true;
    startBtn.textContent = '開始中...';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch('/api/start', {
            method: 'POST',
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (data.success) {
            simulationRunning = true;
            stopBtn.disabled = false;
            addCustomerBtn.disabled = false;
            logEvent('✅ シミュレーション開始しました。「来店」ボタンで顧客を追加できます。', 'success');
        } else {
            startBtn.disabled = false;
            logEvent(`❌ 開始失敗: ${data.message}`, 'error');
        }
    } catch (error) {
        startBtn.disabled = false;
        if (error.name === 'AbortError') {
            logEvent(`❌ 開始がタイムアウトしました`, 'error');
        } else {
            logEvent(`❌ 開始エラー: ${error.message}`, 'error');
        }
    } finally {
        startBtn.textContent = 'シミュレーション開始';
    }
}

async function stopSimulation() {
    stopBtn.disabled = true;
    stopBtn.textContent = '停止中...';

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch('/api/stop', {
            method: 'POST',
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (data.success) {
            simulationRunning = false;
            startBtn.disabled = false;
            addCustomerBtn.disabled = true;
            logEvent('⏸️ シミュレーション停止しました', 'warning');
        } else {
            logEvent(`❌ 停止失敗: ${data.message}`, 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            logEvent(`❌ 停止がタイムアウトしました`, 'error');
        } else {
            logEvent(`❌ 停止エラー: ${error.message}`, 'error');
        }
    } finally {
        stopBtn.disabled = false;
        stopBtn.textContent = '停止';
    }
}

async function addCustomer() {
    if (!simulationRunning) {
        logEvent('⚠️ 先にシミュレーションを開始してください', 'warning');
        return;
    }

    // Temporarily disable button to prevent double-clicks
    const originalText = addCustomerBtn.textContent;
    addCustomerBtn.disabled = true;
    addCustomerBtn.textContent = '来店処理中...';

    try {
        // Add timeout to fetch request
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

        const response = await fetch('/api/add_customer', {
            method: 'POST',
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        const data = await response.json();

        if (data.success) {
            logEvent(`👤 ${data.message}`, 'success');
            logEvent(`🎫 5秒後に券売機で注文予定`, 'info');
        } else {
            logEvent(`❌ 来店失敗: ${data.message}`, 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            logEvent(`❌ 来店処理がタイムアウトしました`, 'error');
        } else {
            logEvent(`❌ 来店エラー: ${error.message}`, 'error');
        }
    } finally {
        // Re-enable button immediately
        addCustomerBtn.disabled = false;
        addCustomerBtn.textContent = originalText;
    }
}

// WebSocket Handlers
function handleConnect() {
    logEvent('🔌 サーバーに接続しました', 'success');
}

function handleDisconnect() {
    logEvent('🔌 サーバーから切断されました', 'warning');
}

function handleStateUpdate(state) {
    updateUI(state);
}

function handleOrderConfirmed(data) {
    logEvent(`🔔 注文確定 (注文#${data.order_id}) → 🔊 指示: ${data.instruction}`, 'voice');
}

// UI Update
function updateUI(state) {
    // Time
    document.getElementById('timeDisplay').textContent = state.time_formatted || '00:00';

    // Customers
    document.getElementById('waitingCustomers').textContent = state.waiting_customers || 0;
    document.getElementById('seatedCustomers').textContent = state.seated_customers || 0;
    document.getElementById('waitingForFood').textContent = state.customers_waiting_for_food || 0;
    document.getElementById('eating').textContent = state.customers_eating || 0;
    document.getElementById('emptySeats').textContent = state.empty_seats || 10;

    // Cooking
    document.getElementById('boilWait').textContent = state.boil_wait || 0;
    document.getElementById('boilProgress').textContent = state.boil_in_progress || 0;
    document.getElementById('plateWait').textContent = state.plate_wait || 0;
    document.getElementById('plateProgress').textContent = state.plate_in_progress || 0;
    document.getElementById('serveWait').textContent = state.serve_wait || 0;
    document.getElementById('serveProgress').textContent = state.serve_in_progress || 0;

    // Staff
    const actionElement = document.getElementById('currentAction');
    actionElement.textContent = state.current_action || 'DO_NOTHING';
    actionElement.className = 'value staff-action';

    // Color code based on action
    if (state.current_action === 'DO_NOTHING') {
        actionElement.style.background = '#9E9E9E';
    } else if (state.current_action === 'DISH_WASH') {
        actionElement.style.background = '#2196F3';
    } else {
        actionElement.style.background = '#4CAF50';
    }

    document.getElementById('pendingAction').textContent =
        state.pending_action || 'なし';

    // Dishes
    document.getElementById('cleanDishes').textContent = state.clean_dishes || 0;
    document.getElementById('dirtyDishes').textContent = state.dirty_dishes || 0;
    document.getElementById('washingDishes').textContent = state.dishes_being_washed || 0;

    // Metrics
    document.getElementById('completedOrders').textContent = state.total_completed || 0;
    document.getElementById('pendingOrders').textContent = state.pending_orders || 0;
}

function logEvent(message, type = 'info') {
    const item = document.createElement('div');
    item.className = `event-item ${type}`;

    const timestamp = new Date().toLocaleTimeString('ja-JP');
    item.textContent = `[${timestamp}] ${message}`;

    eventLog.insertBefore(item, eventLog.firstChild);

    // Keep only last 50 events
    while (eventLog.children.length > 50) {
        eventLog.removeChild(eventLog.lastChild);
    }
}

// Helper function to format time
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}
