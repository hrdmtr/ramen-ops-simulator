# 実運用システム設計

## 概要

強化学習モデルと実店舗オペレーションシステムを分離し、実運用可能なアーキテクチャを設計します。

## アーキテクチャ全体像

```
┌──────────────────────────────────────────────────────────────┐
│                     実店舗環境                                │
│  - 厨房（茹で、盛り付け、配膳）                               │
│  - 座席エリア                                                │
│  - 顧客の流れ                                                │
└──────────────────────────────────────────────────────────────┘
                            ↓ センサー・入力
┌──────────────────────────────────────────────────────────────┐
│              データ収集層（Real-time Data Collection）         │
│                                                              │
│  1. 注文データ（POSシステム）                                 │
│  2. 工程進捗（スタッフ入力 or センサー）                       │
│  3. 座席状態（テーブル管理システム）                          │
│  4. 皿の状態（スタッフ入力 or RFIDタグ）                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│           状態推定層（State Estimator）                       │
│                                                              │
│  - リアルタイム観測データの整形                               │
│  - 欠損データの補完                                          │
│  - シミュレーターと同じフォーマットに変換                      │
│                                                              │
│  実装: src/realtime/state_estimator.py                       │
└──────────────────────────────────────────────────────────────┘
                            ↓ 観測ベクトル
┌──────────────────────────────────────────────────────────────┐
│         AIモデル推論層（AI Inference Engine）                 │
│                                                              │
│  - 学習済みPPOモデルのロード                                  │
│  - 観測 → 行動推論（<100ms）                                 │
│  - 行動番号 → 具体的指示への変換                             │
│                                                              │
│  実装: src/realtime/inference_engine.py                      │
└──────────────────────────────────────────────────────────────┘
                            ↓ 指示
┌──────────────────────────────────────────────────────────────┐
│         指示配信層（Instruction Delivery）                    │
│                                                              │
│  - タブレット/スマホへの通知                                  │
│  - 音声案内                                                  │
│  - 指示履歴の記録                                            │
│  - スタッフのフィードバック受付                               │
│                                                              │
│  実装: src/realtime/instruction_delivery.py                  │
└──────────────────────────────────────────────────────────────┘
                            ↓ 指示実行
┌──────────────────────────────────────────────────────────────┐
│                   スタッフ（人間）                            │
│  - 指示を受け取る                                            │
│  - 実行する                                                  │
│  - フィードバックを返す                                       │
└──────────────────────────────────────────────────────────────┘
```

## 1. データ収集層の実装

### 必要なデータソース

#### A. POSシステム（注文データ）
```python
# 必要な情報
{
    "order_id": "ORD_001",
    "arrival_time": "2026-02-15T12:00:00",
    "seat_number": 3,
    "menu_items": ["ramen"],
    "status": "ordered"  # ordered, cooking, served, completed
}
```

#### B. 工程進捗トラッキング
```python
# スタッフがボタンを押す、またはセンサーで検知
{
    "order_id": "ORD_001",
    "process": "boil",  # boil, plate, serve
    "status": "started" | "completed",
    "timestamp": "2026-02-15T12:01:30"
}
```

#### C. 座席管理
```python
# テーブル管理システムから
{
    "seat_number": 3,
    "status": "occupied" | "eating" | "waiting_for_food",
    "customer_arrival": "2026-02-15T12:00:00",
    "seated_at": "2026-02-15T12:00:15"
}
```

#### D. 皿の状態
```python
# スタッフ入力 or RFIDカウンター
{
    "clean_dishes": 25,
    "dirty_dishes": 5,
    "dishes_being_washed": 0
}
```

### 簡易版データ収集（最小構成）

現実的には、**スタッフがタブレットで簡単に入力**する方式が最も実装しやすいです。

```python
# スタッフが押すボタン
- 「来店」ボタン → 客が入店
- 「着席」ボタン → 客が座った
- 「茹で開始」ボタン → 麺を茹で始めた
- 「盛り付け開始」ボタン
- 「提供完了」ボタン
- 「退店」ボタン → 客が出た
- 「皿洗い開始/完了」ボタン
```

## 2. 状態推定層の実装

シミュレーターの `get_state()` と同じ形式のデータを生成します。

```python
# src/realtime/state_estimator.py

class RealTimeStateEstimator:
    """実店舗のデータから、シミュレーターと同じ形式の状態を推定"""

    def __init__(self):
        # リアルタイムデータストア（Redis等）
        self.active_orders = {}
        self.seated_customers = {}
        self.waiting_customers = []
        self.clean_dishes = 30
        self.dirty_dishes = 0
        self.dishes_being_washed = 0

        # スタッフの現在の行動
        self.float_staff_action = "DO_NOTHING"
        self.float_staff_action_start_time = None
        self.pending_action = None
        self.pending_time_left = 0.0

    def update_from_event(self, event):
        """イベント（スタッフ入力、センサー）から状態を更新"""
        if event["type"] == "order_started":
            self._add_order(event)
        elif event["type"] == "process_started":
            self._update_process(event)
        elif event["type"] == "dish_washed":
            self._update_dishes(event)
        # ... 他のイベント処理

    def get_state_for_ai(self):
        """AIモデルが期待する形式の状態を返す"""
        # シミュレーターの get_state() と同じ形式
        state = {
            "current_action": self.float_staff_action,
            "pending_action": self.pending_action,
            "pending_time_left": self.pending_time_left,

            # WIP
            "boil_wait": self._count_orders_in("boil", "waiting"),
            "boil_in_progress": self._count_orders_in("boil", "in_progress"),
            "plate_wait": self._count_orders_in("plate", "waiting"),
            "plate_in_progress": self._count_orders_in("plate", "in_progress"),
            "serve_wait": self._count_orders_in("serve", "waiting"),
            "serve_in_progress": self._count_orders_in("serve", "in_progress"),

            # Bottleneck
            "bottleneck_wip": self._calculate_bottleneck(),

            # Customers
            "waiting_customers": len(self.waiting_customers),
            "seated_customers": len(self.seated_customers),
            "customers_eating": self._count_eating(),
            "customers_waiting_for_food": self._count_waiting_for_food(),
            "empty_seats": 10 - len(self.seated_customers),

            # Dishes
            "clean_dishes": self.clean_dishes,
            "dirty_dishes": self.dirty_dishes,
            "dishes_being_washed": self.dishes_being_washed,

            # Float staff idle time
            "float_idle_time": self._calculate_idle_time(),
        }
        return state

    def _count_orders_in(self, process, status):
        """特定の工程・状態にある注文数をカウント"""
        count = 0
        for order in self.active_orders.values():
            if order["current_process"] == process and order["status"] == status:
                count += 1
        return count
```

## 3. AI推論層の実装

学習済みモデルを使って、リアルタイムで行動を推論します。

```python
# src/realtime/inference_engine.py

import numpy as np
from stable_baselines3 import PPO
from src.config import ACTION_SPACE, MAX_ARRIVALS_PER_STEP, MAX_WIP_PER_PROCESS, PROCESSES

class AIInferenceEngine:
    """学習済みモデルを使ってリアルタイムで指示を生成"""

    def __init__(self, model_path: str):
        # 学習済みモデルをロード
        self.model = PPO.load(model_path)

        # 前回の到着数（観測に必要）
        self.last_arrivals = 0

    def build_observation(self, state, last_arrivals=0):
        """状態からAIモデルの入力観測ベクトルを構築

        これは eval_with_instruction_log.py の build_observation() と同一
        """
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
        from src.config import MAX_IDLE_TIME
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
        from src.config import MAX_PENDING_TIME
        pending_time = state.get("pending_time_left", 0.0)
        obs.append(min(pending_time / MAX_PENDING_TIME, 1.0))

        # 8. Restaurant/customer state (normalized)
        from src.config import MAX_WAITING_CUSTOMERS, TOTAL_SEATS
        obs.append(min(state.get("waiting_customers", 0) / MAX_WAITING_CUSTOMERS, 1.0))
        obs.append(min(state.get("seated_customers", 0) / TOTAL_SEATS, 1.0))
        obs.append(min(state.get("customers_eating", 0) / TOTAL_SEATS, 1.0))
        obs.append(min(state.get("customers_waiting_for_food", 0) / TOTAL_SEATS, 1.0))
        obs.append(min(state.get("empty_seats", 0) / TOTAL_SEATS, 1.0))

        # 9. Dish state (normalized)
        from src.config import MAX_DISHES
        obs.append(min(state.get("clean_dishes", 0) / MAX_DISHES, 1.0))
        obs.append(min(state.get("dirty_dishes", 0) / MAX_DISHES, 1.0))
        obs.append(min(state.get("dishes_being_washed", 0) / MAX_DISHES, 1.0))

        return np.array(obs, dtype=np.float32)

    def predict_action(self, state):
        """状態から行動を予測"""
        # 観測ベクトルを構築
        obs = self.build_observation(state, self.last_arrivals)

        # モデルで推論（決定論的）
        action_idx, _ = self.model.predict(obs, deterministic=True)

        # 行動番号 → 行動名
        action_name = ACTION_SPACE[action_idx]

        return action_name

    def update_arrivals(self, new_arrivals):
        """到着数を更新（次回の観測に使用）"""
        self.last_arrivals = new_arrivals
```

## 4. 指示配信層の実装

AIの決定をスタッフに伝える部分です。

```python
# src/realtime/instruction_delivery.py

from datetime import datetime
from enum import Enum
import json

class InstructionType(Enum):
    DO_NOTHING = "何もしない"
    BOIL_HELP = "茹で場のヘルプ"
    PLATE_HELP = "盛り付けヘルプ"
    SERVE_HELP = "配膳ヘルプ"
    DISH_WASH = "皿洗い"

class InstructionDeliverySystem:
    """指示をスタッフに配信するシステム"""

    def __init__(self):
        self.instruction_history = []

    def deliver_instruction(self, action_name: str, state: dict):
        """指示をスタッフに配信"""

        # 指示を日本語化
        instruction_text = InstructionType[action_name].value

        # 指示の記録
        instruction_record = {
            "timestamp": datetime.now().isoformat(),
            "action": action_name,
            "instruction_text": instruction_text,
            "state_snapshot": state,
        }
        self.instruction_history.append(instruction_record)

        # 配信（複数チャンネル）
        self._send_to_tablet(instruction_text, state)
        self._send_to_audio(instruction_text)
        self._log_to_database(instruction_record)

        return instruction_record

    def _send_to_tablet(self, instruction: str, state: dict):
        """タブレットに通知を送信"""
        # WebSocket、Firebase Cloud Messaging、等
        notification = {
            "title": "新しい指示",
            "body": instruction,
            "priority": "high",
            "data": {
                "waiting_customers": state.get("waiting_customers", 0),
                "boil_wip": state.get("boil_wait", 0) + state.get("boil_in_progress", 0),
                "clean_dishes": state.get("clean_dishes", 0),
            }
        }
        # 実装例: push_to_device(notification)
        print(f"[タブレット通知] {instruction}")

    def _send_to_audio(self, instruction: str):
        """音声案内（オプション）"""
        # Text-to-Speech API を使用
        # 実装例: tts.speak(instruction)
        print(f"[音声案内] {instruction}")

    def _log_to_database(self, record: dict):
        """データベースに記録（分析用）"""
        # 実装例: db.insert("instruction_log", record)
        pass

    def get_staff_feedback(self, instruction_id: str):
        """スタッフからのフィードバックを受け取る

        例:
        - 「実行できました」
        - 「今は無理です」（別の作業中）
        - 「優先度を変更してください」
        """
        # スタッフがタブレットでボタンを押す
        pass
```

## 5. メインループ（リアルタイム実行）

全てを統合したメインループです。

```python
# src/realtime/main_loop.py

import time
from src.realtime.state_estimator import RealTimeStateEstimator
from src.realtime.inference_engine import AIInferenceEngine
from src.realtime.instruction_delivery import InstructionDeliverySystem

class RealTimeOperationSystem:
    """リアルタイムオペレーションシステム"""

    def __init__(self, model_path: str):
        self.state_estimator = RealTimeStateEstimator()
        self.ai_engine = AIInferenceEngine(model_path)
        self.instruction_system = InstructionDeliverySystem()

        # 30秒ごとに意思決定（シミュレーターと同じ周期）
        self.decision_interval = 30  # seconds

    def run(self):
        """メインループを実行"""
        print("リアルタイムオペレーションシステムを起動しました")

        while True:
            try:
                # 1. 現在の店舗状態を取得
                state = self.state_estimator.get_state_for_ai()

                # 2. AIモデルで行動を予測
                action_name = self.ai_engine.predict_action(state)

                # 3. スタッフに指示を配信
                instruction = self.instruction_system.deliver_instruction(action_name, state)

                # ログ出力
                print(f"[{instruction['timestamp']}] 指示: {instruction['instruction_text']}")
                print(f"  待客: {state['waiting_customers']}人 | "
                      f"茹WIP: {state['boil_wait'] + state['boil_in_progress']} | "
                      f"皿: {state['clean_dishes']}枚")

                # 4. 次の意思決定まで待機
                time.sleep(self.decision_interval)

            except KeyboardInterrupt:
                print("\nシステムを終了します")
                break
            except Exception as e:
                print(f"エラー: {e}")
                time.sleep(5)  # エラー時は5秒待って再試行

    def handle_event(self, event):
        """外部イベント（スタッフ入力等）を処理"""
        self.state_estimator.update_from_event(event)


# 起動スクリプト
if __name__ == "__main__":
    system = RealTimeOperationSystem(
        model_path="models/base_20260215_094011/final_model.zip"
    )
    system.run()
```

## 6. シミュレーターとの違い

### シミュレーターでできること（学習用）
- 未来を予測してステップを進める
- 完全な情報を持っている
- 時間を加速できる
- リセットして何度でも試せる

### 実運用システムでできること（実店舗用）
- **現在の状態のみ**を観測
- **不完全な情報**（欠損や遅延がある）
- **リアルタイム**でしか動作できない
- **やり直しができない**

## 7. 検証方法

### A. シミュレーターとの整合性検証

```python
# tests/test_realtime_consistency.py

def test_state_consistency():
    """シミュレーターと実運用システムで同じ状態表現になるか"""

    # シミュレーターの状態
    from src.sim import RamenShopSimulator
    sim = RamenShopSimulator(scenario="base", seed=42)
    sim_state = sim.get_state()

    # 実運用システムの状態（同じデータから生成）
    from src.realtime.state_estimator import RealTimeStateEstimator
    estimator = RealTimeStateEstimator()
    # ... 同じイベントを再現 ...
    real_state = estimator.get_state_for_ai()

    # 両者が一致するか検証
    assert sim_state["boil_wait"] == real_state["boil_wait"]
    assert sim_state["clean_dishes"] == real_state["clean_dishes"]
    # ...
```

### B. 推論時間の検証

```python
def test_inference_latency():
    """推論が100ms以内に完了するか"""
    import time

    engine = AIInferenceEngine("models/base_20260215_094011/final_model.zip")
    state = {...}  # テスト用の状態

    start = time.time()
    action = engine.predict_action(state)
    latency = time.time() - start

    assert latency < 0.1, f"推論に{latency*1000:.1f}msかかりました（目標: <100ms）"
```

## 8. 段階的な導入戦略

### Phase 1: オフライン検証（シミュレーターのみ）
- シミュレーターで十分な性能が出るか検証
- 実店舗のデータを収集してシミュレーターを調整

### Phase 2: シャドウモード（指示は出さない）
- 実店舗で状態を観測
- AIが指示を「生成するだけ」（スタッフには見せない）
- 実際の人間の判断と比較

### Phase 3: アドバイスモード（スタッフが選択）
- AIの指示を「提案」として表示
- スタッフが受け入れるか拒否するか選べる
- フィードバックを収集

### Phase 4: 自動モード（完全な導入）
- AIの指示を実行する
- 異常時のみ人間が介入

## 9. 必要な追加機能

### A. 安全機能
```python
class SafetyChecker:
    """AIの指示が安全かチェック"""

    def is_safe(self, action: str, state: dict) -> bool:
        # 異常な指示を検出
        if state["clean_dishes"] < 5 and action != "DISH_WASH":
            return False  # 皿が少ないのに洗わない → 危険

        if state["waiting_customers"] > 8 and action == "DO_NOTHING":
            return False  # 多数待ちなのに何もしない → 危険

        return True
```

### B. 説明機能
```python
class ExplainableAI:
    """なぜその指示を出したか説明"""

    def explain(self, action: str, state: dict) -> str:
        if action == "BOIL_HELP":
            boil_wip = state["boil_wait"] + state["boil_in_progress"]
            return f"茹で場に{boil_wip}個の注文が溜まっているため"

        if action == "DISH_WASH":
            clean = state["clean_dishes"]
            return f"きれいな皿が{clean}枚しかないため"

        return "最適な判断として"
```

## 10. まとめ

### 分離は可能か？ → **はい、可能です**

**分離のポイント**:
1. **観測の標準化**: `get_state()` の形式を守る
2. **推論の独立性**: モデルは状態 → 行動のマッピングのみ
3. **明確なインターフェース**: 状態推定、推論、配信を分離

**実装の優先順位**:
1. まず `state_estimator.py` と `inference_engine.py` を実装
2. 簡易的なデータ入力（タブレットボタン）で動作確認
3. 段階的に機能を追加（音声、説明、安全機能）

**次のステップ**:
- この設計ドキュメントをベースに実装を開始するか
- 実店舗のデータ収集方法を具体化するか

どちらを進めますか？
