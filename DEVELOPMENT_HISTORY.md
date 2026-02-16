# 開発履歴とコンテキスト

## プロジェクト概要

ラーメン店のスタッフ配置・指示を最適化する強化学習PoCシステム。**スループット最大化**と**心理的安全性（idle削減）**の両立を目指す。

### 目的
- フロートスタッフの動的配置を強化学習で最適化
- 顧客満足度（待ち時間）とスタッフ満足度（アイドル時間削減）のバランス
- 実運用を考慮した制約の下での学習

## アーキテクチャ

### コアコンポーネント

1. **離散事象シミュレーター** (`src/sim.py`)
   - ポアソン到着プロセスによる顧客生成
   - 多段階処理パイプライン（boil → plate → serve）
   - 30秒の意思決定周期（DECISION_INTERVAL）
   - 10秒の指示反映遅延（INSTRUCTION_DELAY）
   - 顧客ライフサイクル管理（到着 → 着席 → 注文 → 調理 → 提供 → 食事 → 退店）
   - 皿の循環管理（きれい → 使用中 → 汚れ → 洗浄中 → きれい）

2. **Gymnasium環境** (`src/env.py`)
   - PPO学習対応の標準化されたインターフェース
   - 正規化された観測空間（0-1範囲）
   - 離散行動空間（DO_NOTHING, BOIL_HELP, PLATE_HELP, SERVE_HELP, DISH_WASH）

3. **多目的報酬設計** (`src/reward.py`)
   - 完成杯数報酬（+10/杯）
   - 待ち時間ペナルティ（-0.1/秒）
   - アイドル時間ペナルティ（-0.2/秒）
   - WIP過多ペナルティ（-5/個）
   - 指示変更ペナルティ（-2/回）

4. **設定管理** (`src/config.py`)
   - シナリオ定義（base, peak）
   - 工程定義（処理時間、容量制限）
   - 正規化のための最大値定義

## 重要な実装上の決定事項

### 1. ヘルパーシステムの設計（重要な修正）

**初期実装（誤り）**: ヘルパーが処理速度を向上させる（1.5倍速）
```python
# 誤った実装
effective_time = mean / capacity_multiplier  # 90秒 → 60秒
```

**修正後（正しい実装）**: ヘルパーは同時処理容量を増加させる
```python
# 正しい実装
if self.float_staff.current_action == help_action:
    capacity_limit = PROCESS_CAPACITY_LIMITS[process.id]  # 4個
else:
    capacity_limit = PROCESS_CAPACITY_LIMITS_SOLO[process.id]  # 2個

effective_time = mean  # 常に90秒（速度は変わらない）
```

**理由**:
- 物理的制約: 茹で時間は固定（90秒）
- ヘルパーの役割: 同時に複数の麺を茹でられる容量を増やす（2個 → 4個）
- リアリティ: 実際のラーメン店では人が増えても茹で時間は変わらない

**影響範囲**:
- `src/config.py`: `PROCESS_CAPACITY_LIMITS_SOLO` を追加
- `src/sim.py`: `_get_process_capacity()` を削除、`_process_orders()` で動的に容量を決定

**修正日**: 2026-02-15
**修正コミット**: ヘルパー効果の修正（速度ではなく容量）

### 2. 処理フローの2段階構造

各工程は以下の2段階で処理される：

1. **セットアップフェーズ** (5秒)
   - 準備作業（材料配置、器具準備など）
   - `OrderStatus.SETUP` 状態

2. **作業フェーズ** (工程ごとに異なる)
   - 実際の調理作業
   - `OrderStatus.IN_PROGRESS` 状態

**設計意図**: より現実的なタスク遷移をモデル化

### 3. 容量制限の設計

```python
# 物理的な最大容量（ヘルパーあり）
PROCESS_CAPACITY_LIMITS = {
    "boil": 4,    # 茹で釜に4玉まで同時に入る
    "plate": 10,  # 盛り付けは制約少ない
    "serve": 10,  # 配膳も制約少ない
}

# 1人作業時の容量（ヘルパーなし）
PROCESS_CAPACITY_LIMITS_SOLO = {
    "boil": 2,    # 1人では2玉が限界（手が足りない）
    "plate": 10,
    "serve": 10,
}
```

**設計意図**: 茹で工程がボトルネックになりやすい実店舗の状況を再現

### 4. 皿管理システム

皿の状態遷移:
```
きれい皿 (30枚からスタート)
  ↓ (顧客に提供)
使用中
  ↓ (食事完了)
汚れた皿
  ↓ (DISH_WASH指示)
洗浄中 (30秒)
  ↓
きれい皿
```

**重要**: 皿が足りないと新規注文を受けられない（現実的制約）

### 5. 顧客ライフサイクル

```
到着 (ポアソン過程)
  ↓
待ち行列
  ↓ (空席あり)
着席
  ↓
注文確定
  ↓
調理待ち (boil → plate → serve)
  ↓
提供
  ↓
食事中 (180-300秒)
  ↓
退店（皿を汚す）
```

## シナリオ定義

### base シナリオ（通常営業）
- 到着率: 0.5人/30秒（1人/分）
- 想定: 昼営業の平常時
- 特徴: 茹で工程のヘルプが中心となる

### peak シナリオ（ピーク時）
- 到着率: 1.0人/30秒（2人/分）
- 想定: 昼のピーク時
- 特徴: より積極的なヘルプと皿洗いのバランスが必要

## 学習済みモデル

### 最新モデル
**パス**: `models/base_20260215_094011/final_model.zip`

**学習設定**:
- アルゴリズム: PPO
- シナリオ: base
- 学習ステップ数: （過去の学習から継続）
- 特徴: 容量ベースのヘルパーシステムで学習

**性能**（60分営業シミュレーション）:
- 完成料理数: 47杯（47杯/時間）
- 平均待ち時間: 5.1分
- 行動分布:
  - BOIL_HELP: 67.5%
  - DISH_WASH: 32.5%

**傾向**: 茹で工程のヘルプを優先し、適度に皿洗いを実施

## 評価・可視化スクリプト

### 1. 標準評価 (`eval.py`)
シンプルな統計出力
```bash
python eval.py --model-path models/base_20260215_094011/final_model.zip --n-episodes 10
```

### 2. 詳細ログ (`eval_detailed_log.py`)
1分ごとの店舗状況を横表示
```bash
PYTHONPATH=/Users/haradamitsuru/repos/ramen-ops-simulator/src:$PYTHONPATH \
python eval_detailed_log.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 480
```
出力: `results/seat_level_detailed_log.txt`

### 3. 席別詳細ログ (`eval_detailed_with_seats.py`)
10席それぞれの顧客状態を可視化
```bash
PYTHONPATH=/Users/haradamitsuru/repos/ramen-ops-simulator/src:$PYTHONPATH \
python eval_detailed_with_seats.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60
```
出力: `results/detailed_minute_log_with_seats.txt`

**表示内容**:
- 各席の顧客状態（食事中/料理待ち）
- 待ち時間/食事時間
- 注文の進捗状況（boil中、plate待ち、など）
- 待ち行列の状況

### 4. AI指示ログ (`eval_with_instruction_log.py`)
AIの意思決定を時系列で記録
```bash
PYTHONPATH=/Users/haradamitsuru/repos/ramen-ops-simulator/src:$PYTHONPATH \
python eval_with_instruction_log.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60
```
出力:
- `results/ai_instruction_log.csv` (機械可読)
- `results/ai_instruction_log.txt` (人間可読)

**記録内容**:
- タイムスタンプ（HH:MM:SS）
- AI指示（BOIL_HELP, DISH_WASH等）
- 現在の行動と保留中の指示
- 店舗状態（完成数、待客数、在席数）
- 調理状況（各工程のWIP）
- 皿の状態

### 5. 音声案内付き評価 (`eval_with_voice.py`)
リアルタイム音声案内でシミュレーションを実行
```bash
PYTHONPATH=/Users/haradamitsuru/repos/ramen-ops-simulator/src:$PYTHONPATH \
python eval_with_voice.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60 \
  --realtime \
  --time-scale 10.0 \
  --speech-rate 200
```

**オプション**:
- `--realtime`: リアルタイムモード有効化
- `--time-scale`: 時間倍速（1.0=実時間、10.0=10倍速）
- `--speech-rate`: 音声速度（200=通常、600=3倍速、2000=10倍速）
- `--no-voice`: 音声出力を無効化

**特徴**:
- スタッフ名付きの具体的な指示（「Aさん、2玉茹でてください」）
- 音声再生完了の確認ログ
- 音声案内回数のカウント

## 開発の経緯

### Phase 1: 基礎実装
- 離散事象シミュレーターの構築
- Gymnasium環境の実装
- 報酬関数の設計
- PPO学習の基本フロー確立

### Phase 2: ヘルパーシステム修正（重要）
**日付**: 2026-02-15

**問題**: ヘルパーが処理速度を向上させる実装になっていた
- 茹で時間が90秒 → 60秒に短縮されていた
- 非現実的な動作

**修正内容**:
1. `HELP_EFFECT_MULTIPLIERS` を削除
2. `PROCESS_CAPACITY_LIMITS_SOLO` を追加（1人作業時の容量制限）
3. `_get_process_capacity()` を削除（速度は常に1.0）
4. `_process_orders()` で動的に容量を決定
5. 処理時間計算から容量除算を削除

**結果**:
- ヘルパーは同時処理数を増やすのみ（2個 → 4個）
- 処理時間は常に一定（茹で90秒）
- より現実的な動作に

### Phase 3: 可視化の充実
**日付**: 2026-02-15

**追加機能**:
1. 席別詳細ログ（顧客体験の可視化）
2. AI指示ログ（意思決定の透明化）

**目的**:
- AIの学習結果を人間が理解しやすくする
- デバッグと改善点の発見を容易にする
- 実店舗への適用可能性を検証

### Phase 4: 音声案内機能の実装
**日付**: 2026-02-16

**追加機能**:

1. **音声出力システム** (`src/voice.py`)
   - macOS `say`コマンドを使用した日本語TTS
   - 音声速度調整（1倍速〜10倍速対応）
   - 具体的な数量を含む動的指示生成:
     - 茹で: 「Aさん、2玉茹でてください」
     - 盛り付け: 「Bさん、3皿盛り付けしてください」
     - 配膳: 「Cさん、2皿配膳してください」
     - 皿洗い: 「Aさん、皿洗いをお願いします。汚れた皿が15枚あります」
   - 状況に応じた数量計算（2〜4個/皿の範囲で最適化）

2. **スタッフ割り当てシステム** (`src/staff_assignment.py`)
   - ラウンドロビン方式でA/B/Cさんに指示を順番に割り当て
   - タスク状態管理（現在のタスク、開始時刻、可用性）
   - 複数の割り当て戦略をサポート（round_robin, availability, task_based）

3. **音声付き評価スクリプト** (`eval_with_voice.py`)
   - リアルタイムモードでの実行（--realtime フラグ）
   - 時間スケール調整（--time-scale）: 1秒で何秒進めるかを制御
     - 例: `--time-scale 60.0` で1秒で1分進む（60倍速）
   - 音声速度調整（--speech-rate）: 200 wpm（通常） 〜 2000 wpm（10倍速）
   - 音声とシミュレーション速度の独立制御
   - 詳細なデバッグログ出力（音声再生完了の確認）

**使用例**:
```bash
# 通常速度の音声で10倍速シミュレーション（60分を6分で実行）
python eval_with_voice.py --model-path models/xxx/final_model.zip \
  --duration 60 --realtime --time-scale 10.0 --speech-rate 200

# 3倍速音声で60倍速シミュレーション（120分を2分で実行）
python eval_with_voice.py --model-path models/xxx/final_model.zip \
  --duration 120 --realtime --time-scale 60.0 --speech-rate 600

# 最高速で音声なし（8時間を数秒で実行）
python eval_with_voice.py --model-path models/xxx/final_model.zip \
  --duration 480 --no-voice
```

**設計意図**:
- 実運用を想定した音声案内システムの検証
- 具体的な数量指示でスタッフの理解を容易にする
- リアルタイム実行で実際の業務フローを体験
- 時間スケール調整で長時間シミュレーションを短時間で確認

### Phase 5: Webインターフェースによる手動検証システム
**日付**: 2026-02-16

**目的**: シミュレーション環境の正確性を検証し、指示システムの動作を確認するための対話型Webインターフェースの構築

**要件**:
1. ブラウザから手動で顧客を追加
2. 顧客は来店後5秒で自動注文（券売機をシミュレート）
3. 注文確定時に即座に茹で指示を出力
4. リアルタイムで店舗状態を可視化
5. 音声指示の統合

**アーキテクチャ**:
```
┌─────────────┐
│ ブラウザUI  │ ← 来店ボタン、状態表示
└──────┬──────┘
       │ HTTP/WebSocket
┌──────▼──────┐
│ Flask Server│ ← REST API + WebSocket
└──────┬──────┘
       │
┌──────▼──────────────┐
│ Controllable Sim    │
│ - 手動顧客追加      │
│ - 5秒後自動注文     │
│ - 注文時即茹で指示  │
└──────┬──────────────┘
       │
┌──────▼──────┐
│ 音声指示     │
└─────────────┘
```

**実装コンポーネント**:

1. **Webサーバー** (`src/web_server.py`)
   - Flask + Flask-SocketIO
   - REST API:
     - `POST /api/add_customer`: 顧客追加
     - `GET /api/state`: 店舗状態取得
   - WebSocket: リアルタイム状態配信

2. **手動制御可能シミュレーター** (`src/controllable_sim.py`)
   - 既存 `RamenShopSimulator` を拡張
   - `add_manual_customer()`: 外部からの顧客追加API
   - 自動注文タイマー（5秒）
   - 注文確定時の即座茹で指示トリガー

3. **ブラウザUI**
   - `static/index.html`: メインページ
   - `static/style.css`: スタイリング
   - `static/app.js`: WebSocket通信とUI更新
   - 表示内容:
     - 来店ボタン（大きく目立つ）
     - 待ち客数
     - 在席客数
     - 調理状況（boil/plate/serve別のWIP）
     - フロートスタッフの現在行動
     - 皿の状態（きれい/汚れ/洗浄中）

4. **音声指示統合**
   - 注文時の「○玉茹でてください」音声
   - 既存 `src/voice.py` の活用

**データフロー**:
```
1. ユーザーが「来店」ボタンクリック
   ↓
2. POST /api/add_customer
   ↓
3. シミュレーターに顧客追加（待ち行列）
   ↓
4. 5秒経過（券売機で注文）
   ↓
5. 注文確定 → 即座にBOIL_HELP指示
   ↓
6. 音声「○玉茹でてください」
   ↓
7. WebSocketで全体状態をブラウザに配信
```

**期待される成果**:
- シミュレーション環境の動作検証
- 指示タイミングの正確性確認
- 現実の店舗フローとの整合性評価
- デバッグの容易化

**実装完了した内容**:
1. `src/controllable_sim.py`: ControllableSimulatorクラス
   - `add_manual_customer()`: 手動顧客追加API
   - `_process_pending_orders()`: 5秒後の自動注文処理
   - `on_order_confirmed`: 注文確定時のコールバック

2. `src/web_server.py`: Flask + Flask-SocketIO Webサーバー
   - REST API エンドポイント
   - WebSocket リアルタイム通信
   - 音声指示統合（VoiceOutput）

3. `static/index.html`, `static/style.css`, `static/app.js`: ブラウザUI
   - レスポンシブデザイン
   - リアルタイム状態表示
   - イベントログ

4. `WEB_INTERFACE_GUIDE.md`: 使用ガイド

**起動方法**:
```bash
PYTHONPATH=src:$PYTHONPATH python src/web_server.py
# ブラウザで http://localhost:8080 を開く
```

**検証済み項目**:
- ✅ サーバー起動（ポート8080）
- ✅ WebSocket接続
- ✅ 手動顧客追加
- ✅ 5秒後自動注文
- ✅ 音声指示出力（「○玉茹でてください」）
- ✅ リアルタイム状態更新

**重要な修正（2026-02-16）: 専任スタッフ削除による指示ベース設計への移行**

**問題の発見**:
- Webインターフェースのテスト中、盛り付け・配膳の音声指示が一切出ない現象を発見
- 調査の結果、専任スタッフ（FIXED_STAFF）が自動で処理していたため、キューに注文が溜まらず指示が不要だった

**根本原因**:
```python
# 旧設定（自動処理）
FIXED_STAFF = {
    "boil": 1,    # 茹で専任スタッフ1名
    "plate": 1,   # 盛り付け専任スタッフ1名
    "serve": 1,   # 配膳専任スタッフ1名
}
```
この設定では、各工程に専任スタッフがいるため：
- 注文が来るとすぐに処理開始（自動）
- キューに溜まらない
- 音声指示が発動する機会がない（監視ロジックが機能しない）

**選択した解決策（選択肢1）**:
全ての工程で専任スタッフをなくし、明示的な指示がない限り処理が進まない設計に変更

**実装変更**:

1. **設定ファイル修正** (`src/config.py`)
```python
# 新設定（指示ベース）
FIXED_STAFF = {
    "boil": 0,    # No dedicated staff - instruction required
    "plate": 0,   # No dedicated staff - instruction required
    "serve": 0,   # No dedicated staff - instruction required
}
```

2. **シミュレーターロジック修正** (`src/sim.py`)
`_get_process_capacity()` メソッドを修正して、フロートスタッフまたは専任スタッフが割り当てられている場合のみ処理容量を返すように変更：
```python
def _get_process_capacity(self, process_id: str) -> float:
    # Check if float staff is assigned to this process
    help_action = f"{process_id.upper()}_HELP"
    if self.float_staff.current_action == help_action:
        return 1.0  # Float staff is assigned

    # Check fixed staff assignment
    fixed_count = FIXED_STAFF.get(process_id, 0)
    if fixed_count > 0:
        return 1.0  # Fixed staff is present

    # No one is working on this process
    return 0.0
```

3. **監視・指示ロジック追加** (`src/web_server.py`)
`check_and_issue_instructions()` 関数で、盛り付け・配膳キューを監視し、必要に応じて音声指示を発動：
```python
def check_and_issue_instructions(state: dict, current_time: float):
    # Check plate queue
    plate_wait = state.get("plate_wait", 0)
    if plate_wait >= 1 and (current_time - last_instruction_time["plate"]) >= INSTRUCTION_COOLDOWN:
        quantity = max(2, min(4, plate_wait))
        instruction_text = f"{quantity}皿盛り付けしてください"
        voice_output.speak(instruction_text, blocking=False)
        # ... (emit WebSocket event)

    # Check serve queue (similar logic)
    # Check dish washing (similar logic)
```

**テスト結果** (`test_no_fixed_staff.py`):
```
Step 1 (DO_NOTHING):
  t=60s - boil_wait=2, boil_in_progress=0 ✅ 指示なしで待機

Step 2 (BOIL_HELP):
  t=80s - boil_wait=0, boil_in_progress=2 ✅ 指示後に処理開始

Step 3 (継続):
  t=170s - plate_wait=2, plate_in_progress=0 ✅ 盛り付け指示なしで待機

Step 4 (PLATE_HELP):
  t=200s - plate_in_progress=3 ✅ 盛り付け開始
  t=220s - serve_wait=3 ✅ 配膳待ちに移行

Step 5 (SERVE_HELP):
  t=260s - serve_in_progress=3 ✅ 配膳開始
  t=270s - completed=3 ✅ 完成
```

**期待される動作フロー**:
1. 顧客来店 → 5秒後注文確定
2. 🔊 「○玉茹でてください」（注文時）
3. 茹で完了 → キューに溜まる
4. 🔊 「○皿盛り付けしてください」（監視ロジック）
5. 盛り付け完了 → キューに溜まる
6. 🔊 「○皿配膳してください」（監視ロジック）
7. 配膳完了 → 顧客に提供

**メリット**:
- ✅ 全工程で音声指示が必要 → リアリティ向上
- ✅ キューの可視化が可能 → 状況把握が容易
- ✅ ボトルネックの明確化 → 問題箇所の特定が容易
- ✅ より現実的な店舗オペレーションを再現

**デメリット（考慮事項）**:
- 既存の学習済みモデルは専任スタッフありで学習されているため、この設定では再学習が必要
- RL学習時の報酬設計を見直す必要がある（指示を出すタイミングの最適化）

## 現在の状態（2026-02-16 23:00）

### 実装完了（Phase 5まで）
- ✅ 容量ベースのヘルパーシステム
- ✅ 皿管理システム
- ✅ 顧客ライフサイクル管理
- ✅ 多目的報酬関数
- ✅ PPO学習フロー
- ✅ 4種類の評価スクリプト
- ✅ 席別可視化
- ✅ AI指示ログ
- ✅ 音声案内システム（macOS `say` コマンド統合）
- ✅ **Webインターフェース（Phase 5）**
  - Flask + Flask-SocketIO サーバー
  - 手動制御可能シミュレーター (`src/controllable_sim.py`)
  - ブラウザUI（HTML/CSS/JavaScript）
  - リアルタイム状態配信（WebSocket）
  - 手動来店機能
  - 5秒後自動注文機能
  - 注文確定時の即座茹で指示（音声出力）
  - **全工程で指示が必要な設計に移行**（FIXED_STAFF = 0）
  - 盛り付け・配膳の自動監視と音声指示機能

### 動作確認済み
- ✅ ヘルパー効果の正しい動作（容量増加）
- ✅ 60分営業での安定動作（47杯完成）
- ✅ 各種ログの出力
- ✅ CSV/テキスト両形式での出力
- ✅ **Webインターフェースの動作確認**
  - サーバー起動確認（ポート8080）
  - WebSocket接続確認
  - 手動顧客追加機能の動作
  - 5秒後の自動注文トリガー
  - 音声指示の出力
  - リアルタイム状態更新
- ✅ **専任スタッフなし設定の動作確認**（2026-02-16）
  - 全工程で明示的な指示が必要な設計に変更
  - 指示なしでは処理が進まないことを検証

### 学習中のモデル
以下のバックグラウンドタスクが実行中（複数の異なる学習設定を試行中）:
- training_with_improved_dish_penalty.log
- training_base_with_limits.log
- training_peak_with_limits.log
- training_with_dish_penalty.log
- training_120min_episodes.log
- training_final_with_shorter_eating.log
- training_balanced_final.log
- training_corrected_process_times.log
- training_capacity_based_helpers.log

## 既知の課題と改善候補

### 状態空間の拡張
- [ ] 時刻情報（ピーク予測）
- [ ] 到着率トレンド（移動平均）
- [ ] スタッフ疲労度モデル

### 行動空間の拡張
- [ ] ヘルプ時間の可変化（30s/60s/90s）
- [ ] 複数フロートスタッフの同時制御
- [ ] 優先度付けシステム（緊急度の高い注文を優先）

### シミュレーションの改善
- [ ] 実測データに基づく工程時間分布
- [ ] スタッフスキル差の考慮
- [ ] 複数メニューの対応
- [ ] 客席レイアウトの考慮

### 報酬関数の改善
- [ ] 最長待ち時間に対するペナルティ
- [ ] 顧客満足度の明示的モデル化
- [ ] スタッフ疲労の考慮

### 学習の改善
- [ ] カリキュラム学習（easy → hard）
- [ ] マルチタスク学習（複数シナリオ同時）
- [ ] 転移学習（baseシナリオ → peakシナリオ）

## 実運用への展開

### 必要な追加機能
1. **リアルタイム推論**
   - タブレット/スマホアプリでの指示表示
   - 低レイテンシー推論（<100ms）

2. **データ収集**
   - 実店舗でのログ収集
   - センサーデータ統合

3. **A/Bテスト**
   - AI指示 vs 人間判断の比較
   - 段階的なロールアウト

4. **フィードバックループ**
   - 実績データでの再学習
   - オンライン学習

### 懸念事項
- スタッフの受容性（AI指示への抵抗）
- 緊急時の対応（システム停止時のフォールバック）
- プライバシー（顧客データの取り扱い）

## ファイル構成

```
ramen-ops-simulator/
├── src/
│   ├── sim.py               # シミュレーター本体
│   ├── env.py               # Gymnasium環境
│   ├── reward.py            # 報酬関数
│   ├── config.py            # 設定定義
│   ├── voice.py             # 音声出力システム（Phase 4）
│   ├── staff_assignment.py  # スタッフ割り当て（Phase 4）
│   ├── controllable_sim.py  # 手動制御シミュレーター（Phase 5）
│   ├── web_server.py        # Flask Webサーバー（Phase 5）
│   └── logger.py            # ロギング機能
├── static/                  # Webインターフェース（Phase 5）
│   ├── index.html           # メインページ
│   ├── style.css            # スタイリング
│   └── app.js               # フロントエンドロジック
├── train.py                 # 学習スクリプト
├── eval.py                  # 標準評価
├── eval_detailed_log.py     # 詳細ログ（横表示）
├── eval_detailed_with_seats.py  # 席別詳細ログ
├── eval_with_instruction_log.py # AI指示ログ
├── eval_with_voice.py       # 音声案内付き評価（Phase 4）
├── test_no_fixed_staff.py   # 専任スタッフなし設定のテスト（Phase 5）
├── models/                  # 学習済みモデル
│   └── base_20260215_094011/
│       └── final_model.zip
├── results/                 # 評価結果
│   ├── ai_instruction_log.csv
│   ├── ai_instruction_log.txt
│   ├── detailed_minute_log_with_seats.txt
│   └── seat_level_detailed_log.txt
├── requirements.txt         # 依存パッケージ
├── README.md               # プロジェクト説明
├── DEVELOPMENT_HISTORY.md  # このファイル（開発履歴）
└── WEB_INTERFACE_GUIDE.md  # Web使用ガイド（Phase 5）
```

## 依存パッケージ

```
gymnasium>=0.29.0
stable-baselines3>=2.2.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
torch>=2.0.0
tensorboard>=2.14.0
flask>=3.0.0              # Phase 5
flask-socketio>=5.3.0     # Phase 5
flask-cors>=4.0.0         # Phase 5
```

インストール:
```bash
pip install -r requirements.txt
```

### Web機能の追加依存関係
Phase 5のWebインターフェースを使用する場合、以下が追加で必要:
- Flask: Webサーバーフレームワーク
- Flask-SocketIO: WebSocket通信
- Flask-CORS: CORS対応

## 重要な設定値

### タイミング関連
- `DECISION_INTERVAL = 30` (秒): AIの意思決定周期
- `INSTRUCTION_DELAY = 10` (秒): 指示反映までの遅延
- `DISH_WASH_TIME = 30` (秒): 皿洗いにかかる時間

### 容量関連
- `TOTAL_SEATS = 10`: 座席数
- `TOTAL_DISHES = 30`: 総皿数
- `PROCESS_CAPACITY_LIMITS["boil"] = 4`: 茹で釜の最大容量
- `PROCESS_CAPACITY_LIMITS_SOLO["boil"] = 2`: 1人作業時の茹で容量

### 工程時間（平均値）
- 茹で: 90秒
- 盛り付け: 20秒
- 配膳: 15秒
- セットアップ: 5秒（各工程共通）

### 顧客行動
- 食事時間: 180-300秒（ランダム）
- 到着率（base）: 0.5人/30秒
- 到着率（peak）: 1.0人/30秒

## デバッグ Tips

### シミュレーションが遅い場合
```python
# env.pyでレンダリングを無効化
render_mode=None
```

### 学習が不安定な場合
```python
# train.pyで学習率を下げる
learning_rate=1e-4  # デフォルトは3e-4
```

### 報酬が発散する場合
報酬関数の重みを調整（`src/reward.py`）

## 連絡先・参考資料

- プロジェクトリポジトリ: （記載する場合）
- 関連論文: （記載する場合）
- 実店舗データ提供元: （記載する場合）
