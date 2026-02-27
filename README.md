# ラーメン屋オペレーションシミュレーター

ラーメン屋の運営オペレーションを最適化するための強化学習ベースのシミュレーターです。

> **📋 開発履歴・コンテキスト**: 詳細な開発経緯、設計決定、現在の状態については [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md) を参照してください。
>
> **⚡ クイックスタート**: 次回セッション開始時に `/context` コマンドを実行すると、開発履歴を素早く読み込めます。

## 概要

このプロジェクトは、ラーメン店のスタッフ配置・指示を最適化する強化学習PoCです。**スループット最大化**だけでなく、**認知停止（idle）を減らし心理的安全性を高める**ことを目的としています。

### 主な特徴

- 離散事象シミュレーション（DES）による厨房オペレーション模擬
- 30秒に1回の意思決定周期（実際の店舗に即した制約）
- 10秒の指示反映遅延を考慮
- 頻繁な指示変更を抑制（スタッフの混乱を避ける）
- PPO（Proximal Policy Optimization）による学習

### 最適化対象

- フロート（手が空いたスタッフ）への短時間ヘルプ指示
- 工程別の作業支援（皿洗い、盛り付け、配膳、補充など）
- ボトルネック検出と動的な人員配置

### 報酬設計

- ✓ 完成杯数（スループット）
- ✓ 待ち時間削減（顧客満足度）
- ✓ アイドル時間削減（心理的安全性）
- ✓ WIP過多ペナルティ（品質管理）
- ✓ 指示変更ペナルティ（運用安定性）

## プロジェクト構成

```
ramen-ops-simulator/
├── src/
│   ├── config.py            # 設定（工程定義、報酬重み、シナリオなど）
│   ├── sim.py               # 離散事象シミュレーター
│   ├── env.py               # Gymnasium環境（RL用）
│   ├── reward.py            # 報酬計算と内訳
│   ├── voice.py             # 音声出力システム（macOS `say`統合）
│   ├── staff_assignment.py  # スタッフ割り当て（A/B/C ラウンドロビン）
│   ├── controllable_sim.py  # 手動制御可能シミュレーター
│   ├── web_server.py        # Flask Webサーバー
│   └── logger.py            # ロギング機能
├── static/                  # Webインターフェース
│   ├── index.html           # メインページ
│   ├── style.css            # スタイリング
│   └── app.js               # フロントエンドロジック
├── train.py                 # 学習スクリプト（PPO）
├── eval.py                  # 標準評価
├── eval_detailed_log.py     # 詳細ログ（1分ごと横表示）
├── eval_detailed_with_seats.py  # 席別詳細ログ
├── eval_with_instruction_log.py # AI指示ログ
├── eval_with_voice.py       # 音声案内付き評価
├── models/                  # 学習済みモデル
│   └── base_20260215_094011/
│       └── final_model.zip
├── results/                 # 評価結果
├── requirements.txt         # 依存パッケージ
├── README.md
└── DEVELOPMENT_HISTORY.md   # 開発履歴とコンテキスト
```

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

必要なパッケージ：
- `gymnasium>=0.29.0`
- `stable-baselines3>=2.2.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `matplotlib>=3.7.0`
- `torch>=2.0.0`
- `tensorboard>=2.14.0`
- `flask>=3.0.0`（Webインターフェース用）
- `flask-socketio>=5.3.0`（WebSocket通信用）
- `flask-cors>=4.0.0`（CORS対応）

### 2. 設定の確認

`src/config.py` で工程定義やパラメータを確認・調整できます：

```python
# 工程定義（実店舗に合わせて追加・変更可能）
PROCESSES = [
    ProcessConfig(id="boil", mean_time=90.0, std_time=10.0),   # 茹で: 90秒
    ProcessConfig(id="plate", mean_time=20.0, std_time=5.0),   # 盛り付け: 20秒
    ProcessConfig(id="serve", mean_time=15.0, std_time=3.0),   # 配膳: 15秒
]

# 容量制限（重要: ヘルパーは処理速度ではなく容量を増やす）
PROCESS_CAPACITY_LIMITS = {
    "boil": 4,    # ヘルパーありで4玉同時に茹でられる
    "plate": 10,  # 盛り付けは制約少ない
    "serve": 10,  # 配膳も制約少ない
}

PROCESS_CAPACITY_LIMITS_SOLO = {
    "boil": 2,    # 1人では2玉が限界
    "plate": 10,
    "serve": 10,
}

# 行動空間
ACTION_SPACE = [
    "DO_NOTHING",
    "BOIL_HELP",
    "PLATE_HELP",
    "SERVE_HELP",
    "DISH_WASH",
]

# 専任スタッフ（指示ベース設計では0に設定）
FIXED_STAFF = {
    "boil": 0,    # 明示的な指示が必要
    "plate": 0,   # 明示的な指示が必要
    "serve": 0,   # 明示的な指示が必要
}

# 報酬重み（調整可能）
REWARD_WEIGHTS = {
    "completed_bowls": 10.0,
    "avg_wait_time": -0.1,
    "wip_overflow": -5.0,
    "idle_penalty": -0.2,
    "change_penalty": -2.0,
}
```

## 使い方

### 1. Webインターフェースで手動テスト（推奨）

シミュレーション環境の動作を確認するには、まずWebインターフェースを使うことを推奨します。

```bash
# Webサーバーを起動
PYTHONPATH=src:$PYTHONPATH python src/web_server.py
```

ブラウザで `http://localhost:8080` を開き、「来店」ボタンで顧客を追加できます。

**機能**:
- 手動での顧客追加
- 5秒後の自動注文（券売機をシミュレート）
- リアルタイムの店舗状態表示（待ち客数、在席数、調理状況、皿の状態など）
- 音声指示の自動発動（「○玉茹でてください」など）
- 調理工程の進捗率（%）表示
- 退店済み顧客数のカウント

**重要**: このシステムは**指示ベース設計**を採用しており、明示的な指示がない限り処理が進みません（`FIXED_STAFF = 0`）。これにより、より現実的な店舗オペレーションを再現しています。

### 2. 学習（Training）

```bash
# 基本的な学習（デフォルト設定）
python train.py

# シナリオとステップ数を指定
python train.py --scenario peak --total-timesteps 200000

# 並列環境数と学習率を指定
python train.py --n-envs 8 --learning-rate 3e-4
```

主なオプション：
- `--scenario`: `base`（通常営業: 1人/分）または `peak`（ピーク時: 2人/分）
- `--total-timesteps`: 学習ステップ数（デフォルト: 100,000）
- `--n-envs`: 並列環境数（デフォルト: 4）
- `--learning-rate`: 学習率（デフォルト: 3e-4）
- `--output-dir`: モデル保存先（デフォルト: `models/`）

学習中のログはTensorBoardで確認できます：

```bash
tensorboard --logdir logs/
```

### 3. 評価（Evaluation）

#### 3.1 標準評価

```bash
# 学習済みモデルの評価
python eval.py --model-path models/base_20260215_094011/final_model.zip

# シナリオとエピソード数を指定
python eval.py --model-path models/base_20260215_094011/final_model.zip \
               --scenario peak \
               --n-episodes 20

# ランダムベースラインと比較
python eval.py --model-path models/base_20260215_094011/final_model.zip \
               --baseline
```

#### 3.2 詳細ログ評価（1分ごとの店舗状況）

```bash
PYTHONPATH=src:$PYTHONPATH python eval_detailed_log.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 480
```

出力: `results/seat_level_detailed_log.txt`（1分ごとの店舗状況を横表示）

#### 3.3 席別詳細ログ（10席それぞれの顧客状態）

```bash
PYTHONPATH=src:$PYTHONPATH python eval_detailed_with_seats.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60
```

出力: `results/detailed_minute_log_with_seats.txt`

表示内容:
- 各席の顧客状態（食事中/料理待ち）
- 待ち時間/食事時間
- 注文の進捗状況（boil中、plate待ち、など）
- 待ち行列の状況

#### 3.4 AI指示ログ（意思決定の時系列記録）

```bash
PYTHONPATH=src:$PYTHONPATH python eval_with_instruction_log.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60
```

出力:
- `results/ai_instruction_log.csv`（機械可読）
- `results/ai_instruction_log.txt`（人間可読）

記録内容:
- タイムスタンプ（HH:MM:SS）
- AI指示（BOIL_HELP, DISH_WASH等）
- 現在の行動と保留中の指示
- 店舗状態（完成数、待客数、在席数）
- 調理状況（各工程のWIP）
- 皿の状態

#### 3.5 音声案内付き評価（リアルタイムシミュレーション）

```bash
PYTHONPATH=src:$PYTHONPATH python eval_with_voice.py \
  --model-path models/base_20260215_094011/final_model.zip \
  --scenario base \
  --duration 60 \
  --realtime \
  --time-scale 10.0 \
  --speech-rate 200
```

オプション:
- `--realtime`: リアルタイムモード有効化
- `--time-scale`: 時間倍速（1.0=実時間、10.0=10倍速、60.0=60倍速）
- `--speech-rate`: 音声速度（200=通常、600=3倍速、2000=10倍速）
- `--no-voice`: 音声出力を無効化

使用例:
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

特徴:
- スタッフ名付きの具体的な指示（「Aさん、2玉茹でてください」）
- 音声再生完了の確認ログ
- 音声案内回数のカウント

評価結果は `results/` ディレクトリに保存されます

### テスト実行

各モジュールを単独でテストできます：

```bash
# 設定の検証
cd src && python config.py

# シミュレーターのテスト
cd src && python sim.py

# 報酬計算のテスト
cd src && python reward.py

# 環境のテスト
cd src && python env.py
```

## シナリオ

### ベース（通常営業）
- 注文到着率: 1杯/60秒
- 用途: 基本的な学習・評価

### ピーク（混雑時）
- 注文到着率: 1杯/30秒
- 用途: 高負荷時の性能評価

シナリオは `src/config.py` の `SCENARIOS` で定義・追加できます。

## 出力メトリクス

評価時に以下のメトリクスが出力されます：

### エピソードレベル
- 総報酬
- 完成杯数
- 平均待ち時間
- 総アイドル時間
- 指示変更回数

### ステップレベル（CSV）
- 各ステップの行動
- 報酬内訳（完成、待ち時間、WIP、idle、変更）
- 到着数、完成数
- 工程別WIP
- 現在の指示とペンディング指示

## 工程の追加・変更方法

実際の店舗を観察して工程を追加する場合：

1. `src/config.py` の `PROCESSES` に新しい工程を追加：

```python
PROCESSES = [
    ProcessConfig(id="prep", mean_time=60.0, std_time=10.0),      # 追加: 仕込み
    ProcessConfig(id="boil", mean_time=90.0, std_time=10.0),
    ProcessConfig(id="plate", mean_time=20.0, std_time=5.0),
    ProcessConfig(id="toppings", mean_time=25.0, std_time=5.0),   # 追加: トッピング
    ProcessConfig(id="serve", mean_time=15.0, std_time=3.0),
]
```

2. `PROCESS_CAPACITY_LIMITS` と `PROCESS_CAPACITY_LIMITS_SOLO` に容量を追加：

```python
PROCESS_CAPACITY_LIMITS = {
    "prep": 3,        # 追加
    "boil": 4,
    "plate": 10,
    "toppings": 5,    # 追加
    "serve": 10,
}

PROCESS_CAPACITY_LIMITS_SOLO = {
    "prep": 2,        # 追加
    "boil": 2,
    "plate": 10,
    "toppings": 3,    # 追加
    "serve": 10,
}
```

3. 必要に応じて `ACTION_SPACE` にヘルプ行動を追加：

```python
ACTION_SPACE = [
    "DO_NOTHING",
    "PREP_HELP",      # 追加
    "BOIL_HELP",
    "PLATE_HELP",
    "TOPPINGS_HELP",  # 追加
    "SERVE_HELP",
    "DISH_WASH",
]
```

シミュレーター、環境、学習スクリプトは自動的に新しい工程に対応します。

**重要**: ヘルパーは処理**速度**ではなく同時処理**容量**を増やす設計です。これにより、茹で時間などの物理的制約を正確に再現できます。

## 改善候補

このPoCをベースに、以下の拡張が可能です：

### 状態空間の拡張
- [ ] 時刻情報（ピーク時間帯の予測）
- [ ] 直近N分の到着率トレンド
- [ ] 各工程の処理速度推定値
- [ ] スタッフの疲労度モデル
- [ ] 顧客の待ち時間分布

### 行動空間の拡張
- [ ] ヘルプ時間を可変に（30s/60s/90s）
- [ ] 複数フロートスタッフの同時指示
- [ ] 工程順序の動的変更
- [ ] 一時的な休憩指示（疲労回復）

### シミュレーションの改善
- [ ] より正確な工程時間分布（実測データから推定）
- [ ] スタッフのスキル差を考慮
- [ ] 材料切れ・補充のモデル化
- [ ] 複数種類のメニュー（ラーメン、チャーシュー麺など）
- [ ] 座席とテーブル回転率の統合

### 報酬設計の改善
- [ ] 顧客満足度の非線形モデル（待ち時間による離脱）
- [ ] 実際の売上・利益を考慮
- [ ] スタッフのストレス・疲労コスト
- [ ] 長期的な品質評価（リピート率）

### アルゴリズムの拡張
- [ ] マルチタスク学習（複数シナリオ同時学習）
- [ ] オフライン学習（実店舗データから学習）
- [ ] 説明可能なAI（なぜその指示を出したか）
- [ ] Sim-to-Real転移（シミュレーターと実環境のギャップ削減）

### 可視化・UI
- [ ] リアルタイムダッシュボード
- [ ] 3Dシミュレーション表示
- [ ] 学習過程の行動分析（どの状況でどの行動を選ぶか）
- [ ] What-Ifシミュレーション（パラメータを変えたらどうなるか）

## 技術スタック

- **Python 3.10+**
- **Gymnasium**: RL環境の標準インターフェース
- **stable-baselines3**: PPO実装
- **PyTorch**: ニューラルネットワークバックエンド
- **NumPy**: 数値計算
- **Pandas**: データ分析・CSV出力

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更を加える場合は、まずissueを開いて変更内容について議論してください。

## 参考

このプロジェクトは、以下の分野の知見を統合しています：

- 離散事象シミュレーション（DES）
- 強化学習（Reinforcement Learning）
- オペレーションズリサーチ（Operations Research）
- ヒューマンファクターズ（Human Factors）
- サービスオペレーション管理（Service Operations Management）
