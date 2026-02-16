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
│   ├── config.py       # 設定（工程定義、報酬重み、シナリオなど）
│   ├── sim.py          # 離散事象シミュレーター
│   ├── reward.py       # 報酬計算と内訳
│   └── env.py          # Gymnasium環境（RL用）
├── train.py            # 学習スクリプト（PPO）
├── eval.py             # 評価スクリプト（メトリクス出力）
├── requirements.txt    # 依存パッケージ
└── README.md
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

### 2. 設定の確認

`src/config.py` で工程定義やパラメータを確認・調整できます：

```python
# 工程定義（実店舗に合わせて追加・変更可能）
PROCESSES = [
    ProcessConfig(id="boil", mean_time=180.0, std_time=20.0),
    ProcessConfig(id="plate", mean_time=30.0, std_time=5.0),
    ProcessConfig(id="serve", mean_time=20.0, std_time=3.0),
]

# 行動空間
ACTION_SPACE = [
    "DO_NOTHING",
    "DISH_HELP",
    "PLATE_HELP",
    "SERVE_HELP",
    "RESTOCK_HELP",
]

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

### 学習（Training）

```bash
# 基本的な学習（デフォルト設定）
python train.py

# シナリオとステップ数を指定
python train.py --scenario peak --total-timesteps 200000

# 並列環境数と学習率を指定
python train.py --n-envs 8 --learning-rate 3e-4
```

主なオプション：
- `--scenario`: `base`（通常）または `peak`（ピーク時）
- `--total-timesteps`: 学習ステップ数（デフォルト: 100,000）
- `--n-envs`: 並列環境数（デフォルト: 4）
- `--learning-rate`: 学習率（デフォルト: 3e-4）
- `--output-dir`: モデル保存先（デフォルト: `models/`）

学習中のログはTensorBoardで確認できます：

```bash
tensorboard --logdir logs/
```

### 評価（Evaluation）

```bash
# 学習済みモデルの評価
python eval.py --model-path models/base_20250213_120000/final_model.zip

# シナリオとエピソード数を指定
python eval.py --model-path models/base_20250213_120000/final_model.zip \
               --scenario peak \
               --n-episodes 20

# ランダムベースラインと比較
python eval.py --model-path models/base_20250213_120000/final_model.zip \
               --baseline

# リアルタイム表示
python eval.py --model-path models/base_20250213_120000/final_model.zip \
               --render
```

評価結果は `results/` ディレクトリに保存されます：
- `eval_steps_*.csv`: ステップごとの詳細データ
- `eval_summary_*.txt`: エピソード集約結果

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
    ProcessConfig(id="boil", mean_time=180.0, std_time=20.0),
    ProcessConfig(id="prep", mean_time=60.0, std_time=10.0),      # 追加
    ProcessConfig(id="plate", mean_time=30.0, std_time=5.0),
    ProcessConfig(id="toppings", mean_time=25.0, std_time=5.0),   # 追加
    ProcessConfig(id="serve", mean_time=20.0, std_time=3.0),
]
```

2. 必要に応じて `ACTION_SPACE` にヘルプ行動を追加：

```python
ACTION_SPACE = [
    "DO_NOTHING",
    "PREP_HELP",      # 追加
    "PLATE_HELP",
    "TOPPINGS_HELP",  # 追加
    "SERVE_HELP",
]
```

3. `HELP_EFFECT_MULTIPLIERS` で効果を定義：

```python
HELP_EFFECT_MULTIPLIERS = {
    "PREP_HELP": 1.4,
    "PLATE_HELP": 1.5,
    "TOPPINGS_HELP": 1.6,
    "SERVE_HELP": 1.5,
}
```

シミュレーター、環境、学習スクリプトは自動的に新しい工程に対応します。

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
