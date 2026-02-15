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

## 現在の状態（2026-02-15）

### 実装完了
- ✅ 容量ベースのヘルパーシステム
- ✅ 皿管理システム
- ✅ 顧客ライフサイクル管理
- ✅ 多目的報酬関数
- ✅ PPO学習フロー
- ✅ 4種類の評価スクリプト
- ✅ 席別可視化
- ✅ AI指示ログ

### 動作確認済み
- ✅ ヘルパー効果の正しい動作（容量増加）
- ✅ 60分営業での安定動作（47杯完成）
- ✅ 各種ログの出力
- ✅ CSV/テキスト両形式での出力

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
│   ├── sim.py           # シミュレーター本体
│   ├── env.py           # Gymnasium環境
│   ├── reward.py        # 報酬関数
│   └── config.py        # 設定定義
├── train.py             # 学習スクリプト
├── eval.py              # 標準評価
├── eval_detailed_log.py # 詳細ログ（横表示）
├── eval_detailed_with_seats.py  # 席別詳細ログ
├── eval_with_instruction_log.py # AI指示ログ
├── models/              # 学習済みモデル
│   └── base_20260215_094011/
│       └── final_model.zip
├── results/             # 評価結果
│   ├── ai_instruction_log.csv
│   ├── ai_instruction_log.txt
│   ├── detailed_minute_log_with_seats.txt
│   └── seat_level_detailed_log.txt
├── requirements.txt     # 依存パッケージ
├── README.md           # プロジェクト説明
└── DEVELOPMENT_HISTORY.md  # このファイル
```

## 依存パッケージ

```
gymnasium
stable-baselines3[extra]
numpy
```

インストール:
```bash
pip install -r requirements.txt
```

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
