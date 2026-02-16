# Webインターフェース使用ガイド

## 概要

ブラウザから手動で顧客を追加し、シミュレーション環境の動作を検証するためのWebインターフェースです。

## 機能

### 主要機能

1. **手動来店**: ブラウザの「来店」ボタンで顧客を追加
2. **自動注文**: 来店後5秒で自動的に注文確定（券売機をシミュレート）
3. **即座茹で指示**: 注文確定時に音声で茹で指示を出力
4. **リアルタイム可視化**: 店舗状態をWebSocketでリアルタイム更新

### 表示項目

- ⏰ **経過時間**: シミュレーション開始からの時間
- 👥 **顧客状況**: 待ち客数、在席客数、料理待ち、食事中、空席
- 🔥 **調理状況**: 各工程（茹で/盛り付け/配膳）の待ち数と作業中数
- 👨‍🍳 **フロートスタッフ**: 現在の行動と保留中の指示
- 🍽️ **皿の状態**: きれいな皿、汚れた皿、洗浄中
- 📊 **実績**: 完成料理数、注文予定数
- 📋 **イベントログ**: リアルタイムイベント履歴

## 起動方法

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

必要なパッケージ:
- flask>=3.0.0
- flask-socketio>=5.3.0
- flask-cors>=4.0.0

### 2. Webサーバーの起動

```bash
cd /Users/haradamitsuru/repos/ramen-ops-simulator
PYTHONPATH=src:$PYTHONPATH python src/web_server.py
```

または、srcディレクトリから起動:

```bash
cd src
python web_server.py
```

起動すると以下のメッセージが表示されます:

```
============================================================
  Ramen Shop Simulator - Web Interface
============================================================

🌐 Starting server at http://localhost:5000

📝 Usage:
  1. Open http://localhost:5000 in your browser
  2. Click 'Initialize' to start the simulator
  3. Click 'Start Simulation' to begin
  4. Click '来店' button to add customers manually

🔊 Voice instructions will be played when orders are confirmed

============================================================
```

### 3. ブラウザでアクセス

ブラウザで以下のURLにアクセス:

```
http://localhost:5000
```

## 使い方

### 基本的な操作フロー

1. **初期化**
   - 「初期化」ボタンをクリックしてシミュレーターを初期化

2. **シミュレーション開始**
   - 「シミュレーション開始」ボタンをクリック
   - バックグラウンドでシミュレーションループが1秒間隔で実行開始

3. **顧客の追加**
   - 「👤 来店」ボタンをクリックして顧客を追加
   - 顧客は自動的に着席（空席がある場合）
   - 5秒後に自動的に注文確定

4. **注文確定時の動作**
   - 注文が確定すると、即座に茹で指示が発生
   - 音声で「○玉茹でてください」と指示が出力される
   - イベントログに記録される

5. **状態の監視**
   - リアルタイムで店舗状態が更新される
   - 各種メトリクスを確認可能

6. **停止**
   - 「停止」ボタンでシミュレーションループを停止

### 検証ポイント

#### 1. 顧客ライフサイクルの確認

- 来店 → 着席 → 5秒待機 → 注文 → 調理 → 提供 → 食事 → 退店

#### 2. 注文確定時の即座茹で指示

- 注文確定（5秒後）に茹で指示が即座に出力されるか
- 音声案内が正しく再生されるか
- イベントログに記録されるか

#### 3. 調理状況の可視化

- 各工程（茹で/盛り付け/配膳）のWIPが正しく表示されるか
- 待ち数と作業中数が正確にカウントされるか

#### 4. 皿管理の動作

- きれいな皿が減少するか
- 顧客退店時に汚れた皿が増加するか
- 皿洗い指示が必要になるタイミングは適切か

#### 5. 座席管理

- 満席時に顧客が待ち行列に入るか
- 空席ができると待ち客が着席するか

## API エンドポイント

### REST API

- `POST /api/init`: シミュレーター初期化
- `POST /api/start`: シミュレーション開始
- `POST /api/stop`: シミュレーション停止
- `POST /api/add_customer`: 顧客追加
- `GET /api/state`: 現在の状態取得
- `GET /api/metrics`: メトリクスサマリー取得

### WebSocket イベント

- `connect`: クライアント接続
- `disconnect`: クライアント切断
- `state_update`: 状態更新通知（1秒間隔）
- `order_confirmed`: 注文確定通知

## トラブルシューティング

### サーバーが起動しない

```bash
# ポート5000が使用中の場合
lsof -ti:5000 | xargs kill -9

# または別のポートを使用
# web_server.py の最後の行を編集:
# socketio.run(app, host='0.0.0.0', port=8080, debug=True, use_reloader=False)
```

### 音声が出ない

- macOSの`say`コマンドが利用可能か確認:

```bash
say "テスト"
```

- voice.pyの`speak_instruction()`関数を確認

### WebSocketが接続できない

- ブラウザのコンソールでエラーを確認
- CORS設定を確認（flask-corsがインストールされているか）

## ファイル構成

```
ramen-ops-simulator/
├── src/
│   ├── web_server.py          # Flask Webサーバー
│   ├── controllable_sim.py    # 手動制御可能シミュレーター
│   ├── sim.py                  # 基本シミュレーター
│   ├── voice.py                # 音声案内
│   └── config.py               # 設定
├── static/
│   ├── index.html              # メインページ
│   ├── style.css               # スタイル
│   └── app.js                  # フロントエンドロジック
└── WEB_INTERFACE_GUIDE.md      # このファイル
```

## 開発履歴

Phase 5として2026-02-16に実装。詳細は `DEVELOPMENT_HISTORY.md` を参照。

## 今後の拡張候補

- [ ] 複数シナリオの切り替え（base/peak）
- [ ] 手動での指示出し（ボタンでBOIL_HELP/DISH_WASH等を指示）
- [ ] タイムスケール調整（倍速再生）
- [ ] 履歴のCSVエクスポート
- [ ] グラフ表示（時系列データの可視化）
- [ ] 複数顧客の一括追加
- [ ] 座席レイアウトの可視化

## 関連ドキュメント

- [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md): 開発履歴と設計決定
- [README.md](README.md): プロジェクト全体の説明
