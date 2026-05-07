# 公園における社交装置の効果検証 — Happy-to-Chat ベンチを用いた A/B 比較シミュレーション

## 概要

**本リポジトリは、公園空間における「Happy-to-Chat ベンチ」（会話歓迎を明示的に伝える社交装置）の効果を、LLM マルチエージェントシミュレーションで A/B 検証した研究実装である。**

20 体の LLM エージェント（Ollama ローカル推論）が公園内の複数の場所（中央ベンチ・噴水広場・花壇・ブランコ・売店など）を自由に行動する。**唯一の介入条件は中央ベンチに付与される「Anyone sitting here is signaling 'I am open to chat with strangers.'」というプロンプト追記の有無のみ**。物理プロパティ（位置・サイズ・定員）は両条件で完全同一とした。

### 主要結果

- 中央ベンチの平均人数が **3.3〜5.7倍** に増加（2.44名 → 8〜14名）
- エージェント内部思考における「ベンチ」言及比率が **7倍**に上昇
- Ordinary 条件では複数の場所に分散する自然な利用パターン、Happy 条件では中央ベンチへの一極集中が出現

詳細は [`COMPARISON_REPORT.md`](COMPARISON_REPORT.md) / [`PRESENTATION.pdf`](PRESENTATION.pdf) を参照。

### 設計上の特徴

- **行動指示なし**: エージェントには生の数値データ（占有率、距離等）のみを提供。「快適」「混雑」等の定性的評価は与えない。状況解釈・意思決定・コミュニケーションは全てエージェント自身が行う。
- **ペルソナ**: 各エージェントには性別（male/female）がランダムに割り当てられ、LLM プロンプトや近隣エージェント情報に含まれる。
- **拡張機能**: メタ認知エージェント（`metacog/`、Anthropic Claude API 使用）を内包。

## 必要な環境

- Python 3.8以上
- Ollama（LLMサーバー）
- 必要なPythonパッケージ（requirements.txt参照）
- FFmpeg（動画生成を行う場合のみ）

## セットアップ

### 1. 仮想環境の作成とセットアップ

#### macOS/Linux の場合:
```bash
# セットアップスクリプトを実行（推奨）
chmod +x setup_mac.sh
./setup_mac.sh

# または手動でセットアップ
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows の場合:
```cmd
REM セットアップスクリプトを実行（推奨）
setup_win.bat

REM または手動でセットアップ
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 仮想環境の有効化

#### macOS/Linux:
```bash
source venv/bin/activate
```

#### Windows:
```cmd
venv\Scripts\activate.bat
```

### 3. Ollamaのセットアップ

#### Ollamaのインストール
   - [Ollama](https://ollama.ai/)をインストール

#### Ollamaサーバーの起動

Ollamaは通常、インストール時に自動的にバックグラウンドサービスとして起動します。しかし、手動で起動する必要がある場合や、APIサーバーとして明示的に起動したい場合は、以下の手順を実行してください:

**方法1: 別のターミナルで手動起動（推奨）**
```bash
# 新しいターミナルを開いて実行
ollama serve
```

これにより、デフォルトで`http://localhost:11434`でサーバーが起動します。

**方法2: バックグラウンドで起動（macOS/Linux）**
```bash
ollama serve &
```

**Ollamaサーバーの状態確認**
```bash
# サーバーが起動しているか確認
curl http://localhost:11434/api/tags
```

正常に起動していれば、モデルリストがJSON形式で返されます。

#### モデルのダウンロードと選択

**利用可能なモデルの確認**:
```bash
# Ollamaにダウンロード済みのモデル一覧を表示
ollama list
```

**モデルのダウンロード**:
```bash
# 使用するモデルをダウンロード（例: llama3.2）
ollama pull llama3.2

# または他のモデルを使用する場合
ollama pull llama3.1
ollama pull mistral
ollama pull qwen2.5
```

**モデルの選択**:
- シミュレーションは`config.yaml`の`llm.model`で指定されたモデルを使用します
- **重要**: `config.yaml`の`llm.model`を、ご自身の環境にインストール済みのモデル名に変更してください:

```yaml
llm:
  model: "llama3.2"  # 使用したいモデル名に変更（ollama list で確認）
```

**モデル名の指定方法**:
- **推奨**: `ollama list`で表示される形式をそのまま使用（例: `qwen2.5:latest`）
- **省略可能**: タグが`:latest`の場合は、タグを省略しても動作します（例: `qwen2.5`）
- **例**:
  - `qwen2.5:latest` ✅ （推奨）
  - `qwen2.5` ✅ （`:latest`を省略）
  - `llama3.2:latest` ✅
  - `llama3.2` ✅
  - `gpt-oss:20b` ✅ （特定のタグを指定）

**注意**: 
- `config.yaml`の`llm.model`で指定したモデル名と一致するモデルが使用されます
- 指定したモデルが存在しない場合、シミュレーション開始時にエラーメッセージが表示されます
- 複数のモデルがダウンロードされていても、`config.yaml`で指定した1つのモデルのみが使用されます
- `ollama list`で表示される`NAME`列の値をそのまま使用するのが最も確実です

### 4. 設定ファイルの確認
   - `config.yaml`でパラメータを確認・調整
   - **特に `llm.model` を、ご自身の環境にインストール済みのモデル名に変更してください**（`ollama list` で確認できます）

### 5. （任意）メタ認知機能を使う場合の API キー設定
   `metacog/` 配下のメタ認知エージェントは Anthropic Claude API を使用するため、以下の環境変数を設定してください（メタ認知機能を使わない場合は不要）。
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
   ```

## 使用方法

**注意**: 
- 実行前に仮想環境を有効化してください
- Ollamaサーバーが起動していることを確認してください（`ollama serve`を別のターミナルで実行）

### 基本的な実行

```bash
# 仮想環境を有効化（まだ有効化していない場合）
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate.bat  # Windows

# シミュレーションを実行
python main.py
```

### 可視化を有効にする

**注意**: WSL環境やGUIが利用できない環境では、`--visualize`オプションは動作しません。代わりに`--save-frames`を使用してください。

```bash
python main.py --visualize
```

### フレームを保存する

```bash
python main.py --save-frames
```

### カスタム設定ファイルを使用

```bash
python main.py --config custom_config.yaml
```

## 設定ファイル（config.yaml）

主要なパラメータ:

- **simulation**: シミュレーション設定
  - `duration`: シミュレーションステップ数
  - `half_space_size`: 空間の半分のサイズ（例: 25 → 座標範囲は -25 〜 +25）
  - `half_place_size`: 場所の半分のサイズ（各場所の`half_size`が優先されます）

- **agents**: エージェント設定
  - `num_agents`: エージェント数
  - `communication_radius`: 通信半径
  - `memory_limit`: 保存する最大メモリ数（デフォルト: 20）
  - `memory_size`: LLM推論に使用するメモリ数（デフォルト: 5）
  - `message_history_limit`: 保存する最大メッセージ数（デフォルト: 10）
  - `message_context_size`: LLM推論に使用するメッセージ数（デフォルト: 3）

- **places**: 場所設定（複数場所対応、将来的にカフェや図書館などにも拡張可能）
  - 場所のリスト。各場所は以下の設定を持ちます:
    - `name`: 場所名（必須、例: "left_bar", "right_bar", "cafe", "library"）
    - `type`: 場所の種類（必須、例: "bar", "cafe", "library"）
    - `center_x`: 場所の中心X座標（必須）
    - `center_y`: 場所の中心Y座標（必須）
    - `half_size`: 場所の半分のサイズ（必須、場所は中心から±half_sizeの範囲）
    - `capacity`: その場所の収容上限人数（必須、占有率の計算に使用）

  例（現在の設定）:
  ```yaml
  places:
    - name: "left_bar"
      type: "bar"  # 場所の種類: bar, cafe, library など
      center_x: -15
      center_y: 0
      half_size: 5
      capacity: 12
    - name: "right_bar"
      type: "bar"
      center_x: 15
      center_y: 0
      half_size: 5
      capacity: 10
  ```

  将来の拡張例（カフェや図書館など）:
  ```yaml
  places:
    - name: "cafe"
      type: "cafe"
      center_x: -15
      center_y: 0
      half_size: 5
      capacity: 12
    - name: "library"
      type: "library"
      center_x: 15
      center_y: 0
      half_size: 5
      capacity: 10
  ```

- **llm**: LLM設定（詳細は `config.yaml` を参照）
  - `model`: Ollamaモデル名
  - `base_url`: Ollama APIエンドポイント
  - `temperature`: サンプリング温度（低いほど一貫した行動、高いほど多様な行動）
  - `max_tokens`: 最大トークン数
  - `repeat_penalty`: 反復ペナルティ（同じトークンの繰り返しを抑制）
  - `repeat_last_n`: 反復チェック範囲（繰り返し検出の対象トークン数。`0`で無効、`-1`で`num_ctx`を使用）
  - `min_p`: Min-Pサンプリング（指定確率未満のトークンを除外）

### 座標系について

このシミュレーションは**原点中心の座標系**を使用します:

- **フィールド**: 原点 (0, 0) を中心に、`-half_space_size` 〜 `+half_space_size` の範囲
- **場所**: 各場所は独立した中心位置（`center_x`, `center_y`）を持ち、その中心から`±half_size`の範囲（両端を含む）

**エージェントの知識**:
- エージェントは**すべての場所の位置情報**を知っています（プロンプトに含まれます）
- ただし、場所の占有状況（エージェント数、収容上限、占有率）は、その場所内にいるエージェントのみが直接受け取ります

**モデルの選択について**:
- `config.yaml` の `llm.model` で使用するモデルを指定
- 利用可能なモデルを確認するには: `ollama list`
- モデルをダウンロードするには: `ollama pull <モデル名>`

## 出力

- `output/`: 可視化フレームと統計グラフが保存されます
  - 可視化では、エージェントの**性別を色**（男=青、女=赤）、**場所内/外をマーカー形状**（場所内=★、場所外=●）で表現
- `output/messages.jsonl`: エージェント間メッセージ履歴
- `output/memory_reasoning.jsonl`: エージェントの記憶と推論ログ
- `simulation.log`: シミュレーションログ

## シミュレーション結果の可視化ツール

`visualization/` ディレクトリに、シミュレーション結果を閲覧・動画化するためのツールが含まれています。いずれも `output/` ディレクトリに保存されたフレーム画像（`frame_*.png`）とログファイル（`messages.jsonl`、`memory_reasoning.jsonl`）を読み込んで使用します。

### ブラウザビューア（viewer.html）

シミュレーション結果をブラウザ上でインタラクティブに再生・閲覧できるHTMLビューアです。左側にフレーム画像、右上にエージェント間メッセージ（会話内容+内省）、右下に各エージェントの行動理由と記憶が表示されます。

**使い方**:

1. `visualization/viewer.html` をブラウザで開く
2. シミュレーション結果が入ったディレクトリ（例: `output/`）を選択する
   - **Chrome / Edge（推奨）**: 「ディレクトリを選択」ボタンからフォルダごと選択
   - **Firefox / Safari**: 「ファイルを選択」ボタンから `messages.jsonl` を選択（ただしPNG画像が読み込めない場合があります）
3. ステップごとの画像・メッセージ・行動理由が統合表示される

**操作方法**:

| 操作 | 方法 |
|---|---|
| 再生 / 停止 | 「▶ 再生」ボタン または `Space` キー |
| 前のステップ | 「⏮ 前へ」ボタン または `←` キー |
| 次のステップ | 「⏭ 次へ」ボタン または `→` キー |
| ステップジャンプ | スライダーをドラッグ |
| 再生速度調整 | 速度スライダー（0.5x〜3.0x） |

### 動画生成（generate_video.py）

viewer.html と同様のレイアウト（左: フレーム画像、右上: メッセージ、右下: 思考）でMP4動画を生成するスクリプトです。

**前提条件**: FFmpeg および Pillow がインストールされている必要があります。Pillow は `requirements.txt` に含まれています。

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

**使い方**:

```bash
# 基本的な使い方（output/ ディレクトリから動画を生成）
python visualization/generate_video.py output/

# 出力ファイル名とフレームレートを指定
python visualization/generate_video.py output/ -o result.mp4 --fps 20

# DPIを指定（デフォルト: 150）
python visualization/generate_video.py output/ --dpi 200
```

**オプション**:

| オプション | デフォルト | 説明 |
|---|---|---|
| `data_dir`（必須） | — | シミュレーション結果のディレクトリパス |
| `-o`, `--output` | `simulation.mp4` | 出力MP4ファイル名 |
| `--fps` | `10` | フレームレート（1秒あたりのフレーム数） |
| `--dpi` | `150` | 画像解像度（DPI） |

## LLMエージェント設計

### 基本設計

- 各エージェントは毎ステップ、**Message / Memory / Action** をLLMから生成し、同期的に行動
- 同一のLLMエンジンを全エージェントで共有し、差分は各自のMemoryと相互作用履歴のみ（個性は履歴から創発）
- プロンプトは「最適化タスク」を明示せず、状況説明＋数値データ＋近傍メッセージ＋自己状態を与える構成

### ペルソナ（性別）

- 各エージェントには初期化時に **male** または **female** がランダム（1/2の確率）で割り当てられる
- LLMプロンプトの冒頭（`You are Agent N (male/female)`）および自己状態欄（`Gender:`）に性別が含まれる
- 近隣エージェント情報にも相手の性別が表示される（例: `Agent 3 (female) is outside the places`）
- エージェントは相手の性別を認識した上でメッセージ内容や行動を決定する

### Action（行動）

4方向の離散選択 + 滞在:
- `up`: Y+1（上方向）
- `down`: Y-1（下方向）
- `left`: X-1（左方向）
- `right`: X+1（右方向）
- `stay`: 現在位置に滞在

### Memory（記憶）

- LLMが出力した `memory` フィールドが次ステップの「Previous Memory」として自己フィードバック
- 内部状態が履歴依存で進化し、個性が創発する仕組み
- `memory_limit`: 保存する最大メモリ数（古いものから削除）
- `memory_size`: LLM推論時に参照する直近のメモリ数

### 場所内限定情報

**各場所内のエージェントのみ**が以下の数値データを直接受け取る:
- **現在のエージェント数**（Number of agents here）
- **収容上限**（Capacity）
- **占有率**（Occupancy rate = エージェント数 / 収容上限）

定性的な評価（快適/不快等）は一切含まれず、数値の解釈はエージェント自身に委ねられる。

**場所外のエージェント**はこれらの情報を受け取らない（会話や推論で間接的に学ぶ）

**注意**: エージェントはすべての場所の位置情報を知っていますが、各場所の占有状況はその場所内にいるエージェントのみが知ることができます。

### コミュニケーション

- 通信半径内（デフォルト: 5セル）のエージェント間でメッセージ交換が可能
- **同一領域条件**: 以下の条件を満たすエージェント間でのみ通信可能
  - **両方とも場所外**にいる、または
  - **両方とも同じ場所内**にいる
  
  **通信制限**:
  - ❌ 場所内のエージェント ↔ 場所外のエージェント: **通信不可**
  - ❌ 異なる場所内のエージェント同士: **通信不可**
  - ✅ 同じ場所内のエージェント同士: **通信可能**
  - ✅ 両方とも場所外のエージェント同士: **通信可能**

- `message_history_limit`: 保存する最大メッセージ数（古いものから削除）
- `message_context_size`: LLM推論時に参照する直近のメッセージ数

### シミュレーションステップの実行順序

各ステップは以下の順序で実行されます:

1. **Phase 1: メッセージ決定** - 全エージェントがLLMでメッセージを決定
2. **Phase 2: メッセージ送信** - 意思決定時点での近傍エージェントにメッセージを送信
3. **Phase 3: 行動決定** - 全エージェントがLLMで行動（move/stay）を決定
4. **Phase 4: 移動実行** - エージェントが決定した方向に移動

この順序により、メッセージは移動前の位置関係に基づいて送信されます。

### LLM出力形式

**メッセージ決定（Phase 1）**:
```json
{
    "message": "近傍エージェントへのメッセージ（任意、最大200語）",
    "reasoning": "メッセージ送信の理由"
}
```

**行動決定（Phase 3）**:
```json
{
    "action": "move" or "stay",
    "direction": "up", "down", "left", or "right",
    "memory": "次ステップのために記憶したいこと",
    "reasoning": "決定理由"
}
```


## ライセンス

GNU General Public License v3.0

詳細は [LICENSE.txt](LICENSE.txt) を参照してください。

