# T5Gemma-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/T5Gemma-TTS-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

![Architecture](figures/architecture.png)

Encoder-Decoder LLMアーキテクチャに基づく多言語Text-to-Speechモデル **T5Gemma-TTS** の学習および推論コードです。このリポジトリでは、データ前処理、学習（LoRAファインチューニングを含む）、推論のためのスクリプトを提供しています。

モデルの詳細、音声サンプル、技術情報については、[モデルカード](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)を参照してください。

## 特徴

- **Multilingual TTS**: 英語、中国語、日本語をサポート
- **Voice Cloning**: 参照音声からのzero-shot voice cloningをサポート
- **Duration Control**: 生成音声の長さを明示的に制御可能（未指定時は自動推定）
- **Flexible Training**: スクラッチからの学習、学習済みモデルのファインチューニング、LoRAファインチューニングをサポート
- **Multiple Inference Options**: コマンドライン、HuggingFaceフォーマット、Gradio Web UI

## インストール

```bash
git clone https://github.com/Aratako/T5Gemma-TTS.git
cd T5Gemma-TTS
pip install -r requirements.txt
```

**注意**: GPUサポートには、`pip install`を実行する前にCUDA対応のPyTorchをインストールしてください：
```bash
pip install "torch<=2.8.0" torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 既知の問題

- **Windows**: 開発者の環境において、Windows上での推論時に生成時間が不安定になったり、処理がハングしたりする問題が確認されています。原因は調査中です。同様の問題が発生した場合は、WSL2、またはDockerを使用することを推奨します。

## 動作確認済み環境

以下の環境で開発者が動作確認を行っています。他の環境でも動作する可能性がありますが、保証はしていません。

| プラットフォーム | 状態 | 備考 |
|----------|--------|-------|
| Linux + CUDA | ✅ 確認済み | |
| Windows + CUDA (Docker) | ✅ 確認済み | ネイティブWindowsには既知の問題あり |
| Apple Silicon (MPS) | ✅ 確認済み | M4 Max MacBook Pro |

## クイックスタート

### 基本的な推論（HuggingFaceフォーマット）

```bash
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "こんにちは、これは音声合成のテストです。"
```

### Voice Cloning

```bash
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "こんにちは、これは音声合成のテストです。" \
    --reference_text "これは参照音声です。" \
    --reference_speech path/to/reference.wav
```

### Duration Control

```bash
# 目標の長さを秒単位で指定
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "こんにちは、これは音声合成のテストです。" \
    --target_duration 5.0
```

**注意**: `--target_duration`を指定しない場合、音素数と言語ごとの音素長に基づいて適切な長さが自動計算されます。この計算は簡易的なものなので、結果が期待通りでない場合は手動で長さを指定してください。

## 推論

### HuggingFaceフォーマットを使用

```bash
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "素早い茶色のキツネは怠け者の犬を飛び越えます。" \
    --output_dir ./generated_tts
```

### .pthチェックポイントを使用

```bash
python inference_commandline.py \
    --model_root . \
    --model_name trained \
    --target_text "素早い茶色のキツネは怠け者の犬を飛び越えます。"
```

LoRA Checkpointの場合：

```bash
python inference_commandline.py \
    --model_root . \
    --model_name lora \
    --target_text "素早い茶色のキツネは怠け者の犬を飛び越えます。"
```

### Gradio Web UI

```bash
python inference_gradio.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --port 7860
```

デフォルトでは、日本語音声の品質のためにXCodec2-Variant（NandemoGHS/Anime-XCodec2-44.1kHz-v2）が使用されます。英語および中国語音声には、元のXCodec2モデルの使用を推奨します。

```bash
# 元のXCodec2モデルを使用する場合は、オリジナルのxcodec2ライブラリを使用する必要があります
pip install xcodec2==0.1.5 --no-deps

python inference_gradio.py \
    --model_dir t5gemma_voice_hf \
    --xcodec2_model_name HKUSTAudio/xcodec2 \
    --xcodec2_sample_rate 16000 \
    --port 7860
```

#### 低VRAM向けオプション（Gradio / HF推論）

- `--cpu_codec`: XCodec2トークナイザをCPUに載せる。VRAM使用量を約**3.5GB**削減（エンコード／デコードは遅くなる）。
- `--cpu_whisper`: Whisper（参照音声を自動書き起こしする場合）をCPUに載せる。VRAM使用量を約**5GB**削減。書き起こしは少し遅くなる。
- `--low_vram`: 上記2つをまとめて有効化し、`torch.compile`を無効化するプリセット。

品質は変わらず、GPUメモリと速度のトレードオフだけが変わります。

### Docker（Windowsユーザー向け推奨）

Windowsで問題が発生する場合、Dockerを使用することで安定したLinux環境で実行できます：

```bash
# docker-composeを使用（推奨）
docker compose up --build

# または手動でビルド・実行
docker build -t t5gemma-tts .
docker run --gpus all -p 7860:7860 t5gemma-tts
```

#### Docker設定オプション

環境変数でDockerの設定をカスタマイズできます：

```bash
# CUDAバージョンを指定（cu118, cu121, cu124, cu128）
CUDA_VERSION=cu121 docker compose up --build

# 別のモデルを使用
MODEL_DIR=your-org/your-model docker compose up

# ポートを変更
PORT=8080 docker compose up

# 追加の引数を渡す
EXTRA_ARGS="--no_compile --share" docker compose up
```

### 推論パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `--target_text` | (必須) | 合成するテキスト |
| `--reference_speech` | None | Voice Cloning用の参照音声パス |
| `--reference_text` | None | 参照音声の書き起こし（未指定時はWhisperで自動書き起こし） |
| `--target_duration` | None | 目標音声長（秒単位、未指定時は自動推定） |
| `--top_k` | 30 | Top-kサンプリングパラメータ |
| `--top_p` | 0.9 | Top-p（nucleus）サンプリングパラメータ |
| `--temperature` | 0.8 | Sampling temperature |
| `--seed` | 1 | ランダムシード（再現性のため） |

## 学習

### データ前処理

前処理スクリプトを使用して学習データを準備します。Emilia-YODASの英語サブセットの例：

```bash
python examples/data_preprocess/prepare_emilia_en.py \
    --output-dir datasets/emilia-yodas-en_0-9 \
    --data-files '{"train": "Emilia-YODAS/**/EN-B00000*.tar"}' \
    --encoder-devices auto \
    --valid-ratio 0.005 \
    --hf-num-proc 8
```

生成されるファイル：
- `text/` - テキスト書き起こし
- `xcodec2_1cb/` - XCodec2音声トークン
- `manifest_final/` - train / valid マニフェスト
- `neighbors/` - 音声プロンプト用のneighborファイル

### スクラッチからの学習例

```bash
NUM_GPUS=8 examples/training/t5gemma_2b-2b.sh
```

### 学習済みモデルのファインチューニング例

フルファインチューニング：

```bash
NUM_GPUS=8 examples/training/t5gemma_2b-2b-ft.sh
```

LoRA：

```bash
NUM_GPUS=1 examples/training/t5gemma_2b-2b-ft-lora.sh
```

### 学習パラメータ

主な学習パラメータ（完全な設定は学習スクリプトを参照）：

| パラメータ | 説明 |
|-----------|-------------|
| `--t5gemma_model_name` | ベースT5Gemmaモデル（例：`google/t5gemma-2b-2b-ul2`） |
| `--xcodec2_model_name` | 音声コーデックモデル |
| `--lr` | 学習率（ScaledAdamのデフォルト：0.035） |
| `--gradient_accumulation_steps` | 勾配累積ステップ数 |
| `--use_lora` | LoRA学習を有効化（1で有効） |
| `--lora_rank` | LoRAランク（デフォルト：8） |

## モデル変換

### .pthからHuggingFaceフォーマットへ変換

通常のCheckpoint：

```bash
python scripts/export_t5gemma_voice_hf.py \
    --ckpt trained.pth \
    --out t5gemma_voice_hf \
    --base_repo google/t5gemma-2b-2b-ul2
```

LoRA Checkpoint（アダプターをマージ）：

```bash
python scripts/export_t5gemma_voice_hf_lora.py \
    --ckpt lora.pth \
    --out t5gemma_voice_hf_lora_merged \
    --base_repo google/t5gemma-2b-2b-ul2 \
    --save_adapter_dir lora-adapter
```

## プロジェクトの構成

```
T5Gemma-TTS/
├── main.py                      # 学習のエントリーポイント
├── inference_commandline.py     # CLI推論（.pthフォーマット）
├── inference_commandline_hf.py  # CLI推論（HuggingFaceフォーマット）
├── inference_gradio.py          # Gradio Web UI
├── config.py                    # 設定と引数
├── requirements.txt             # 依存関係
│
├── models/                      # モデルアーキテクチャ
│   └── t5gemma.py               # T5Gemma-TTSモデル
│
├── data/                        # データ読み込み
│   ├── combined_dataset.py      # 複数データセットのローダー
│   └── tokenizer.py             # AudioTokenizer (XCodec2)
│
├── steps/                       # 学習インフラ
│   ├── trainer.py               # 分散学習トレーナー
│   └── optim.py                 # ScaledAdamオプティマイザ
│
├── scripts/                     # ユーティリティスクリプト
│   ├── export_t5gemma_voice_hf.py      # HFフォーマットへエクスポート
│   └── export_t5gemma_voice_hf_lora.py # LoRAをHFフォーマットへエクスポート
│
├── hf_export/                   # HuggingFaceモデルラッパー
│   ├── configuration_t5gemma_voice.py
│   └── modeling_t5gemma_voice.py
│
└── examples/
    ├── training/                # 学習シェルスクリプト
    │   ├── t5gemma_2b-2b.sh           # スクラッチからの学習
    │   ├── t5gemma_2b-2b-ft.sh        # フルファインチューニング
    │   └── t5gemma_2b-2b-ft-lora.sh   # LoRA
    └── data_preprocess/         # データ前処理
        └── prepare_emilia_en.py       # Emilia-YODASの英語データ準備を行う例
```

## 制限事項

- **推論速度**: リアルタイムTTSアプリケーションには最適化されていません。音声トークンの自己回帰生成には時間がかかるため、低遅延が求められるユースケースには不向きです。
- **Duration Control**: 明示的な長さ指定をサポートしていますが、制御は完璧ではありません。生成された音声が指定した長さと異なる場合や、長さが合っていても発話のペースや自然さが最適でない場合があります。
- **音声品質**: 品質は学習データの特性に依存します。学習データに含まれていない声質、アクセント、話し方では性能が低下する可能性があります。

## ライセンス

- **コード**: [MITライセンス](LICENSE)
- **モデル**: ライセンスの詳細は[モデルカード](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)を参照してください

## 謝辞

このプロジェクトは以下の研究に基づいています：

- [VoiceStar](https://arxiv.org/abs/2505.19462) - アーキテクチャの参考とベースとなるコード
- [T5Gemma](https://huggingface.co/google/t5gemma-2b-2b-ul2) - ベースモデル
- [XCodec2](https://huggingface.co/HKUSTAudio/xcodec2)および[XCodec2-Variant](https://huggingface.co/NandemoGHS/Anime-XCodec2-44.1kHz-v2) - 音声コーデック

## 引用

```bibtex
@misc{t5gemma-tts,
  author = {Aratako},
  title = {T5Gemma-TTS: An Encoder-Decoder LLM-based TTS Model},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/T5Gemma-TTS}}
}
```
