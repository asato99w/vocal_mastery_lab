# UVR-MDX-NET CoreML PoC

UVR-MDX-NET ボーカル分離モデルのCoreML変換と検証用PoC。

## ディレクトリ構成

```
uvr_coreml_clean/
├── README.md
├── requirements.txt
├── models/
│   ├── onnx/           # 元のONNXモデル
│   └── coreml/         # 変換後のCoreMLモデル
├── scripts/
│   ├── convert_to_coreml.py   # ONNX→CoreML変換
│   ├── convert_audio.py       # 音声形式変換
│   ├── separate_vocals.py     # Python参照実装
│   └── verify_conversion.py   # 変換検証
├── swift/
│   ├── Package.swift
│   └── Sources/
│       ├── VocalSeparator.swift
│       └── main.swift
├── test_audio/
│   ├── raw/            # 元ファイル（mp3等）
│   └── {sample_name}/  # サンプルごとのディレクトリ
│       ├── mix.wav     # ミックス音源 (44.1kHz)
│       ├── vocal.wav   # ボーカル（正解データ、任意）
│       └── accomp.wav  # 伴奏（正解データ、任意）
└── output/
    ├── python/         # Python出力
    └── coreml/         # CoreML出力
```

## セットアップ

### Python環境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### モデル準備

ONNXモデルを `models/onnx/` に配置してください。

## 使用方法

### 1. テスト音源の準備

```bash
# 元ファイルをraw/に配置
cp input.mp3 test_audio/raw/

# サンプルディレクトリを作成し、wav変換
mkdir -p test_audio/my_sample
python scripts/convert_audio.py test_audio/raw/input.mp3 -o test_audio/my_sample/mix.wav --duration 15
```

### 2. ONNX → CoreML 変換

```bash
python scripts/convert_to_coreml.py UVR-MDX-NET-Voc_FT
```

### 3. ボーカル分離（Python）

```bash
python scripts/separate_vocals.py test_audio/hollow_crown/mix.wav
```

出力: `output/python/vocals.wav`, `output/python/instrumental.wav`

### 4. 変換検証

Python (ONNX) と CoreML の出力を比較:

```bash
python scripts/verify_conversion.py test_audio/hollow_crown/mix.wav
```

### 5. Swift実行（macOS）

```bash
cd swift
swift build
swift run VocalSeparator
```

## モデル仕様

| パラメータ | 値 |
|-----------|-----|
| 入力形状 | (1, 4, 3072, 256) |
| 出力形状 | (1, 4, 3072, 256) |
| サンプルレート | 44100 Hz |
| FFTサイズ | 6144 |
| ホップ長 | 1024 |

## 検証結果

正常な変換の場合:

- 相関係数: > 0.9999
- SNR: > 50 dB
- 最大絶対差: < 0.001

## ライセンス

UVR-MDX-NET モデルは MIT License で公開されています。
