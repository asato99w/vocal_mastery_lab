# UVR CoreML Implementation

UVR/MDX-Net音源分離モデルのCoreML実装

## ディレクトリ構造

```
uvr_coreml/
├── python/              # Python変換スクリプト
│   ├── download_model.py       # UVR ONNXモデルダウンロード
│   ├── convert_to_coreml.py    # ONNX → CoreML変換
│   ├── quantize_model.py       # モデル量子化
│   └── test_conversion.py      # 変換検証
├── swift/               # Swift実装
│   ├── STFTProcessor.swift     # vDSP STFT実装
│   ├── VocalSeparator.swift    # 音源分離メインクラス
│   └── CoreMLWrapper.swift     # CoreMLモデルラッパー
├── models/              # 変換済みモデル（.gitignore）
├── scripts/             # ユーティリティスクリプト
│   └── setup.sh               # 環境セットアップ
└── tests/               # テストファイル
    └── test_audio/            # テストオーディオサンプル
```

## セットアップ

### Python環境

```bash
# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージインストール
pip install -r requirements.txt
```

### モデルダウンロードと変換

```bash
# 1. UVR ONNXモデルダウンロード
python python/download_model.py

# 2. CoreML変換
python python/convert_to_coreml.py

# 3. 8-bit量子化
python python/quantize_model.py
```

## 実装状況

- [x] プロジェクト構造整備
- [ ] モデルダウンロードスクリプト
- [ ] CoreML変換スクリプト
- [ ] 量子化スクリプト
- [ ] STFT実装（Swift）
- [ ] CoreMLラッパー（Swift）
- [ ] 統合テスト

## パフォーマンス目標

| 指標 | 目標値 |
|------|--------|
| モデルサイズ | < 50MB |
| 処理時間（3分音源） | < 10秒 |
| メモリ使用量 | < 200MB |
| SDR（分離品質） | > 6dB |
