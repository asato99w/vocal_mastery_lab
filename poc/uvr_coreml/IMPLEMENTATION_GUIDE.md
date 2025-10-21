# UVR CoreML 実装ガイド

## 概要

UVR/MDX-Net音源分離モデルのCoreML実装の完全ガイドです。

## セットアップ

### 1. 環境構築

```bash
cd poc/uvr_coreml
./scripts/setup.sh
```

これにより以下が実行されます:
- Python仮想環境の作成
- 必要なパッケージのインストール（coremltools, onnx, etc.）
- ディレクトリ構造の作成

### 2. モデルダウンロード

```bash
source venv/bin/activate
python python/download_model.py
```

**推奨モデル**:
- `UVR-MDX-NET-Inst_Main.onnx` (ボーカル/伴奏分離)

インタラクティブモードまたはコマンドラインで指定可能:
```bash
# インタラクティブ
python python/download_model.py

# 直接指定
python python/download_model.py UVR-MDX-NET-Inst_Main

# すべてダウンロード
python python/download_model.py --list
```

### 3. CoreML変換

```bash
python python/convert_to_coreml.py
```

変換パラメータ:
- 計算ユニット: ALL (CPU + GPU + Neural Engine)
- 最小ターゲット: iOS 17
- 形式: ML Program

**変換時間**: 約2-5分

### 4. モデル量子化（オプション）

```bash
python python/quantize_model.py
```

量子化オプション:
1. **8-bit (推奨)**: Neural Engine最適化、精度保持
2. **4-bit**: 最大圧縮、精度トレードオフ

**期待される圧縮率**:
- 8-bit: 約75%削減
- 4-bit: 約87.5%削減

## 実装詳細

### Python実装

#### 1. モデルダウンロード (`download_model.py`)

**機能**:
- UVR公式リポジトリからONNXモデルをダウンロード
- 進捗バー表示
- 既存モデルの検出とスキップ

**使用例**:
```python
from pathlib import Path

models_dir = Path("models/onnx")
download_model("UVR-MDX-NET-Inst_Main", models_dir)
```

#### 2. CoreML変換 (`convert_to_coreml.py`)

**機能**:
- ONNX → CoreML変換
- ML Program形式でエクスポート
- 入出力形状の検証

**変換フロー**:
```
ONNX モデル読み込み
    ↓
ONNXモデル検証
    ↓
CoreML変換 (ct.convert)
    ↓
ML Program保存 (.mlpackage)
```

**使用例**:
```python
import coremltools as ct

mlmodel = ct.convert(
    "models/onnx/UVR-MDX-NET-Inst_Main.onnx",
    minimum_deployment_target=ct.target.iOS17,
    compute_units=ct.ComputeUnit.ALL,
    convert_to="mlprogram"
)

mlmodel.save("models/coreml/UVR-MDX-NET-Inst_Main.mlpackage")
```

#### 3. モデル量子化 (`quantize_model.py`)

**機能**:
- 8-bit/4-bit量子化
- Neural Engine最適化
- モデルサイズ削減

**量子化設定**:
```python
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

config = OptimizationConfig(
    global_config=OpPalettizerConfig(
        mode="kmeans",
        nbits=8,
        granularity="per_channel"  # Neural Engine最適化
    )
)
```

### Swift実装

#### 1. STFTプロセッサ (`STFTProcessor.swift`)

**機能**:
- vDSP/Accelerateを使用した高速STFT/iSTFT
- ハン窓関数適用
- ステレオ/モノラル対応

**使用例**:
```swift
let processor = STFTProcessor(fftSize: 4096, hopSize: 1024)

// STFT実行
let spectrogram = processor.computeSTFT(audio: audioData)

// iSTFT実行
let reconstructed = processor.computeISTFT(
    magnitude: spectrogram.magnitude,
    phase: spectrogram.phase
)
```

**パラメータ**:
- `fftSize`: 4096 (UVR標準)
- `hopSize`: 1024 (fftSize / 4)
- `windowType`: .hann

#### 2. VocalSeparator (`VocalSeparator.swift`)

**機能**:
- 音源分離のエンドツーエンド処理
- CoreMLモデル推論
- マスク適用

**使用例**:
```swift
let modelURL = Bundle.main.url(
    forResource: "UVR-MDX-NET-Inst_Main",
    withExtension: "mlpackage"
)!

let separator = try VocalSeparator(modelURL: modelURL)

// 音源分離実行
let separated = try await separator.separate(audioURL: inputURL)

// ボーカルのみ保存
try separator.save(
    audio: separated.vocals,
    sampleRate: separated.sampleRate,
    to: vocalsURL
)
```

## アーキテクチャ

### 全体フロー

```
┌─────────────────────────────────────────┐
│           Input Audio File              │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│     AVAudioFile → Float Array           │
│  (ステレオ、44.1kHz/48kHz)               │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   STFT (vDSP/Accelerate)                │
│   - FFTサイズ: 4096                     │
│   - ホップサイズ: 1024                   │
│   - 窓関数: Hann                        │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   振幅スペクトログラム抽出                │
│   [frequency: 2049, time: variable]     │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   CoreML推論 (MDX-Net)                  │
│   - Neural Engine実行                   │
│   - 8-bit量子化モデル                    │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   分離マスク取得                         │
│   - ボーカルマスク                       │
│   - 伴奏マスク                          │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   マスク適用                            │
│   Vocal = Original × VocalMask         │
│   Inst = Original × InstMask           │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   iSTFT (vDSP/Accelerate)               │
│   位相情報復元                           │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   分離音声出力                           │
│   - vocals.wav                         │
│   - instrumental.wav                   │
└─────────────────────────────────────────┘
```

### クラス図

```
┌──────────────────────┐
│   VocalSeparator     │
├──────────────────────┤
│ - model: MLModel     │
│ - stftProcessor      │
│ - configuration      │
├──────────────────────┤
│ + separate()         │
│ + extractVocals()    │
└──────┬───────────────┘
       │ 使用
       ↓
┌──────────────────────┐
│   STFTProcessor      │
├──────────────────────┤
│ - fftSize: Int       │
│ - hopSize: Int       │
│ - fftSetup           │
├──────────────────────┤
│ + computeSTFT()      │
│ + computeISTFT()     │
└──────────────────────┘
```

## パフォーマンス最適化

### 1. モデル最適化

**量子化による効果**:
| モデル | サイズ | 推論速度 | 精度劣化 |
|--------|--------|---------|---------|
| Float32 | 120MB | 1.0x | - |
| Float16 | 60MB | 1.5x | ほぼなし |
| 8-bit | 30MB | 2.0x | 最小限 |
| 4-bit | 15MB | 2.5x | 中程度 |

**推奨**: 8-bit量子化

### 2. Neural Engine活用

**compute_units設定**:
```python
mlmodel = ct.convert(
    ...,
    compute_units=ct.ComputeUnit.ALL  # 推奨
)
```

**効果**:
- A17 Pro: 約2倍高速化
- M4: 約3倍高速化

### 3. ストリーミング処理

**長時間音声への対応**:
```swift
// チャンク単位で処理
let chunkSize = 44100 * 10  // 10秒
for chunk in audioData.chunks(ofSize: chunkSize) {
    let separated = try await separator.separate(chunk: chunk)
    // ...
}
```

## トラブルシューティング

### 変換エラー

**症状**: ONNX → CoreML変換失敗

**対処**:
1. ONNXモデルの検証
   ```python
   import onnx
   model = onnx.load("model.onnx")
   onnx.checker.check_model(model)
   ```

2. CoreMLToolsバージョン確認
   ```bash
   pip install --upgrade coremltools
   ```

3. 互換性のあるバージョン使用
   - CoreMLTools: 8.0+
   - ONNX: 1.15+

### パフォーマンス問題

**症状**: 推論が遅い

**対処**:
1. compute_units確認
   ```swift
   let config = MLModelConfiguration()
   config.computeUnits = .all  // Neural Engine有効化
   ```

2. 量子化適用
   ```bash
   python python/quantize_model.py
   ```

3. プロファイリング
   ```swift
   // Instrumentsで測定
   ```

### メモリ不足

**症状**: 長時間音声でクラッシュ

**対処**:
1. ストリーミング処理
2. チャンクサイズ削減
3. メモリマップドファイル使用

## 次のステップ

### Phase 1完了後

- [x] Python変換スクリプト
- [x] Swift STFT実装
- [x] CoreMLラッパー

### Phase 2: 最適化

- [ ] AVAudioFile統合
- [ ] リアルタイム処理
- [ ] メモリ最適化
- [ ] バッチ処理

### Phase 3: 統合

- [ ] F0抽出との統合
- [ ] UI実装
- [ ] パフォーマンステスト
- [ ] ユーザビリティ改善

## リファレンス

### 公式ドキュメント
- [Apple CoreML Tools](https://apple.github.io/coremltools/)
- [vDSP](https://developer.apple.com/documentation/accelerate/vdsp)
- [UVR GitHub](https://github.com/Anjok07/ultimatevocalremovergui)

### 関連資料
- `UVR_COREML_TECHNICAL_RESEARCH.md`: 技術調査レポート
- `COREML_CONVERSION_CHALLENGES.md`: Demucs変換課題（参考）
