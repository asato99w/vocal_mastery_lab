# ノイズ問題詳細調査 - 進捗レポート

## 現状サマリー

**問題**: Swift実装の音声出力に「聞くに耐えないレベルのノイズ」が存在
**Python参照実装**: 高品質な出力 (Max 1.000, 相関 0.926)
**Swift実装**: ノイズあり (Max 0.089, 相関 0.994)

## 調査フェーズ

### フェーズ1: 振幅スケーリング問題の発見と修正

**発見**:
- Swift実装が元々11倍振幅不足 (Max 0.089 vs Python 1.000)
- iFFTのスケーリングを削除する修正を試行
- 結果: 4倍改善したが、まだ2.7倍不足 (Max 0.363)

**結論**:
- iFFTは1/Nスケーリングが**必要** (vDSPはスケーリングしないため)
- 振幅不足だけが問題ではなく、**本質的なノイズが残存**

### フェーズ2: STFT/iSTFTの詳細検証

**実験**: 440Hz正弦波でSTFT/iSTFT round-trip test

**Python (librosa)**:
```
入力: Max 1.000
STFT: Max magnitude 1012.5
iSTFT: Max 1.000
再構成誤差: RMS 0.000000 (完全再構成)
```

**Swift (vDSP) - 修正前**:
```
入力: Max 1.000
STFT: Max magnitude 852.5 (Python比 84%)
iSTFT: Max 1.000 (1/Nスケーリング後)
再構成誤差: RMS 2894.0 (完全に誤り)
```

**重大な発見**:
- Swift STFTのmagnitudeがPythonより16%低い
- iSTFTでの再構成が完全に失敗
- **共役対称性の実装に問題がある可能性**

### フェーズ3: モデル入出力比較 (完了)

**目的**: ONNX vs CoreML で同じ入力に対する出力を比較

**Python ONNX モデル入出力** (440Hz テスト信号):
```
入力範囲: -327.4 ~ 315.0
出力範囲: -274.5 ~ 246.2
440Hz bin入力: 315.0
440Hz bin出力: 246.2
```

**Swift CoreML モデル入出力**:
```
入力範囲: -345.7 ~ 236.9
出力範囲: -301.3 ~ 201.4
440Hz bin入力: 236.9
440Hz bin出力: 117.1
```

**比較結果**:
- モデル入力の相関係数: **0.012** (ほぼ無相関！)
- モデル出力の相関係数: **-0.26** (負の相関！)
- 440Hz bin入力の差: 78 (Python 315 vs Swift 237)
- 440Hz bin出力の差: 129 (Python 246 vs Swift 117)

**結論**: モデル入力が完全に壊れている → STFT実装に致命的な問題

### フェーズ4: STFT詳細比較 (完了)

**Python librosa STFT vs Swift vDSP STFT**:

**Magnitude比較**:
```
Python Max: 509.7
Swift Max: 852.6
相関係数: 0.90 (高相関だが...)
```

**実部・虚部比較** (440Hz信号):
```
実部相関: 0.012 (無相関!)
虚部相関: -0.991 (完全に反転!)
```

**DC bin (bin 0)の異常**:
```
Python magnitude: 15.95
Swift magnitude: 0.00058 (2.7万倍小さい!)
Python phase: 0.0
Swift phase: π (180度反転!)
```

**低周波成分 (bin 0-19)の異常**:
- Swift magnitudeが全て0.001以下
- Pythonは15～20の値
- 比率が1万～2.7万倍の差

## 発見された根本原因

### 🔴 **vDSP_DFT_Execute の誤用**

**問題のコード** (STFTProcessor.swift:229-232):
```swift
// 入力を実数部にコピー
real = input
imaginary = [0, 0, ..., 0]  // ゼロ初期化

// ❌ 間違い: vDSP_DFT_Execute は複素数→複素数のFFT
vDSP_DFT_Execute(setup, real, imaginary, &real, &imaginary)
```

**問題点**:
1. `vDSP_DFT_Execute`は**複素数入力→複素数出力**のDFT
2. 実数信号のFFTには`vDSP_DFT_zrop`（実数→複素数）を使うべき
3. 虚部をゼロで初期化した複素数として処理しているため:
   - DC成分が正しく計算されない
   - 低周波成分が欠落
   - 位相が反転（虚部相関-0.99）
   - 高周波成分のmagnitudeが過大評価

**証拠**:
- DC bin magnitude: 2.7万倍の差
- 実部相関: 0.012 (無相関)
- 虚部相関: -0.991 (完全反転)
- iSTFT再構成エラー: RMS 2894 (完全に失敗)

### 1. STFT Magnitude差異 (16%) - 原因特定済み
```python
Python librosa FFT: Max magnitude 1012.5 (正しい実数FFT)
Swift vDSP FFT: Max magnitude 852.5 (誤った複素数DFT)
```

**根本原因**:
- vDSP_DFT_Execute を実数FFTに誤用
- 正しくは vDSP_DFT_zrop を使用すべき

### 2. iFFT共役対称性の実装

現在の実装:
```swift
for i in 1..<usedBins-1 {
    let mirrorIndex = fftSize - i
    real[mirrorIndex] = real[i]
    imaginary[mirrorIndex] = -imaginary[i]  // 共役
}
```

**潜在的問題**:
- `usedBins` が2048の場合、ミラーリングが不完全
- DC (i=0) とNyquist (i=2048) の扱いが不適切
- fftSize=4096, usedBins=2048 の場合、後半が未初期化

### 3. Window関数の正規化

**librosa**: Overlap-Add時にwindow^2で自動正規化
**vDSP Swift**: 手動でwindowSum正規化

```swift
windowSum[outputIndex] += windowBuffer[i] * windowBuffer[i]
output[i] /= windowSum[i]
```

この正規化が不十分な可能性

## 次のステップ

### 即時対応 (実行中)
1. ✅ Python ONNX モデル出力をダンプ
2. 🔄 Swift CoreML モデル出力をダンプ (実行中)
3. ⏳ 両者を比較してモデルレベルでの差異を確認

### 詳細調査 (予定)
4. STFT magnitude差異の原因特定
   - Window関数のスケーリング確認
   - FFT正規化の比較
   
5. iFFT共役対称性の修正
   - usedBins=2048の場合の正しい実装
   - DC/Nyquistの適切な処理
   
6. Overlap-Add正規化の見直し
   - librosaと同等の正規化を実装

## 技術的発見

### vDSP FFT/iFFT の動作
```
Forward FFT: スケーリングなし
Inverse FFT: スケーリングなし (1/N が必要)
```

### NumPy/librosa の動作
```
Forward FFT: スケーリングなし
Inverse FFT: 自動正規化済み (irfft)
```

### Window正規化
```
Hann window: mean(window^2) = 0.375
4倍overlap (hop=fft/4): 理論的に完全再構成可能
```

## 結論

**根本原因を特定しました**:

### 🎯 主要原因: vDSP FFTの誤用

1. **vDSP_DFT_Execute を実数FFTに誤用**:
   - `vDSP_DFT_Execute`は複素数→複素数のDFT
   - 実数信号には`vDSP_DFT_zrop`（実数→複素数）を使うべき
   - この誤用により:
     - DC成分が2.7万倍小さくなる
     - 低周波成分が完全に欠落
     - 位相が反転（虚部相関-0.99）
     - モデル入力が完全に壊れる（相関0.012）

2. **連鎖的な影響**:
   - 壊れたSTFT → 壊れたモデル入力
   - 壊れたモデル入力 → 壊れたモデル出力（相関-0.26）
   - 壊れたマスク → 誤ったiSTFT入力
   - iSTFT再構成の完全失敗（RMS誤差2894）
   - 最終的に「聞くに耐えないノイズ」

### ✅ 証拠

**定量的証拠**:
- DC bin magnitude差: 2.7万倍
- 低周波成分 (0-19 bin): 1万～2.7万倍の差
- 実部相関: 0.012 (無相関)
- 虚部相関: -0.991 (完全反転)
- モデル入力相関: 0.012 (無相関)
- モデル出力相関: -0.26 (負の相関)
- iSTFT再構成誤差: RMS 2894 vs Python 0.0

### 📋 次のステップ

1. **STFTProcessor.swiftの完全書き直し**:
   - `vDSP_DFT_zrop`を使用した正しい実数FFT実装
   - `vDSP_DFT_zrop`のiFFT版を使用
   - Appleの公式ドキュメントに従った実装

2. **検証**:
   - 440Hz信号でSTFT/iSTFT round-trip test
   - PythonのSTFT出力と完全一致を確認
   - モデル入力の相関が0.99以上になることを確認

3. **最終確認**:
   - Hollow Crownでの音源分離実行
   - Python出力との品質比較
