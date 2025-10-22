# vDSP 実数FFT API調査

## 問題の整理

現在の実装は`vDSP_DFT_Execute`を使用していますが、これは**複素数→複素数**のDFTです。
実数信号のFFTには専用APIを使用する必要があります。

## vDSP実数FFT API

### 従来のAPI（vDSP_fft_zrip）

AppleのAccelerateフレームワークには2系統のFFT APIがあります：

1. **古いAPI**: `vDSP_fft_zrip` (split complex形式)
2. **新しいAPI**: `vDSP_DFT_zrop` (packed形式)

### vDSP_DFT_zrop（推奨）

**実数→複素数FFT**に特化したAPI。

**特徴**:
- 実数入力を直接受け取る
- 出力は複素数（実部・虚部分離）
- Nスケーリングなし（NumPyと同じ）
- librosaとの互換性が高い

**セットアップ**:
```swift
// Forward FFT (実数 → 複素数)
let fftSetup = vDSP_DFT_zrop_CreateSetup(
    nil,
    vDSP_Length(fftSize),
    .FORWARD
)

// Inverse FFT (複素数 → 実数)
let ifftSetup = vDSP_DFT_zrop_CreateSetup(
    nil,
    vDSP_Length(fftSize),
    .INVERSE
)
```

**実行**:
```swift
// Forward FFT
vDSP_DFT_Execute(fftSetup, input, stride, &realOut, &imagOut)

// Inverse FFT
vDSP_DFT_Execute(ifftSetup, realIn, imagIn, &output, &dummy)
```

**問題点**: `vDSP_DFT_zrop`は実際には存在しない可能性。
正しいAPIは`vDSP_DFT_zop`（複素数→複素数）のみかもしれません。

### 正しいアプローチ: vDSP_fft_zrip（split complex）

実数FFTの標準的な方法：

```swift
// セットアップ
let log2n = vDSP_Length(log2(Float(fftSize)))
let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2))!

// 実数 → split complex変換
var realPart = [Float](repeating: 0, count: fftSize/2)
var imagPart = [Float](repeating: 0, count: fftSize/2)
var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)

// 実数データをinterleaved形式に
var complexBuffer = [Float](repeating: 0, count: fftSize)
// input[0], 0, input[1], 0, input[2], 0, ...

// FFT実行
vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))

// iFFT実行
vDSP_fft_zrip(fftSetup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))

// スケーリング（iFFTの場合）
var scale = Float(1.0 / Float(fftSize))
vDSP_vsmul(output, 1, &scale, &output, 1, vDSP_Length(fftSize))
```

## 結論

**採用すべきAPI**: `vDSP_fft_zrip` (split complex形式)

**理由**:
1. Appleの公式実数FFT API
2. NumPy/librosaと同じアルゴリズム
3. 実績のある安定したAPI
4. split complex形式は実部・虚部が分離されているので扱いやすい

**注意点**:
1. iFFT後に1/Nスケーリングが必要
2. データレイアウトがNumPyと異なる（interleaved → split complex）
3. fftSize は2の累乗である必要がある
