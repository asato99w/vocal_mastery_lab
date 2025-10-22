# vDSP_fft_zrip データレイアウト調査

## 公式ドキュメントの要約

### vDSP_fft_zrip の仕様

**関数シグネチャ**:
```c
void vDSP_fft_zrip(
    FFTSetup          setup,
    DSPSplitComplex  *ioData,
    vDSP_Stride       stride,
    vDSP_Length       log2n,
    FFTDirection      direction
);
```

**重要な仕様**:
1. **入力**: Split complex形式 (realp, imagp)
2. **実数FFTの特殊なパッキング**:
   - 入力の実数データは、偶数インデックスをrealp、奇数インデックスをimagpに配置
   - N個の実数 → N/2個のsplit complex
   - 例: [r0, r1, r2, r3, r4, r5, r6, r7] → realp[0,1,2,3] = [r0,r2,r4,r6], imagp[0,1,2,3] = [r1,r3,r5,r7]

3. **出力のパッキング**:
   - DC成分: realp[0]
   - Nyquist成分: imagp[0]
   - 他の周波数: realp[k], imagp[k] (k=1...N/2-1)

### 正しい実数FFTの手順

**Forward FFT (実数 → 周波数)**:
```swift
// 1. 実数データをsplit complexにパック
let N = 8  // 実数サンプル数
let halfN = N / 2
var realPart = [Float](repeating: 0, count: halfN)
var imagPart = [Float](repeating: 0, count: halfN)

// 偶数インデックス → realp、奇数インデックス → imagp
for i in 0..<halfN {
    realPart[i] = input[2*i]      // r0, r2, r4, r6
    imagPart[i] = input[2*i + 1]  // r1, r3, r5, r7
}

// 2. FFT実行
var splitComplex = DSPSplitComplex(realp: &realPart, imagp: &imagPart)
vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFT_FORWARD)

// 3. 結果の解釈
// realp[0] = DC成分
// imagp[0] = Nyquist成分
// realp[k], imagp[k] = 周波数kの実部・虚部 (k=1...N/2-1)
```

**Inverse FFT (周波数 → 実数)**:
```swift
// 1. 周波数データをsplit complexに配置
// realp[0] = DC成分
// imagp[0] = Nyquist成分
// realp[k], imagp[k] = 周波数kの実部・虚部

// 2. iFFT実行
vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFT_INVERSE)

// 3. スケーリング
var scale = Float(0.5)  // 1/(2N) のうちの1/2
vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(halfN))
vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(halfN))

// 4. split complexからアンパック
for i in 0..<halfN {
    output[2*i]     = realPart[i]  // 偶数インデックス
    output[2*i + 1] = imagPart[i]  // 奇数インデックス
}
```

## 現在の実装の問題点

**現在のperformFFT** (STFTProcessor.swift:244-253):
```swift
// ❌ 間違い: vDSP_ctozを使用している
input.withUnsafeBytes { inputBytes in
    let inputPtr = inputBytes.baseAddress!.assumingMemoryBound(to: DSPComplex.self)
    vDSP_ctoz(inputPtr, 2, &splitComplex, 1, vDSP_Length(halfSize))
}
```

**問題**:
- vDSP_ctozは複素数配列の変換用
- 実数FFTには直接的なパッキングが必要

## 正しい実装

**performFFT修正版**:
```swift
private func performFFT(_ input: [Float]) -> (magnitude: [Float], phase: [Float]) {
    guard let setup = fftSetup else {
        fatalError("FFT setup not initialized")
    }

    let halfSize = fftSize / 2
    var realPart = [Float](repeating: 0, count: halfSize)
    var imagPart = [Float](repeating: 0, count: halfSize)

    // 実数データをsplit complexにパック (偶数→realp, 奇数→imagp)
    for i in 0..<halfSize {
        realPart[i] = input[2*i]
        imagPart[i] = input[2*i + 1]
    }

    // FFT実行
    realPart.withUnsafeMutableBufferPointer { realBuffer in
        imagPart.withUnsafeMutableBufferPointer { imagBuffer in
            var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
            vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_FORWARD))
        }
    }

    // 結果をmagnitude/phaseに変換
    var magnitude = [Float](repeating: 0, count: frequencyBins)
    var phase = [Float](repeating: 0, count: frequencyBins)

    // DC成分 (bin 0)
    magnitude[0] = abs(realPart[0])
    phase[0] = realPart[0] >= 0 ? 0 : Float.pi

    // Nyquist成分 (bin N/2)
    magnitude[halfSize] = abs(imagPart[0])
    phase[halfSize] = imagPart[0] >= 0 ? 0 : Float.pi

    // 中間成分 (bin 1 ~ N/2-1)
    for i in 1..<halfSize {
        let re = realPart[i]
        let im = imagPart[i]
        magnitude[i] = sqrtf(re * re + im * im)
        phase[i] = atan2f(im, re)
    }

    return (magnitude, phase)
}
```

**performIFFT修正版**:
```swift
private func performIFFT(magnitude: [Float], phase: [Float]) -> [Float] {
    guard let setup = fftSetup else {
        fatalError("FFT setup not initialized")
    }

    let inputBins = magnitude.count
    let halfSize = fftSize / 2

    var realPart = [Float](repeating: 0, count: halfSize)
    var imagPart = [Float](repeating: 0, count: halfSize)

    // DC成分
    realPart[0] = magnitude[0] * cos(phase[0])

    // Nyquist成分
    imagPart[0] = inputBins > halfSize ? magnitude[halfSize] * cos(phase[halfSize]) : 0

    // 中間成分
    let usedBins = min(inputBins, halfSize)
    for i in 1..<usedBins {
        realPart[i] = magnitude[i] * cos(phase[i])
        imagPart[i] = magnitude[i] * sin(phase[i])
    }

    // iFFT実行
    realPart.withUnsafeMutableBufferPointer { realBuffer in
        imagPart.withUnsafeMutableBufferPointer { imagBuffer in
            var splitComplex = DSPSplitComplex(realp: realBuffer.baseAddress!, imagp: imagBuffer.baseAddress!)
            vDSP_fft_zrip(setup, &splitComplex, 1, log2n, FFTDirection(FFT_INVERSE))
        }
    }

    // スケーリング (1/2N = 1/2 * 1/N)
    var scale = Float(0.5)
    vDSP_vsmul(realPart, 1, &scale, &realPart, 1, vDSP_Length(halfSize))
    vDSP_vsmul(imagPart, 1, &scale, &imagPart, 1, vDSP_Length(halfSize))

    // Split complexからアンパック
    var output = [Float](repeating: 0, count: fftSize)
    for i in 0..<halfSize {
        output[2*i] = realPart[i]
        output[2*i + 1] = imagPart[i]
    }

    return output
}
```

## 次のステップ

1. STFTProcessor.swiftのperformFFTとperformIFFTを修正
2. 440Hz信号でround-trip testを実行
3. PythonのSTFT出力と比較
