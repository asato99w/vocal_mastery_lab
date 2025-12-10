# ピッチ検出パラメータチューニングガイド

## 現在のパラメータ設定

### FFT設定 (RealtimePitchDetector.swift)

```swift
// Buffer Configuration
private let bufferSize = 4096      // FFTバッファサイズ (2^12)
private let hopSize = 2048          // ホップサイズ (50% overlap)

// Frequency Range
let minFreq = 100.0  // 最小周波数 (G2) - ファイル分析用
let maxFreq = 500.0  // 最大周波数 (B4) - ファイル分析用

// HPS Configuration
let numHarmonics = 5  // ハーモニック次数 (1-5倍音)

// Thresholds
let rmsThreshold = 0.001        // RMS振幅閾値
let magnitudeThreshold = 0.001  // スペクトラムピーク閾値
let confidenceThreshold = 0.3   // 信頼度閾値
```

## 問題別パラメータ調整案

### 1. オクターブエラー修正 (優先度: 🔴 CRITICAL)

**問題**: Track10_Note1 (104.58 Hz) を 204.60 Hz と誤検出

**原因推定**:
- 基本周波数が弱く、第2倍音(2f0)が強い
- HPSが倍音を基本周波数と誤認識
- 低周波数帯域でのFFT解像度不足

**調整案A: ハーモニック次数を増やす**
```swift
// 現在
let numHarmonics = 5

// 提案
let numHarmonics = 7  // より多くの倍音を使用してF0を強化
```

**効果**: 多くの倍音を乗算することで、真の基本周波数のピークが強調される

**調整案B: 低周波数用の適応的バッファサイズ**
```swift
// 現在
private let bufferSize = 4096  // 固定

// 提案: 周波数帯域別の適応的サイズ
func getAdaptiveBufferSize(estimatedFreq: Double) -> Int {
    if estimatedFreq < 150.0 {
        return 8192  // 低周波数: より大きいバッファ
    } else if estimatedFreq < 300.0 {
        return 4096  // 中周波数: 標準
    } else {
        return 2048  // 高周波数: 小さいバッファ
    }
}
```

**効果**: 低周波数での周波数解像度が向上

**調整案C: サブハーモニックチェック**
```swift
// 新規追加: オクターブダウンをチェック
func checkSubharmonic(peakFreq: Double, magnitudes: [Float]) -> Double {
    let subharmonicFreq = peakFreq / 2.0
    let subharmonicBin = Int(subharmonicFreq * Double(bufferSize) / sampleRate)

    if subharmonicBin >= minBin && subharmonicBin < maxBin {
        let subharmonicMagnitude = magnitudes[subharmonicBin]
        let harmonicMagnitude = magnitudes[Int(peakFreq * Double(bufferSize) / sampleRate)]

        // サブハーモニックが十分強い場合、それを基本周波数とする
        if subharmonicMagnitude > harmonicMagnitude * 0.5 {
            return subharmonicFreq
        }
    }

    return peakFreq
}
```

**効果**: オクターブ上を検出した場合、1オクターブ下をチェックして修正

### 2. 高周波数バイアス軽減 (優先度: 🟡 IMPORTANT)

**問題**: 系統的に3-7 Hz高い周波数を検出 (55-85 cents error)

**原因推定**:
- FFTビンの量子化誤差
- パラボリック補間の精度不足
- ウィンドウ関数の影響

**調整案A: 補間アルゴリズムの改善**
```swift
// 現在: シンプルなパラボリック補間
let offset = 0.5 * (alpha - gamma) / denominator

// 提案: Quinn's first estimator (より正確)
func quinnsFirstEstimator(magnitudes: [Float], peakBin: Int) -> Double {
    let k = peakBin
    guard k > 0 && k < magnitudes.count - 1 else { return Double(k) }

    let alpha = Double(magnitudes[k - 1])
    let beta = Double(magnitudes[k])
    let gamma = Double(magnitudes[k + 1])

    // Quinn's interpolation formula
    let delta1 = alpha / beta
    let delta2 = gamma / beta

    let d1 = delta1 / (1.0 + delta1)
    let d2 = -delta2 / (1.0 + delta2)

    let delta = (d1 + d2) / 2.0 + tau(d1 * d1) - tau(d2 * d2)

    return Double(k) + delta
}

private func tau(_ x: Double) -> Double {
    let p1 = log(3.0 * x * x) / (4.0 * Double.pi * Double.pi)
    return 0.25 * log(x) + 0.25 * p1
}
```

**効果**: ビン間の周波数推定精度が向上

**調整案B: 周波数依存の補正テーブル**
```swift
// 評価結果から作成した補正テーブル
let frequencyCorrectionTable: [Double: Double] = [
    100.0: 0.98,   // 100 Hz帯域: 2%低く補正
    150.0: 0.99,   // 150 Hz帯域: 1%低く補正
    200.0: 1.00,   // 200 Hz帯域: 補正なし
    300.0: 1.00,   // 300 Hz帯域: 補正なし
]

func applyCorrectionFactor(frequency: Double) -> Double {
    // 線形補間で補正係数を取得
    // ... implementation
    return frequency * correctionFactor
}
```

**効果**: 既知のバイアスを経験的に補正

**調整案C: ウィンドウ関数の変更**
```swift
// 現在: Hann window
vDSP_hann_window(&window, vDSP_Length(bufferSize), Int32(vDSP_HANN_NORM))

// 提案: Blackman-Harris window (よりスペクトラル漏れが少ない)
vDSP_blkman_window(&window, vDSP_Length(bufferSize), Int32(vDSP_BLKMAN_NORM))
```

**効果**: スペクトラル漏れの削減により、ピーク位置の精度が向上

### 3. 信頼度メトリクスの改善 (優先度: 🟡 IMPORTANT)

**問題**: 全サンプルで信頼度100%、失敗ケースでも高信頼度

**原因推定**:
- 現在の信頼度計算が単純すぎる(ピーク/平均のみ)
- スペクトラムの明瞭度を考慮していない
- 複数フレームでの安定性をチェックしていない

**調整案A: 多要素信頼度**
```swift
// 現在: 単純なピーク/平均比
let avgMagnitude = magnitudes[minBin..<maxBin].reduce(0, +) / Float(maxBin - minBin)
let confidence = min(Double(maxMagnitude / (avgMagnitude + 0.001)), 1.0)

// 提案: 複数要素を組み合わせた信頼度
func calculateMultiFactorConfidence(
    maxMagnitude: Float,
    avgMagnitude: Float,
    hps: [Float],
    peakBin: Int
) -> Double {
    // 1. Peak-to-Average Ratio (0.0-1.0)
    let peakRatio = Double(maxMagnitude / (avgMagnitude + 0.001))
    let peakScore = min(peakRatio / 10.0, 1.0)  // Normalize

    // 2. Peak Prominence (ピークが周辺より突出しているか)
    let leftNeighbor = peakBin > 0 ? hps[peakBin - 1] : 0
    let rightNeighbor = peakBin < hps.count - 1 ? hps[peakBin + 1] : 0
    let avgNeighbor = (leftNeighbor + rightNeighbor) / 2.0
    let prominence = Double(maxMagnitude / (avgNeighbor + 0.001))
    let prominenceScore = min(prominence / 5.0, 1.0)

    // 3. Spectral Flatness (スペクトラムがフラットでないほど良い)
    let geometricMean = exp(magnitudes.map { log(Double($0) + 0.001) }.reduce(0, +) / Double(magnitudes.count))
    let arithmeticMean = Double(avgMagnitude)
    let spectralFlatness = geometricMean / arithmeticMean
    let clarityScore = 1.0 - spectralFlatness  // フラットでないほど高スコア

    // 4. Harmonic Consistency (倍音が整っているか)
    var harmonicScore = 0.0
    let fundamentalBin = peakBin
    var harmonicsFound = 0
    for harmonic in 2...5 {
        let harmonicBin = fundamentalBin * harmonic
        if harmonicBin < magnitudes.count {
            let harmonicMagnitude = magnitudes[harmonicBin]
            if harmonicMagnitude > avgMagnitude * 2.0 {
                harmonicsFound += 1
            }
        }
    }
    harmonicScore = Double(harmonicsFound) / 4.0  // 4個の倍音のうち何個見つかったか

    // 重み付け平均
    let confidence = peakScore * 0.3 +
                     prominenceScore * 0.3 +
                     clarityScore * 0.2 +
                     harmonicScore * 0.2

    return min(confidence, 1.0)
}
```

**効果**: より信頼性の高い信頼度スコアで、誤検出を識別可能

**調整案B: 時間的安定性チェック**
```swift
// 複数フレームでの周波数安定性をチェック
var recentDetections: [DetectedPitch] = []  // 直近5フレームの結果

func calculateTemporalStability() -> Double {
    guard recentDetections.count >= 3 else { return 0.5 }

    let frequencies = recentDetections.map { $0.frequency }
    let avgFreq = frequencies.reduce(0, +) / Double(frequencies.count)

    // 標準偏差を計算
    let variance = frequencies.map { pow($0 - avgFreq, 2) }.reduce(0, +) / Double(frequencies.count)
    let stdDev = sqrt(variance)

    // 変動が小さいほど高スコア (10 Hz以内で95%以上)
    let stabilityScore = exp(-stdDev / 10.0)

    return stabilityScore
}
```

**効果**: 安定した検出結果により高い信頼度を与える

### 4. 周波数帯域別最適化 (優先度: 🟢 RECOMMENDED)

**問題**: 低周波数帯域(<150 Hz)で精度が著しく低下(50%合格率)

**調整案: 適応的パラメータセット**
```swift
struct PitchDetectionParameters {
    let bufferSize: Int
    let numHarmonics: Int
    let minMagnitudeThreshold: Float
    let confidenceThreshold: Double
}

func getParametersForFrequencyRange(_ estimatedFreq: Double) -> PitchDetectionParameters {
    if estimatedFreq < 150.0 {
        // Low frequency: 大きいバッファ、多くの倍音
        return PitchDetectionParameters(
            bufferSize: 8192,
            numHarmonics: 7,
            minMagnitudeThreshold: 0.0005,
            confidenceThreshold: 0.25
        )
    } else if estimatedFreq < 300.0 {
        // Mid-low frequency: 標準設定
        return PitchDetectionParameters(
            bufferSize: 4096,
            numHarmonics: 5,
            minMagnitudeThreshold: 0.001,
            confidenceThreshold: 0.3
        )
    } else {
        // High frequency: 小さいバッファ、少ない倍音
        return PitchDetectionParameters(
            bufferSize: 2048,
            numHarmonics: 3,
            minMagnitudeThreshold: 0.002,
            confidenceThreshold: 0.4
        )
    }
}
```

**効果**: 各周波数帯域で最適なパラメータを使用

## パラメータ調整実験計画

### フェーズ1: オクターブエラー修正 (1週間)

**実験1-A: ハーモニック次数の影響**
- `numHarmonics`: 3, 5, 7, 9 を比較
- 評価指標: Track10_Note1の誤差、全体の合格率
- 期待結果: numHarmonics=7で最良

**実験1-B: サブハーモニックチェック**
- サブハーモニック閾値: 0.3, 0.5, 0.7 を比較
- 評価指標: オクターブエラー数、低周波数帯域精度
- 期待結果: 閾値0.5でバランス良好

**実験1-C: 適応的バッファサイズ**
- 低周波数用バッファ: 6144, 8192, 16384 を比較
- 評価指標: <150 Hz帯域の合格率、処理時間
- 期待結果: 8192でパフォーマンスと精度のバランス

**成功基準**:
- オクターブエラー完全解決(0件)
- <150 Hz帯域の合格率が70%以上に向上
- 全体合格率が80%以上

### フェーズ2: 高周波数バイアス軽減 (1週間)

**実験2-A: 補間アルゴリズム**
- パラボリック vs Quinn's estimator vs Jain's method
- 評価指標: 平均誤差、中周波数帯域の精度
- 期待結果: Quinn's estimatorで平均誤差30%削減

**実験2-B: ウィンドウ関数**
- Hann vs Hamming vs Blackman-Harris vs Kaiser
- 評価指標: ピーク検出精度、スペクトラル漏れ
- 期待結果: Blackman-Harrisで精度向上

**実験2-C: 補正テーブル**
- 評価データから作成した補正係数を適用
- 評価指標: 補正後の平均誤差、バイアス削減
- 期待結果: 系統的バイアスが50%以上削減

**成功基準**:
- 平均誤差が50 cents以下に改善
- 高周波数バイアス(55-85 cents)が半減
- 全体合格率が85%以上

### フェーズ3: 信頼度メトリクス改善 (1週間)

**実験3-A: 多要素信頼度**
- 各要素の重み付けを最適化(grid search)
- 評価指標: 信頼度とエラーの相関係数
- 期待結果: 相関係数0.7以上

**実験3-B: 時間的安定性**
- ウィンドウサイズ: 3, 5, 7フレーム を比較
- 評価指標: 安定性スコアと実際の精度の相関
- 期待結果: 5フレームで最良のバランス

**成功基準**:
- 失敗ケースで信頼度<0.5
- 成功ケースで信頼度>0.7
- 信頼度とエラーの負の相関(r < -0.6)

### フェーズ4: 周波数帯域別最適化 (2週間)

**実験4-A: 適応的パラメータ**
- 3つの周波数帯域で個別最適化
- 評価指標: 各帯域の合格率、全体スコア
- 期待結果: 各帯域で85%以上の合格率

**実験4-B: 機械学習ベースの最適化**
- ベイズ最適化またはグリッドサーチで全パラメータを最適化
- 探索空間: bufferSize, numHarmonics, thresholds
- 評価指標: vocadito全40トラックでの総合スコア
- 期待結果: スコア75/100 → 85/100

**成功基準**:
- 全周波数帯域で80%以上の合格率
- 総合スコア85/100以上
- リアルタイム性能を維持(<100ms レイテンシ)

## 実装チェックリスト

### Phase 1: オクターブエラー修正
- [ ] `numHarmonics` をパラメータ化
- [ ] サブハーモニックチェック関数の実装
- [ ] 適応的バッファサイズの実装
- [ ] ユニットテストの作成
- [ ] vocaditoテストでの検証
- [ ] パフォーマンステスト

### Phase 2: 高周波数バイアス軽減
- [ ] Quinn's estimator実装
- [ ] ウィンドウ関数の切り替え機能
- [ ] 補正テーブルの実装
- [ ] A/Bテストの実施
- [ ] 最適パラメータの決定

### Phase 3: 信頼度メトリクス改善
- [ ] 多要素信頼度関数の実装
- [ ] 時間的安定性チェックの実装
- [ ] 信頼度閾値の再調整
- [ ] vocaditoテストでの検証
- [ ] UIでの信頼度表示の改善

### Phase 4: 統合と最適化
- [ ] 適応的パラメータセットの実装
- [ ] パラメータ最適化スクリプトの作成
- [ ] 全テストスイートの実行
- [ ] パフォーマンス最適化
- [ ] ドキュメント更新

## 期待される改善結果

### ベースライン (現在)
- 総合スコア: **61.0 / 100**
- 合格率: **76.7%**
- 平均誤差: **72.4 cents**
- オクターブエラー: **1件**
- 低周波数帯域: **50.0%**

### Phase 1完了後
- 総合スコア: **70.0 / 100** (+9.0)
- 合格率: **80.0%** (+3.3%)
- 平均誤差: **60.0 cents** (-12.4)
- オクターブエラー: **0件** (-1)
- 低周波数帯域: **70.0%** (+20.0%)

### Phase 2完了後
- 総合スコア: **78.0 / 100** (+8.0)
- 合格率: **86.7%** (+6.7%)
- 平均誤差: **40.0 cents** (-20.0)
- 高周波数バイアス: **50%削減**

### Phase 3完了後
- 総合スコア: **80.0 / 100** (+2.0)
- 信頼度-誤差相関: **r = -0.7**
- 誤検出の信頼度: **<0.5**

### Phase 4完了後 (最終目標)
- 総合スコア: **85.0 / 100** (+5.0)
- 合格率: **90.0%** (+3.3%)
- 平均誤差: **30.0 cents** (-10.0)
- 全帯域で: **80%以上**

## 参考資料

### 論文・文献
1. "YIN, a fundamental frequency estimator for speech and music" (de Cheveigné & Kawahara, 2002)
2. "A smarter way to find pitch" (McLeod & Wyvill, 2005)
3. "Accurate Short-Term Analysis of the Fundamental Frequency" (Quinn, 1997)

### 実装リファレンス
- Librosa (Python): `librosa.pyin()`, `librosa.piptrack()`
- Essentia (C++): F0推定アルゴリズム集
- Aubio (C): `aubio_pitch_yin()`, `aubio_pitch_mcomb()`

### vocaditoデータセット
- 論文: "The NUS Sung and Spoken Lyrics Corpus" (Duan et al., 2013)
- 評価基準: F0トラッキング精度、ノートレベル精度
