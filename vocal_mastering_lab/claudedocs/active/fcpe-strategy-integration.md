# FCPE Strategy Integration Design

## 概要

FCPE (Fast Context-based Pitch Estimation) を VocalisStudio の `PitchDetectionStrategy` として統合する設計ドキュメント。

## 現状

### 既存のStrategy実装 (`feature/pitch-detection-strategy`ブランチ)

```
VocalisStudio/Infrastructure/Analysis/
├── PitchStrategyFactory.swift
├── Strategies/
│   ├── YINStrategy.swift      # 生波形 → ピッチ
│   └── PYINStrategy.swift     # 生波形 → ピッチ (HMM)
```

**共通インターフェース** (`VocalisDomain/ServiceInterfaces/PitchDetectionStrategy.swift`):
```swift
public protocol PitchDetectionStrategy {
    var name: String { get }
    var requiresOctaveCorrection: Bool { get }
    func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame]
}
```

### FCPE CoreML PoC結果

| 指標 | FCPE | CREPE | 備考 |
|------|------|-------|------|
| RPA (50cents以内) | 99.1% | 99.2% | ほぼ同等 |
| 処理時間 | 0.14s | 19.7s | **133倍高速** |
| モデルサイズ | ~15MB | ~80MB | 5倍小さい |

**結論**: FCPEはCREPEと同等精度で133倍高速。ファイル分析用途に最適。

## FCPEパイプライン

```
[音声波形]
    ↓ wav2mel (Mel Spectrogram変換)
[Mel Spectrogram: (1, T, 128)]
    ↓ CoreML Model (fcpe_core_fp32.mlpackage)
[Logits: (1, T, 360)]
    ↓ local_argmax decoder
[F0 Hz: (T,)]
```

## 課題と解決策

### 課題1: Mel Spectrogram前処理 ✅ 解決済み

**問題**: FCPEはMelスペクトログラムを入力とする。YIN/pYINは生波形を直接処理。

**PoC結果** (`mel_spectrogram_poc.py`):
| 実装 | Argmax一致率 | F0精度 (5cents) |
|------|-------------|-----------------|
| librosa版 | 99.9% | 100% |
| numpy/scipy版 | 99.9% | 100% |

**解決策: Swift側でMel変換を実装**

```swift
class MelSpectrogramTransformer {
    // torchfcpe互換のMel変換パラメータ
    static let sampleRate: Int = 16000
    static let nMels: Int = 128
    static let nFFT: Int = 1024
    static let winSize: Int = 1024
    static let hopLength: Int = 160  // 10ms at 16kHz
    static let fmin: Float = 0
    static let fmax: Float = 8000
    static let clipVal: Float = 1e-5

    // 事前計算済みMelフィルターバンク行列 (128 x 513)
    // mel_filterbank_16k_128.bin からロード
    private let melFilterbank: [[Float]]

    func transform(samples: [Float]) -> [[Float]] {
        // 1. カスタムパディング (torchfcpe互換)
        let padLeft = (winSize - hopLength) / 2  // = 432
        let padRight = max((winSize - hopLength + 1) / 2, winSize - samples.count - padLeft)

        // 2. 反射パディング (padRight < samples.count の場合)
        let padded = reflectPad(samples, left: padLeft, right: padRight)

        // 3. STFT (center=False, Hann window)
        // 4. Magnitude: sqrt(real² + imag² + 1e-9)
        // 5. Mel変換: melFilterbank × magnitude
        // 6. Log圧縮: log(clamp(x, min=1e-5))
    }
}
```

**重要な発見**:
- Melフィルターバンクには**Slaney正規化**が必要（librosa.filters.melのデフォルト）
- 事前計算されたMelフィルターバンク行列をバイナリファイルとして同梱
- `mel_filterbank_16k_128.bin` (128×513 float32 = ~263KB)

### 課題2: local_argmax デコーダー

**FCPEのlogits→f0変換** (Python実装より):
```python
def fcpe_logits_to_f0(logits, threshold=0.006):
    f0_min = 32.7
    f0_max = 1975.5
    out_dims = 360

    # Cent table: 32.7Hz〜1975.5Hzを360ビンで表現
    f0_mel_min = 1200 * np.log2(f0_min / 10.0)  # ~2051 cents
    f0_mel_max = 1200 * np.log2(f0_max / 10.0)  # ~9151 cents
    cent_table = np.linspace(f0_mel_min, f0_mel_max, out_dims)

    for t in range(logits.shape[0]):
        if max_conf[t] <= threshold:
            f0_hz[t] = 0.0  # Unvoiced
            continue

        # local_argmax: 9ビンの加重平均
        idx = max_idx[t]
        local_indices = np.arange(idx - 4, idx + 5)
        local_indices = np.clip(local_indices, 0, out_dims - 1)

        local_logits = logits[t, local_indices]
        local_cents = cent_table[local_indices]

        cents = np.sum(local_cents * local_logits) / np.sum(local_logits)
        f0_hz[t] = 10.0 * (2.0 ** (cents / 1200.0))
```

**Swift実装**:
```swift
struct FCPEDecoder {
    static let f0Min: Double = 32.7
    static let f0Max: Double = 1975.5
    static let outDims: Int = 360
    static let threshold: Float = 0.006

    private let centTable: [Float]  // Pre-computed

    func decode(logits: [[Float]]) -> [Float] {
        // 上記Pythonロジックと同等
    }
}
```

### 課題3: CoreMLモデル統合

**モデルファイル**: `fcpe_core_fp32.mlpackage`
- 入力: `mel_spectrogram` **(1, 128, T)** Float32 ← 注意: (batch, mels, time)
- 出力: `f0_logits` **(1, T, 360)** Float32

**形状変換の注意**:
```
Mel計算結果: (128, T) → CoreML入力: (1, 128, T)  // バッチ次元追加のみ
CoreML出力: (1, T, 360) → デコード入力: (T, 360)  // バッチ次元除去
```

**リソース配置**:
```
VocalisStudio/Resources/
└── Models/
    ├── fcpe_core_fp32.mlpackage
    └── mel_filterbank_16k_128.bin  # 事前計算Melフィルターバンク
```

## 実装設計

### 拡張可能なアーキテクチャ

将来的に他のMLモデル（CREPE、RMVPE等）を追加する可能性を考慮し、拡張可能な設計を採用します。

```
┌─────────────────────────────────────────────────────────────┐
│                  PitchDetectionStrategy                      │
│                    (既存プロトコル)                           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────┴───────┐    ┌───────┴───────┐    ┌───────┴───────┐
│  YINStrategy  │    │ PYINStrategy  │    │NeuralPitch    │
│  (生波形処理)  │    │ (生波形+HMM)  │    │Strategy       │
└───────────────┘    └───────────────┘    │(MLモデル共通) │
                                          └───────┬───────┘
                                                  │ uses
                              ┌───────────────────┼───────────────────┐
                              │                   │                   │
                    ┌─────────┴─────────┐ ┌───────┴───────┐ ┌─────────┴─────────┐
                    │ AudioPreprocessor │ │   MLModel     │ │   PitchDecoder    │
                    │    (Protocol)     │ │   (CoreML)    │ │    (Protocol)     │
                    └─────────┬─────────┘ └───────────────┘ └─────────┬─────────┘
                              │                                       │
        ┌─────────────────────┼─────────────────────┐                │
        │                     │                     │                │
┌───────┴───────┐    ┌───────┴───────┐    ┌───────┴───────┐        │
│ FCPEPre       │    │ CREPEPre      │    │ RMVPEPre      │        │
│ processor     │    │ processor     │    │ processor     │        │
│ (Mel 16k/128) │    │ (Raw 16k)     │    │ (Mel 16k/80)  │        │
└───────────────┘    └───────────────┘    └───────────────┘        │
                                                                    │
                              ┌─────────────────────────────────────┤
                              │                                     │
                    ┌─────────┴─────────┐              ┌────────────┴────────────┐
                    │   FCPEDecoder     │              │   CREPEDecoder          │
                    │   (local_argmax)  │              │   (weighted_argmax)     │
                    └───────────────────┘              └─────────────────────────┘
```

### コアプロトコル定義

```swift
// MARK: - 前処理プロトコル
/// 音声波形をMLモデル入力に変換する
public protocol AudioPreprocessor {
    /// モデルが期待するサンプルレート
    var targetSampleRate: Double { get }

    /// 前処理パラメータの識別子（キャッシュキー用）
    var configurationId: String { get }

    /// 音声波形を前処理してMLMultiArrayに変換
    /// - Parameters:
    ///   - samples: 入力波形
    ///   - sampleRate: 入力サンプルレート
    /// - Returns: MLModel入力用のMultiArray
    func preprocess(samples: [Float], sampleRate: Double) throws -> MLMultiArray
}

// MARK: - デコーダープロトコル
/// MLモデル出力をF0周波数に変換する
public protocol PitchDecoder {
    /// モデル出力をF0 Hz配列にデコード
    /// - Parameter output: MLModel出力
    /// - Returns: F0周波数配列（無声区間は0.0）
    func decode(output: MLMultiArray) throws -> [Float]

    /// フレームあたりのサンプル数（hop_length相当）
    var samplesPerFrame: Int { get }
}

// MARK: - モデル設定
/// ニューラルピッチ検出モデルの設定
public struct NeuralModelConfiguration {
    public let name: String
    public let modelFileName: String
    public let modelFileExtension: String
    public let requiresOctaveCorrection: Bool

    public static let fcpe = NeuralModelConfiguration(
        name: "FCPE",
        modelFileName: "fcpe_core_fp32",
        modelFileExtension: "mlpackage",
        requiresOctaveCorrection: false
    )

    // 将来の拡張例
    // public static let crepe = NeuralModelConfiguration(...)
    // public static let rmvpe = NeuralModelConfiguration(...)
}
```

### 汎用NeuralPitchStrategy

```swift
import Foundation
import CoreML
import VocalisDomain

/// 複数のMLベースピッチ検出モデルに対応する汎用Strategy
public final class NeuralPitchStrategy: PitchDetectionStrategy {

    // MARK: - Properties

    public let name: String
    public let requiresOctaveCorrection: Bool

    private let preprocessor: AudioPreprocessor
    private let decoder: PitchDecoder
    private let model: MLModel
    private let configuration: NeuralModelConfiguration

    // MARK: - Initialization

    public init(
        configuration: NeuralModelConfiguration,
        preprocessor: AudioPreprocessor,
        decoder: PitchDecoder
    ) throws {
        self.configuration = configuration
        self.name = configuration.name
        self.requiresOctaveCorrection = configuration.requiresOctaveCorrection
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.model = try Self.loadModel(configuration: configuration)
    }

    private static func loadModel(configuration: NeuralModelConfiguration) throws -> MLModel {
        guard let modelURL = Bundle.main.url(
            forResource: configuration.modelFileName,
            withExtension: configuration.modelFileExtension
        ) else {
            throw NeuralPitchError.modelNotFound(configuration.modelFileName)
        }

        let compiledURL = try MLModel.compileModel(at: modelURL)
        return try MLModel(contentsOf: compiledURL)
    }

    // MARK: - PitchDetectionStrategy

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        do {
            // 1. 前処理（リサンプリング + モデル固有変換）
            let input = try preprocessor.preprocess(samples: samples, sampleRate: sampleRate)

            // 2. CoreML推論
            let output = try runInference(input: input)

            // 3. デコード
            let f0Hz = try decoder.decode(output: output)

            // 4. PitchFrameに変換
            return buildPitchFrames(
                f0Hz: f0Hz,
                sampleRate: preprocessor.targetSampleRate
            )
        } catch {
            print("[\(name)] Error: \(error)")
            return []
        }
    }

    private func runInference(input: MLMultiArray) throws -> MLMultiArray {
        let inputFeature = try MLDictionaryFeatureProvider(
            dictionary: ["mel_spectrogram": MLFeatureValue(multiArray: input)]
        )
        let output = try model.prediction(from: inputFeature)
        guard let logits = output.featureValue(for: "f0_logits")?.multiArrayValue else {
            throw NeuralPitchError.invalidOutput
        }
        return logits
    }

    private func buildPitchFrames(f0Hz: [Float], sampleRate: Double) -> [PitchFrame] {
        let hopSeconds = Double(decoder.samplesPerFrame) / sampleRate
        return f0Hz.enumerated().map { index, frequency in
            PitchFrame(
                time: Double(index) * hopSeconds,
                frequency: Double(frequency),
                confidence: frequency > 0 ? 1.0 : 0.0
            )
        }
    }
}

// MARK: - Errors

public enum NeuralPitchError: Error {
    case modelNotFound(String)
    case preprocessingFailed(String)
    case invalidOutput
    case decodingFailed(String)
}
```

### FCPE固有実装

```swift
// MARK: - FCPE前処理

public final class FCPEPreprocessor: AudioPreprocessor {

    // パラメータ（torchfcpe互換）
    public let targetSampleRate: Double = 16000.0
    public var configurationId: String { "fcpe_mel_16k_128" }

    private static let nMels = 128
    private static let nFFT = 1024
    private static let winSize = 1024
    private static let hopLength = 160
    private static let fmin: Float = 0
    private static let fmax: Float = 8000
    private static let clipVal: Float = 1e-5

    private let melFilterbank: [[Float]]  // 128 x 513

    public init() throws {
        self.melFilterbank = try Self.loadMelFilterbank()
    }

    private static func loadMelFilterbank() throws -> [[Float]] {
        guard let url = Bundle.main.url(forResource: "mel_filterbank_16k_128", withExtension: "bin") else {
            throw NeuralPitchError.preprocessingFailed("Mel filterbank not found")
        }
        // バイナリからロード (128 x 513)
        let data = try Data(contentsOf: url)
        // ... 変換処理
    }

    public func preprocess(samples: [Float], sampleRate: Double) throws -> MLMultiArray {
        // 1. リサンプリング
        let resampled = resample(samples, from: sampleRate, to: targetSampleRate)

        // 2. torchfcpe互換パディング
        let padLeft = (Self.winSize - Self.hopLength) / 2
        let padRight = max((Self.winSize - Self.hopLength + 1) / 2,
                          Self.winSize - resampled.count - padLeft)
        let padded = reflectPad(resampled, left: padLeft, right: padRight)

        // 3. STFT (center=False, Hann window)
        // 4. Magnitude: sqrt(real² + imag² + 1e-9)
        // 5. Mel変換: melFilterbank × magnitude
        // 6. Log圧縮: log(clamp(x, min=clipVal))

        // 7. MLMultiArray形状: (1, 128, T)
        return melMultiArray
    }
}

// MARK: - FCPEデコーダー

public final class FCPEDecoder: PitchDecoder {

    public let samplesPerFrame: Int = 160  // hop_length

    private static let f0Min: Double = 32.7
    private static let f0Max: Double = 1975.5
    private static let outDims: Int = 360
    private static let threshold: Float = 0.006

    private let centTable: [Float]  // 事前計算

    public init() {
        // cent table: 32.7Hz〜1975.5Hzを360ビンで表現
        let f0MelMin = 1200 * log2(Self.f0Min / 10.0)
        let f0MelMax = 1200 * log2(Self.f0Max / 10.0)
        self.centTable = (0..<Self.outDims).map { i in
            Float(f0MelMin + (f0MelMax - f0MelMin) * Double(i) / Double(Self.outDims - 1))
        }
    }

    public func decode(output: MLMultiArray) throws -> [Float] {
        // logits shape: (1, T, 360)
        // local_argmax: 9ビンの加重平均
        // 無声判定: max_conf <= threshold → 0.0
        // ...
    }
}
```

### Factoryパターン

```swift
public enum NeuralPitchStrategyFactory {

    public static func create(for type: NeuralModelType) throws -> NeuralPitchStrategy {
        switch type {
        case .fcpe:
            return try NeuralPitchStrategy(
                configuration: .fcpe,
                preprocessor: try FCPEPreprocessor(),
                decoder: FCPEDecoder()
            )
        // 将来の拡張
        // case .crepe:
        //     return try NeuralPitchStrategy(
        //         configuration: .crepe,
        //         preprocessor: CREPEPreprocessor(),
        //         decoder: CREPEDecoder()
        //     )
        }
    }
}

public enum NeuralModelType {
    case fcpe
    // case crepe
    // case rmvpe
}
```

### PitchStrategyFactory統合

```swift
enum PitchStrategyFactory {
    static func createStrategy(for algorithm: PitchDetectionAlgorithm) -> PitchDetectionStrategy {
        switch algorithm {
        case .yin:
            return YINStrategy()
        case .pyinDefault:
            return PYINStrategy(configuration: .default, name: "pYIN")
        // ... 既存のpYIN variants

        // ニューラルモデル
        case .fcpe:
            do {
                return try NeuralPitchStrategyFactory.create(for: .fcpe)
            } catch {
                print("[Factory] FCPE creation failed, fallback to pYIN: \(error)")
                return PYINStrategy(configuration: .default, name: "pYIN")
            }
        }
    }
}


### PitchDetectionAlgorithm更新

```swift
public enum PitchDetectionAlgorithm: String, Codable, CaseIterable {
    case yin
    case pyinDefault
    case pyinHighDetection
    case pyinBalanced
    case pyinAggressive

    // 新規追加
    case fcpe

    public var displayName: String {
        switch self {
        // ...
        case .fcpe: return "FCPE (Neural)"
        }
    }
}
```

## 実装フェーズ

### Phase 1: 基盤実装 (推定: 2-3時間)

1. `MelSpectrogramTransformer.swift` - Mel変換
2. `FCPEDecoder.swift` - logits→f0デコーダー
3. ユニットテスト（Pythonの結果と比較）

### Phase 2: Strategy統合 (推定: 1-2時間)

1. `FCPEStrategy.swift` - Strategy実装
2. `PitchDetectionAlgorithm`にcase追加
3. `PitchStrategyFactory`更新
4. CoreMLモデルをResourcesに追加

### Phase 3: 精度検証 (推定: 1時間)

1. vocadito_1テストデータでYIN/pYIN/FCPEを比較
2. RPA/GPE/FPE指標で評価
3. 処理時間比較

### Phase 4: UI統合 (推定: 30分)

1. 設定画面にFCPEオプション追加
2. ローカライズ（日本語/英語）

## テスト戦略

### Unit Tests

```swift
// MelSpectrogramTransformerTests.swift
func testTransform_withSinusoid_producesExpectedMelBins() {
    // 440Hz正弦波のMelスペクトログラムを検証
}

// FCPEDecoderTests.swift
func testDecode_withKnownLogits_matchesPythonOutput() {
    // Pythonで計算したlogitsをデコードして一致確認
}

// FCPEStrategyTests.swift
func testDetectPitch_with440HzSine_returnsCorrectFrequency() {
    let strategy = FCPEStrategy()
    let samples = generateSinusoid(frequency: 440.0, duration: 1.0)
    let frames = strategy.detectPitch(samples: samples, sampleRate: 44100)

    XCTAssertFalse(frames.isEmpty)
    XCTAssertEqual(frames[0].frequency, 440.0, accuracy: 5.0)
}
```

### Integration Tests

```swift
func testFCPE_withVocadito1_achievesExpectedRPA() async {
    // vocadito_1_f0.csv (ground truth) と比較
    // 期待: RPA > 95%
}
```

## リスクと軽減策

| リスク | 影響 | 軽減策 |
|--------|------|--------|
| Mel変換精度の差異 | 検出精度低下 | Pythonと同一パラメータ使用、クロス検証 |
| CoreMLモデルロード失敗 | 機能不全 | Fallback to pYIN、エラーハンドリング |
| 処理時間増加 | UX劣化 | バックグラウンド処理、プログレス表示 |
| モデルサイズ増加 | アプリサイズ増 | モデル圧縮（量子化）検討 |

## 参考資料

- [torchfcpe GitHub](https://github.com/CNChTu/FCPE)
- FCPE論文: "Fast Context-based Pitch Estimation"
- `pitch_detection_poc/fcpe_poc/` - PoC実装とテスト結果
- `testdata_metrics_fp32_v2.json` - PyTorch vs CoreML 100%一致確認

## 次のステップ

### ✅ 完了済み
1. Mel Spectrogram PoC検証 - 99.9%の精度達成
2. Melフィルターバンク行列のエクスポート

### 🔜 次の作業
1. Phase 1: `MelSpectrogramTransformer.swift`の実装
   - `mel_filterbank_16k_128.bin`をロード
   - 反射パディング実装
   - vDSP使用のSTFT実装
   - Mel変換とログ圧縮

2. Phase 2: `FCPEDecoder.swift`の実装
   - centTable事前計算
   - local_argmaxデコード

3. Phase 3: `FCPEStrategy.swift`の統合

### 📁 関連ファイル
- `pitch_detection_poc/fcpe_poc/mel_spectrogram_poc.py` - PoC実装
- `pitch_detection_poc/fcpe_poc/mel_filterbank_16k_128.bin` - Melフィルターバンク行列
- `pitch_detection_poc/fcpe_poc/fcpe_core_fp32.mlpackage` - CoreMLモデル
