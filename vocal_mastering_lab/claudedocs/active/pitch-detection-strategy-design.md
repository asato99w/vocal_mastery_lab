# ピッチ検出アルゴリズム Strategy パターン設計

## 概要

VocalisStudioのピッチ検出アルゴリズムをStrategy パターンで切り替え可能にする設計。
現在のYINアルゴリズムに加え、pYINなど複数のアルゴリズムを簡単に導入・比較できる構造を目指す。

## 現状分析

### 現在の実装 (`AudioFileAnalyzer.swift`)

```
AudioFileAnalyzer
├── YINアルゴリズム直接実装 (detectPitchUsingYIN)
│   ├── パラメータ: bufferSize=2048, threshold=0.25
│   └── 周波数範囲: 80Hz〜1200Hz
├── スペクトログラム解析 (FFT)
└── レイテンシ補正 (pitchDetectionLatencyOffset)
```

### 現在のオクターブ補正 (`OctaveCorrectionService.swift`)

- **配置**: Domain層（VocalisDomain パッケージ）
- **役割**: ターゲットノート（スケール音）に基づいてオクターブエラーを補正
- **前提**: YINはオクターブエラー（倍音検出）が起きやすいため必須

### POCでの評価結果

| アルゴリズム | RPA勝利数 | GPE | FPE | 特徴 |
|------------|----------|-----|-----|------|
| YIN-default | 2/10 | 11.5% | 14.7¢ | 高検出率、オクターブエラーあり |
| pYIN (default) | 0/10 | **1.4%** | **8.8¢** | 低検出率、高精度 |
| pYIN-balanced | **5/10** | 2.8% | 9.5¢ | バランス型、推奨 |
| pYIN-aggressive | 3/10 | 4.2% | 10.1¢ | 高検出率重視 |

**重要な発見**: pYINはHMM/Viterbiデコーディングにより時間的連続性を考慮するため、オクターブエラー（GPE）が大幅に低い。

## 設計

### Protocol定義

```swift
/// ピッチ検出アルゴリズムの共通インターフェース
public protocol PitchDetectionStrategy {
    /// アルゴリズム識別名
    var name: String { get }

    /// オクターブ補正が必要かどうか
    /// - YIN: true (オクターブエラーが起きやすい)
    /// - pYIN: false (HMMで時間的連続性を考慮済み)
    var requiresOctaveCorrection: Bool { get }

    /// ピッチ検出を実行
    /// - Parameters:
    ///   - samples: 音声サンプル配列
    ///   - sampleRate: サンプリングレート
    /// - Returns: 検出されたピッチフレームの配列
    func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame]
}

/// ピッチ検出結果の1フレーム
public struct PitchFrame {
    public let timestamp: Double      // 時刻（秒）
    public let frequency: Float?      // 検出周波数（無音時はnil）
    public let confidence: Float      // 信頼度 (0.0〜1.0)
    public let amplitude: Float       // 正規化振幅 (0.0〜1.0)
}
```

### Strategy実装

#### YINStrategy

```swift
public final class YINStrategy: PitchDetectionStrategy {
    public let name = "YIN"
    public let requiresOctaveCorrection = true  // オクターブ補正必要

    private let configuration: Configuration

    public struct Configuration {
        let bufferSize: Int
        let hopSize: Int
        let threshold: Float
        let minFrequency: Double
        let maxFrequency: Double

        public static let `default` = Configuration(
            bufferSize: 2048,
            hopSize: 2205,  // 50ms at 44100Hz
            threshold: 0.25,
            minFrequency: 80.0,
            maxFrequency: 1200.0
        )
    }

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        // 現在のdetectPitchUsingYIN実装を移植
    }
}
```

#### pYINStrategy

```swift
public final class PYINStrategy: PitchDetectionStrategy {
    public let name: String
    public let requiresOctaveCorrection = false  // HMMで補正済み

    private let configuration: Configuration

    public struct Configuration {
        // PitchDetectionPOCのPYINDetector.Configurationと同等
        let bufferSize: Int
        let hopSize: Int
        let minFrequency: Double
        let maxFrequency: Double
        let silenceThreshold: Float
        let thresholdDistribution: [Float]
        let hmmTransitionWidth: Float
        let voicedBias: Float

        public static let `default` = Configuration(...)
        public static let balanced = Configuration(...)
        public static let aggressive = Configuration(...)
    }

    public init(configuration: Configuration = .default, name: String = "pYIN") {
        self.configuration = configuration
        self.name = name
    }

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        // PitchDetectionPOCのPYINDetector実装を移植
    }
}
```

### AudioFileAnalyzer修正

```swift
public class AudioFileAnalyzer: AudioFileAnalyzerProtocol {
    // Strategy injection
    private let pitchStrategy: PitchDetectionStrategy
    private let octaveCorrectionService: OctaveCorrectionService

    public init(
        pitchStrategy: PitchDetectionStrategy = YINStrategy(),
        octaveCorrectionService: OctaveCorrectionService = OctaveCorrectionService()
    ) {
        self.pitchStrategy = pitchStrategy
        self.octaveCorrectionService = octaveCorrectionService
    }

    private func analyzePitch(...) async throws -> PitchAnalysisData {
        // 1. Strategyでピッチ検出
        let frames = pitchStrategy.detectPitch(samples: samples, sampleRate: sampleRate)

        // 2. PitchAnalysisDataに変換
        var pitchData = convertToPitchAnalysisData(frames)

        // 3. 必要に応じてオクターブ補正
        if pitchStrategy.requiresOctaveCorrection {
            pitchData = octaveCorrectionService.applyCorrection(
                to: pitchData,
                segments: noteSegments  // スケール設定から取得
            )
        }

        return pitchData
    }
}
```

### レイヤー配置

```
VocalisDomain (Domain層)
├── Protocols/
│   └── PitchDetectionStrategy.swift  [NEW]
├── ValueObjects/
│   └── PitchFrame.swift  [NEW]
└── Services/
    └── OctaveCorrectionService.swift  [既存]

VocalisStudio (Infrastructure層)
├── Services/
│   └── Analysis/
│       ├── AudioFileAnalyzer.swift  [修正]
│       ├── Strategies/  [NEW]
│       │   ├── YINStrategy.swift
│       │   └── PYINStrategy.swift
│       └── ...
```

## 実装フェーズ

### Phase 1: Protocol定義とYIN移行

1. `PitchDetectionStrategy` protocolをDomain層に作成
2. `PitchFrame` value objectを作成
3. `YINStrategy`を作成し、既存ロジックを移植
4. `AudioFileAnalyzer`をStrategy利用に修正
5. 単体テスト作成・実行

**確認ポイント**: 既存の動作が変わらないこと

### Phase 2: pYIN統合

1. `PYINStrategy`を作成（POCから移植）
2. Configuration presetsを追加（default, balanced, aggressive）
3. `requiresOctaveCorrection = false`の動作確認
4. 統合テスト作成・実行

**確認ポイント**: pYINでオクターブ補正がスキップされること

### Phase 3: 設定UI統合（オプション）

1. 設定画面にアルゴリズム選択を追加
2. UserDefaultsに選択を保存
3. DependencyContainerで切り替え

## テスト戦略

### Unit Tests

```swift
// YINStrategyTests.swift
func testDetectPitch_withSinusoid_returnsCorrectFrequency() {
    let strategy = YINStrategy()
    let samples = generateSinusoid(frequency: 440.0, duration: 1.0)
    let frames = strategy.detectPitch(samples: samples, sampleRate: 44100)

    XCTAssertFalse(frames.isEmpty)
    XCTAssertEqual(frames[0].frequency!, 440.0, accuracy: 5.0)
}

func testRequiresOctaveCorrection_forYIN_returnsTrue() {
    let strategy = YINStrategy()
    XCTAssertTrue(strategy.requiresOctaveCorrection)
}

// PYINStrategyTests.swift
func testRequiresOctaveCorrection_forPYIN_returnsFalse() {
    let strategy = PYINStrategy()
    XCTAssertFalse(strategy.requiresOctaveCorrection)
}
```

### Integration Tests

```swift
// AudioFileAnalyzerIntegrationTests.swift
func testAnalyze_withYINStrategy_appliesOctaveCorrection() async throws {
    let analyzer = AudioFileAnalyzer(pitchStrategy: YINStrategy())
    // ...オクターブ補正が適用されることを確認
}

func testAnalyze_withPYINStrategy_skipsOctaveCorrection() async throws {
    let analyzer = AudioFileAnalyzer(pitchStrategy: PYINStrategy())
    // ...オクターブ補正がスキップされることを確認
}
```

## 考慮事項

### パフォーマンス

- pYINはYINより計算コストが高い（HMM/Viterbi処理）
- リアルタイム用途ではYINを維持、ファイル解析ではpYINを検討
- 必要に応じてバックグラウンド処理や進捗表示を調整

### 後方互換性

- デフォルトは現在のYINを維持
- 既存の録音データは変更なしで再生可能
- 設定変更は新規解析時のみ適用

### 将来の拡張

- 他のアルゴリズム追加（CREPE, SPICE等）
- アルゴリズム自動選択（音声特性に基づく）
- ハイブリッドアプローチ（YINでリアルタイム、pYINで後処理）

## 参照

- [YIN論文](https://www.sciencedirect.com/science/article/pii/S1057739802000124): "YIN, a fundamental frequency estimator for speech and music" (de Cheveigné & Kawahara, 2002)
- [pYIN論文](https://ieeexplore.ieee.org/document/6853678): "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions" (Mauch & Dixon, 2014)
- PitchDetectionPOC: `/Users/asatokazu/Documents/dev/mine/music/vocalis-studio/PitchDetectionPOC/`
