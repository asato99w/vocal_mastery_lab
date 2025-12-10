# AVAudioPlayerNode 実装プラン

**作成日**: 2025-12-06
**目的**: AVAudioUnitSamplerの遅延問題を解決するため、AVAudioPlayerNodeベースのスケールプレイヤーを実装する

## 1. 背景

### 1.1 現在の問題
- AVAudioUnitSamplerは内部処理により約100msの遅延が発生
- 固定オフセット補正（80ms）で改善したが、まだ約30msの残差あり
- 遅延値が環境（シミュレータ/実機）や音色によって変動する可能性

### 1.2 PlayerNodeの利点
- `scheduleBuffer(at:)`で正確な再生時刻を指定可能
- 遅延が`outputLatency`のみで予測可能
- タイムスタンプ記録と再生が完全に同期可能

## 2. アーキテクチャ設計

### 2.1 設計方針: 既存の音色選択と統合

**重要な発見**: 既存の `ScaleSoundType` に `sineWave` が存在し、`midiProgram` が `nil` を返す設計になっている。

```swift
// 既存コード: ScaleSoundType.swift
public enum ScaleSoundType {
    case acousticGrandPiano  // midiProgram = 0
    case electricPiano       // midiProgram = 4
    case marimba             // midiProgram = 12
    case sineWave            // midiProgram = nil ← これを活用！
    // ...
}
```

**統合戦略**: 新しいenumを追加する代わりに、`midiProgram` の有無でエンジンを自動選択

- **`midiProgram != nil`** → AVAudioUnitSampler（SF2）を使用
- **`midiProgram == nil`** → AVAudioPlayerNode（PCMバッファ）を使用

### 2.2 アーキテクチャ図

```
┌─────────────────────────────────────────────────────────┐
│                     ScaleSoundType                       │
│  (既存enum - midiProgramプロパティを活用)                  │
│                                                         │
│  ┌─────────────────────┐   ┌─────────────────────┐      │
│  │ midiProgram != nil  │   │ midiProgram == nil  │      │
│  │ (Piano, Marimba...) │   │ (SineWave, Synth..) │      │
│  └──────────┬──────────┘   └──────────┬──────────┘      │
└─────────────┼──────────────────────────┼────────────────┘
              │                          │
              ▼                          ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│  AVAudioEngineScalePlayer│   │ AVAudioPlayerNodeScale  │
│  (現在の実装)             │   │ Player (新規実装)        │
│                         │   │                         │
│  - AVAudioUnitSampler   │   │  - AVAudioPlayerNode    │
│  - SF2サウンドバンク      │   │  - PCMバッファ生成       │
│  - 80ms固定オフセット補正 │   │  - outputLatencyのみ補正 │
└─────────────────────────┘   └─────────────────────────┘
              │                          │
              └────────────┬─────────────┘
                           │
                    ┌──────▼──────┐
                    │ HybridScale │
                    │ Player      │
                    │             │
                    │ soundTypeに │
                    │ 基づいて    │
                    │ 内部で切替  │
                    └─────────────┘
```

### 2.3 実装選択肢

#### 選択肢A: HybridScalePlayer（推奨）

単一のScalePlayerが内部で両方のエンジンを持ち、音色設定に応じて切り替える。

```swift
// Infrastructure/Audio/HybridScalePlayer.swift
public class HybridScalePlayer: ScalePlayerProtocol {
    private let samplerPlayer: AVAudioEngineScalePlayer
    private let synthPlayer: AVAudioPlayerNodeScalePlayer
    private let settingsRepository: AudioSettingsRepositoryProtocol

    private var activePlayer: ScalePlayerProtocol {
        let settings = settingsRepository.get()
        if settings.scaleSoundType.midiProgram != nil {
            return samplerPlayer
        } else {
            return synthPlayer
        }
    }

    // ScalePlayerProtocolのメソッドはactivePlayerに委譲
    func play() async throws {
        try await activePlayer.play()
    }

    func stop() {
        activePlayer.stop()
    }
    // ...
}
```

**利点**:
- UIの変更不要（既存の音色選択画面がそのまま使える）
- DependencyContainerの変更が最小限
- ユーザーは音色を選ぶだけでエンジンが自動的に最適化される

#### 選択肢B: DependencyContainerでの切り替え

音色設定を読み取り、適切な実装を返す。

```swift
// DependencyContainer.swift
func makeScalePlayer() -> ScalePlayerProtocol {
    let settings = audioSettingsRepository.get()

    if settings.scaleSoundType.midiProgram != nil {
        // SF2音色 → Sampler
        return AVAudioEngineScalePlayer(
            settingsRepository: audioSettingsRepository
        )
    } else {
        // 合成音 → PlayerNode
        return AVAudioPlayerNodeScalePlayer(
            settingsRepository: audioSettingsRepository
        )
    }
}
```

**注意**: この方法では音色変更時にScalePlayerの再生成が必要

### 2.4 ScaleSoundType拡張（将来）

将来的に合成音の種類を増やす場合:

```swift
public enum ScaleSoundType: String, Codable, CaseIterable, Hashable {
    // SF2ベース（Sampler使用）
    case acousticGrandPiano     // midiProgram = 0
    case electricPiano          // midiProgram = 4
    case marimba                // midiProgram = 12
    // ...

    // 合成音ベース（PlayerNode使用）
    case sineWave               // midiProgram = nil
    case synthPiano             // midiProgram = nil（将来追加）
    case synthBell              // midiProgram = nil（将来追加）

    /// General MIDI Program Number (nil for synthesized sounds)
    public var midiProgram: UInt8? {
        switch self {
        case .acousticGrandPiano: return 0
        case .electricPiano: return 4
        case .marimba: return 12
        // SF2音色は番号を返す

        case .sineWave, .synthPiano, .synthBell:
            return nil  // 合成音はnilを返す
        }
    }

    /// Whether this sound type uses synthesis (PlayerNode) or sampler (SF2)
    public var usesSynthesis: Bool {
        return midiProgram == nil
    }
}
```

### 2.5 UI変更不要

既存の音色選択UIがそのまま機能:

```swift
// AudioOutputSettingsView.swift（変更不要）
Picker("sound.type", selection: $viewModel.scaleSoundType) {
    ForEach(ScaleSoundType.allCases, id: \.self) { type in
        Text(type.displayNameKey.localized)
            .tag(type)
    }
}
```

ユーザーが「Sine Wave」を選択すると、自動的にPlayerNodeベースの再生に切り替わる。

### 2.4 新規クラス

```swift
/// PlayerNodeベースのスケールプレイヤー
/// 正確なタイミング制御が可能
public class AVAudioPlayerNodeScalePlayer: ScalePlayerProtocol {
    private let engine: AVAudioEngine
    private let playerNode: AVAudioPlayerNode
    private let bufferCache: ScaleBufferCache
    // ...
}

/// 音声バッファのキャッシュ管理
class ScaleBufferCache {
    private var cache: [CacheKey: AVAudioPCMBuffer] = [:]

    struct CacheKey: Hashable {
        let midiNote: UInt8
        let soundType: ScaleSoundType
        let duration: TimeInterval
    }
}

/// 音声バッファ生成器
protocol AudioBufferGenerator {
    func generateBuffer(
        for note: MIDINote,
        duration: TimeInterval,
        sampleRate: Double
    ) -> AVAudioPCMBuffer
}

/// サイン波生成器
class SineWaveGenerator: AudioBufferGenerator { }

/// ピアノ風合成音生成器（複数倍音 + ADSR）
class PianoSynthGenerator: AudioBufferGenerator { }
```

## 3. 実装フェーズ

### Phase 1: 基本実装（サイン波のみ）

**目標**: 最小限の実装で正確なタイミングを検証

**実装内容**:
1. `SineWaveGenerator`クラスの実装
2. `AVAudioPlayerNodeScalePlayer`の基本実装
3. 単音再生のテスト

**コード例**:
```swift
class SineWaveGenerator: AudioBufferGenerator {
    func generateBuffer(
        for note: MIDINote,
        duration: TimeInterval,
        sampleRate: Double
    ) -> AVAudioPCMBuffer {
        let frameCount = AVAudioFrameCount(duration * sampleRate)
        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        )!
        buffer.frameLength = frameCount

        let frequency = note.frequency
        let data = buffer.floatChannelData![0]

        for i in 0..<Int(frameCount) {
            let t = Double(i) / sampleRate
            // 基本サイン波
            var sample = sin(2.0 * .pi * frequency * t)
            // ADSRエンベロープ適用
            sample *= envelope(t: t, duration: duration)
            data[i] = Float(sample)
        }

        return buffer
    }

    private func envelope(t: Double, duration: Double) -> Double {
        let attack = 0.01   // 10ms
        let decay = 0.05    // 50ms
        let sustain = 0.7   // 70%
        let release = 0.1   // 100ms

        let releaseStart = duration - release

        if t < attack {
            return t / attack
        } else if t < attack + decay {
            let decayProgress = (t - attack) / decay
            return 1.0 - (1.0 - sustain) * decayProgress
        } else if t < releaseStart {
            return sustain
        } else {
            let releaseProgress = (t - releaseStart) / release
            return sustain * (1.0 - releaseProgress)
        }
    }
}
```

**成果物**:
- サイン波でのスケール再生
- タイミング精度の検証結果

### Phase 2: タイミング精度検証

**目標**: PlayerNodeのタイミング精度を計測

**検証内容**:
1. `scheduleBuffer(at:)`での再生時刻精度
2. outputLatencyとの関係
3. Samplerとの比較

**計測方法**:
```swift
// 再生スケジュール時刻を記録
let scheduleTime = AVAudioTime(
    hostTime: mach_absolute_time() + hostTimeOffset
)
playerNode.scheduleBuffer(buffer, at: scheduleTime)

// 実際の再生時刻をTap検出で計測
// TimingOffset = scheduleTime - actualPlayTime
```

**成果物**:
- タイミング精度の計測結果
- Samplerとの比較レポート

### Phase 3: 音色拡張

**目標**: ピアノ風の合成音を実装

**実装内容**:
1. 倍音合成の実装
2. ADSR エンベロープの調整
3. 音色パラメータの設定

**コード例**:
```swift
class PianoSynthGenerator: AudioBufferGenerator {
    // 倍音構成（ピアノ風）
    private let harmonics: [(partial: Int, amplitude: Double)] = [
        (1, 1.0),    // 基音
        (2, 0.5),    // 第2倍音
        (3, 0.25),   // 第3倍音
        (4, 0.125),  // 第4倍音
        (5, 0.0625), // 第5倍音
    ]

    func generateBuffer(
        for note: MIDINote,
        duration: TimeInterval,
        sampleRate: Double
    ) -> AVAudioPCMBuffer {
        // ... バッファ生成

        for i in 0..<Int(frameCount) {
            let t = Double(i) / sampleRate
            var sample = 0.0

            // 倍音合成
            for (partial, amplitude) in harmonics {
                let freq = frequency * Double(partial)
                sample += sin(2.0 * .pi * freq * t) * amplitude
            }

            // 正規化
            sample /= harmonics.map { $0.amplitude }.reduce(0, +)

            // ピアノ風エンベロープ（急速なアタック、長い減衰）
            sample *= pianoEnvelope(t: t, duration: duration)

            data[i] = Float(sample)
        }

        return buffer
    }

    private func pianoEnvelope(t: Double, duration: Double) -> Double {
        let attack = 0.005  // 5ms（ピアノは急速）
        let decay = duration * 0.8  // 長い減衰

        if t < attack {
            return t / attack
        } else {
            let decayProgress = (t - attack) / decay
            return exp(-3.0 * decayProgress)  // 指数減衰
        }
    }
}
```

**成果物**:
- ピアノ風合成音
- 音色切り替え機能

### Phase 4: キャッシュ最適化

**目標**: メモリ効率とパフォーマンスの最適化

**実装内容**:
1. バッファキャッシュの実装
2. 事前生成戦略
3. メモリ管理

**コード例**:
```swift
class ScaleBufferCache {
    private var cache: [CacheKey: AVAudioPCMBuffer] = [:]
    private let maxCacheSize: Int = 50  // 最大50バッファ
    private let queue = DispatchQueue(label: "buffer-cache")

    /// スケール用のバッファを事前生成
    func preloadScale(
        notes: [MIDINote],
        duration: TimeInterval,
        soundType: ScaleSoundType,
        generator: AudioBufferGenerator
    ) {
        queue.async {
            for note in notes {
                let key = CacheKey(
                    midiNote: note.value,
                    soundType: soundType,
                    duration: duration
                )
                if self.cache[key] == nil {
                    let buffer = generator.generateBuffer(
                        for: note,
                        duration: duration,
                        sampleRate: 44100
                    )
                    self.cache[key] = buffer
                }
            }
        }
    }

    /// キャッシュからバッファを取得（なければ生成）
    func getBuffer(
        for note: MIDINote,
        duration: TimeInterval,
        soundType: ScaleSoundType,
        generator: AudioBufferGenerator
    ) -> AVAudioPCMBuffer {
        let key = CacheKey(
            midiNote: note.value,
            soundType: soundType,
            duration: duration
        )

        if let cached = cache[key] {
            return cached
        }

        let buffer = generator.generateBuffer(
            for: note,
            duration: duration,
            sampleRate: 44100
        )
        cache[key] = buffer

        // キャッシュサイズ管理
        if cache.count > maxCacheSize {
            evictOldest()
        }

        return buffer
    }
}
```

**成果物**:
- 効率的なバッファキャッシュ
- メモリ使用量の最適化

### Phase 5: 統合とテスト

**目標**: 既存システムへの統合

**実装内容**:
1. HybridScalePlayerの実装
2. DependencyContainerでHybridScalePlayerを返すよう変更
3. 統合テスト

**HybridScalePlayer実装**:
```swift
// Infrastructure/Audio/HybridScalePlayer.swift
public class HybridScalePlayer: ScalePlayerProtocol {
    private let samplerPlayer: AVAudioEngineScalePlayer
    private let synthPlayer: AVAudioPlayerNodeScalePlayer
    private let settingsRepository: AudioSettingsRepositoryProtocol

    public init(settingsRepository: AudioSettingsRepositoryProtocol) {
        self.settingsRepository = settingsRepository
        self.samplerPlayer = AVAudioEngineScalePlayer(settingsRepository: settingsRepository)
        self.synthPlayer = AVAudioPlayerNodeScalePlayer(settingsRepository: settingsRepository)
    }

    private var activePlayer: ScalePlayerProtocol {
        let settings = settingsRepository.get()
        if settings.scaleSoundType.midiProgram != nil {
            return samplerPlayer
        } else {
            return synthPlayer
        }
    }

    // ScalePlayerProtocolの全メソッドをactivePlayerに委譲
    public func loadScale(from settings: ScaleSettings) {
        activePlayer.loadScale(from: settings)
    }

    public func play() async throws {
        try await activePlayer.play()
    }

    public func stop() {
        activePlayer.stop()
    }

    // ... 他のメソッドも同様に委譲
}
```

**DependencyContainer変更**:
```swift
// DependencyContainer.swift
func makeScalePlayer() -> ScalePlayerProtocol {
    return HybridScalePlayer(settingsRepository: audioSettingsRepository)
}
```

**UI変更不要**:
既存の音色選択UIがそのまま機能する。ユーザーが「Sine Wave」を選択すると
自動的にPlayerNodeベースの再生に切り替わる。

**ローカライズ文字列追加（音色説明の更新）**:
```
// en.lproj/Localizable.strings
"sound.sine_wave.desc" = "Pure synthesized tone with precise timing";

// ja.lproj/Localizable.strings
"sound.sine_wave.desc" = "正確なタイミングの合成音";
```

**成果物**:
- HybridScalePlayer（音色に応じてエンジン自動選択）
- 既存UIとの完全な統合
- ユーザーは音色を選ぶだけでエンジンが自動最適化

## 4. タイムスタンプ記録の改善

### 4.1 現在の問題
- Samplerでは`startNote()`からオーディオ出力まで約100msの遅延
- Tap検出または固定オフセットで補償が必要

### 4.2 PlayerNodeでの解決

```swift
class AVAudioPlayerNodeScalePlayer: ScalePlayerProtocol {
    func playNote(_ note: MIDINote, duration: TimeInterval) async throws {
        guard let startTime = recordingStartTime else { return }

        // 現在時刻を取得
        let now = Date()
        let timestamp = now.timeIntervalSince(startTime)

        // バッファを取得
        let buffer = bufferCache.getBuffer(
            for: note,
            duration: duration,
            soundType: currentSoundType,
            generator: generator
        )

        // 即座に再生開始（遅延なし）
        playerNode.scheduleBuffer(buffer, at: nil)

        // タイムスタンプを記録（outputLatency補正のみ）
        let compensatedTimestamp = timestamp + outputLatency
        let event = ScalePlaybackEvent(
            timestamp: compensatedTimestamp,
            note: note,
            eventType: .noteStart
        )
        recordedEvents.append(event)

        // ノート終了まで待機
        try await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
    }
}
```

**利点**:
- Sampler内部遅延が存在しない
- `outputLatency`のみの補正で正確なタイミング
- Tap検出が不要になりシンプル化

## 5. リスクと対策

### 5.1 音質
**リスク**: 合成音がSF2音色より劣る可能性
**対策**:
- 倍音合成で自然な音色を目指す
- ユーザーがSampler/PlayerNodeを選択可能に

### 5.2 メモリ使用量
**リスク**: バッファキャッシュによるメモリ増加
**対策**:
- キャッシュサイズ制限
- 使用頻度の低いバッファを削除
- スケール開始前に必要なバッファのみ事前生成

### 5.3 CPU使用量
**リスク**: リアルタイムバッファ生成の負荷
**対策**:
- 事前生成とキャッシュで再生時の生成を回避
- バックグラウンドスレッドでの生成

## 6. 成功基準

### 6.1 タイミング精度
- TimingOffset（ScaleBarTime - PitchTime）が±20ms以内
- シミュレータと実機で一貫した精度

### 6.2 音質
- ユーザーが違和感なくスケール練習できる音質
- 音程が明確に識別できる

### 6.3 パフォーマンス
- スケール開始時の遅延が200ms以内
- メモリ使用量が追加で10MB以内

## 7. スケジュール

| Phase | 内容 | 期間（目安） |
|-------|------|-------------|
| Phase 1 | 基本実装（サイン波） | 1日 |
| Phase 2 | タイミング精度検証 | 1日 |
| Phase 3 | 音色拡張 | 1-2日 |
| Phase 4 | キャッシュ最適化 | 1日 |
| Phase 5 | 統合とテスト | 1日 |
| **合計** | | **5-6日** |

## 8. 次のステップ

1. **Phase 1を開始**: SineWaveGeneratorとAVAudioPlayerNodeScalePlayerの基本実装
2. **タイミング計測**: 実装後にTimingOffsetを計測してSamplerと比較
3. **判断**: 精度が十分であればPhase 3以降に進む、不十分なら固定オフセット補正を継続

## 9. 代替案

### 9.1 Sampler + 100ms補正
- 現在の80msを100msに変更
- 最もシンプルな解決策
- 環境による変動リスクあり

### 9.2 ハイブリッドアプローチ
- 通常はSampler（豊富な音色）
- タイミング重視の場合はPlayerNode
- 複雑だが両方の利点を活かせる

### 9.3 SF2からPCM変換
- SF2音色をPCMバッファに事前変換
- PlayerNodeで再生
- 音質を維持しつつ正確なタイミング
- 実装が複雑

## 10. 結論

AVAudioPlayerNodeへの移行は、タイミング精度の問題を根本的に解決できる可能性がある。まずPhase 1-2で基本実装と精度検証を行い、効果を確認してから本格的な実装に進むことを推奨する。

固定オフセット補正（100ms）も引き続き有効な選択肢であり、PlayerNode実装の結果次第で最終的なアプローチを決定する。
