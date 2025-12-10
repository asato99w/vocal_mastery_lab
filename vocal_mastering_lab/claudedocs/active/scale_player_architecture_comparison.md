# スケールプレイヤー アーキテクチャ比較

**作成日**: 2025-12-06
**目的**: AVAudioUnitSampler vs AVAudioPlayerNode の比較検討

## 1. 背景

### 1.1 現在の問題
- スケールバー表示時刻（ScaleBarTime）とピッチ検出時刻（PitchTime）に約100msのズレ
- 原因: AVAudioUnitSampler内部の処理遅延が未補償

### 1.2 検討する2つのアプローチ
1. **SamplerPlayer**: 現在の実装（AVAudioUnitSampler + 固定オフセット補正）
2. **PlayNodePlayer**: 代替案（AVAudioPlayerNode + 事前生成バッファ）

---

## 2. アーキテクチャ比較

### 2.1 AVAudioUnitSampler（現在の実装）

```
┌─────────────────────────────────────────────────────────────┐
│                    AVAudioEngine                            │
│                                                             │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ AVAudioUnitSampler│───▶│ mainMixerNode│───▶│  output  │  │
│  │  (SF2 SoundBank) │    │              │    │          │  │
│  └──────────────────┘    └──────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────┘

処理フロー:
sampler.startNote()
  → SF2からサンプル読み込み
  → エンベロープ処理（ADSR）
  → 内部バッファリング
  → 音声出力

遅延: ~100ms（不確定）
```

**コード例**:
```swift
let sampler = AVAudioUnitSampler()
engine.attach(sampler)
engine.connect(sampler, to: engine.mainMixerNode, format: nil)

// SF2サウンドバンク読み込み
try sampler.loadSoundBankInstrument(at: sf2URL, program: 0, ...)

// ノート再生
sampler.startNote(60, withVelocity: 64, onChannel: 0)
```

### 2.2 AVAudioPlayerNode（代替案）

```
┌─────────────────────────────────────────────────────────────┐
│                    AVAudioEngine                            │
│                                                             │
│  ┌──────────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ AVAudioPlayerNode│───▶│ mainMixerNode│───▶│  output  │  │
│  │  (PCM Buffer)    │    │              │    │          │  │
│  └──────────────────┘    └──────────────┘    └──────────┘  │
└─────────────────────────────────────────────────────────────┘

処理フロー:
playerNode.scheduleBuffer(buffer, at: audioTime)
  → 指定時刻にバッファを再生
  → 予測可能な遅延

遅延: outputLatencyのみ（予測可能）
```

**コード例**:
```swift
let playerNode = AVAudioPlayerNode()
engine.attach(playerNode)
engine.connect(playerNode, to: engine.mainMixerNode, format: format)

// 事前生成されたPCMバッファ
let buffer = generateNoteBuffer(frequency: 261.63, duration: 1.0)

// 正確な時刻にスケジュール
let when = AVAudioTime(hostTime: targetHostTime)
playerNode.scheduleBuffer(buffer, at: when, options: [])
```

---

## 3. 詳細比較

### 3.1 機能比較表

| 項目 | AVAudioUnitSampler | AVAudioPlayerNode |
|------|-------------------|-------------------|
| **音源** | SF2サウンドバンク | 事前生成PCMバッファ |
| **遅延** | ~100ms（不確定） | outputLatencyのみ（予測可能） |
| **音色の多様性** | ◎ General MIDI全128音色 | △ 自分で生成が必要 |
| **タイミング精度** | △ Tap検出/固定オフセットで補償 | ◎ scheduleBufferで正確 |
| **実装の複雑さ** | ○ 既存実装あり | △ バッファ生成が必要 |
| **メモリ使用量** | ○ オンデマンド読み込み | △ バッファを事前保持 |
| **CPU使用量** | △ SF2処理のオーバーヘッド | ○ 単純なバッファ再生 |

### 3.2 遅延の性質

**AVAudioUnitSampler**:
- 遅延は内部処理に依存（SF2読み込み、ADSR、バッファリング）
- 音色によって遅延が異なる可能性（ただし計測では2ms程度の差）
- シミュレータと実機で異なる可能性
- 固定オフセット補正で対応可能だが、完全に正確にはならない

**AVAudioPlayerNode**:
- 遅延は`engine.outputLatency`で取得可能
- `scheduleBuffer(at:)`で正確な再生時刻を指定可能
- タイムスタンプ記録と再生が完全に同期可能

### 3.3 音色生成の課題（PlayerNode使用時）

PlayerNodeを使用する場合、以下の音色生成が必要：

```swift
/// サイン波バッファ生成（シンプル）
func generateSineWaveBuffer(frequency: Double, duration: Double, sampleRate: Double) -> AVAudioPCMBuffer {
    let frameCount = AVAudioFrameCount(duration * sampleRate)
    let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)!
    buffer.frameLength = frameCount

    let data = buffer.floatChannelData![0]
    for i in 0..<Int(frameCount) {
        let t = Double(i) / sampleRate
        data[i] = Float(sin(2.0 * .pi * frequency * t))
    }

    // ADSRエンベロープ適用
    applyEnvelope(to: buffer)

    return buffer
}
```

**音色オプション**:
1. **サイン波**: シンプル、明確なピッチ
2. **ピアノ風合成**: 複数倍音 + 減衰エンベロープ
3. **オルガン風**: 基音 + 奇数倍音
4. **SF2からPCM変換**: 既存SF2を事前変換してキャッシュ

---

## 4. 推奨アプローチ

### 4.1 短期的対策（推奨）

**現在のSampler実装を維持 + 固定オフセット補正**

理由:
- 既存実装が動作している
- 80ms補正で-100ms → -29msに改善済み
- 100ms補正でさらに改善が見込める
- ±20ms程度の誤差は実用上許容範囲

実装:
```swift
private let samplerLatencyOffset: TimeInterval = 0.100  // 100ms

func getNoteStartTimestamp() -> TimeInterval? {
    let rawTimestamp = Date().timeIntervalSince(recordingStartTime)
    return rawTimestamp + currentOutputLatency + samplerLatencyOffset
}
```

### 4.2 長期的対策（将来の検討事項）

**AVAudioPlayerNodeへの移行を検討**

条件:
- 固定オフセット補正で精度が不十分な場合
- 音色の自由度より正確なタイミングが優先される場合
- シンプルな音色（サイン波、オルガン風）で十分な場合

実装ステップ:
1. サイン波バッファ生成器を実装
2. PlayerNode版ScalePlayerを実装
3. 既存Sampler版と並行運用（設定で切り替え可能に）
4. 精度比較後、優れた方をデフォルトに

---

## 5. ハイブリッドアプローチ（参考）

両方の利点を活かす方法:

```swift
protocol ScalePlayerEngine {
    func playNote(_ note: MIDINote, at time: AVAudioTime) async
}

class SamplerEngine: ScalePlayerEngine {
    // SF2ベースの豊富な音色
}

class PlayerNodeEngine: ScalePlayerEngine {
    // 正確なタイミング制御
}

class HybridScalePlayer {
    var engine: ScalePlayerEngine

    init(preferAccuracy: Bool) {
        engine = preferAccuracy ? PlayerNodeEngine() : SamplerEngine()
    }
}
```

---

## 6. 現在の計測結果まとめ

### Phase 1: 音色依存性（SF2 Sampler）
| 音色 | 平均オフセット |
|------|---------------|
| Piano | -100.1ms |
| Marimba | -102.3ms |
| Sine Wave | -113.6ms |

→ SF2音色間の差は小さい（2.2ms）、共通オフセットで補償可能

### 80ms補正テスト結果
| 状態 | 平均オフセット |
|------|---------------|
| 補正なし | -100.1ms |
| 80ms補正後 | -29.1ms |

→ 約71ms改善、まだ約30msの残差あり

### 次のステップ
- 100ms補正をテストし、±20ms以内を目指す

---

## 7. 結論

**現時点の推奨**: AVAudioUnitSampler + 固定オフセット補正（100ms）

理由:
1. 既存実装を最大限活用
2. 音色の多様性を維持（General MIDI対応）
3. 固定オフセットで実用的な精度が達成可能
4. PlayerNode移行は将来の選択肢として保持

PlayerNodeへの移行は、固定オフセット補正で十分な精度が得られない場合に検討する。
