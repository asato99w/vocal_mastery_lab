# 音声レベルメーター機能 仕様書

## 概要

録音画面にリアルタイムの音声レベル（ボリュームメーター）を表示する機能。
ユーザーがマイク入力レベルを視覚的に確認できるようにする。

## UI仕様

### 表示位置
- `RealtimeDisplayArea` 内に追加
- 周波数スペクトラムの上部、または横に配置

### 表示形式

**オプション1: 横バーメーター**
```
音声レベル: [████████░░░░░░░░░░░░] -12 dB
```

**オプション2: 縦バーメーター**
```
 ▓
 ▓
 ▓
 █
 █
 █
 █
-∞ dB
```

### 視覚デザイン
- **カラーグラデーション**:
  - 緑: 適正レベル (-40dB 〜 -12dB)
  - 黄: 注意レベル (-12dB 〜 -6dB)
  - 赤: クリップ危険 (-6dB 〜 0dB)
- **更新頻度**: 30fps (約33ms間隔)
- **スムージング**: 急激な変化を緩和するためのローパスフィルター適用

### 表示タイミング
- 録音中 (`recordingState == .recording`)
- 再生中 (`isPlayingRecording == true`)

## 実装概要

### 1. Infrastructure層: 音声レベル取得

**ファイル**: `Infrastructure/Audio/AudioLevelMonitor.swift` (新規)

```swift
import AVFoundation
import Combine

/// プロトコル定義
protocol AudioLevelMonitorProtocol {
    var audioLevelPublisher: AnyPublisher<Float, Never> { get }
    func startMonitoring()
    func stopMonitoring()
}

/// AVAudioRecorder の metering 機能を使用
class AudioLevelMonitor: AudioLevelMonitorProtocol {
    private var audioRecorder: AVAudioRecorder?
    private var timer: Timer?
    private let audioLevelSubject = CurrentValueSubject<Float, Never>(-160.0)

    var audioLevelPublisher: AnyPublisher<Float, Never> {
        audioLevelSubject.eraseToAnyPublisher()
    }

    func startMonitoring() {
        // AVAudioRecorder の isMeteringEnabled = true を設定
        // Timer で updateMeters() を定期的に呼び出し
        // averagePowerForChannel(0) で dB 値を取得
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
}
```

**代替案**: `AVAudioEngine` + `installTap(onBus:)` でオーディオバッファから計算

```swift
// AVAudioEngineを使用する場合
inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
    let channelData = buffer.floatChannelData?[0]
    // RMS (Root Mean Square) を計算してdB変換
    let rms = sqrt(sumOfSquares / Float(frameLength))
    let db = 20 * log10(rms)
}
```

### 2. Presentation層: ViewModel

**ファイル**: `RecordingViewModel.swift` に追加

```swift
// プロパティ追加
@Published var audioLevel: Float = -160.0  // dB値 (-160 〜 0)

// メソッド追加
private func startAudioLevelMonitoring() {
    audioLevelMonitor.startMonitoring()
    audioLevelMonitor.audioLevelPublisher
        .receive(on: DispatchQueue.main)
        .sink { [weak self] level in
            self?.audioLevel = level
        }
        .store(in: &cancellables)
}

private func stopAudioLevelMonitoring() {
    audioLevelMonitor.stopMonitoring()
}
```

### 3. Presentation層: View

**ファイル**: `RealtimeDisplayArea.swift` に追加

```swift
struct AudioLevelMeterView: View {
    let level: Float  // dB値

    private var normalizedLevel: CGFloat {
        // -60dB 〜 0dB を 0.0 〜 1.0 に正規化
        let clamped = max(-60, min(0, level))
        return CGFloat((clamped + 60) / 60)
    }

    private var meterColor: Color {
        if normalizedLevel > 0.9 { return .red }
        if normalizedLevel > 0.7 { return .yellow }
        return .green
    }

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                // 背景
                Rectangle()
                    .fill(Color.gray.opacity(0.3))

                // レベルバー
                Rectangle()
                    .fill(meterColor)
                    .frame(width: geometry.size.width * normalizedLevel)
            }
        }
        .frame(height: 20)
        .cornerRadius(4)
    }
}
```

### 4. 依存性注入

**ファイル**: `DependencyContainer.swift` に追加

```swift
lazy var audioLevelMonitor: AudioLevelMonitorProtocol = {
    AudioLevelMonitor()
}()
```

## テスト戦略

### Unit Tests

1. **AudioLevelMonitor テスト**
   - `startMonitoring()` 後に `audioLevelPublisher` が値を発行
   - `stopMonitoring()` 後に値の発行が停止
   - dB値が有効範囲内 (-160 〜 0)

2. **ViewModel テスト**
   - 録音開始時に `audioLevel` が更新される
   - 録音停止時に監視が停止される

### UI Tests

1. **レベルメーター表示テスト**
   - 録音中にレベルメーターが表示される
   - 非録音時にレベルメーターが非表示

## 実装ステップ

1. [ ] TDD Red: AudioLevelMonitor のテスト作成
2. [ ] TDD Green: AudioLevelMonitor 実装
3. [ ] TDD Refactor: コード品質改善
4. [ ] TDD Red: ViewModel テスト作成
5. [ ] TDD Green: ViewModel 実装
6. [ ] TDD Red: UI Tests 作成
7. [ ] TDD Green: View 実装
8. [ ] コミット

## 参考情報

- **iOS バージョン**: iOS 15.0+
- **AVFoundation ドキュメント**: [Apple Developer](https://developer.apple.com/documentation/avfoundation)
