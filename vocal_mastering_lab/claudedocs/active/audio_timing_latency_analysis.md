# Audio Timing Latency Analysis & Implementation Plan

**作成日**: 2025-12-05
**ステータス**: Active - 実装予定

## 概要

VocalisStudioのオーディオタイミングシステムにおける遅延源の分析と、補償実装の計画書。

## 1. 問題の整理

### 1.1 ズレの種類と方向

#### 入力遅延（ピッチ検出）
```
実際の発声時刻: T
記録される時刻: T + 23ms（右にズレ）

原因: FFT処理遅延（固定値 ~23ms）
```

#### 出力遅延（スケール再生）
```
MIDI送信時刻:     T - outputLatency
実際に音が鳴る時刻: T

記録される時刻: T - outputLatency（左にズレ）※outputLatency未補償のため
```

### 1.2 グラフ上の結果

ユーザーが完璧にスケールに合わせて歌った場合:
```
スケールバー:  |████████|                    ← T - outputLatency に記録
検出ピッチ:                    ●●●●●●●●     ← T + 23ms に記録

                ←── outputLatency + 23ms ──→
                    （Bluetooth時: ~173ms）
```

### 1.3 再生時の挙動

#### スケールバー単体
```
録音時: 実際より outputLatency 分早く記録（左）
再生時: カーソルより outputLatency 分遅れて音が聞こえる（右）

→ 相殺される（同じデバイスなら正しいタイミングで聞こえる）
```

#### ピッチ単体
```
録音時: 実際より 23ms 遅く記録（右）
再生時: カーソルより outputLatency 分遅れて音が聞こえる（さらに右）

→ 増幅される（23ms + outputLatency 分ズレて聞こえる）
```

### 1.4 補償の方針

| 補償対象 | 副作用 | 方針 |
|----------|--------|------|
| **入力遅延（23ms）** | なし | ✅ 単独で実装可能 |
| **outputLatency（録音時）** | 再生時とセットで実装必要 | ⚠️ 同時実装 |
| **outputLatency（再生時）** | 録音時とセットで実装必要 | ⚠️ 同時実装 |

## 2. 実装計画

### Step 1: 入力遅延補償（単独実装可）

**目的**: ピッチのタイムスタンプを23ms早める

```swift
// PitchDetectionResult生成時
let compensatedTimestamp = rawTimestamp.addingTimeInterval(-0.023)
```

**効果**:
- グラフ上でピッチとスケールの相対位置が改善
- 再生時の体験に悪影響なし

### Step 2: outputLatency補償（録音時・再生時を同時実装）

#### 2.1 録音時（スケールタイムスタンプ）
```swift
// TapBasedTimestampStrategy
let adjustedTimestamp = baseTimestamp.addingTimeInterval(outputLatency)
```

#### 2.2 再生時（カーソル位置）
```swift
// AnalysisViewModel
let newTime = rawTime - outputLatency

// 後戻り防止
if newTime >= currentTime {
    currentTime = newTime
}
// newTime < currentTime の場合は更新しない（停止して待機）
```

**後戻り防止の理由**:
- オーディオルート変更時（有線→Bluetooth）にoutputLatencyが急増
- 補償なしだとカーソルが後ろにジャンプする
- 停止して待機することで自然な体験を維持

## 3. 検証計画

### 3.1 入力遅延補償の検証

1. **実装前の計測**
   - 現在のピッチとスケールバーのズレを計測
   - 複数回録音して平均値を確認

2. **実装後の検証**
   - 同条件で再計測
   - ズレが23ms程度改善されていることを確認

### 3.2 outputLatency補償の検証

1. **録音時の検証**
   - 内蔵スピーカーで録音・再生
   - AirPodsで録音・再生
   - スケールバーとカーソルの同期を体感確認

2. **再生時の検証**
   - 通常再生での視聴覚同期確認
   - オーディオルート変更時の挙動確認（後戻りしないこと）

3. **クロスデバイス検証**
   - 内蔵スピーカーで録音 → Bluetoothで再生
   - Bluetoothで録音 → 内蔵スピーカーで再生
   - それぞれの体験を確認

### 3.3 outputLatency値の変動調査

実装前に確認すべき事項:
```swift
// outputLatencyの実際の挙動を調査
// - 再生中に値が変動するか
// - どの程度の頻度で変動するか
// - 変動幅はどの程度か

Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
    print("outputLatency: \(AVAudioSession.sharedInstance().outputLatency * 1000)ms")
}
```

## 4. 実装詳細

### 4.1 LatencyMeasurementService

```swift
protocol LatencyMeasurementServiceProtocol {
    var inputLatency: TimeInterval { get }
    var outputLatency: TimeInterval { get }
    var pitchDetectionLatency: TimeInterval { get }  // 固定値 0.023
}

final class LatencyMeasurementService: LatencyMeasurementServiceProtocol {
    private let audioSession = AVAudioSession.sharedInstance()

    var inputLatency: TimeInterval { audioSession.inputLatency }
    var outputLatency: TimeInterval { audioSession.outputLatency }
    let pitchDetectionLatency: TimeInterval = 0.023
}
```

### 4.2 再生時の後戻り防止実装

```swift
// AnalysisViewModel.swift

private var lastCurrentTime: TimeInterval = 0

func updatePlaybackPosition() {
    let rawTime = audioPlayer.currentTime
    let outputLatency = latencyService.outputLatency
    let compensatedTime = max(0, rawTime - outputLatency)

    // 後戻り防止: 前回より小さい場合は更新しない
    if compensatedTime >= lastCurrentTime {
        currentTime = compensatedTime
        lastCurrentTime = compensatedTime
    }
    // else: カーソルを停止して待機（rawTimeが追いつくまで）
}

func resetPlaybackPosition() {
    // シーク時や再生開始時にリセット
    lastCurrentTime = 0
}
```

## 5. 参考資料

- [pitch_bar_timing_investigation.md](./pitch_bar_timing_investigation.md) - 過去の調査レポート
- [Apple Developer: AVAudioSession](https://developer.apple.com/documentation/avfaudio/avaudiosession)

## 6. 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-12-05 | 初版作成 |
| 2025-12-05 | 実装方針を整理、検証計画を追加 |
| 2025-12-05 | **実装完了**: Step 1 (入力遅延補償), Step 2 (outputLatency補償 - 録音時・再生時) |
| 2025-12-05 | タイミング調査用ロギング追加（YIN NOTE_CHANGEマーカー、scale_barログ） |

## 7. 実装詳細（完了）

### 7.1 入力遅延補償（AudioFileAnalyzer.swift）

```swift
// Latency compensation: FFT window center offset
private var pitchDetectionLatencyOffset: Double {
    Double(yinBufferSize / 2) / sampleRate  // ~23.2ms for 2048 buffer at 44.1kHz
}

// タイムスタンプ生成時に適用
let rawTimestamp = Double(position) / sampleRate
let compensatedTimestamp = max(0, rawTimestamp - pitchDetectionLatencyOffset)
```

### 7.2 outputLatency補償 - 録音時（ScaleTimestampStrategy.swift）

```swift
// Output latency compensation
private var outputLatency: TimeInterval {
    AVAudioSession.sharedInstance().outputLatency
}

// タイムスタンプ計算時にoutputLatencyを加算
let rawTimestamp = detectionTime.timeIntervalSince(startTime)
let compensatedTimestamp = rawTimestamp + currentOutputLatency
```

### 7.3 outputLatency補償 - 再生時（AnalysisViewModel.swift）

```swift
// 再生カーソル位置の補正
let rawTime = self.audioPlayer.currentTime
let compensatedTime = max(0, rawTime - self.outputLatency)

// 後戻り防止
if compensatedTime >= self.lastDisplayedTime {
    self.currentTime = compensatedTime
    self.lastDisplayedTime = compensatedTime
}
```
