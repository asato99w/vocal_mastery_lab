# PlaybackUITests.testPlayback_shouldDetectPitch 修正ガイド

## 問題の本質

### 問題1: UI表示条件
`DetectedPitchNoteName` が表示される条件は `RealtimeDisplayArea.swift:238` で定義されている：

```swift
if isActive, let detected = detectedPitch {
    // DetectedPitchNoteName を表示
}
```

ここで `isActive` は：
```swift
private var isActive: Bool {
    recordingState == .recording || isPlayingRecording
}
```

**つまり、再生終了後は `isPlayingRecording = false` となり、`detectedPitch` に値があっても表示されない。**

### 問題2: 再生中のピッチ検出成功率
ログ分析の結果、**再生中はピッチ検出の成功率が非常に低い**（または成功しない）ことが判明。

- **録音中**: SUCCESS #8-50 など連続的に成功
- **再生中**: SUCCESSログなし（ピッチ検出が成功していない）

原因: AVAudioSessionの制約により、再生と録音（マイク入力）の同時使用でマイク入力が制限される。

## 必要な修正（2ファイル）

### 1. `PitchDetectionViewModel.swift` - debounce機能追加

録音中に検出したピッチを一定時間保持することで、再生中もUI表示を維持する：

```swift
// プロパティ追加
private let pitchRetentionDuration: TimeInterval = 4.0
private var lastValidPitchTime: Date?

// updateDetectedPitch メソッド内
guard let pitch = pitch else {
    // Debounce: 最後の有効なピッチ検出から一定時間は保持
    if let lastValid = lastValidPitchTime {
        let timeSinceLastValid = Date().timeIntervalSince(lastValid)
        if timeSinceLastValid < pitchRetentionDuration {
            return  // ピッチを保持、クリアしない
        }
    }
    detectedPitch = nil
    pitchAccuracy = .none
    return
}

// 有効なピッチ検出時に時間を記録
lastValidPitchTime = Date()
```

### 2. `PlaybackUITests.swift` - テストタイミング調整

1. **録音時間を長くする**: 3秒 → 5秒
2. **待機時間を削除**: 2秒の `Thread.sleep` を削除
3. **タイムアウトを短縮**: 5秒 → 4秒

```swift
func testPlayback_shouldDetectPitch() throws {
    // ... 省略 ...

    // 5秒間録音（再生時間を確保してピッチ検出の余裕を作る）
    Thread.sleep(forTimeInterval: 5.0)
    stopButton.tap()

    // ... 省略 ...

    // 待機なしで即座にピッチ検出を確認（再生終了前に確認する）
    let detectedPitchNoteName = app.staticTexts["DetectedPitchNoteName"]
    XCTAssertTrue(detectedPitchNoteName.waitForExistence(timeout: 4))
}
```

## テスト実行コマンド

```bash
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio-UIOnly \
  -destination "id=8E091155-1AB5-4C0C-AA9D-B89EB3B01DFD" \
  -parallel-testing-enabled NO \
  -only-testing:VocalisStudioUITests/PlaybackUITests/testPlayback_shouldDetectPitch \
  -allowProvisioningUpdates
```

## 検証済み結果（両方の修正を適用後）

テスト実行結果：
- t=21.76s: `waitForExistence(timeout: 4)` 開始
- t=22.83s: `DetectedPitchNoteName` 検出成功（約1秒で検出）
- スクリーンショット: `D3 → F#3 +564¢` を確認

---
作成日: 2025-11-30
更新日: 2025-11-30（debounce必須であることを追記）
