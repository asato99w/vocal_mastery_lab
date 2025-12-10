# 録音一覧ページ UI 再設計仕様

## 変更概要

録音一覧ページ（RecordingListView）のユーザーインタラクションを改善し、より直感的で機能的な UI に変更する。

## 現在の仕様

### 現在の問題点
- 再生位置の調整ができない
- 再生/停止以外の操作がボタンに集約されている

### 現在の動作
```
録音リスト表示
└── 各録音行（HStack）
    ├── 再生/停止ボタン（左）
    ├── 録音情報（中央）
    │   ├── 日時（formattedDate）
    │   └── 長さ（duration.formatted）
    ├── 分析ボタン（NavigationLink）
    └── 削除ボタン（ゴミ箱アイコン）
```

**Note**: 現在は録音時のスケール設定（startNote、notePattern、tempo等）は表示されていない

## 新仕様

### 変更内容

#### 1. 再生位置調節バーの追加
- **位置**: 各録音行内、録音情報の下部
- **機能**:
  - 現在の再生位置を視覚的に表示
  - ドラッグで任意の位置にシーク可能
  - 再生中はリアルタイムで進捗を表示
- **UI要素**:
  - Slider コンポーネント（SwiftUI）
  - 最小値: 0.0（開始）
  - 最大値: 録音の総時間（秒）
  - 現在値: 再生位置（秒）
  - 左側: 現在時刻表示（例: "0:15"）
  - 右側: 総時間表示（例: "1:23"）

#### 2. 削除ボタンの維持
- **現状**: すでに削除ボタンが存在（ゴミ箱アイコン）
- **変更**: 位置はそのまま維持（各録音行の右端）
- **動作**: 確認ダイアログを表示してから削除

### 新しい UI レイアウト

```
録音リスト表示
└── 各録音行（VStack - 縦方向レイアウト）
    ├── 上部（HStack）
    │   ├── 再生/停止ボタン（左）
    │   ├── 録音情報（中央）
    │   │   ├── 日時（formattedDate）
    │   │   └── スケール名 ※新規（例: "C4 五声音階"、スケールなしの場合は非表示）
    │   ├── 分析ボタン（NavigationLink）※既存
    │   └── 削除ボタン（ゴミ箱アイコン）※既存
    └── 下部（VStack）※新規追加
        ├── 再生位置調節バー（Slider）
        └── 時間表示（HStack）
            ├── 現在時刻（左）例: "0:15"
            └── 総時間（右）例: "1:23"
```

**変更点**:
- 長さ（duration）の表示を削除
- スケール名を追加（scaleSettings から生成）
  - 表示形式: "[開始音名] [音階パターン名]" 例: "C4 五声音階"
  - スケールなし録音（scaleSettings が nil）の場合は非表示

## 技術仕様

### 影響を受けるファイル

#### 1. `Recording.swift` (Domain)
- スケール名を生成する computed property を追加
  - `var scaleDisplayName: String?` - スケール設定から表示用の名前を生成
  - 例: "C4 五声音階"、"D4 七声音階"
  - scaleSettings が nil の場合は nil を返す

#### 2. `RecordingListView.swift`
- 録音情報の表示内容を変更
  - duration 表示を削除
  - スケール名表示を追加（optional binding で nil の場合は非表示）
- 再生位置調節バーのレイアウトを追加
- 時間表示（現在時刻/総時間）を追加

#### 3. `MIDINote.swift` (Domain)
- 音名を表示用文字列に変換する computed property を追加
  - `var noteName: String` - MIDI番号から音名を生成
  - 例: 60 → "C4"、62 → "D4"

#### 4. `NotePattern.swift` (Domain)
- 音階パターンの表示名を追加
  - `var displayName: String` - 音階パターンの日本語名
  - 例: fiveToneScale → "五声音階"、sevenToneScale → "七声音階"

#### 5. `RecordingListViewModel.swift`
- 再生位置の状態管理を追加
  - `@Published var currentPlaybackPosition: [RecordingId: TimeInterval]`
- シーク機能の実装
  - `func seek(to position: TimeInterval, for recordingId: RecordingId)`
- 再生位置の定期更新
  - Combine タイマーで 0.1 秒ごとに更新

#### 6. `AudioPlayerProtocol.swift`
- シーク機能の追加
  - `func seek(to position: TimeInterval) async throws`
- 現在位置の取得
  - `var currentTime: TimeInterval { get }`

#### 7. `AVAudioPlayerWrapper.swift`
- シーク機能の実装
  - `AVAudioPlayer.currentTime` プロパティの活用

### データフロー

#### 再生位置更新フロー
```
1. ユーザーがスライダーをドラッグ
   ↓
2. RecordingListView が値変更を検知
   ↓
3. RecordingListViewModel.seek(to:for:) を呼び出し
   ↓
4. AudioPlayerProtocol.seek(to:) を実行
   ↓
5. AVAudioPlayerWrapper が実際のシーク処理
   ↓
6. 再生位置が更新される
```

#### 分析ページ遷移フロー（変更なし）
```
1. ユーザーが「分析」ボタン（NavigationLink）をタップ
   ↓
2. AnalysisView へ遷移
```

## UI/UX 考慮事項

### アクセシビリティ
- 再生位置調節バーに VoiceOver 対応
  - 現在位置と総時間を読み上げ
  - 10秒単位での移動をサポート
- 分析ボタンに明確なラベル
  - `.accessibilityLabel("分析を表示")`

### パフォーマンス
- 再生位置の更新頻度を最適化（0.1秒ごと）
- スライダードラッグ中は更新を一時停止

### エッジケース
- 再生中の録音でのみシーク可能
- 停止中はシーク不可（グレーアウト）
- 録音が存在しない場合のエラーハンドリング

## 実装優先度

### Phase 1: 再生位置調節バーの実装
- [ ] AudioPlayerProtocol にシーク機能を追加
- [ ] AVAudioPlayerWrapper にシーク実装
- [ ] RecordingListViewModel に位置管理機能追加
- [ ] RecordingListView に Slider UI を追加
- [ ] 単体テストの追加

### Phase 2: 統合とテスト
- [ ] UI テストの更新
- [ ] アクセシビリティテスト
- [ ] パフォーマンステスト

## 参考デザイン

### Apple Music / Podcast アプリ
- 標準的な再生位置調節バーのデザイン
- 時間表示のレイアウト

### 既存の VocalisStudio UI
- 既存の ColorPalette とデザイントークンを使用
- 一貫性のあるボタンスタイル

## 備考

- この変更は既存の録音再生機能に影響を与えない
- Clean Architecture の原則を維持（Presentation 層のみの変更が主）
- TDD アプローチで実装（テストファースト）

## 現在の実装調査結果

### すでに実装済みの機能
1. **MIDINote.noteName** ✅
   - `MIDINote.swift` 47-49行目
   - MIDI番号から音名への変換機能が存在（例: 60 → "C4"）

2. **AudioPlayerProtocol** ✅
   - `AudioPlayerProtocol.swift` に必要なメソッドが存在
   - `seek(to: TimeInterval)`: シーク機能（9行目）
   - `currentTime: TimeInterval`: 現在位置取得（11行目）
   - `duration: TimeInterval`: 総時間取得（12行目）

3. **RecordingListViewModel** の基本構造 ✅
   - `playingRecordingId`: 再生中の録音ID管理（11行目）
   - `audioPlayer`: AudioPlayerProtocol のインスタンス（14行目）

### 追加が必要な実装

#### 1. NotePattern に日本語表示名を追加
**ファイル**: `NotePattern.swift`
```swift
// 追加が必要
public var displayName: String {
    switch self {
    case .fiveToneScale:
        return "五声音階"
    }
}
```

#### 2. Recording にスケール名生成機能を追加
**ファイル**: `Recording.swift`
```swift
// 追加が必要
public var scaleDisplayName: String? {
    guard let settings = scaleSettings else { return nil }
    return "\(settings.startNote.noteName) \(settings.notePattern.displayName)"
}
```

#### 3. RecordingListViewModel に再生位置管理を追加
**ファイル**: `RecordingListViewModel.swift`

```swift
// 追加が必要
@Published public private(set) var currentPlaybackPosition: [RecordingId: TimeInterval] = [:]
private var positionUpdateTask: Task<Void, Never>?
private var cancellables = Set<AnyCancellable>()

// 再生位置の定期更新
private func startPositionTracking() {
    positionUpdateTask?.cancel()
    positionUpdateTask = Task { @MainActor in
        while !Task.isCancelled {
            if let recordingId = playingRecordingId {
                currentPlaybackPosition[recordingId] = audioPlayer.currentTime
            }
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1秒ごと
        }
    }
}

// シーク機能
public func seek(to position: TimeInterval, for recordingId: RecordingId) {
    guard playingRecordingId == recordingId else { return }
    audioPlayer.seek(to: position)
    currentPlaybackPosition[recordingId] = position
}

// playRecording() 内で startPositionTracking() を呼び出す
// stopPlayback() 内で positionUpdateTask?.cancel() を呼び出す
```

#### 4. RecordingListView の UI 変更
**ファイル**: `RecordingListView.swift` の RecordingRow

**変更箇所**: 114-122行目（録音情報表示部分）
```swift
// 現在
VStack(alignment: .leading, spacing: 4) {
    Text(recording.formattedDate)
        .font(.headline)
        .foregroundColor(ColorPalette.text)

    Text(recording.duration.formatted)
        .font(.subheadline)
        .foregroundColor(ColorPalette.text.opacity(0.6))
}

// 変更後
VStack(alignment: .leading, spacing: 4) {
    Text(recording.formattedDate)
        .font(.headline)
        .foregroundColor(ColorPalette.text)

    if let scaleName = recording.scaleDisplayName {
        Text(scaleName)
            .font(.subheadline)
            .foregroundColor(ColorPalette.text.opacity(0.6))
    }
}
```

**追加箇所**: 再生位置調節バーと時間表示（RecordingRow body の下部）
```swift
// HStack の後に追加
if isPlaying {
    VStack(spacing: 4) {
        Slider(
            value: Binding(
                get: { viewModel.currentPlaybackPosition[recording.id] ?? 0.0 },
                set: { viewModel.seek(to: $0, for: recording.id) }
            ),
            in: 0...recording.duration.seconds
        )
        .tint(ColorPalette.primary)

        HStack {
            Text(formatTime(viewModel.currentPlaybackPosition[recording.id] ?? 0.0))
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
            Spacer()
            Text(recording.duration.formatted)
                .font(.caption)
                .foregroundColor(ColorPalette.text.opacity(0.6))
        }
    }
    .padding(.top, 4)
}

// ヘルパーメソッド
private func formatTime(_ seconds: TimeInterval) -> String {
    let minutes = Int(seconds) / 60
    let secs = Int(seconds) % 60
    return String(format: "%d:%02d", minutes, secs)
}
```

**変更が必要な箇所**: RecordingRow の引数に ViewModel を追加
```swift
struct RecordingRow: View {
    let recording: Recording
    let isPlaying: Bool
    let viewModel: RecordingListViewModel // 追加
    let audioPlayer: AudioPlayerProtocol
    let analyzeRecordingUseCase: AnalyzeRecordingUseCase
    let onTap: () -> Void
    let onDelete: () -> Void

    // ...
}

// RecordingListView の recordingList 内で RecordingRow を生成する箇所
RecordingRow(
    recording: recording,
    isPlaying: viewModel.playingRecordingId == recording.id,
    viewModel: viewModel, // 追加
    audioPlayer: audioPlayer,
    analyzeRecordingUseCase: analyzeRecordingUseCase,
    onTap: { ... },
    onDelete: { ... }
)
```

## 実装プラン（詳細版）

### Phase 1: Domain層の拡張（スケール名表示）
1. **NotePattern.displayName を追加**
   - テスト: `NotePatternTests` に displayName のテスト追加
   - 実装: `NotePattern.swift` に computed property 追加

2. **Recording.scaleDisplayName を追加**
   - テスト: `RecordingTests` に scaleDisplayName のテスト追加
   - 実装: `Recording.swift` に computed property 追加

3. **RecordingListView の表示変更**
   - テスト: ViewInspector または UI テストで表示内容確認
   - 実装: duration 表示を scaleDisplayName に変更

### Phase 2: 再生位置調節機能の実装
4. **RecordingListViewModel の拡張**
   - テスト: `RecordingListViewModelTests` に以下を追加
     - 再生位置トラッキングのテスト
     - シーク機能のテスト
   - 実装:
     - `currentPlaybackPosition` プロパティ追加
     - `startPositionTracking()` メソッド追加
     - `seek(to:for:)` メソッド追加
     - `playRecording()` と `stopPlayback()` の修正

5. **RecordingListView に Slider UI 追加**
   - テスト: ViewInspector または UI テストで Slider の存在確認
   - 実装:
     - RecordingRow に viewModel 引数追加
     - Slider と時間表示の UI 追加
     - formatTime() ヘルパーメソッド追加

### Phase 3: 統合とテスト
6. **UI テストの更新**
   - RecordingListUITests の更新（もし存在すれば）
   - 表示内容の検証

7. **アクセシビリティテスト**
   - VoiceOver 対応の確認
   - Slider の accessibilityLabel 設定

8. **パフォーマンステスト**
   - 0.1秒ごとの更新によるパフォーマンス影響確認
   - 必要に応じて更新頻度調整

## テストファースト実装プラン

### 既存テスト分析

#### 1. NotePatternTests.swift
**場所**: `Packages/VocalisDomain/Tests/VocalisDomainTests/ValueObjects/NotePatternTests.swift`

**現在のテスト** (28行):
- `testFiveToneScale_Intervals()` - intervals プロパティのテスト
- `testFiveToneScale_AscendingDescending()` - ascendingDescending() のテスト

**追加が必要なテスト**:
```swift
func testFiveToneScale_DisplayName() {
    // Given
    let pattern = NotePattern.fiveToneScale

    // When
    let displayName = pattern.displayName

    // Then
    XCTAssertEqual(displayName, "五声音階")
}
```

#### 2. RecordingTests.swift
**場所**: `Packages/VocalisDomain/Tests/VocalisDomainTests/Entities/RecordingTests.swift`

**現在のテスト** (79行):
- `testInit_DefaultValues()` - 初期化テスト
- `testIdentifiable()` - ID のユニーク性テスト
- `testFormattedDate()` - 日付フォーマットテスト
- `testCodable()` - エンコード/デコードテスト

**追加が必要なテスト**:
```swift
func testScaleDisplayName_WithScaleSettings() {
    // Given
    let settings = ScaleSettings(
        startNote: MIDINote(60),  // C4
        endNote: MIDINote(72),
        notePattern: .fiveToneScale,
        tempo: .medium,
        ascendingCount: 12
    )
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/test.m4a"),
        duration: Duration(seconds: 100),
        scaleSettings: settings
    )

    // When
    let displayName = recording.scaleDisplayName

    // Then
    XCTAssertEqual(displayName, "C4 五声音階")
}

func testScaleDisplayName_WithoutScaleSettings() {
    // Given
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/test.m4a"),
        duration: Duration(seconds: 100),
        scaleSettings: nil
    )

    // When
    let displayName = recording.scaleDisplayName

    // Then
    XCTAssertNil(displayName)
}
```

#### 3. RecordingListViewModelTests.swift
**場所**: `VocalisStudioTests/Presentation/ViewModels/RecordingListViewModelTests.swift`

**現在のテスト** (275行):
- 初期化、録音読込、再生、停止、削除のテスト
- MockAudioPlayer を使用したテスト

**MockAudioPlayer の現在の実装** (76行):
- `currentTime`, `duration`, `seek(to:)` は既に実装済み ✅
- 新機能追加は不要

**追加が必要なテスト**:

```swift
// MARK: - Playback Position Tracking Tests

func testStartPositionTracking_WhenPlaybackStarts_ShouldUpdatePosition() async {
    // Given
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )
    mockAudioPlayer._duration = 10.0
    mockAudioPlayer._currentTime = 0.0

    // When
    await sut.playRecording(recording)

    // Simulate time progression
    mockAudioPlayer._currentTime = 2.5
    try? await Task.sleep(nanoseconds: 150_000_000) // 0.15秒待機

    // Then
    XCTAssertNotNil(sut.currentPlaybackPosition[recording.id])
    XCTAssertGreaterThan(sut.currentPlaybackPosition[recording.id] ?? 0, 0)
}

func testStopPositionTracking_WhenPlaybackStops_ShouldStopUpdating() async {
    // Given
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )
    mockAudioPlayer._duration = 10.0

    await sut.playRecording(recording)

    // When
    await sut.stopPlayback()

    let positionBeforeWait = sut.currentPlaybackPosition[recording.id] ?? 0
    try? await Task.sleep(nanoseconds: 200_000_000) // 0.2秒待機
    let positionAfterWait = sut.currentPlaybackPosition[recording.id] ?? 0

    // Then
    XCTAssertEqual(positionBeforeWait, positionAfterWait,
                   "Position should not update after stopping")
}

// MARK: - Seek Tests

func testSeek_WhenPlaying_ShouldUpdateAudioPlayerAndPosition() async {
    // Given
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )
    mockAudioPlayer._duration = 10.0
    await sut.playRecording(recording)

    // When
    let seekPosition: TimeInterval = 5.0
    sut.seek(to: seekPosition, for: recording.id)

    // Then
    XCTAssertTrue(mockAudioPlayer.seekCalled)
    XCTAssertEqual(mockAudioPlayer._currentTime, seekPosition)
    XCTAssertEqual(sut.currentPlaybackPosition[recording.id], seekPosition)
}

func testSeek_WhenNotPlaying_ShouldDoNothing() {
    // Given
    let recording = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )
    mockAudioPlayer.seekCalled = false

    // When
    sut.seek(to: 5.0, for: recording.id)

    // Then
    XCTAssertFalse(mockAudioPlayer.seekCalled,
                   "Should not seek when not playing")
}

func testSeek_WhenDifferentRecordingPlaying_ShouldDoNothing() async {
    // Given
    let recording1 = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )
    let recording2 = Recording(
        fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
        duration: Duration(seconds: 10.0),
        scaleSettings: ScaleSettings.mvpDefault
    )

    await sut.playRecording(recording1)
    mockAudioPlayer.seekCalled = false

    // When
    sut.seek(to: 5.0, for: recording2.id)

    // Then
    XCTAssertFalse(mockAudioPlayer.seekCalled,
                   "Should not seek when different recording is playing")
}
```

### TDDサイクル実装手順

#### Phase 1: NotePattern.displayName の追加

**🔴 Red (1分)**:
1. `NotePatternTests.swift` に `testFiveToneScale_DisplayName()` を追加
2. テスト実行 → コンパイルエラー（displayName プロパティが存在しない）
3. エラーメッセージ確認: "Value of type 'NotePattern' has no member 'displayName'"

**🟢 Green (1分)**:
1. `NotePattern.swift` に最小実装を追加:
```swift
public var displayName: String {
    switch self {
    case .fiveToneScale:
        return "五声音階"
    }
}
```
2. テスト実行 → パス ✅

**🔵 Refactor (30秒)**:
1. コードレビュー（十分シンプルなので変更不要）
2. テスト再実行 → パス ✅

#### Phase 2: Recording.scaleDisplayName の追加

**🔴 Red (2分)**:
1. `RecordingTests.swift` に以下を追加:
   - `testScaleDisplayName_WithScaleSettings()`
   - `testScaleDisplayName_WithoutScaleSettings()`
2. テスト実行 → コンパイルエラー（scaleDisplayName プロパティが存在しない）
3. エラーメッセージ確認

**🟢 Green (1分)**:
1. `Recording.swift` に実装を追加:
```swift
public var scaleDisplayName: String? {
    guard let settings = scaleSettings else { return nil }
    return "\(settings.startNote.noteName) \(settings.notePattern.displayName)"
}
```
2. テスト実行 → パス ✅

**🔵 Refactor (30秒)**:
1. コードレビュー（十分シンプルなので変更不要）
2. テスト再実行 → パス ✅

#### Phase 3: RecordingListViewModel の位置管理機能追加

**🔴 Red (5分)**:
1. `RecordingListViewModelTests.swift` に以下のテストを追加:
   - `testStartPositionTracking_WhenPlaybackStarts_ShouldUpdatePosition()`
   - `testStopPositionTracking_WhenPlaybackStops_ShouldStopUpdating()`
2. テスト実行 → コンパイルエラー（currentPlaybackPosition プロパティが存在しない）
3. エラーメッセージ確認

**🟢 Green (5分)**:
1. `RecordingListViewModel.swift` にプロパティと基本実装を追加:
```swift
@Published public private(set) var currentPlaybackPosition: [RecordingId: TimeInterval] = [:]
private var positionUpdateTask: Task<Void, Never>?

private func startPositionTracking() {
    positionUpdateTask?.cancel()
    positionUpdateTask = Task { @MainActor in
        while !Task.isCancelled {
            if let recordingId = playingRecordingId {
                currentPlaybackPosition[recordingId] = audioPlayer.currentTime
            }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }
}

private func stopPositionTracking() {
    positionUpdateTask?.cancel()
    positionUpdateTask = nil
}
```
2. `playRecording()` メソッド内で `startPositionTracking()` を呼び出し
3. `stopPlayback()` メソッド内で `stopPositionTracking()` を呼び出し
4. テスト実行 → パス ✅

**🔵 Refactor (2分)**:
1. 命名とコード構造を確認
2. メモリリーク防止を確認（Task のキャンセル処理）
3. テスト再実行 → パス ✅

#### Phase 4: シーク機能の追加

**🔴 Red (3分)**:
1. `RecordingListViewModelTests.swift` に以下のテストを追加:
   - `testSeek_WhenPlaying_ShouldUpdateAudioPlayerAndPosition()`
   - `testSeek_WhenNotPlaying_ShouldDoNothing()`
   - `testSeek_WhenDifferentRecordingPlaying_ShouldDoNothing()`
2. テスト実行 → コンパイルエラー（seek メソッドが存在しない）
3. エラーメッセージ確認

**🟢 Green (2分)**:
1. `RecordingListViewModel.swift` に実装を追加:
```swift
public func seek(to position: TimeInterval, for recordingId: RecordingId) {
    guard playingRecordingId == recordingId else { return }
    audioPlayer.seek(to: position)
    currentPlaybackPosition[recordingId] = position
}
```
2. テスト実行 → パス ✅

**🔵 Refactor (1分)**:
1. エッジケース処理を確認
2. テスト再実行 → パス ✅

#### Phase 5: 全テスト実行と確認

**最終確認 (2分)**:
```bash
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisDomainTests \
  -only-testing:VocalisStudioTests/RecordingListViewModelTests \
  -allowProvisioningUpdates
```

**期待結果**:
- NotePatternTests: 1個のテスト追加 → 3個のテストすべてパス ✅
- RecordingTests: 2個のテスト追加 → 6個のテストすべてパス ✅
- RecordingListViewModelTests: 5個のテスト追加 → 18個のテストすべてパス ✅

### 見積もり時間

| Phase | TDD サイクル | 時間 |
|-------|------------|------|
| Phase 1 | Red → Green → Refactor | 2.5分 |
| Phase 2 | Red → Green → Refactor | 3.5分 |
| Phase 3 | Red → Green → Refactor | 12分 |
| Phase 4 | Red → Green → Refactor | 6分 |
| Phase 5 | 全テスト実行 | 2分 |
| **合計** | | **26分** |

### 注意事項

1. **テストファーストを厳守**: 必ず「テスト作成 → 実行(失敗) → 実装 → 実行(成功)」の順序
2. **1つずつ進める**: 複数のテストを同時に書かない
3. **最小実装**: Greenフェーズでは必要最小限のコードのみ
4. **既存テストの確認**: 新機能追加後も既存テストがすべてパスすることを確認
5. **UIテストは後回し**: 今回はUnit/Integrationテストのみ実施

## UIテスト実装プラン（事後対応）

### 既存UIテストファイル分析

**ファイル**: `VocalisStudioUITests/RecordingListUITests.swift` (193行)

**現在のテスト**:
1. `testRecordingListNavigation()` - ナビゲーションと分析画面遷移テスト
2. `testDeleteRecording()` - 録音削除機能テスト

**テスト対象の変更内容**:
- ✅ 削除機能: 変更なし（既存テスト影響なし）
- ✅ 分析ページ遷移: NavigationLink のまま（既存テスト影響なし）
- ⚠️ 新機能: 再生位置調節バーとスケール名表示（新規テスト追加が必要）

### 追加が必要なUIテスト

#### Test 1: スケール名表示の確認

```swift
/// Test: Recording list displays scale name when scale settings exist
/// Expected: ~10 seconds execution time
@MainActor
func testRecordingList_DisplaysScaleName_WhenScaleSettingsExist() throws {
    let app = launchAppWithResetRecordingCount()

    // 1. Create recording with scale settings
    let homeRecordButton = app.buttons["HomeRecordButton"]
    XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
    homeRecordButton.tap()

    // Navigate to scale settings
    let settingsButton = app.buttons["RecordingSettingsButton"]
    XCTAssertTrue(settingsButton.waitForExistence(timeout: 5))
    settingsButton.tap()

    // Enable scale
    let scaleToggle = app.switches["ScaleToggle"]
    XCTAssertTrue(scaleToggle.waitForExistence(timeout: 3))
    if scaleToggle.value as? String == "0" {
        scaleToggle.tap()
    }

    // Go back and start recording
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let startButton = app.buttons["StartRecordingButton"]
    XCTAssertTrue(startButton.waitForExistence(timeout: 5))
    startButton.tap()

    let stopButton = app.buttons["StopRecordingButton"]
    XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
    Thread.sleep(forTimeInterval: 1.0)
    stopButton.tap()

    let playButton = app.buttons["PlayLastRecordingButton"]
    XCTAssertTrue(playButton.waitForExistence(timeout: 5))

    // 2. Navigate to recording list
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let homeListButton = app.buttons["HomeListButton"]
    XCTAssertTrue(homeListButton.waitForExistence(timeout: 5))
    homeListButton.tap()

    Thread.sleep(forTimeInterval: 2.0)

    // Screenshot: Recording list with scale name
    let screenshot = app.screenshot()
    let attachment = XCTAttachment(screenshot: screenshot)
    attachment.name = "scale_name_01_list_with_scale"
    attachment.lifetime = .keepAlways
    add(attachment)

    // 3. Verify scale name is displayed
    // Note: Exact text depends on scale settings (e.g., "C4 五声音階")
    let scaleNameLabels = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "五声音階"))
    XCTAssertTrue(scaleNameLabels.firstMatch.waitForExistence(timeout: 3),
                  "Scale name should be displayed in recording list")
}

/// Test: Recording list does not display scale name when no scale settings
/// Expected: ~10 seconds execution time
@MainActor
func testRecordingList_DoesNotDisplayScaleName_WhenNoScaleSettings() throws {
    let app = launchAppWithResetRecordingCount()

    // 1. Create recording WITHOUT scale settings
    let homeRecordButton = app.buttons["HomeRecordButton"]
    XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
    homeRecordButton.tap()

    // Navigate to scale settings
    let settingsButton = app.buttons["RecordingSettingsButton"]
    XCTAssertTrue(settingsButton.waitForExistence(timeout: 5))
    settingsButton.tap()

    // Disable scale
    let scaleToggle = app.switches["ScaleToggle"]
    XCTAssertTrue(scaleToggle.waitForExistence(timeout: 3))
    if scaleToggle.value as? String == "1" {
        scaleToggle.tap()
    }

    // Go back and start recording
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let startButton = app.buttons["StartRecordingButton"]
    XCTAssertTrue(startButton.waitForExistence(timeout: 5))
    startButton.tap()

    let stopButton = app.buttons["StopRecordingButton"]
    XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
    Thread.sleep(forTimeInterval: 1.0)
    stopButton.tap()

    let playButton = app.buttons["PlayLastRecordingButton"]
    XCTAssertTrue(playButton.waitForExistence(timeout: 5))

    // 2. Navigate to recording list
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let homeListButton = app.buttons["HomeListButton"]
    XCTAssertTrue(homeListButton.waitForExistence(timeout: 5))
    homeListButton.tap()

    Thread.sleep(forTimeInterval: 2.0)

    // Screenshot: Recording list without scale name
    let screenshot = app.screenshot()
    let attachment = XCTAttachment(screenshot: screenshot)
    attachment.name = "scale_name_02_list_without_scale"
    attachment.lifetime = .keepAlways
    add(attachment)

    // 3. Verify scale name is NOT displayed (only date should be shown)
    let scaleNameLabels = app.staticTexts.matching(NSPredicate(format: "label CONTAINS %@", "五声音階"))
    XCTAssertFalse(scaleNameLabels.firstMatch.exists,
                   "Scale name should NOT be displayed when no scale settings")
}
```

#### Test 2: 再生位置調節バーの動作確認

```swift
/// Test: Playback position slider appears during playback
/// Expected: ~15 seconds execution time
@MainActor
func testRecordingList_ShowsPlaybackSlider_DuringPlayback() throws {
    let app = launchAppWithResetRecordingCount()

    // 1. Create a recording
    let homeRecordButton = app.buttons["HomeRecordButton"]
    XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
    homeRecordButton.tap()

    let startButton = app.buttons["StartRecordingButton"]
    XCTAssertTrue(startButton.waitForExistence(timeout: 5))
    startButton.tap()

    let stopButton = app.buttons["StopRecordingButton"]
    XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
    Thread.sleep(forTimeInterval: 2.0) // Record for 2 seconds
    stopButton.tap()

    let playButton = app.buttons["PlayLastRecordingButton"]
    XCTAssertTrue(playButton.waitForExistence(timeout: 5))

    // 2. Navigate to recording list
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let homeListButton = app.buttons["HomeListButton"]
    XCTAssertTrue(homeListButton.waitForExistence(timeout: 5))
    homeListButton.tap()

    Thread.sleep(forTimeInterval: 2.0)

    // Screenshot: Before playback
    let screenshot1 = app.screenshot()
    let attachment1 = XCTAttachment(screenshot: screenshot1)
    attachment1.name = "playback_slider_01_before_playback"
    attachment1.lifetime = .keepAlways
    add(attachment1)

    // 3. Verify slider does NOT exist before playback
    let playbackSliders = app.sliders.matching(NSPredicate(format: "identifier BEGINSWITH %@", "PlaybackPositionSlider_"))
    XCTAssertFalse(playbackSliders.firstMatch.exists,
                   "Playback slider should NOT exist before playback starts")

    // 4. Start playback
    let playButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "PlayRecordingButton_"))
    XCTAssertTrue(playButtons.firstMatch.waitForExistence(timeout: 3))
    playButtons.firstMatch.tap()

    // Wait for playback to start
    Thread.sleep(forTimeInterval: 0.5)

    // Screenshot: During playback
    let screenshot2 = app.screenshot()
    let attachment2 = XCTAttachment(screenshot: screenshot2)
    attachment2.name = "playback_slider_02_during_playback"
    attachment2.lifetime = .keepAlways
    add(attachment2)

    // 5. Verify slider appears during playback
    XCTAssertTrue(playbackSliders.firstMatch.waitForExistence(timeout: 3),
                  "Playback slider SHOULD appear during playback")

    // 6. Verify time labels exist
    let currentTimeLabels = app.staticTexts.matching(NSPredicate(format: "label MATCHES %@", "\\d+:\\d{2}"))
    XCTAssertGreaterThanOrEqual(currentTimeLabels.count, 2,
                                "Should display current time and total duration")

    // 7. Stop playback
    let stopPlaybackButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "StopPlaybackButton_"))
    XCTAssertTrue(stopPlaybackButtons.firstMatch.waitForExistence(timeout: 3))
    stopPlaybackButtons.firstMatch.tap()

    Thread.sleep(forTimeInterval: 0.5)

    // Screenshot: After playback stopped
    let screenshot3 = app.screenshot()
    let attachment3 = XCTAttachment(screenshot: screenshot3)
    attachment3.name = "playback_slider_03_after_playback"
    attachment3.lifetime = .keepAlways
    add(attachment3)
}

/// Test: Playback position slider can be adjusted by user
/// Expected: ~15 seconds execution time
@MainActor
func testRecordingList_CanAdjustPlaybackPosition_UsingSlider() throws {
    let app = launchAppWithResetRecordingCount()

    // 1. Create a recording (longer duration for better testing)
    let homeRecordButton = app.buttons["HomeRecordButton"]
    XCTAssertTrue(homeRecordButton.waitForExistence(timeout: 5))
    homeRecordButton.tap()

    let startButton = app.buttons["StartRecordingButton"]
    XCTAssertTrue(startButton.waitForExistence(timeout: 5))
    startButton.tap()

    let stopButton = app.buttons["StopRecordingButton"]
    XCTAssertTrue(stopButton.waitForExistence(timeout: 10))
    Thread.sleep(forTimeInterval: 3.0) // Record for 3 seconds
    stopButton.tap()

    let playButton = app.buttons["PlayLastRecordingButton"]
    XCTAssertTrue(playButton.waitForExistence(timeout: 5))

    // 2. Navigate to recording list
    app.navigationBars.buttons.element(boundBy: 0).tap()
    Thread.sleep(forTimeInterval: 0.5)

    let homeListButton = app.buttons["HomeListButton"]
    XCTAssertTrue(homeListButton.waitForExistence(timeout: 5))
    homeListButton.tap()

    Thread.sleep(forTimeInterval: 2.0)

    // 3. Start playback
    let playButtons = app.buttons.matching(NSPredicate(format: "identifier BEGINSWITH %@", "PlayRecordingButton_"))
    XCTAssertTrue(playButtons.firstMatch.waitForExistence(timeout: 3))
    playButtons.firstMatch.tap()

    Thread.sleep(forTimeInterval: 0.5)

    // Screenshot: Initial playback state
    let screenshot1 = app.screenshot()
    let attachment1 = XCTAttachment(screenshot: screenshot1)
    attachment1.name = "slider_adjust_01_initial_state"
    attachment1.lifetime = .keepAlways
    add(attachment1)

    // 4. Find and adjust the slider
    let playbackSliders = app.sliders.matching(NSPredicate(format: "identifier BEGINSWITH %@", "PlaybackPositionSlider_"))
    XCTAssertTrue(playbackSliders.firstMatch.waitForExistence(timeout: 3))

    let slider = playbackSliders.firstMatch

    // Adjust slider to 50% position
    slider.adjust(toNormalizedSliderPosition: 0.5)

    Thread.sleep(forTimeInterval: 0.5)

    // Screenshot: After slider adjustment
    let screenshot2 = app.screenshot()
    let attachment2 = XCTAttachment(screenshot: screenshot2)
    attachment2.name = "slider_adjust_02_after_adjustment"
    attachment2.lifetime = .keepAlways
    add(attachment2)

    // 5. Verify time label updated
    // Note: Exact verification depends on recording duration
    // We just verify that time labels still exist and are updating
    let currentTimeLabels = app.staticTexts.matching(NSPredicate(format: "label MATCHES %@", "\\d+:\\d{2}"))
    XCTAssertGreaterThanOrEqual(currentTimeLabels.count, 2,
                                "Time labels should still be displayed after slider adjustment")
}
```

### UIテスト実装タイミング

**実施タイミング**: Phase 1-4 (Unit/Integrationテスト) 完了後

**実装順序**:
1. Unit/Integration テスト完了 ✅
2. UI実装完了 ✅
3. UIテスト追加 ← ここから開始
4. UIテスト実行と修正
5. 全テスト実行（Unit + Integration + UI）

**見積もり時間**:
| テスト | 作成時間 | 実行・デバッグ時間 |
|--------|---------|------------------|
| スケール名表示テスト (2個) | 10分 | 5分 |
| 再生位置バーテスト (2個) | 15分 | 10分 |
| **合計** | **25分** | **15分** |

**総計**: 約40分

### UIテスト実行コマンド

```bash
# 新規追加したテストのみ実行
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioUITests/RecordingListUITests/testRecordingList_DisplaysScaleName_WhenScaleSettingsExist \
  -only-testing:VocalisStudioUITests/RecordingListUITests/testRecordingList_DoesNotDisplayScaleName_WhenNoScaleSettings \
  -only-testing:VocalisStudioUITests/RecordingListUITests/testRecordingList_ShowsPlaybackSlider_DuringPlayback \
  -only-testing:VocalisStudioUITests/RecordingListUITests/testRecordingList_CanAdjustPlaybackPosition_UsingSlider \
  -allowProvisioningUpdates

# 既存テストの回帰確認
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio \
  -destination 'platform=iOS Simulator,name=iPhone 16' \
  -only-testing:VocalisStudioUITests/RecordingListUITests \
  -allowProvisioningUpdates
```

**期待結果**:
- 新規テスト4個: すべてパス ✅
- 既存テスト2個: すべてパス（回帰なし）✅
- 合計6個のUIテスト: すべてパス ✅

### UIテスト注意事項

1. **accessibility identifier の追加**: 新しいUI要素に識別子を設定
   - `PlaybackPositionSlider_{recordingId}` - 再生位置調節バー
   - スケール名表示には既存の accessibility が自動適用される

2. **既存テストへの影響**:
   - `testRecordingListNavigation()`: NavigationLink は変更なし → 影響なし ✅
   - `testDeleteRecording()`: 削除ボタンは変更なし → 影響なし ✅

3. **スクリーンショット**: すべてのUIテストでスクリーンショットを取得し、検証に活用

4. **タイミング調整**: UIの更新待機時間は適宜調整（Thread.sleep）

## 更新履歴

- 2025-11-07: 初版作成
- 2025-11-07: 現在の実装調査結果と詳細な実装プランを追加
- 2025-11-07: テストファースト実装プランを追加（既存テスト分析、TDDサイクル手順、見積もり時間）
