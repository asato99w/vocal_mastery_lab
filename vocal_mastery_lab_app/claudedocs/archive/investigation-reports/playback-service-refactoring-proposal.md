# PlaybackService リファクタリング提案

## 概要

再生機能の状態管理を中央集権化する`PlaybackService`アーキテクチャの提案。

## 背景と問題

### 発生したバグ (2024年12月 commit: 9b77efc で修正)

1. **画面遷移時に音声が再生され続ける**
   - RecordingListView、AnalysisViewから画面遷移しても音声が停止しない

2. **分析画面から戻った後、同じ録音が再生できない**
   - `selectedRecording`がリセットされず、`selectAndPlay()`が早期リターン

3. **一時停止位置が画面間で共有される**
   - 分析画面で一時停止→一覧に戻る→途中から再生される

### 根本原因: 状態の重複管理

```
現状のアーキテクチャ:

┌─────────────────────────────────────────────────────┐
│                AudioPlayer (Singleton)               │
│  isPlaying, currentTime, audioPlayer instance        │
└─────────────────────────────────────────────────────┘
          ▲                    ▲
          │ uses               │ uses
          │                    │
┌─────────────────────────┐    ┌─────────────────────────┐
│ RecordingListViewModel  │    │   AnalysisViewModel     │
│ ───────────────────────│    │ ───────────────────────│
│ isPlaying (独自)         │    │ isPlaying (独自)         │
│ currentTime (独自)       │    │ currentTime (独自)       │
│ playingRecordingId      │    │ playbackState           │
│ selectedRecording       │    │ playbackTimer           │
│ currentPlaybackPosition │    │                         │
└─────────────────────────┘    └─────────────────────────┘

問題点:
- 各ViewModelが独立して再生状態を追跡
- 画面遷移時に状態の不整合が発生
- 状態リセットが複数箇所で必要
```

## 提案するアーキテクチャ

### PlaybackService パターン

```
┌─────────────────────────────────────────────────────┐
│                  PlaybackService                     │
│  ─────────────────────────────────────────────────  │
│  @Published isPlaying: Bool                         │
│  @Published currentTime: Double                     │
│  @Published currentRecordingId: RecordingId?        │
│  @Published playbackState: PlaybackState            │
│  ─────────────────────────────────────────────────  │
│  play(recording:) → async                           │
│  pause()                                            │
│  resume()                                           │
│  stop()                                             │
│  seek(to:)                                          │
│  ─────────────────────────────────────────────────  │
│  (内部) AudioPlayer                                  │
│  (内部) Timer for position tracking                  │
└─────────────────────────────────────────────────────┘
          ▲                    ▲
          │ subscribe          │ subscribe
          │ (Combine)          │ (Combine)
          │                    │
┌─────────────────────────┐    ┌─────────────────────────┐
│ RecordingListViewModel  │    │   AnalysisViewModel     │
│ ───────────────────────│    │ ───────────────────────│
│ playbackService の状態を │    │ playbackService の状態を │
│ そのまま公開             │    │ そのまま公開             │
└─────────────────────────┘    └─────────────────────────┘
```

### 実装イメージ

```swift
// PlaybackService.swift
@MainActor
public class PlaybackService: ObservableObject {
    // MARK: - Published State (Single Source of Truth)
    @Published public private(set) var isPlaying: Bool = false
    @Published public private(set) var currentTime: Double = 0.0
    @Published public private(set) var currentRecordingId: RecordingId?
    @Published public private(set) var duration: Double = 0.0

    // MARK: - Private
    private let audioPlayer: AudioPlayerProtocol
    private var positionTimer: Timer?

    // MARK: - Public Methods
    public func play(recording: Recording) async {
        // 既存の再生を停止
        await stop()

        currentRecordingId = recording.id
        duration = recording.duration.seconds
        isPlaying = true

        startPositionTracking()

        do {
            try await audioPlayer.play(url: recording.fileURL)
            // 自然終了時の処理
            handlePlaybackComplete()
        } catch {
            handlePlaybackError(error)
        }
    }

    public func pause() {
        audioPlayer.pause()
        isPlaying = false
        stopPositionTracking()
    }

    public func resume() {
        audioPlayer.resume()
        isPlaying = true
        startPositionTracking()
    }

    public func stop() async {
        await audioPlayer.stop()
        isPlaying = false
        currentTime = 0.0
        currentRecordingId = nil
        stopPositionTracking()
    }

    public func seek(to time: Double) {
        currentTime = time
        audioPlayer.seek(to: time)
    }
}

// ViewModel での使用例
@MainActor
public class RecordingListViewModel: ObservableObject {
    private let playbackService: PlaybackService

    // 状態はServiceから直接公開
    var isPlaying: Bool { playbackService.isPlaying }
    var currentTime: Double { playbackService.currentTime }
    var currentRecordingId: RecordingId? { playbackService.currentRecordingId }

    func selectAndPlay(_ recording: Recording) async {
        await playbackService.play(recording: recording)
    }
}

// View での onDisappear 処理がシンプルに
.onDisappear {
    Task {
        await playbackService.stop()
    }
}
```

## メリット

1. **Single Source of Truth**
   - 再生状態が一箇所で管理される
   - 状態の不整合が発生しない

2. **画面遷移時の処理がシンプル**
   - `stop()` を呼ぶだけで完結
   - 各ViewModelでの状態リセットが不要

3. **テスト容易性**
   - PlaybackServiceをモック化するだけでテスト可能
   - 状態変化の検証が容易

4. **拡張性**
   - 新しい画面追加時も同じパターン
   - 再生キュー、履歴などの機能追加が容易

## デメリット

1. **既存コードへの影響が大きい**
   - RecordingListViewModel の大幅変更
   - AnalysisViewModel の大幅変更
   - テストコードの修正

2. **DIコンテナの変更**
   - PlaybackServiceの登録
   - 依存関係の更新

## 実装時の注意点

1. **段階的移行**
   - まずPlaybackServiceを作成
   - 既存のViewModelと並行稼働
   - 動作確認後に完全移行

2. **既存テストの保護**
   - MockPlaybackServiceの作成
   - 既存テストケースの維持

## 判断基準

以下の場合にリファクタリングを検討:

- [ ] 再生機能に関するバグが再発した
- [ ] 新しい画面で再生機能が必要になった
- [ ] 再生キューやプレイリスト機能を追加する
- [ ] バックグラウンド再生を実装する

## 現在のステータス

**見送り** (2024年12月時点)

理由:
- 現在のバグ修正で問題は解決済み
- 既存機能への影響が大きい
- 過度な早期最適化を避ける (YAGNI)

## 関連コミット

- `9b77efc`: Stop playback on navigation from list and analysis pages
