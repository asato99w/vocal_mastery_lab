# ピッチ偏差分析機能 設計ドキュメント

## 概要

録音された声のピッチを、スケール再生時の目標音程と比較し、偏差（セント値）を表示する機能。

### 目的
- ユーザーが自分の歌唱精度を客観的に把握できる
- 各ノートに対してどれだけ正確に歌えたかをセント単位で表示
- 時系列でのピッチ変動と目標音程の関係を可視化

## 設計方針

### アプローチ: 実測タイムスタンプ記録

**選択理由**: タイミングをデータから計算するアプローチは後々複雑になるため避ける。録音時に実際の再生タイムスタンプを記録するアプローチを採用。

```
❌ 計算アプローチ（採用しない）
ScaleSettings → calculateTimingFromSettings() → 理論上のタイミング

✅ 実測アプローチ（採用）
再生中 → recordActualTimestamp() → 実際のタイミング
```

### メリット
1. **正確性**: 実際の再生タイミングを使用するため、計算誤差がない
2. **シンプル**: 後から複雑なタイミング計算ロジックを実装する必要がない
3. **拡張性**: テンポ変更やルバート対応も自然に対応可能
4. **デバッグ容易**: 実測データなので問題の特定が容易

## データ構造

### ScalePlaybackEvent（新規作成）

```swift
/// スケール再生中の各イベント（ノート開始/終了）を記録
public struct ScalePlaybackEvent: Codable, Equatable {
    /// 録音開始からの経過時間（秒）
    public let timestamp: TimeInterval

    /// 再生されたMIDIノート
    public let note: MIDINote

    /// イベントタイプ
    public let eventType: EventType

    public enum EventType: String, Codable {
        case noteStart      // ノート開始
        case noteEnd        // ノート終了
        case chordStart     // コード開始
        case chordEnd       // コード終了
    }
}
```

### ScalePlaybackTimeline（新規作成）

```swift
/// 録音セッション中のすべての再生イベントのコレクション
public struct ScalePlaybackTimeline: Codable, Equatable {
    /// すべての再生イベント（時系列順）
    public let events: [ScalePlaybackEvent]

    /// 録音開始時刻（基準点）
    public let recordingStartTime: Date

    /// 指定時刻における目標ノートを取得
    public func targetNote(at timestamp: TimeInterval) -> MIDINote? {
        // timestamp時点でアクティブなノートを検索
        // noteStart <= timestamp < noteEnd となるイベントを探す
    }

    /// 指定時刻における目標周波数を取得
    public func targetFrequency(at timestamp: TimeInterval) -> Double? {
        guard let note = targetNote(at: timestamp) else { return nil }
        return note.frequency
    }
}
```

### Recording Entity 拡張

```swift
public struct Recording: Equatable, Identifiable, Codable, Hashable {
    // 既存プロパティ
    public let id: RecordingId
    public let fileURL: URL
    public let createdAt: Date
    public let duration: Duration
    public let scaleSettings: ScaleSettings?
    public var title: String?

    // 新規追加
    /// スケール再生のタイムライン（nil = スケールなし録音）
    public let playbackTimeline: ScalePlaybackTimeline?
}
```

## 実装箇所

### 1. AVAudioEngineScalePlayer 拡張

タイムスタンプ記録機能を追加:

```swift
public class AVAudioEngineScalePlayer: ScalePlayerProtocol {
    // 新規追加
    private var playbackEvents: [ScalePlaybackEvent] = []
    private var recordingStartTime: Date?

    /// タイムスタンプ記録を開始
    public func startTimestampRecording(recordingStartTime: Date) {
        self.recordingStartTime = recordingStartTime
        self.playbackEvents = []
    }

    /// 記録されたタイムラインを取得
    public func getPlaybackTimeline() -> ScalePlaybackTimeline? {
        guard let startTime = recordingStartTime else { return nil }
        return ScalePlaybackTimeline(
            events: playbackEvents,
            recordingStartTime: startTime
        )
    }

    // playNote() 内で記録
    private func playNote(_ note: MIDINote, duration: TimeInterval) async throws {
        let timestamp = Date().timeIntervalSince(recordingStartTime ?? Date())

        // ノート開始を記録
        playbackEvents.append(ScalePlaybackEvent(
            timestamp: timestamp,
            note: note,
            eventType: .noteStart
        ))

        sampler.startNote(note.value, withVelocity: 64, onChannel: 0)
        try await Task.sleep(nanoseconds: UInt64(duration * 0.9 * 1_000_000_000))
        sampler.stopNote(note.value, onChannel: 0)

        // ノート終了を記録
        let endTimestamp = Date().timeIntervalSince(recordingStartTime ?? Date())
        playbackEvents.append(ScalePlaybackEvent(
            timestamp: endTimestamp,
            note: note,
            eventType: .noteEnd
        ))

        try await Task.sleep(nanoseconds: UInt64(duration * 0.1 * 1_000_000_000))
    }
}
```

### 2. StartRecordingWithScaleUseCase 連携

録音開始時にタイムスタンプ記録を開始:

```swift
func execute(settings: ScaleSettings) async throws -> RecordingSession {
    // 録音開始時刻を取得
    let recordingStartTime = Date()

    // ScalePlayerにタイムスタンプ記録開始を指示
    scalePlayer.startTimestampRecording(recordingStartTime: recordingStartTime)

    // ... 既存の録音開始ロジック
}
```

### 3. StopRecordingUseCase 連携

録音停止時にタイムラインを取得してRecordingに保存:

```swift
func execute() async throws -> Recording {
    // タイムラインを取得
    let timeline = scalePlayer.getPlaybackTimeline()

    // Recordingを作成（タイムライン付き）
    let recording = Recording(
        fileURL: recordingURL,
        duration: duration,
        scaleSettings: settings,
        playbackTimeline: timeline  // 新規追加
    )

    // ... 保存処理
}
```

## ピッチ偏差計算

### 計算式

```swift
/// セント値での偏差計算
/// 正の値 = 検出ピッチが高い
/// 負の値 = 検出ピッチが低い
func calculateDeviation(detected: Double, expected: Double) -> Double {
    return 1200.0 * log2(detected / expected)
}
```

### 実装例（AnalysisView）

```swift
// 各ピッチポイントについて偏差を計算
for pitchPoint in pitchData.points {
    // その時点での目標周波数を取得
    guard let targetFreq = timeline.targetFrequency(at: pitchPoint.timestamp) else {
        continue // スケール再生外の時点はスキップ
    }

    // 偏差を計算（セント）
    let deviation = 1200.0 * log2(pitchPoint.frequency / targetFreq)

    // 許容範囲内かチェック（例: ±50セント）
    let isAccurate = abs(deviation) <= 50
}
```

## UI表示

### カラオケ風音程バー表示（メインUI）

カラオケの採点画面のように、目標音程を「バー」として表示し、ユーザーの声のピッチがバーに対してどれだけ一致しているかを視覚的に表現。

#### 基本コンセプト

```
時間 →
      ┌────────────────────────────────────────────────────────┐
  E4  │                          ████████                      │ ← 目標バー（E4）
      │                        ∿∿████∿∿∿∿                      │ ← 検出ピッチ（重なり=緑）
  D4  │              ████████████                              │ ← 目標バー（D4）
      │            ∿∿██████████∿∿                              │
  C4  │  ████████████                                          │ ← 目標バー（C4）
      │∿∿██████████∿∿                                          │
      └────────────────────────────────────────────────────────┘
           0s        1s        2s        3s        4s
```

#### 詳細レイアウト

```
┌──────────────────────────────────────────────────────────────┐
│  分析結果                                           [共有] ▼ │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─ 音程バー表示 ──────────────────────────────────────────┐ │
│  │                                                        │ │
│  │ G4 ─┼─────────────────────────────────────────────────│ │
│  │     │                                    ▓▓▓▓▓▓▓▓     │ │ ← 目標（グレー）
│  │     │                                  ●━━━━━━━●      │ │ ← 検出（色付き）
│  │ F4 ─┼─────────────────────────────────────────────────│ │
│  │     │                         ▓▓▓▓▓▓▓▓               │ │
│  │     │                       ●━━━━━━━●                 │ │
│  │ E4 ─┼─────────────────────────────────────────────────│ │
│  │     │              ▓▓▓▓▓▓▓▓                          │ │
│  │     │            ●━━━━━━━●                            │ │
│  │ D4 ─┼─────────────────────────────────────────────────│ │
│  │     │   ▓▓▓▓▓▓▓▓                                     │ │
│  │     │ ●━━━━━━━●                                       │ │
│  │ C4 ─┼─────────────────────────────────────────────────│ │
│  │     │                                                 │ │
│  │     └─────┴─────┴─────┴─────┴─────┴─────┴─────┴──────│ │
│  │          1s    2s    3s    4s    5s    6s    7s       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ スコア ───────────────────────────────────────────────┐ │
│  │                                                        │ │
│  │   🎯 音程精度: 87%        📊 平均偏差: +8 セント       │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─ ノート別評価 ─────────────────────────────────────────┐ │
│  │  C4  ████████████████████ 95% 優秀                    │ │
│  │  D4  ██████████████░░░░░░ 78% 良好                    │ │
│  │  E4  ████████████████████ 92% 優秀                    │ │
│  │  F4  ████████████░░░░░░░░ 65% 許容範囲                │ │
│  │  G4  ██████████████████░░ 88% 優秀                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

#### 視覚要素の説明

| 要素 | 表示 | 説明 |
|-----|------|------|
| 目標バー | `▓▓▓▓▓▓▓▓` (グレー) | スケールで再生されたノートの区間 |
| 検出ピッチ（一致） | `━━━━━━` (緑) | 目標に近い（±25セント以内） |
| 検出ピッチ（ずれ） | `━━━━━━` (黄/赤) | 目標からずれている |
| 無音区間 | (空白) | 声が検出されなかった区間 |

#### 色による精度表示

```
目標バーと検出ピッチの重なり具合で色が変化：

  ████████████  ← 目標バー（常にグレー）
  ━━━━━━━━━━━━  ← 検出（色で精度表示）

  ±10セント以内  → 🟢 緑（完璧）
  ±25セント以内  → 🔵 青（良好）
  ±50セント以内  → 🟡 黄（許容範囲）
  ±50セント超    → 🔴 赤（要改善）
```

#### インタラクション

```
1. 横スクロール
   - 長い録音は左右にスワイプで時間軸をスクロール
   - ピンチでズームイン/アウト

2. タップで詳細表示
   - バーをタップすると、そのノートの詳細情報をポップオーバー表示
   ┌─────────────────┐
   │ D4 (293.7 Hz)   │
   │ 検出: 298.2 Hz  │
   │ 偏差: +26 セント │
   │ 評価: 良好 🔵    │
   └─────────────────┘

3. 再生連動
   - 録音を再生中、現在位置にインジケーター表示
   - 再生位置に合わせてビューが自動スクロール
```

#### SwiftUI実装イメージ

```swift
struct PitchBarView: View {
    let timeline: ScalePlaybackTimeline
    let pitchData: PitchAnalysisData

    var body: some View {
        ScrollView(.horizontal) {
            ZStack {
                // 背景グリッド（ノート名）
                PitchGridBackground()

                // 目標バー（スケールのノート区間）
                ForEach(timeline.noteSegments) { segment in
                    TargetNoteBar(segment: segment)
                }

                // 検出ピッチライン
                PitchDetectionPath(
                    pitchData: pitchData,
                    timeline: timeline
                )
            }
        }
    }
}

struct TargetNoteBar: View {
    let segment: NoteSegment

    var body: some View {
        Rectangle()
            .fill(Color.gray.opacity(0.3))
            .frame(
                width: segment.duration * pixelsPerSecond,
                height: noteBarHeight
            )
            .position(
                x: segment.startTime * pixelsPerSecond,
                y: pitchToY(segment.note.frequency)
            )
    }
}
```

### 色分け基準

| 偏差範囲 | 評価 | 色 |
|---------|------|-----|
| ±10セント以内 | 優秀 | 緑 |
| ±25セント以内 | 良好 | 青 |
| ±50セント以内 | 許容範囲 | 黄 |
| ±50セント超 | 要改善 | 赤 |

## ファイル構成

```
VocalisDomain/
├── ValueObjects/
│   └── ScalePlaybackEvent.swift      # 新規作成
├── Entities/
│   ├── ScalePlaybackTimeline.swift   # 新規作成
│   └── Recording.swift               # 拡張
└── RepositoryInterfaces/
    └── ScalePlayerProtocol.swift     # 拡張

VocalisStudio/
├── Infrastructure/Audio/
│   └── AVAudioEngineScalePlayer.swift  # 拡張
├── Application/UseCases/
│   ├── StartRecordingWithScaleUseCase.swift  # 拡張
│   └── StopRecordingUseCase.swift            # 拡張
└── Presentation/Views/Analysis/
    └── PitchDeviationView.swift      # 新規作成
```

## 実装順序

1. **Phase 1: データ構造** (VocalisDomain)
   - [ ] ScalePlaybackEvent ValueObject
   - [ ] ScalePlaybackTimeline Entity
   - [ ] Recording拡張（playbackTimelineプロパティ追加）

2. **Phase 2: タイムスタンプ記録** (Infrastructure)
   - [ ] ScalePlayerProtocol拡張
   - [ ] AVAudioEngineScalePlayer拡張

3. **Phase 3: UseCase連携** (Application)
   - [ ] StartRecordingWithScaleUseCase拡張
   - [ ] StopRecordingUseCase拡張

4. **Phase 4: UI実装** (Presentation)
   - [ ] PitchDeviationView作成
   - [ ] AnalysisView統合

## 考慮事項

### 既存データとの互換性

- `playbackTimeline`はOptionalなので、既存の録音データは`nil`として処理
- 新しい録音のみタイムラインデータが保存される

### パフォーマンス

- イベント数は通常100〜500程度（1分の録音で約50ノート × 2イベント）
- メモリ・ストレージへの影響は軽微

### テスト戦略

- ScalePlaybackEvent: 初期化、Codable変換
- ScalePlaybackTimeline: targetNote検索、境界条件
- AVAudioEngineScalePlayer: タイムスタンプ記録の正確性
- E2E: 録音→保存→読み込み→分析の一連の流れ

---

## 設計レビュー（2024-11-27）

### ✅ カラオケ風UI実現に必要なデータ

| 必要なデータ | 現在の設計 | 判定 |
|-------------|-----------|------|
| 目標バーの開始時刻 | `ScalePlaybackEvent.timestamp` (noteStart) | ✅ OK |
| 目標バーの終了時刻 | `ScalePlaybackEvent.timestamp` (noteEnd) | ✅ OK |
| 目標バーのノート（音程） | `ScalePlaybackEvent.note: MIDINote` | ✅ OK |
| 検出ピッチの時刻 | `PitchAnalysisData.timeStamps` | ✅ 既存 |
| 検出ピッチの周波数 | `PitchAnalysisData.frequencies` | ✅ 既存 |
| 検出の信頼度 | `PitchAnalysisData.confidences` | ✅ 既存 |

### ⚠️ 改善が必要な点

#### 1. ScalePlaybackTimelineに「ノート区間」取得メソッドが必要

現在の設計では`events`（開始/終了イベントの配列）のみ。
カラオケ風UIで「バー」を描画するには、開始〜終了をペアにした**セグメント**形式が便利。

```swift
/// 追加すべきメソッド
public struct ScalePlaybackTimeline {
    // ... 既存 ...

    /// ノート再生区間のリスト（バー描画用）
    public var noteSegments: [NoteSegment] {
        // noteStartとnoteEndをペアにして返す
    }
}

/// ノート再生区間（1つの目標バー）
public struct NoteSegment: Identifiable {
    public let id: UUID
    public let note: MIDINote          // 音程（Y軸位置）
    public let startTime: TimeInterval  // 開始時刻（X軸左端）
    public let endTime: TimeInterval    // 終了時刻（X軸右端）

    public var duration: TimeInterval { endTime - startTime }
    public var frequency: Double { note.frequency }
}
```

#### 2. 既存PitchAnalysisDataとの統合方法

現在の`PitchAnalysisData.targetNotes`は使われていない。
カラオケ風UIでは、**時刻ベースで目標ノートを検索**する必要がある。

```swift
// 現在のPitchAnalysisView（L924付近）
if let settings = scaleSettings {
    let targetFrequencies = getTargetFrequencies(from: settings)
    // → 全目標周波数を取得しているが、時間軸との対応がない
}

// 必要な実装
if let timeline = recording.playbackTimeline {
    // 各検出ピッチに対して、その時点の目標ノートを特定
    for (i, timestamp) in pitchData.timeStamps.enumerated() {
        if let targetNote = timeline.targetNote(at: timestamp) {
            let deviation = calculateDeviation(
                detected: Double(pitchData.frequencies[i]),
                expected: targetNote.frequency
            )
        }
    }
}
```

#### 3. UIコンポーネントの追加

現在の`PitchAnalysisView`を拡張または新規Viewを作成:

```
VocalisStudio/Presentation/Views/Analysis/
├── AnalysisView.swift           # 既存（メインコンテナ）
├── PitchAnalysisView.swift      # 既存（折れ線グラフ）
├── PitchBarView.swift           # 新規（カラオケ風バー表示）
├── TargetNoteBar.swift          # 新規（目標バー部品）
├── PitchDeviationPath.swift     # 新規（検出ピッチ描画）
└── DeviationScoreView.swift     # 新規（スコア/評価表示）
```

### 📋 更新した実装順序

1. **Phase 1: データ構造** (VocalisDomain)
   - [ ] ScalePlaybackEvent ValueObject
   - [ ] NoteSegment ValueObject（バー描画用）
   - [ ] ScalePlaybackTimeline Entity（noteSegments計算プロパティ含む）
   - [ ] Recording拡張（playbackTimelineプロパティ追加）

2. **Phase 2: タイムスタンプ記録** (Infrastructure)
   - [ ] ScalePlayerProtocol拡張
   - [ ] AVAudioEngineScalePlayer拡張

3. **Phase 3: UseCase連携** (Application)
   - [ ] StartRecordingWithScaleUseCase拡張
   - [ ] StopRecordingUseCase拡張

4. **Phase 4: UI実装** (Presentation)
   - [ ] PitchBarView作成（カラオケ風メイン表示）
   - [ ] TargetNoteBar作成（目標バー）
   - [ ] PitchDeviationPath作成（検出ピッチ描画）
   - [ ] DeviationScoreView作成（スコア表示）
   - [ ] AnalysisView統合（既存PitchAnalysisViewと切り替え or 追加タブ）

### 🔍 実装時の注意点

1. **既存機能への影響を最小化**
   - `Recording.playbackTimeline`はOptionalなので後方互換性あり
   - 既存の`PitchAnalysisView`は残し、新UIは別Viewとして追加

2. **パフォーマンス考慮**
   - `noteSegments`は計算プロパティとしてキャッシュ推奨
   - 長い録音（数分）でもスムーズにスクロールできるよう、
     表示範囲外のバーはCanvas描画をスキップ

3. **テスト戦略**
   - NoteSegment生成のユニットテスト（境界条件）
   - 偏差計算の精度テスト
   - UIテストは既存パターンに従う

---

作成日: 2024-11-27
ステータス: 設計レビュー完了、実装待ち
