# 音程バー表示 + ピッチデータ永続化 統合設計

## 概要

次の2つの機能の実装順序と統合方法についてまとめる：

1. **音程バー表示（カラオケ風UI）** - ピッチ偏差分析の可視化
2. **ピッチデータ永続化** - 分析結果のファイルベースキャッシュ

## 現状分析

### 既存アーキテクチャ

```
AnalysisView
    ↓
AnalysisViewModel
    ↓
AnalyzeRecordingUseCase
    ↓ (キャッシュチェック)
AnalysisCache (インメモリ・LRU・最大10件)
    ↓ (キャッシュミス時)
AudioFileAnalyzer
    ├─ analyzePitch() → PitchAnalysisData   [重い: YIN O(n²)]
    └─ analyzeSpectrogram() → SpectrogramData [軽い: FFT O(n log n)]
```

### 既存データ構造

| 構造体 | 用途 | 永続化 |
|--------|------|--------|
| `AnalysisResult` | ピッチ + スペクトログラム + スケール設定 | ❌ インメモリのみ |
| `PitchAnalysisData` | 検出ピッチデータ | ❌ インメモリのみ |
| `SpectrogramData` | 周波数スペクトル | ❌ インメモリのみ |
| `Recording` | 録音メタデータ | ✅ UserDefaults |

### 問題点

1. **分析結果が消失** - アプリ終了でキャッシュがクリア、毎回YIN再分析（1-2秒）
2. **目標音程の時間対応がない** - 検出ピッチと目標ノートの時間軸での対応付けがない
3. **カラオケUIに必要なデータがない** - ノート再生区間（開始〜終了）の情報がない

## 機能間の依存関係

```
┌─────────────────────────────────────────────────────────────────┐
│                         機能A: 音程バー表示                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 必要データ:                                                │ │
│  │   1. ScalePlaybackTimeline (目標ノートの時刻情報)          │ │
│  │   2. PitchAnalysisData (検出ピッチ)                       │ │
│  │   3. NoteSegment (バー描画用の区間データ)                  │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    PitchAnalysisData を使用
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       機能B: ピッチデータ永続化                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 対象:                                                      │ │
│  │   1. PitchAnalysisData (YIN分析結果)                      │ │
│  │   2. [将来] ScalePlaybackTimeline                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 依存関係の結論

**音程バー表示 は ピッチデータ永続化 に依存しない**（逆も同様）

- 音程バー表示: `PitchAnalysisData`を使用するが、データの取得元（メモリ/ファイル）は関係ない
- ピッチデータ永続化: UIの表示方法に依存しない

**→ どちらを先に実装しても問題なし**

## 推奨実装順序

### オプション1: 永続化 → 音程バー（推奨）

```
Phase 1: ピッチデータ永続化
├── 1.1 Codable対応 (MIDINote, PitchAnalysisData)
├── 1.2 FilePitchDataCache 作成
├── 1.3 AnalyzeRecordingUseCase 更新
└── 1.4 テスト作成・検証

Phase 2: 音程バー表示
├── 2.1 ScalePlaybackEvent/Timeline データ構造
├── 2.2 AVAudioEngineScalePlayer タイムスタンプ記録
├── 2.3 UseCase連携
└── 2.4 PitchBarView UI実装
```

**メリット**:
- 永続化は独立した小さな変更（リスク低）
- 永続化完了後は分析が高速化し、UI開発時の体験が向上
- 後から音程バーUIを追加する際、既存の永続化をそのまま活用

### オプション2: 音程バー → 永続化

```
Phase 1: 音程バー表示
├── 1.1 ScalePlaybackEvent/Timeline データ構造
├── 1.2 AVAudioEngineScalePlayer タイムスタンプ記録
├── 1.3 UseCase連携
└── 1.4 PitchBarView UI実装

Phase 2: ピッチデータ永続化
├── 2.1 Codable対応
├── 2.2 FilePitchDataCache 作成
├── 2.3 AnalyzeRecordingUseCase 更新
└── 2.4 ScalePlaybackTimeline も永続化対象に追加
```

**メリット**:
- ユーザー価値の高い機能（UI）を先に提供
- 永続化時にScalePlaybackTimelineも一緒に永続化できる

## 統合時の考慮事項

### 1. ScalePlaybackTimeline の永続化

音程バー表示を先に実装した場合、永続化フェーズで追加対応が必要：

```swift
/// 永続化対象（Phase 2で追加）
public struct PersistedAnalysisData: Codable {
    let pitchData: PitchAnalysisData
    let playbackTimeline: ScalePlaybackTimeline?  // 音程バー用
}
```

### 2. 既存Recordingとの整合性

`Recording`エンティティは現在`playbackTimeline`を持っていない。
音程バー実装時に追加が必要：

```swift
public struct Recording: Equatable, Identifiable, Codable, Hashable {
    // 既存
    public let id: RecordingId
    public let fileURL: URL
    public let createdAt: Date
    public let duration: Duration
    public let scaleSettings: ScaleSettings?
    public var title: String?

    // 新規追加（音程バー用）
    public let playbackTimeline: ScalePlaybackTimeline?
}
```

### 3. キャッシュの整合性

```
┌─────────────────────────────────────────────────────────────┐
│                    キャッシュレイヤー構造                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: インメモリキャッシュ（AnalysisCache）               │
│           - 全AnalysisResult保持                            │
│           - LRU・最大10件                                   │
│           - アプリ終了で消失                                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: ファイルキャッシュ（FilePitchDataCache）           │
│           - PitchAnalysisDataのみ保持                       │
│           - 永続（アプリ再起動後も存続）                      │
│           - 容量: 約12KB/録音                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 録音メタデータ（Recording in UserDefaults）        │
│           - playbackTimeline含む                            │
│           - 永続（Recordingと同じライフサイクル）             │
└─────────────────────────────────────────────────────────────┘
```

## 詳細実装計画

### Phase 1: ピッチデータ永続化（約2時間）

#### 1.1 Codable対応 (20分)

**ファイル**: `VocalisDomain/Sources/VocalisDomain/ValueObjects/`

```
- MIDINote.swift: Codable追加（既存の Codable extension を確認）
- PitchAnalysisData.swift: Codable追加
```

#### 1.2 FilePitchDataCache (20分)

**ファイル**: `VocalisStudio/Infrastructure/Analysis/FilePitchDataCache.swift`

```swift
/// File-based persistent cache for pitch analysis data
public class FilePitchDataCache {
    private let cacheDirectory: URL  // Documents/PitchCache

    func get(_ id: RecordingId) -> PitchAnalysisData?
    func set(_ id: RecordingId, pitchData: PitchAnalysisData)
    func delete(_ id: RecordingId)
    func clearAll()
}
```

#### 1.3 AnalyzeRecordingUseCase更新 (15分)

**ファイル**: `VocalisStudio/Application/UseCases/AnalyzeRecordingUseCase.swift`

```
キャッシュ優先度:
1. インメモリキャッシュ（フル結果） → あればそのまま返却
2. ファイルキャッシュ（ピッチのみ） → あればスペクトログラムのみ再計算
3. フル分析実行 → ピッチ結果をファイルキャッシュに保存
```

#### 1.4 関連更新 (30分)

- AudioFileAnalyzer: `analyzePitchOnly()`, `analyzeSpectrogramOnly()` 追加
- FileRecordingRepository: 削除時にキャッシュも削除
- DependencyContainer: FilePitchDataCache 注入
- テスト作成

### Phase 2: 音程バー表示（約4時間）

#### 2.1 データ構造 (40分)

**新規ファイル**: `VocalisDomain/`

```
- ValueObjects/ScalePlaybackEvent.swift
- ValueObjects/NoteSegment.swift
- Entities/ScalePlaybackTimeline.swift
- Recording.swift 拡張（playbackTimeline追加）
```

#### 2.2 タイムスタンプ記録 (40min)

**拡張ファイル**: `VocalisStudio/Infrastructure/Audio/`

```
- ScalePlayerProtocol.swift: タイムスタンプ記録メソッド追加
- AVAudioEngineScalePlayer.swift: 実装
```

#### 2.3 UseCase連携 (30分)

**拡張ファイル**: `VocalisStudio/Application/UseCases/`

```
- StartRecordingWithScaleUseCase: タイムスタンプ記録開始
- StopRecordingUseCase: タイムライン取得・Recording保存
```

#### 2.4 UI実装 (2時間)

**新規ファイル**: `VocalisStudio/Presentation/Views/Analysis/`

```
- PitchBarView.swift: メインコンテナ
- TargetNoteBar.swift: 目標バー
- PitchDeviationPath.swift: 検出ピッチ描画
- DeviationScoreView.swift: スコア表示
- AnalysisView.swift 更新（タブ/切り替え追加）
```

## テスト戦略

### Phase 1 テスト

| テストクラス | 内容 |
|-------------|------|
| `PitchAnalysisDataCodableTests` | エンコード/デコードの正確性 |
| `FilePitchDataCacheTests` | 保存/読み込み/削除/クリア |
| `AnalyzeRecordingUseCaseTests` | キャッシュヒット/ミスフロー |

### Phase 2 テスト

| テストクラス | 内容 |
|-------------|------|
| `ScalePlaybackEventTests` | イベント初期化、Codable |
| `ScalePlaybackTimelineTests` | targetNote検索、noteSegments生成 |
| `NoteSegmentTests` | 区間計算、境界条件 |
| `PitchBarViewTests` | UI表示（スナップショット） |

### 統合テスト

1. 録音 → 分析 → アプリ再起動 → キャッシュヒット確認
2. 録音削除 → キャッシュファイル削除確認
3. スケール付き録音 → 分析 → 音程バー表示確認

## 工数見積もり

| Phase | 内容 | 工数 |
|-------|------|------|
| 1 | ピッチデータ永続化 | 約2時間 |
| 2 | 音程バー表示 | 約4時間 |
| **合計** | | **約6時間** |

## リスクと対策

| リスク | 対策 |
|--------|------|
| キャッシュファイル破損 | JSONDecodeエラー時は再分析にフォールバック |
| Recording Codable互換性 | playbackTimelineはOptionalで後方互換性維持 |
| 長い録音のUI性能 | 表示範囲外のバーはCanvas描画スキップ |
| タイムスタンプ精度 | Date()使用で十分な精度（ミリ秒単位） |

## 結論と推奨

### 推奨: オプション1（永続化 → 音程バー）

**理由**:

1. **リスク分散**: 永続化は独立した小さな変更で、失敗時の影響が限定的
2. **開発体験向上**: 永続化完了後は分析が高速化し、UI開発時のイテレーションが速くなる
3. **自然な拡張**: 永続化の仕組みができれば、ScalePlaybackTimelineの永続化も自然に追加可能
4. **既存コードへの影響最小**: 永続化はインフラ層の変更が中心で、UI層への影響がない

### 次のアクション

1. `pitch-data-persistence-plan.md` に従って永続化を実装
2. 完了後、`PITCH_DEVIATION_ANALYSIS_DESIGN.md` に従って音程バーを実装
3. 必要に応じてScalePlaybackTimelineの永続化を追加

---

作成日: 2024-11-27
関連ドキュメント:
- `pitch-data-persistence-plan.md` - 永続化の詳細実装プラン
- `PITCH_DEVIATION_ANALYSIS_DESIGN.md` - 音程バーUIの詳細設計
