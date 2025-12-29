# ボーカル抽出機能 UI設計プラン

## 概要
BGMと一緒に録音した音声からボーカルトラックを分離する機能のUI設計

## 決定事項

| 項目 | 決定内容 |
|------|---------|
| 入力ソース | アプリ内録音のみ |
| 目的 | BGMからボーカルを分離して分析に使用 |
| データ管理 | 別エンティティ（ExtractedAudio） |
| 分析フロー | **パターンA（厳格）**: 抽出必須、抽出済み音声のみ分析可能 |
| アクセス方法 | RecordingListViewのメニューから |

---

## 分析フロー設計

### パターンA（採用）: 抽出必須

```
Recording(未抽出)
    │
    ├─[抽出] → VocalExtractionView → ExtractedAudio作成
    │                                      │
    │                                      ▼
    │                              Recording(抽出済み)
    │                                      │
    │                              ─[分析]→ AnalysisView
    │                                      (ExtractedAudioを分析)
    │
    └─[分析] → ❌ 不可（メニューに表示しない）
```

**メニュー表示ルール:**
- 未抽出: 「🎤 ボーカル抽出」表示、「📊 分析」非表示
- 抽出済み: 「📊 分析」表示、「🎤 再抽出」表示

### 将来拡張: パターンB（柔軟）への移行

**変更コスト: 低〜中**

| 変更箇所 | 内容 | コスト |
|----------|------|--------|
| AnalysisView入力 | `ExtractedAudio` → `Recording + ExtractedAudio?` | 低 |
| RecordingListView | メニュー条件緩和（未抽出でも分析可能に） | 低 |
| トラック選択UI | 新規追加（ボーカル/元音声） | 中 |
| AnalysisViewModel | 両方の入力型に対応 | 低〜中 |

**拡張性を確保する設計ポイント:**

```swift
// AnalysisViewの入力を抽象化しておく
protocol AnalyzableAudio {
    var fileURL: URL { get }
    var duration: Duration { get }
}

// 現在はExtractedAudioのみが準拠
extension ExtractedAudio: AnalyzableAudio {}

// 将来、Recordingも準拠させることで拡張可能
// extension Recording: AnalyzableAudio {}
```

---

## UI設計

### 全体フロー図

```
┌─────────────────────────────────────────────────────────────────┐
│                          HomeView                                │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐                       │
│   │  録音   │   │ リスト  │   │  設定   │                       │
│   └────┬────┘   └────┬────┘   └─────────┘                       │
└────────┼─────────────┼──────────────────────────────────────────┘
         │             │
         ▼             ▼
┌────────────────┐   ┌────────────────────────────────────────────┐
│  RecordingView │   │           RecordingListView                 │
│                │   │                                              │
│  🎤 録音中...  │   │  ┌────────────────────────────────────────┐│
│                │   │  │ 🎵 カラオケ練習_12/21              [⋮]││
│  [停止]        │   │  │    📅 2024/12/21 15:30  ⏱️ 3:45        ││
│                │   │  │    状態: 未抽出                        ││
└────────────────┘   │  └────────────────────────────────────────┘│
                     │                                              │
                     │  ┌────────────────────────────────────────┐│
                     │  │ 🎵 練習曲A_12/20                   [⋮]││
                     │  │    📅 2024/12/20 10:15  ⏱️ 2:30        ││
                     │  │    ✅ 抽出済み                         ││
                     │  └────────────────────────────────────────┘│
                     └────────────────────────────────────────────┘
                                        │
                     ┌──────────────────┴──────────────────┐
                     │                                      │
                     ▼                                      ▼
        ┌────────────────────────┐            ┌────────────────────────┐
        │  VocalExtractionView   │            │      AnalysisView      │
        │                        │            │                        │
        │  録音: カラオケ練習    │            │  ※抽出済み音声のみ     │
        │  ⏱️ 3:45               │            │    アクセス可能        │
        │                        │            │                        │
        │  [抽出開始]            │            │  📊 ピッチグラフ       │
        │                        │            │  📈 スペクトログラム   │
        │  処理中:               │            │                        │
        │  ████████░░░░ 60%      │            │  [▶] 再生コントロール  │
        │                        │            │                        │
        │  プレビュー:           │            └────────────────────────┘
        │  ▶ 元の音声            │
        │  ▶ ボーカル            │
        │                        │
        │  [保存] [キャンセル]   │
        └────────────────────────┘
```

### RecordingListView メニュー

| 録音状態 | メニュー項目 |
|----------|-------------|
| 未抽出 | ▶ 再生 / 🎤 ボーカル抽出 / ✏ 名前変更 / 🗑 削除 |
| 抽出済み | ▶ 再生 / 📊 分析 / 🎤 再抽出 / ✏ 名前変更 / 🗑 削除 |

### 状態バッジ

| 状態 | 表示 |
|------|------|
| 未抽出 | グレー背景「未抽出」 |
| 抽出済み | グリーン背景「✅ 抽出済み」 |

---

## データ構造

### ExtractedAudio エンティティ（新規）

```swift
public struct ExtractedAudio: Identifiable, Codable, Equatable {
    public let id: UUID
    public let sourceRecordingId: RecordingId  // 元の録音への参照
    public let type: ExtractionType            // vocal
    public let fileURL: URL
    public let createdAt: Date
    public let duration: Duration
}

public enum ExtractionType: String, Codable {
    case vocal
}
```

> **注記**: 伴奏抽出機能は未実装のため、ExtractionTypeはvocalのみ対応

### ExtractedAudioRepository プロトコル（新規）

```swift
public protocol ExtractedAudioRepositoryProtocol {
    func save(_ extractedAudio: ExtractedAudio) async throws
    func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio]
    func findAll() async throws -> [ExtractedAudio]
    func delete(_ id: UUID) async throws
}
```

### Recording エンティティ

**変更なし** - 既存のまま維持

---

## VocalExtractionView 詳細

### 状態遷移

```
┌──────────┐   選択    ┌──────────┐   開始    ┌──────────┐
│  未選択  │ ───────→ │  選択済  │ ───────→ │  処理中  │
└──────────┘          └──────────┘          └──────────┘
                            ↑                     │
                            │                     ▼
                            │              ┌──────────┐
                            └── やり直し ──│   完了   │
                                           └──────────┘
                                                │
                                                ▼ 保存
                                           ┌──────────┐
                                           │ 保存完了 │
                                           └──────────┘
```

### ViewModel 状態

```swift
enum VocalExtractionState {
    case idle
    case processing(progress: Double, stage: String)
    case completed(result: ExtractionResult)
    case error(message: String)
}

struct ExtractionResult {
    let vocalURL: URL
    let instrumentalURL: URL
    let duration: Duration
}
```

---

## 実装ファイル一覧

### 新規作成

| レイヤー | ファイルパス |
|----------|-------------|
| Domain | `Packages/VocalisDomain/Sources/VocalisDomain/Entities/ExtractedAudio.swift` |
| Domain | `Packages/VocalisDomain/Sources/VocalisDomain/Repositories/ExtractedAudioRepositoryProtocol.swift` |
| Domain | `Packages/VocalisDomain/Sources/VocalisDomain/Services/VocalExtractionServiceProtocol.swift` |
| Infrastructure | `VocalMasteryLab/Infrastructure/Repositories/ExtractedAudioRepository.swift` |
| Infrastructure | `VocalMasteryLab/Infrastructure/Audio/VocalExtractor.swift` |
| Application | `VocalMasteryLab/Application/UseCases/ExtractVocalUseCase.swift` |
| Presentation | `VocalMasteryLab/Presentation/Views/VocalExtractionView.swift` |
| Presentation | `VocalMasteryLab/Presentation/ViewModels/VocalExtractionViewModel.swift` |
| Presentation | `VocalMasteryLab/Presentation/Components/ExtractionStatusBadge.swift` |

### 変更

| ファイルパス | 変更内容 |
|-------------|---------|
| `VocalMasteryLab/Presentation/Views/RecordingListView.swift` | 抽出状態表示、メニュー項目追加 |
| `VocalMasteryLab/Presentation/Views/AnalysisView.swift` | ExtractedAudio入力対応 |
| `VocalMasteryLab/Presentation/ViewModels/RecordingListViewModel.swift` | 抽出状態取得ロジック |
| `VocalMasteryLab/App/DependencyContainer.swift` | 新規依存性の登録 |

---

## 技術検討事項

### ボーカル分離エンジン候補

| エンジン | 特徴 | iOS対応 |
|----------|------|---------|
| Demucs (Meta) | 高品質、Core ML変換可能 | ○ |
| Spleeter (Deezer) | 軽量、TensorFlow Lite | △ |
| 独自軽量モデル | カスタマイズ可能 | ○ |

### 処理時間の目安（プログレス表示用）

| 段階 | 進捗 |
|------|------|
| モデルロード | 0-10% |
| 音声読み込み | 10-20% |
| 分離処理 | 20-90% |
| 出力生成 | 90-100% |

---

## 次のステップ

1. ✅ UI配置の決定 → 録音リストからアクセス
2. ✅ データ管理方式の決定 → 別エンティティ（ExtractedAudio）
3. ✅ フロー設計 → パターンA（抽出必須）、将来B拡張可能
4. 🔲 ボーカル分離エンジンの技術調査・選定
5. 🔲 実装開始
