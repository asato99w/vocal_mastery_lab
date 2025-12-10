# AnalysisView.swift 分割計画

## 現状分析

### ファイル概要
- **ファイル**: `VocalisStudio/Presentation/Views/AnalysisView.swift`
- **総行数**: 1,709行
- **構造体数**: 14個
- **問題**: 単一ファイルに過多な責務が集中

### 現在のファイル構造

```
AnalysisView.swift (1,709行)
├── AnalysisView (L1-280)                    # メインView
│   ├── body                                 # レイアウト切り替え
│   ├── landscapeLayout                      # 横向きレイアウト
│   ├── portraitLayout                       # 縦向きレイアウト
│   └── expandedGraphFullScreen              # 全画面表示
│
├── CompactPlaybackControl (L284-302)        # 再生コントロール（小）
├── RecordingInfoPanel (L306-401)            # 録音情報パネル（横向き）
├── RecordingInfoCompact (L405-473)          # 録音情報パネル（縦向き）
├── InfoPill (L477-494)                      # 情報ピル
│
├── StatisticsSheetView (L498-778)           # ★ 統計シート（280行）
│   ├── overallSection
│   ├── positionSection
│   ├── pitchSection
│   └── フォーマットヘルパー
│
├── DeviationBarView (L782-831)              # 偏差バー
├── StatisticsSectionView (L835-857)         # 統計セクション
├── StatisticsRow (L861-879)                 # 統計行
├── InfoRow (L882-896)                       # 情報行
│
├── PlaybackControl (L901-967)               # 再生コントロール
├── SpectrogramView (L971-1181)              # ★ スペクトログラム（210行）
├── PitchAnalysisView (L1186-1602)           # ★ ピッチグラフ（416行）
│
└── Preview (L1607-1709)                     # プレビュー
```

## 分割計画

### Phase 1: 統計関連コンポーネント分離（優先度: 高）

**新規ファイル**: `Analysis/StatisticsComponents.swift`

| コンポーネント | 行数 | 説明 |
|---------------|------|------|
| `StatisticsSheetView` | 280 | メイン統計シート |
| `DeviationBarView` | 50 | 偏差表示バー |
| `StatisticsSectionView` | 23 | セクションコンテナ |
| `StatisticsRow` | 19 | 統計行 |

**効果**: 約372行を分離

### Phase 2: 録音情報コンポーネント分離（優先度: 中）

**新規ファイル**: `Analysis/RecordingInfoComponents.swift`

| コンポーネント | 行数 | 説明 |
|---------------|------|------|
| `RecordingInfoPanel` | 96 | 横向き用情報パネル |
| `RecordingInfoCompact` | 69 | 縦向き用情報パネル |
| `InfoPill` | 18 | 情報ピル |
| `InfoRow` | 15 | 情報行 |

**効果**: 約198行を分離

### Phase 3: 再生コントロール分離（優先度: 中）

**新規ファイル**: `Analysis/PlaybackComponents.swift`

| コンポーネント | 行数 | 説明 |
|---------------|------|------|
| `PlaybackControl` | 67 | メイン再生コントロール |
| `CompactPlaybackControl` | 19 | コンパクト版 |

**効果**: 約86行を分離

### Phase 4: ビジュアライゼーション分離（優先度: 低）

**Option A: 既存ファイルへの統合**
- `SpectrogramView` → 既存の `Components/Spectrogram/` ディレクトリへ移動
- `PitchAnalysisView` → 既存の `Components/PitchGraph/` ディレクトリへ移動

**Option B: 新規ファイル作成**
- `Analysis/SpectrogramView.swift` (210行)
- `Analysis/PitchAnalysisView.swift` (416行)

**効果**: 約626行を分離

## 分割後の構造

```
Presentation/Views/
├── Analysis/
│   ├── AnalysisView.swift           # ~280行（メインView + レイアウト）
│   ├── StatisticsComponents.swift   # ~370行（統計関連）
│   ├── RecordingInfoComponents.swift # ~200行（録音情報）
│   └── PlaybackComponents.swift     # ~90行（再生コントロール）
│
└── Components/
    ├── Spectrogram/
    │   └── SpectrogramView.swift    # 移動または統合
    └── PitchGraph/
        └── PitchAnalysisView.swift  # 移動または統合
```

## 分割効果の見積もり

| フェーズ | 分離行数 | 残り行数 | 削減率 |
|---------|---------|---------|--------|
| 現状 | - | 1,709 | - |
| Phase 1 | 372 | 1,337 | 22% |
| Phase 2 | 198 | 1,139 | 33% |
| Phase 3 | 86 | 1,053 | 38% |
| Phase 4 | 626 | 427 | 75% |

**最終目標**: AnalysisView.swift を約280-430行に削減

## 実装手順

### Phase 1 詳細手順

1. **新規ファイル作成**
   ```
   VocalisStudio/Presentation/Views/Analysis/StatisticsComponents.swift
   ```

2. **移動対象**
   - `StatisticsSheetView` (L498-778)
   - `DeviationBarView` (L782-831)
   - `StatisticsSectionView` (L835-857)
   - `StatisticsRow` (L861-879)

3. **import文追加**
   ```swift
   import SwiftUI
   import VocalisDomain
   ```

4. **テスト確認**
   - ビルド成功確認
   - UIテスト実行（AnalysisUITests）

### 依存関係の確認

```
StatisticsSheetView
├── RecordingStatistics (VocalisDomain)
├── ColorPalette (Theme)
├── Recording (VocalisDomain)
└── DeviationBarView, StatisticsSectionView, StatisticsRow (内部)

RecordingInfoPanel / RecordingInfoCompact
├── Recording (VocalisDomain)
├── ScaleSettings (VocalisDomain)
├── ColorPalette (Theme)
└── InfoPill (内部)

SpectrogramView
├── SpectrogramData (VocalisDomain)
├── SpectrogramRenderer, SpectrogramCoordinateSystem (Components)
├── SpectrogramScrollManager (Components)
└── ColorPalette (Theme)

PitchAnalysisView
├── PitchAnalysisData, ScaleSettings (VocalisDomain)
├── PitchGraphRenderer, PitchGraphCoordinateSystem (Components)
├── SpectrogramScrollManager (再利用)
└── ColorPalette (Theme)
```

## リスク評価

| リスク | 影響度 | 対策 |
|--------|--------|------|
| ビルドエラー | 中 | 段階的な移動、各フェーズでテスト |
| プレビュー壊れ | 低 | プレビュー専用コードは最後に移動 |
| テスト失敗 | 低 | UIテストは View 名を参照しないため影響小 |
| パフォーマンス | 低 | 分割は実行時パフォーマンスに影響なし |

## 推奨事項

1. **Phase 1から開始**: 最も独立性が高く、影響範囲が限定的
2. **各フェーズでテスト実行**: リグレッション防止
3. **コミットは小さく**: 各コンポーネント移動ごとにコミット
4. **プレビューは最後**: デバッグ用コードは最後に対応

## 補足: try! の修正

Phase 4 実装時に以下を修正:

```swift
// 現在 (L1692-1695)
scaleSettings: ScaleSettings(
    startNote: try! MIDINote(60),
    endNote: try! MIDINote(72),
    tempo: try! Tempo(secondsPerNote: 0.5)
)

// 修正案
scaleSettings: ScaleSettings.preview  // 静的プロパティとして定義
```

---

## 実装完了報告

### 実装日: 2025-11-30

### 実装結果

| フェーズ | 新規ファイル | 行数 | ステータス |
|---------|-------------|------|-----------|
| Phase 1 | StatisticsComponents.swift | 396 | ✅ 完了 |
| Phase 2 | RecordingInfoComponents.swift | 202 | ✅ 完了 |
| Phase 3 | PlaybackComponents.swift | 101 | ✅ 完了 |
| Phase 4 | VisualizationComponents.swift | 645 | ✅ 完了 |

### 分割後のファイル構造

```
Presentation/Views/
├── AnalysisView.swift              # 386行 (1,709行から77%削減)
└── Analysis/
    ├── StatisticsComponents.swift   # 396行
    │   ├── StatisticsSheetView
    │   ├── DeviationBarView
    │   ├── StatisticsSectionView
    │   └── StatisticsRow
    │
    ├── RecordingInfoComponents.swift # 202行
    │   ├── RecordingInfoPanel
    │   ├── RecordingInfoCompact
    │   └── InfoPill
    │
    ├── PlaybackComponents.swift     # 101行
    │   ├── CompactPlaybackControl
    │   └── PlaybackControl
    │
    └── VisualizationComponents.swift # 645行
        ├── SpectrogramView
        └── PitchAnalysisView
```

### 合計行数
- **リファクタリング前**: 1,709行 (単一ファイル)
- **リファクタリング後**: 1,730行 (5ファイルに分散)
  - AnalysisView.swift: 386行 (メインView + レイアウト + Preview)
  - StatisticsComponents.swift: 396行
  - RecordingInfoComponents.swift: 202行
  - PlaybackComponents.swift: 101行
  - VisualizationComponents.swift: 645行

### ビルド確認
- ✅ 全フェーズでビルド成功確認済み
- ✅ 既存機能に影響なし

---

作成日: 2025-11-30
完了日: 2025-11-30
状態: ✅ 完了
