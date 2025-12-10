# スペクトログラム機能リファクタリングプラン

**作成日**: 2025-11-16
**対象ファイル**: `VocalisStudio/Presentation/Views/AnalysisView.swift`
**対象範囲**: SpectrogramView関連のコードと機能

## 1. 現状分析

### 1.1 ファイル構成

**AnalysisView.swift** (全1305行)

```
AnalysisView (6-261行)          - メインビュー
├─ CompactPlaybackControl (262-283行)
├─ RecordingInfoPanel (284-316行)
├─ RecordingInfoCompact (317-350行)
├─ InfoRow (351-369行)
├─ PlaybackControl (370-439行)
├─ SpectrogramView (440-967行)  ← リファクタリング対象 (527行)
├─ PitchAnalysisView (968-1277行)
└─ Preview (1278-1305行)
```

### 1.2 SpectrogramView内部構造

**SpectrogramView struct** (440-967行、約527行)

#### 状態管理
- `@State private var paperTop: CGFloat` - Y軸スクロール位置
- `@State private var lastPaperTop: CGFloat` - 前回のY軸位置
- `@State private var canvasOffsetX: CGFloat` - X軸スクロールオフセット

#### プロパティ
- `currentTime: Double` - 現在の再生時刻
- `spectrogramData: SpectrogramData?` - スペクトログラムデータ
- `isExpanded: Bool` - フルスクリーン表示フラグ
- `onExpand/onCollapse/onPlayPause/onSeek` - コールバック

#### 主要関数 (8個)

| 関数名 | 行数 | 役割 | カテゴリ |
|--------|------|------|----------|
| `calculateCanvasHeight` | 713-723 | Canvas高さ計算 | 座標系 |
| `frequencyToCanvasY` | 732-735 | 周波数→Y座標変換 | 座標系 |
| `getMaxFrequency` | 741-744 | 最大周波数取得 | 座標系 |
| `drawFrequencyLabelsOnCanvas` | 754-809 | 周波数ラベル描画 | 描画 |
| `drawSpectrogramOnCanvas` | 822-914 | スペクトログラム描画 | 描画 |
| `drawPlaceholder` | 919-921 | プレースホルダー描画 | 描画 |
| `drawPlaybackPosition` | 924-935 | 再生位置(赤線)描画 | 描画 |
| `drawSpectrogramTimeAxis` | 938-962 | 時間軸ラベル描画 | 描画 |

#### body内の主要ロジック
- Canvas計算（476-504行）
- Canvas描画とレイヤー構成（507-552行）
- スクロール初期化（587-618行）
- スクロール再初期化（620-648行）
- ジェスチャー処理（649-683行）
- 時間変更時の処理（684-696行）

### 1.3 問題点

#### 責務の肥大化
- **SpectrogramViewが複数の責務を持つ**
  - Canvas座標計算ロジック
  - スクロール状態管理
  - 描画ロジック（5種類の描画関数）
  - ジェスチャー処理
  - 初期化ロジック

#### コードの可読性
- **527行の巨大なstruct**
  - 関数が多すぎて全体構造が把握しにくい
  - body内のロジックが複雑（約200行）
  - コメントが多く必要な状態

#### テスタビリティ
- **座標計算ロジックのテストが困難**
  - View内に埋め込まれているため、単体テストができない
  - 座標変換の正確性を検証しにくい

#### 再利用性
- **描画ロジックの再利用が困難**
  - 他のビューで同様のスペクトログラム表示が必要な場合、コピペが必要
  - ロジックの共通化ができていない

## 2. リファクタリング目標

### 2.1 設計原則

- **Single Responsibility Principle (SRP)**
  - 各クラス/structは単一の責務を持つ

- **Separation of Concerns (SoC)**
  - ビューロジック、座標計算、描画ロジックを分離

- **Testability First**
  - 座標計算ロジックを独立させて単体テスト可能にする

### 2.2 具体的目標

1. **SpectrogramViewを200行以下にする**
   - 現在527行 → 目標200行（約60%削減）

2. **座標計算ロジックを独立させる**
   - 新規クラス: `SpectrogramCoordinateSystem`
   - テスト可能な構造にする

3. **描画ロジックをグループ化する**
   - 新規クラス: `SpectrogramRenderer`
   - 描画関数を集約する

4. **スクロール管理を独立させる**
   - 新規クラス: `SpectrogramScrollManager`
   - スクロール状態とロジックを分離

## 3. リファクタリング戦略

### 3.1 段階的アプローチ

**Phase 1: 座標系の抽出** (優先度: 高)
- `SpectrogramCoordinateSystem`クラスを作成
- 座標計算ロジックを移動
- 単体テストを追加

**Phase 2: 描画ロジックの抽出** (優先度: 高)
- `SpectrogramRenderer`クラスを作成
- 描画関数を移動
- 描画パラメータを構造化

**Phase 3: スクロール管理の抽出** (優先度: 中)
- `SpectrogramScrollManager`を作成
- スクロール状態管理を分離
- ジェスチャー処理を整理

**Phase 4: 定数の整理** (優先度: 低)
- `SpectrogramConstants`を作成
- マジックナンバーを定数化

### 3.2 安全性の確保

1. **既存機能の動作保証**
   - リファクタリング前後でUIテストを実行
   - スクリーンショット比較で視覚的検証

2. **段階的な移行**
   - 1つのPhaseごとにコミット
   - 各Phase後にテスト実行

3. **ロールバック可能性**
   - 各Phaseを独立したブランチで実施
   - 問題があれば即座に戻せる構造

## 4. 新規クラス設計

### 4.1 SpectrogramCoordinateSystem

**責務**: Canvas座標系の計算と変換

```swift
/// Spectrogram canvas coordinate system
/// Handles all coordinate calculations and conversions
public class SpectrogramCoordinateSystem {
    // MARK: - Constants
    private let basePixelsPerKHz: CGFloat = 576.0
    private let maxCanvasHeight: CGFloat = 10000.0
    private let maxFrequency: Double = 6000.0

    // MARK: - Canvas Dimensions

    /// Calculate canvas height based on frequency range
    /// - Parameters:
    ///   - maxFreq: Maximum frequency in Hz
    ///   - viewportHeight: Viewport height (unused, kept for compatibility)
    /// - Returns: Canvas height in points
    public func calculateCanvasHeight(maxFreq: Double, viewportHeight: CGFloat) -> CGFloat

    /// Convert frequency (Hz) to Canvas Y coordinate
    /// - Parameters:
    ///   - frequency: Frequency in Hz
    ///   - canvasHeight: Total canvas height
    ///   - maxFreq: Maximum frequency
    /// - Returns: Y coordinate in canvas space
    public func frequencyToCanvasY(frequency: Double, canvasHeight: CGFloat, maxFreq: Double) -> CGFloat

    /// Get maximum frequency for display
    /// - Returns: Maximum frequency (6kHz)
    public func getMaxFrequency() -> Double

    // MARK: - Time Axis

    /// Calculate canvas width based on data duration
    /// - Parameters:
    ///   - dataDuration: Recording duration in seconds
    ///   - leftPadding: Left padding for canvas
    /// - Returns: Canvas width in points
    public func calculateCanvasWidth(dataDuration: Double, leftPadding: CGFloat) -> CGFloat

    /// Convert time to canvas X coordinate
    /// - Parameters:
    ///   - time: Time in seconds
    ///   - pixelsPerSecond: Pixel density for time axis
    ///   - leftPadding: Left padding offset
    /// - Returns: X coordinate in canvas space
    public func timeToCanvasX(time: Double, pixelsPerSecond: CGFloat, leftPadding: CGFloat) -> CGFloat
}
```

**テスト可能性**:
- 純粋な計算ロジックのため、単体テスト容易
- 各変換関数の正確性を検証可能

### 4.2 SpectrogramRenderer

**責務**: スペクトログラムの描画処理

```swift
/// Spectrogram rendering engine
/// Handles all drawing operations for spectrogram visualization
public class SpectrogramRenderer {
    private let coordinateSystem: SpectrogramCoordinateSystem

    public init(coordinateSystem: SpectrogramCoordinateSystem) {
        self.coordinateSystem = coordinateSystem
    }

    // MARK: - Drawing Functions

    /// Draw spectrogram heatmap
    public func drawSpectrogram(
        context: GraphicsContext,
        data: SpectrogramData,
        canvasWidth: CGFloat,
        canvasHeight: CGFloat,
        maxFreq: Double,
        leftPadding: CGFloat
    )

    /// Draw frequency axis labels
    public func drawFrequencyLabels(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        maxFreq: Double,
        viewportHeight: CGFloat,
        paperTop: CGFloat
    )

    /// Draw time axis labels
    public func drawTimeAxis(
        context: GraphicsContext,
        canvasSize: CGSize,
        leftPadding: CGFloat
    )

    /// Draw playback position indicator (red line)
    public func drawPlaybackPosition(
        context: GraphicsContext,
        viewportSize: CGSize
    )

    /// Draw placeholder when no data
    public func drawPlaceholder(
        context: GraphicsContext,
        size: CGSize
    )
}
```

**利点**:
- 描画ロジックの集約
- 再利用可能な描画エンジン
- テスト時にモック化可能

### 4.3 SpectrogramScrollManager

**責務**: スクロール状態管理とジェスチャー処理

```swift
/// Spectrogram scroll state manager
/// Manages 2D scrolling (frequency axis + time axis)
public class SpectrogramScrollManager: ObservableObject {
    // MARK: - Published State

    /// Y-axis scroll position (frequency axis)
    @Published public var paperTop: CGFloat = 0

    /// X-axis scroll offset (time axis)
    @Published public var canvasOffsetX: CGFloat = 0

    // MARK: - Internal State

    private var lastPaperTop: CGFloat = 0

    // MARK: - Initialization

    /// Initialize scroll position for frequency axis (bottom-aligned)
    public func initializeFrequencyScroll(viewportHeight: CGFloat, canvasHeight: CGFloat)

    /// Initialize scroll position for time axis (playhead-centered)
    public func initializeTimeScroll(
        currentTime: Double,
        viewportWidth: CGFloat,
        pixelsPerSecond: CGFloat,
        canvasLeftPadding: CGFloat
    )

    // MARK: - Gesture Handling

    /// Handle vertical drag gesture (frequency scrolling)
    public func handleVerticalDrag(
        translation: CGFloat,
        viewportHeight: CGFloat,
        canvasHeight: CGFloat
    )

    /// Handle horizontal drag gesture (time seeking)
    public func handleHorizontalDrag(
        translation: CGFloat,
        currentTime: Double,
        pixelsPerSecond: CGFloat,
        onSeek: (Double) -> Void
    )

    /// Finalize drag gesture
    public func endDrag()

    // MARK: - Time Synchronization

    /// Update scroll position to follow playback time
    public func syncToPlaybackTime(
        currentTime: Double,
        viewportWidth: CGFloat,
        pixelsPerSecond: CGFloat,
        canvasLeftPadding: CGFloat
    )
}
```

**利点**:
- スクロール状態の集約管理
- SwiftUI `@Published`でビューと自動連携
- ジェスチャー処理の分離

### 4.4 SpectrogramConstants

**責務**: 定数の集約管理

```swift
/// Spectrogram display constants
public struct SpectrogramConstants {
    // MARK: - Zoom & Density

    /// Pixels per kHz for frequency axis (9.6x zoom from original)
    public static let basePixelsPerKHz: CGFloat = 576.0

    /// Pixels per second for time axis (6x zoom from original)
    public static let pixelsPerSecond: CGFloat = 300.0

    /// Maximum canvas height to prevent memory issues
    public static let maxCanvasHeight: CGFloat = 10000.0

    // MARK: - Frequency Range

    /// Maximum frequency for display (matches data range)
    public static let maxFrequency: Double = 6000.0

    /// Frequency label interval
    public static let frequencyLabelInterval: Double = 100.0

    // MARK: - Time Axis

    /// Time label interval
    public static let timeLabelInterval: Double = 0.5

    /// Time label Y position offset from bottom
    public static let timeLabelBottomOffset: CGFloat = 10.0

    // MARK: - UI Dimensions

    /// Frequency label text width
    public static let frequencyLabelWidth: CGFloat = 45.0

    /// Frequency label text height
    public static let frequencyLabelHeight: CGFloat = 16.0

    /// Frequency label clipping margin
    public static let frequencyLabelClipMargin: CGFloat = 8.0
}
```

**利点**:
- マジックナンバーの排除
- 定数の一元管理
- 変更時の影響範囲の明確化

## 5. リファクタリング後のSpectrogramView

### 5.1 簡素化されたコード構造

```swift
struct SpectrogramView: View {
    // MARK: - Properties
    let currentTime: Double
    let spectrogramData: SpectrogramData?
    var isExpanded: Bool = false
    var onExpand: (() -> Void)? = nil
    var onCollapse: (() -> Void)? = nil
    var onPlayPause: (() -> Void)? = nil
    var onSeek: ((Double) -> Void)? = nil

    // MARK: - Dependencies
    @StateObject private var scrollManager = SpectrogramScrollManager()
    private let coordinateSystem = SpectrogramCoordinateSystem()
    private let renderer: SpectrogramRenderer

    init(...) {
        self.renderer = SpectrogramRenderer(coordinateSystem: coordinateSystem)
    }

    // MARK: - Body
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("analysis.spectrogram_title".localized)
                .font(.subheadline)
                .fontWeight(.semibold)
                .accessibilityIdentifier("SpectrogramTitle")

            GeometryReader { geometry in
                spectrogramCanvas(viewportSize: geometry.size)
            }
            .background(Color.black.opacity(0.1))
            .cornerRadius(8)
        }
        .contentShape(Rectangle())
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("SpectrogramView")
        .onTapGesture {
            onPlayPause?()
        }
    }

    // MARK: - Canvas

    private func spectrogramCanvas(viewportSize: CGSize) -> some View {
        // Canvas構築ロジック（簡素化）
        // scrollManager、coordinateSystem、rendererを使用
    }

    // MARK: - Initialization

    private func initializeScrollPosition(viewportSize: CGSize, canvasSize: CGSize) {
        scrollManager.initializeFrequencyScroll(
            viewportHeight: viewportSize.height,
            canvasHeight: canvasSize.height
        )
        scrollManager.initializeTimeScroll(
            currentTime: currentTime,
            viewportWidth: viewportSize.width,
            pixelsPerSecond: SpectrogramConstants.pixelsPerSecond,
            canvasLeftPadding: viewportSize.width / 2
        )
    }
}
```

**改善点**:
- **527行 → 約150-200行** (60%以上削減)
- 依存性注入パターンでテスト容易性向上
- 各責務が明確に分離
- body内のロジックが大幅に簡素化

### 5.2 期待される効果

| 項目 | リファクタリング前 | リファクタリング後 |
|------|-------------------|-------------------|
| コード行数 | 527行 | 150-200行 |
| 関数数 | 8個 (View内) | 2-3個 (View内) |
| テスタビリティ | 低 (View依存) | 高 (独立クラス) |
| 可読性 | 中 (複雑) | 高 (明確な分離) |
| 再利用性 | 低 (View固有) | 高 (独立クラス) |
| 保守性 | 中 | 高 (責務明確) |

## 6. ファイル構成

### 6.1 新規ファイル配置

```
VocalisStudio/
├── Presentation/
│   ├── Views/
│   │   └── AnalysisView.swift (既存、リファクタリング)
│   └── Components/
│       └── Spectrogram/
│           ├── SpectrogramCoordinateSystem.swift (新規)
│           ├── SpectrogramRenderer.swift (新規)
│           ├── SpectrogramScrollManager.swift (新規)
│           └── SpectrogramConstants.swift (新規)
└── Tests/
    └── PresentationTests/
        └── Components/
            └── Spectrogram/
                ├── SpectrogramCoordinateSystemTests.swift (新規)
                ├── SpectrogramRendererTests.swift (新規)
                └── SpectrogramScrollManagerTests.swift (新規)
```

### 6.2 既存ファイルへの影響

**変更が必要なファイル**:
- `AnalysisView.swift` - SpectrogramViewのリファクタリング

**影響を受ける可能性があるファイル**:
- UIテスト - スペクトログラム関連のテストケース
- スクリーンショット比較 - 視覚的検証

## 7. 実装計画

### 7.1 Phase 1: 座標系の抽出

**目的**: 座標計算ロジックを独立させてテスト可能にする

**タスク**:
1. ✅ `SpectrogramCoordinateSystem.swift`を作成
2. ✅ 座標計算関数を移動:
   - `calculateCanvasHeight()`
   - `frequencyToCanvasY()`
   - `getMaxFrequency()`
   - `calculateCanvasWidth()` (新規)
   - `timeToCanvasX()` (新規)
3. ✅ `SpectrogramView`を更新して新クラスを使用
4. ✅ 単体テスト`SpectrogramCoordinateSystemTests.swift`を作成
5. ✅ UIテスト実行で動作確認
6. ✅ コミット

**成功基準**:
- すべてのUIテストがパス
- 単体テストで座標計算の正確性を検証
- コード行数が100行以上削減

### 7.2 Phase 2: 描画ロジックの抽出

**目的**: 描画関数を集約して再利用可能にする

**タスク**:
1. ✅ `SpectrogramRenderer.swift`を作成
2. ✅ 描画関数を移動:
   - `drawSpectrogramOnCanvas()`
   - `drawFrequencyLabelsOnCanvas()`
   - `drawSpectrogramTimeAxis()`
   - `drawPlaybackPosition()`
   - `drawPlaceholder()`
3. ✅ `SpectrogramView`を更新して新クラスを使用
4. ✅ 単体テスト`SpectrogramRendererTests.swift`を作成
5. ✅ UIテスト実行で視覚的検証
6. ✅ スクリーンショット比較
7. ✅ コミット

**成功基準**:
- すべてのUIテストがパス
- スクリーンショット差分なし
- コード行数がさらに150行以上削減

### 7.3 Phase 3: スクロール管理の抽出

**目的**: スクロール状態管理を分離して管理しやすくする

**タスク**:
1. ✅ `SpectrogramScrollManager.swift`を作成
2. ✅ スクロール状態を移動:
   - `paperTop`, `lastPaperTop`, `canvasOffsetX`
3. ✅ スクロールロジックを移動:
   - 初期化処理
   - ジェスチャー処理
   - 時間同期処理
4. ✅ `SpectrogramView`を更新して新クラスを使用
5. ✅ 単体テスト`SpectrogramScrollManagerTests.swift`を作成
6. ✅ UIテスト実行でスクロール動作確認
7. ✅ コミット

**成功基準**:
- すべてのUIテストがパス
- スクロール動作が正常
- コード行数がさらに100行以上削減

### 7.4 Phase 4: 定数の整理

**目的**: マジックナンバーを排除して保守性を向上

**タスク**:
1. ✅ `SpectrogramConstants.swift`を作成
2. ✅ 定数を集約:
   - ズーム・密度関連
   - 周波数範囲
   - 時間軸
   - UI寸法
3. ✅ 既存コードを更新して定数を使用
4. ✅ ビルド確認
5. ✅ コミット

**成功基準**:
- すべてのビルドがパス
- マジックナンバーが排除されている

## 8. リスクと対策

### 8.1 リスク

| リスク | 影響度 | 対策 |
|--------|--------|------|
| 既存機能の破壊 | 高 | UIテスト・スクリーンショット比較で検証 |
| パフォーマンス低下 | 中 | クラス初期化コストを測定、必要に応じて最適化 |
| テスト追加の工数 | 中 | Phase 1で集中的にテスト追加、以降は段階的 |
| 移行中のバグ混入 | 中 | Phase単位でコミット、問題時は即ロールバック |

### 8.2 対策詳細

**既存機能の破壊を防ぐ**:
- 各Phase後に`AnalysisUITests`を実行
- スクリーンショット抽出して視覚的比較
- リファクタリング前後でピクセル単位の差異がないことを確認

**パフォーマンス低下を防ぐ**:
- `SpectrogramCoordinateSystem`のインスタンス化コストを測定
- 必要に応じて`static let`で共有インスタンス化
- 描画パフォーマンスをInstrumentsで測定

**テストカバレッジの確保**:
- 座標計算関数は100%カバレッジ目標
- 描画関数は主要パスのみテスト（完全なピクセル検証は不要）
- スクロール管理は状態遷移を重点的にテスト

## 9. 完了基準

### 9.1 技術的基準

- ✅ `SpectrogramView`が200行以下
- ✅ 座標計算ロジックが独立クラス化され、単体テスト済み
- ✅ 描画ロジックが独立クラス化
- ✅ スクロール管理が独立クラス化
- ✅ マジックナンバーが定数化
- ✅ すべてのUIテストがパス
- ✅ スクリーンショット比較で差異なし

### 9.2 品質基準

- ✅ コードレビュー完了
- ✅ ドキュメント更新（本プラン含む）
- ✅ 単体テストカバレッジ80%以上（新規クラス）
- ✅ パフォーマンス低下なし（描画60fps維持）

### 9.3 運用基準

- ✅ リファクタリング完了後1週間の安定動作確認
- ✅ 次回機能追加時の開発効率向上を実感

## 10. 期待される成果

### 10.1 コード品質向上

- **可読性**: 527行→200行でコード理解が容易
- **保守性**: 責務分離で変更影響範囲が明確
- **テスタビリティ**: 座標計算ロジックの単体テスト可能

### 10.2 開発効率向上

- **デバッグ効率**: 問題箇所の特定が容易
- **機能追加**: 新規描画機能の追加が簡単
- **再利用性**: 他のビューでスペクトログラム表示可能

### 10.3 長期的メリット

- **技術的負債の削減**: クリーンな設計への移行
- **チーム開発の効率化**: コード構造が明確
- **将来の拡張性**: 新機能追加の基盤が整う

## 11. 参考資料

- **現在のコード**: `VocalisStudio/Presentation/Views/AnalysisView.swift` (440-967行)
- **座標系ガイド**: `claudedocs/spectrogram-resolution-zoom-guide.md`
- **解像度改善**: `claudedocs/spectrogram-resolution-improvement.md`
- **UIテスト実行**: `claudedocs/test-scheme-management.md`
- **スクリーンショット抽出**: `claudedocs/UITEST_SCREENSHOT_EXTRACTION.md`

## 12. 補足事項

### 12.1 他のViewへの影響

**PitchAnalysisView** (968-1277行)
- 現時点では影響なし
- 将来的に同様のリファクタリングを検討

**AnalysisView全体**
- SpectrogramViewの改善により、全体の可読性も向上
- 将来的にPitchAnalysisViewも同様の構造にできる

### 12.2 次のステップ

本リファクタリング完了後、以下を検討：
1. PitchAnalysisViewの同様のリファクタリング
2. 座標系クラスの共通化（Spectrogram/Pitch共通基盤）
3. レンダラーの抽象化（描画エンジンの汎用化）
