# ピッチグラフ軸・フルスクリーン実装計画

## 概要

スペクトログラムで実装された縦軸・横軸の扱い、フルスクリーン表示機能をピッチグラフにも適用する計画書です。

**作成日**: 2025-11-20
**ステータス**: 計画段階

---

## 1. 現状分析

### 1.1 スペクトログラムの実装（参考）

スペクトログラムは以下のアーキテクチャで実装されています：

```
Presentation/Components/Spectrogram/
├── SpectrogramConstants.swift      # 定数管理
├── SpectrogramCoordinateSystem.swift  # 座標系計算
├── SpectrogramRenderer.swift       # 描画ロジック
└── SpectrogramScrollManager.swift  # スクロール状態管理
```

**主な特徴**:
- **キャンバスベース**: 大きなキャンバスを作成し、ビューポートでクリップ
- **2軸スクロール**: 縦（周波数）・横（時間）の独立スクロール
- **軸ラベル固定**:
  - Y軸（周波数）: Y方向スクロール、X方向固定
  - X軸（時間）: X方向スクロール、Y方向固定
  - 再生位置（赤線）: 完全固定
- **ドラッグジェスチャー**: 縦横の方向を判定して動作を切り替え

### 1.2 ピッチグラフの現状

現在のPitchAnalysisViewは単純なCanvas描画で、以下の制限があります：

| 項目 | スペクトログラム | ピッチグラフ（現状） |
|------|----------------|-------------------|
| キャンバス | データサイズに応じた大きなキャンバス | ビューポートサイズのみ |
| 縦スクロール | ✅ 周波数軸スクロール | ❌ なし |
| 横スクロール | ✅ 時間軸スクロール | ❌ なし（再生位置中心固定） |
| 軸ラベル | ✅ 固定表示 | ❌ なし |
| フルスクリーン | ✅ 詳細な座標系 | ⚠️ 簡易実装 |
| ズーム | ✅ 高密度表示 | ❌ 固定密度 |

### 1.3 現在のピッチグラフ実装（AnalysisView.swift:720-799）

```swift
private func drawPitchGraph(context: GraphicsContext, size: CGSize, data: PitchAnalysisData) {
    // 周波数範囲は検出データに基づいて動的に決定
    let baseMinFreq = frequencies.min() ?? 200.0
    let baseMaxFreq = frequencies.max() ?? 800.0

    // Expanded viewで拡大
    let minFreq = isExpanded ? max(100.0, baseMinFreq - 100) : baseMinFreq
    let maxFreq = isExpanded ? min(2000.0, baseMaxFreq + 200) : baseMaxFreq

    // 時間軸は再生位置を中心に固定ウィンドウ
    let pixelsPerSecond: CGFloat = isExpanded ? 30 : 50
    let timeWindow = Double(graphWidth / pixelsPerSecond)

    // 再生位置が常に中心
    let centerX = leftMargin + graphWidth / 2
}
```

**問題点**:
1. 周波数範囲がデータ依存で不安定
2. 軸ラベルがない
3. スクロール機能がない
4. フルスクリーン時の表示が制限的

---

## 2. 実装目標

### 2.1 機能要件

1. **縦軸（周波数）スクロール**: 周波数範囲全体をスクロール可能に
2. **横軸（時間）スクロール**: 再生位置中心の自動スクロール + 手動シーク
3. **軸ラベル固定**:
   - Y軸ラベル: 左端に固定
   - X軸ラベル: 下端に固定
4. **フルスクリーン対応**: 詳細な表示と操作

### 2.2 非機能要件

1. **パフォーマンス**: スペクトログラムと同等のスムーズな描画
2. **メモリ効率**: キャンバスサイズの適切な制限
3. **コード再利用**: 共通コンポーネントの活用

---

## 3. 設計

### 3.1 アーキテクチャ

```
Presentation/Components/
├── Spectrogram/                    # 既存
│   ├── SpectrogramConstants.swift
│   ├── SpectrogramCoordinateSystem.swift
│   ├── SpectrogramRenderer.swift
│   └── SpectrogramScrollManager.swift
├── PitchGraph/                     # 新規作成
│   ├── PitchGraphConstants.swift
│   ├── PitchGraphCoordinateSystem.swift
│   └── PitchGraphRenderer.swift
└── Shared/                         # 共通化検討
    └── GraphScrollManager.swift    # SpectrogramScrollManagerを共通化
```

### 3.2 定数設計（PitchGraphConstants.swift）

```swift
public struct PitchGraphConstants {
    // MARK: - Frequency Range

    /// Minimum frequency for pitch graph (Hz)
    /// ボーカル分析用: 80Hz（男性低音域）
    public static let minFrequency: Double = 80.0

    /// Maximum frequency for pitch graph (Hz)
    /// ボーカル分析用: 1000Hz（女性高音域+マージン）
    public static let maxFrequency: Double = 1000.0

    // MARK: - Display Density

    /// Pixels per 100Hz for frequency axis
    /// 920Hz範囲を適切に表示（スペクトログラムより高密度）
    public static let pixelsPerHundredHz: CGFloat = 60.0

    /// Pixels per second for time axis
    /// スペクトログラムと同等の時間軸密度
    public static let pixelsPerSecond: CGFloat = 300.0

    // MARK: - Canvas Limits

    /// Maximum canvas height
    public static let maxCanvasHeight: CGFloat = 5000.0

    /// Minimum canvas width
    public static let minCanvasWidth: CGFloat = 100.0

    // MARK: - Labels

    /// Frequency label interval (Hz)
    public static let frequencyLabelInterval: Double = 100.0

    /// Time label interval (seconds)
    public static let timeLabelInterval: Double = 0.5

    // MARK: - Margins

    /// Left margin for Y-axis labels
    public static let leftMargin: CGFloat = 50.0

    /// Bottom margin for X-axis labels
    public static let bottomMargin: CGFloat = 30.0

    /// Top margin
    public static let topMargin: CGFloat = 10.0

    /// Right margin
    public static let rightMargin: CGFloat = 10.0
}
```

### 3.3 座標系設計（PitchGraphCoordinateSystem.swift）

```swift
public class PitchGraphCoordinateSystem {
    // MARK: - Canvas Dimensions

    /// Calculate canvas height based on frequency range
    public func calculateCanvasHeight() -> CGFloat {
        let freqRange = PitchGraphConstants.maxFrequency - PitchGraphConstants.minFrequency
        let pixelsPerHz = PitchGraphConstants.pixelsPerHundredHz / 100.0
        let canvasHeight = CGFloat(freqRange) * pixelsPerHz
        return min(PitchGraphConstants.maxCanvasHeight, canvasHeight)
    }

    /// Calculate canvas width based on data duration
    public func calculateCanvasWidth(dataDuration: Double, leftPadding: CGFloat) -> CGFloat {
        let dataWidth = CGFloat(dataDuration) * PitchGraphConstants.pixelsPerSecond
        return max(dataWidth + leftPadding, PitchGraphConstants.minCanvasWidth)
    }

    // MARK: - Frequency Conversions

    /// Convert frequency to canvas Y coordinate
    /// Y=0: maxFreq (top), Y=canvasHeight: minFreq (bottom)
    public func frequencyToCanvasY(frequency: Double, canvasHeight: CGFloat) -> CGFloat {
        let minFreq = PitchGraphConstants.minFrequency
        let maxFreq = PitchGraphConstants.maxFrequency
        let ratio = (maxFreq - frequency) / (maxFreq - minFreq)
        return CGFloat(ratio) * canvasHeight
    }

    /// Convert canvas Y to frequency
    public func canvasYToFrequency(y: CGFloat, canvasHeight: CGFloat) -> Double {
        let minFreq = PitchGraphConstants.minFrequency
        let maxFreq = PitchGraphConstants.maxFrequency
        let ratio = Double(y / canvasHeight)
        return maxFreq - ratio * (maxFreq - minFreq)
    }

    // MARK: - Time Conversions

    /// Convert time to canvas X coordinate
    public func timeToCanvasX(time: Double, leftPadding: CGFloat) -> CGFloat {
        return CGFloat(time) * PitchGraphConstants.pixelsPerSecond + leftPadding
    }
}
```

### 3.4 レンダラー設計（PitchGraphRenderer.swift）

```swift
public class PitchGraphRenderer {
    private let coordinateSystem: PitchGraphCoordinateSystem

    // MARK: - Main Drawing

    /// Draw pitch line graph
    public func drawPitchLine(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        data: PitchAnalysisData,
        leftPadding: CGFloat
    ) {
        // 折れ線グラフの描画
        // 信頼度に応じたドットサイズ
    }

    /// Draw target scale lines (reference frequencies)
    public func drawTargetScaleLines(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        settings: ScaleSettings,
        leftPadding: CGFloat,
        canvasWidth: CGFloat
    ) {
        // ターゲット周波数の水平線
    }

    // MARK: - Axis Labels

    /// Draw frequency labels (Y-axis) - Y-scrollable, X-fixed
    public func drawFrequencyLabels(
        context: GraphicsContext,
        canvasHeight: CGFloat,
        viewportHeight: CGFloat,
        paperTop: CGFloat
    ) {
        // 周波数ラベル（100Hz間隔）
        // X方向は固定、Y方向はスクロール追従
    }

    /// Draw time labels (X-axis) - X-scrollable, Y-fixed
    public func drawTimeAxis(
        context: GraphicsContext,
        viewportHeight: CGFloat,
        leftPadding: CGFloat,
        canvasWidth: CGFloat
    ) {
        // 時間ラベル（0.5秒間隔）
        // Y方向は固定（ビューポート下端）、X方向はスクロール追従
    }

    // MARK: - Playhead

    /// Draw playback position line - fully fixed
    public func drawPlaybackPosition(
        context: GraphicsContext,
        viewportWidth: CGFloat,
        viewportHeight: CGFloat
    ) {
        // 再生位置（白線）- 完全固定
    }

    // MARK: - Placeholder

    /// Draw placeholder when no data
    public func drawPlaceholder(context: GraphicsContext, size: CGSize)
}
```

### 3.5 ScrollManager共通化

`SpectrogramScrollManager`は汎用的なため、そのまま再利用可能です。
ピッチグラフでも同じスクロールロジック（Y軸周波数、X軸時間）を使用します。

---

## 4. 実装計画

### Phase 1: コンポーネント作成（1-2日）

1. **PitchGraphConstants.swift**: 定数定義
2. **PitchGraphCoordinateSystem.swift**: 座標系計算
3. **PitchGraphRenderer.swift**: 描画ロジック

### Phase 2: PitchAnalysisView改修（2-3日）

1. **キャンバスベース化**:
   - 固定サイズCanvas → データサイズ依存Canvas
   - ビューポートクリッピング

2. **スクロール対応**:
   - SpectrogramScrollManager（またはGraphScrollManager）の導入
   - 縦ドラッグ: 周波数スクロール
   - 横ドラッグ: シーク

3. **軸ラベル固定**:
   - Y軸: X方向補正
   - X軸: Y方向補正
   - 再生位置: 両軸補正

4. **フルスクリーン対応**:
   - isExpandedに応じた初期位置設定
   - 適切なスクロール範囲

### Phase 3: テスト・調整（1-2日）

1. **ユニットテスト**: PitchGraphCoordinateSystemTests
2. **UIテスト**: フルスクリーン動作確認
3. **パフォーマンス確認**: 長い録音データでのスムーズ度
4. **デザイン調整**: ラベル間隔、色、線の太さ

---

## 5. 技術的考慮事項

### 5.1 周波数範囲の違い

| 項目 | スペクトログラム | ピッチグラフ |
|------|----------------|-------------|
| 範囲 | 0-6000Hz | 80-1000Hz |
| 用途 | 倍音分析 | 基本周波数分析 |
| 密度 | 576pt/kHz | 60pt/100Hz (600pt/kHz) |

ピッチグラフは狭い周波数範囲を高密度で表示するため、ピクセル密度を高く設定。

### 5.2 Canvas高さの計算

```
周波数範囲 = 1000Hz - 80Hz = 920Hz
ピクセル密度 = 60pt/100Hz = 0.6pt/Hz
Canvas高さ = 920Hz × 0.6pt/Hz = 552pt
```

スペクトログラム（3456pt）より小さく、メモリ効率が良い。

### 5.3 スクロール範囲

- **縦スクロール**: 全周波数範囲（80-1000Hz）
- **横スクロール**: 録音全長（再生位置自動追従 + 手動シーク）

### 5.4 既存機能との互換性

現在のピッチグラフで表示しているターゲットスケールライン機能は維持。
スケール設定に基づく周波数参照線を新しい座標系で描画。

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|--------|------|------|
| パフォーマンス低下 | 描画がカクつく | データポイント間引き、Canvas最適化 |
| 座標系の不整合 | 表示ずれ | ユニットテストで座標変換を検証 |
| ScrollManager競合 | スクロール動作不良 | 独立インスタンスで管理 |
| フルスクリーン遷移 | 位置リセット問題 | onChange(of: isExpanded)で再初期化 |

---

## 7. 成功基準

1. **機能**: 縦横スクロールが正常動作
2. **表示**: 軸ラベルが固定表示される
3. **操作**: ドラッグでスクロール/シークが可能
4. **品質**: スペクトログラムと同等のスムーズさ
5. **互換**: 既存のターゲットスケールライン機能が動作

---

## 8. 参考資料

### 関連ドキュメント
- `spectrogram-resolution-zoom-guide.md`: スペクトログラム解像度・ズームガイド
- `fullscreen-graph-implementation-plan.md`: フルスクリーン実装計画（スペクトログラム）

### 関連コード
- `SpectrogramView`: AnalysisView.swift:440-645
- `PitchAnalysisView`: AnalysisView.swift:655-918
- `SpectrogramConstants`: Presentation/Components/Spectrogram/SpectrogramConstants.swift
- `SpectrogramCoordinateSystem`: Presentation/Components/Spectrogram/SpectrogramCoordinateSystem.swift
- `SpectrogramRenderer`: Presentation/Components/Spectrogram/SpectrogramRenderer.swift
- `SpectrogramScrollManager`: Presentation/Components/Spectrogram/SpectrogramScrollManager.swift

---

## 9. 次のステップ

1. [ ] この計画のレビュー・承認
2. [ ] Phase 1: コンポーネント作成
3. [ ] Phase 2: PitchAnalysisView改修
4. [ ] Phase 3: テスト・調整
5. [ ] ドキュメント更新

---

**最終更新**: 2025-11-20
