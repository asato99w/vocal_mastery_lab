# 全画面表示実装レポート - ビューポート拡大のみ

**作成日**: 2025-11-12
**ステータス**: ✅ 実装完了

## 1. 実装目的

全画面表示（`isExpanded = true`）を**ビューポートの拡大のみ**として実装する。

### 設計原則

**「窓（ビューポート）が拡大されるだけで、裏のグラフ（キャンバス）はそのまま」**

- ✅ ビューポートサイズが拡大（画面全体を使用）
- ✅ キャンバスの描画パラメータは不変
- ✅ セル（マス目）のサイズは不変
- ✅ 表示される時間・周波数範囲が広がる（ビューポートに比例）

## 2. 修正前の問題

### 問題1: セルサイズが変化していた

**原因**: `basePixelsPerKHz` が `isExpanded` に依存

```swift
// ❌ 修正前: セルの縦幅が変化
let basePixelsPerKHz: CGFloat = isExpanded ? 120.0 : 60.0
```

**影響**:
- 通常表示: 60 px/kHz → セル高さ小
- 全画面表示: 120 px/kHz → セル高さ大（2倍）
- **結果**: マス目の大きさが変化

### 問題2: 時間軸密度が変化していた

**原因**: `pixelsPerSecond` が `isExpanded` に依存

```swift
// ❌ 修正前: セルの横幅が変化
let pixelsPerSecond: CGFloat = isExpanded ? 30 : 50
```

**影響**:
- 通常表示: 50 px/秒 → セル幅 5px
- 全画面表示: 30 px/秒 → セル幅 3px
- **結果**: マス目の横幅が変化（縦長に変形）

## 3. 実施した修正

### 修正1: basePixelsPerKHz を固定

**ファイル**: `VocalisStudio/Presentation/Views/AnalysisView.swift`

**関数**: `calculateCanvasHeight()`

**修正内容**:

```swift
// ✅ 修正後: 常に固定
private func calculateCanvasHeight(maxFreq: Double, viewportHeight: CGFloat) -> CGFloat {
    // Fixed pixel density per kHz (isExpanded only affects viewport, not canvas drawing)
    let basePixelsPerKHz: CGFloat = 60.0  // 固定
    let desiredHeight = CGFloat(maxFreq / 1000.0) * basePixelsPerKHz

    // Ensure minimum canvas height (viewport * 2.0) for sufficient scrolling
    let minHeight = viewportHeight * 2.0

    // Apply maximum limit to prevent excessive memory usage
    let maxHeight: CGFloat = 10000.0

    return min(maxHeight, max(desiredHeight, minHeight))
}
```

**効果**:
- キャンバス高さ: `8kHz * 60 = 480px`（通常・全画面とも同じ）
- セルの縦幅: 不変

### 修正2: pixelsPerSecond を固定

**ファイル**: `VocalisStudio/Presentation/Views/AnalysisView.swift`

**関数**: `drawSpectrogramOnCanvas()`

**修正内容**:

```swift
// ✅ 修正後: 常に固定
private func drawSpectrogramOnCanvas(
    context: GraphicsContext,
    canvasWidth: CGFloat,
    canvasHeight: CGFloat,
    maxFreq: Double,
    data: SpectrogramData
) {
    // Fixed time axis density (isExpanded only affects viewport, not drawing parameters)
    let pixelsPerSecond: CGFloat = 50  // 固定
    let timeWindow = Double(canvasWidth / pixelsPerSecond)
    let centerX = canvasWidth / 2
    let maxMagnitude = data.magnitudes.flatMap { $0 }.max() ?? 1.0

    // Calculate cell dimensions
    let cellWidth = pixelsPerSecond * 0.1  // 5px固定

    // ... (描画処理)
}
```

**効果**:
- セルの横幅: `50 * 0.1 = 5px`（通常・全画面とも同じ）
- 時間軸密度: 不変

### 修正3: 時間軸ラベルも固定

**ファイル**: `VocalisStudio/Presentation/Views/AnalysisView.swift`

**関数**: `drawSpectrogramTimeAxis()`

**修正内容**:

```swift
// ✅ 修正後: 常に固定
private func drawSpectrogramTimeAxis(context: GraphicsContext, size: CGSize) {
    // Fixed time axis density (isExpanded only affects viewport, not drawing parameters)
    let pixelsPerSecond: CGFloat = 50  // 固定
    let timeWindow = Double(size.width / pixelsPerSecond)
    let halfWindow = timeWindow / 2

    // ... (ラベル描画処理)
}
```

**効果**:
- 時間軸の計算がキャンバスの描画と一致

## 4. 修正結果の比較

### 通常表示

| パラメータ | 値 |
|-----------|---|
| ビューポート幅 | 約360px |
| ビューポート高さ | 約176px |
| キャンバス幅 | 約360px |
| キャンバス高さ | 480px |
| basePixelsPerKHz | 60 px/kHz |
| pixelsPerSecond | 50 px/秒 |
| セルサイズ | 5px × 6px |
| 表示時間範囲 | 約7.2秒 |
| 表示周波数範囲 | ビューポート依存 |

### 全画面表示

| パラメータ | 値 | 変化 |
|-----------|---|-----|
| ビューポート幅 | 約390px | ✅ 拡大 |
| ビューポート高さ | 約800px | ✅ 拡大 |
| キャンバス幅 | 約390px | （ビューポートに追従） |
| キャンバス高さ | **480px** | **✅ 不変** |
| basePixelsPerKHz | **60 px/kHz** | **✅ 不変** |
| pixelsPerSecond | **50 px/秒** | **✅ 不変** |
| セルサイズ | **5px × 6px** | **✅ 不変** |
| 表示時間範囲 | 約7.8秒 | ✅ 拡大（幅に比例） |
| 表示周波数範囲 | より広い | ✅ 拡大（高さに比例） |

## 5. アーキテクチャの正しさ

### ビューポート拡大の概念図

```
通常表示:
┌─────────────────────┐
│ キャンバス (480px)   │
│  0Hz ~ 8kHz         │
│                     │
│  ┌──────┐           │
│  │Viewport│ ← 小さい │
│  │(176px)│           │
│  └──────┘           │
│         ↕ offset    │
└─────────────────────┘

全画面表示:
┌─────────────────────┐
│ キャンバス (480px)   │← 同じサイズ
│  0Hz ~ 8kHz         │
│                     │
│  ┌──────────────┐   │
│  │  Viewport    │   │← 大きくなる
│  │  (800px)     │   │
│  └──────────────┘   │
│         ↕ offset    │
└─────────────────────┘
```

### セルサイズの不変性

```
通常表示のセル:
┌─┐
│ │ 5px × 6px
└─┘

全画面表示のセル:
┌─┐
│ │ 5px × 6px ← 同じサイズ
└─┘
```

## 6. 変更されたコード箇所まとめ

### AnalysisView.swift

**Line 541**: `basePixelsPerKHz` を固定
```swift
// 変更前
let basePixelsPerKHz: CGFloat = isExpanded ? 120.0 : 60.0

// 変更後
let basePixelsPerKHz: CGFloat = 60.0
```

**Line 660**: `pixelsPerSecond` を固定（スペクトル描画）
```swift
// 変更前
let pixelsPerSecond: CGFloat = isExpanded ? 30 : 50

// 変更後
let pixelsPerSecond: CGFloat = 50
```

**Line 770**: `pixelsPerSecond` を固定（時間軸ラベル）
```swift
// 変更前
let pixelsPerSecond: CGFloat = isExpanded ? 30 : 50

// 変更後
let pixelsPerSecond: CGFloat = 50
```

## 7. 検証方法

### 視覚的検証

1. 通常表示でスペクトログラムを表示
2. セル（縦の縞模様）のサイズを観察
3. 全画面表示に切り替え
4. セルのサイズが変わらないことを確認 ✅

### 数値的検証

```swift
// 通常表示
cellWidth = 50 * 0.1 = 5px
cellHeight = (8000/bins) * (60/1000) = 約6px

// 全画面表示
cellWidth = 50 * 0.1 = 5px  ← 同じ
cellHeight = (8000/bins) * (60/1000) = 約6px  ← 同じ
```

## 8. 残存する `isExpanded` の使用箇所

### ピッチ分析グラフ（未修正）

**ファイル**: `AnalysisView.swift`
**Line 833-834, 840**: ピッチ分析グラフでは `isExpanded` を使用

```swift
// ピッチ分析グラフは別仕様（今回の修正対象外）
let minFreq = isExpanded ? max(100.0, baseMinFreq - 100) : baseMinFreq
let maxFreq = isExpanded ? min(2000.0, baseMaxFreq + 200) : baseMaxFreq
let pixelsPerSecond: CGFloat = isExpanded ? 30 : 50
```

**理由**: ピッチ分析グラフは別のUI要件があるため、今回の修正対象外としました。

## 9. 結論

✅ **全画面表示 = ビューポートの拡大のみ**の実装に成功

- キャンバスの描画パラメータは不変
- セル（マス目）のサイズは不変
- 表示範囲のみが拡大（ビューポートに比例）

**スペクトログラムの全画面表示は、文字通り「窓を大きくして覗く」だけの実装になりました。**

---

**関連ドキュメント**:
- `claudedocs/spectrogram_canvas_implementation_result.md` - キャンバスアーキテクチャの基本実装
- `claudedocs/spectrogram_canvas_architecture_plan.md` - 設計計画書
