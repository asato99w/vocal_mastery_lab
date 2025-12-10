# スペクトログラム キャンバスアーキテクチャ実装計画

**作成日**: 2025-11-12
**目的**: ビューポートが巨大キャンバスを覗く正しいアーキテクチャの実装

## 1. 根本的な設計原則

### ❌ 誤った理解（以前の実装）
- ビューポート内にグラフをフィットさせて描画
- 可視範囲に応じてセル高さやラベル間隔を調整
- 周波数 → ビューポートY座標に変換

### ✅ 正しい理解
- **巨大なキャンバスを作成**（0Hz～MaxHzまでの全高さ）
- **すべてキャンバス座標系で描画**（スペクトルもラベルも）
- **ビューポートは`.clipped()`で覗くだけ**（描画しない）
- **スクロールはキャンバスの`.offset(y:)`のみ**
- **セル高さ・ラベル間隔はキャンバス基準で固定**

## 2. 座標系の定義

### キャンバス座標系

```
Y = 0                          ← MaxHz (例: 10000Hz)
│
│  ┌──────────────┐
│  │ Viewport     │ ← ビューポートが覗いている範囲
│  │ (clipped)    │   (例: 130Hz～132Hz)
│  └──────────────┘
│
Y = canvasHeight               ← 0Hz
```

### 周波数 → キャンバスY座標の変換

```swift
func frequencyToCanvasY(frequency: Double, canvasHeight: CGFloat, maxFreq: Double) -> CGFloat {
    let ratio = (maxFreq - frequency) / maxFreq
    return CGFloat(ratio) * canvasHeight
}
```

**重要**: この変換は**キャンバス全体の高さ**を基準とする。ビューポート高さは無関係。

## 3. キャンバスの設計

### 3.1 キャンバスの高さ

```swift
// 周波数範囲あたりのピクセル密度（例: 10px/kHz）
let pixelsPerKHz: CGFloat = 10.0
let maxFreq: Double = 10000.0  // 10kHz

// キャンバス全体の高さ
let canvasHeight = CGFloat(maxFreq / 1000.0) * pixelsPerKHz
// 例: 10kHz → 100ピクセル
```

### 3.2 描画対象

キャンバスに描画するもの：
1. **Y軸周波数ラベル** - キャンバス全体に固定間隔で配置
2. **スペクトル** - 各セルをキャンバス座標で配置
3. **時間軸ラベル** - キャンバスと一緒にスクロール

## 4. 実装ステップ

### Step 1: キャンバス高さの計算

```swift
// AnalysisView.swift
private func calculateCanvasHeight(maxFreq: Double) -> CGFloat {
    // 周波数あたりのピクセル密度
    let pixelsPerKHz: CGFloat = isExpanded ? 20.0 : 10.0
    return CGFloat(maxFreq / 1000.0) * pixelsPerKHz
}
```

### Step 2: キャンバス座標での描画

```swift
// Canvas内で描画
Canvas { context, size in
    // size = ビューポートサイズ（例: 300pt）

    // キャンバス全体の高さを計算
    let canvasHeight = calculateCanvasHeight(maxFreq: getMaxFrequency())
    // 例: 10000Hz → 100pt または 200pt

    // Y軸ラベルをキャンバス座標で描画
    drawFrequencyLabels(
        context: context,
        canvasHeight: canvasHeight,
        maxFreq: getMaxFrequency()
    )

    // スペクトルをキャンバス座標で描画
    drawSpectrogram(
        context: context,
        canvasHeight: canvasHeight,
        maxFreq: getMaxFrequency()
    )
}
.frame(height: canvasHeight)  // キャンバスの実サイズ
.offset(y: canvasOffsetY)      // スクロール
.clipped()                     // ビューポートで切り抜き
```

### Step 3: Y軸ラベルの描画（キャンバス座標）

```swift
private func drawFrequencyLabels(
    context: GraphicsContext,
    canvasHeight: CGFloat,
    maxFreq: Double
) {
    // 固定間隔（例: 1kHz）
    let labelInterval: Double = 1000.0

    var freq: Double = 0
    while freq <= maxFreq {
        // キャンバス座標でY位置を計算
        let canvasY = frequencyToCanvasY(
            frequency: freq,
            canvasHeight: canvasHeight,
            maxFreq: maxFreq
        )

        // ラベルを描画（キャンバス上の絶対位置）
        let labelText = freq >= 1000 ? "\(Int(freq/1000))k" : "\(Int(freq))Hz"
        context.draw(
            Text(labelText).font(.caption2).foregroundColor(.white),
            at: CGPoint(x: 20, y: canvasY)
        )

        freq += labelInterval
    }
}
```

**重要**: `canvasY` はキャンバス全体（0～canvasHeight）の座標。ビューポートサイズは無関係。

### Step 4: スペクトルの描画（キャンバス座標）

```swift
private func drawSpectrogram(
    context: GraphicsContext,
    canvasHeight: CGFloat,
    maxFreq: Double
) {
    guard let data = spectrogramData else { return }

    // 各周波数ビンについて
    for freqIndex in 0..<data.frequencyBins.count {
        let binFreq = Double(data.frequencyBins[freqIndex])

        // 周波数ビンの範囲（低～高）
        let binFreqHigh: Double
        if freqIndex + 1 < data.frequencyBins.count {
            binFreqHigh = Double(data.frequencyBins[freqIndex + 1])
        } else {
            binFreqHigh = binFreq + 100  // デフォルト幅
        }

        // キャンバス座標でY位置を計算
        let yTop = frequencyToCanvasY(
            frequency: binFreqHigh,
            canvasHeight: canvasHeight,
            maxFreq: maxFreq
        )
        let yBottom = frequencyToCanvasY(
            frequency: binFreq,
            canvasHeight: canvasHeight,
            maxFreq: maxFreq
        )
        let cellHeight = yBottom - yTop

        // 時間軸方向の各セルを描画
        for (timeIndex, timestamp) in data.timeStamps.enumerated() {
            let timeOffset = timestamp - currentTime
            let x = centerX + CGFloat(timeOffset) * pixelsPerSecond
            let cellWidth = pixelsPerSecond * 0.1

            // マグニチュード取得
            let magnitude: Float
            if timeIndex < data.magnitudes.count &&
               freqIndex < data.magnitudes[timeIndex].count {
                magnitude = data.magnitudes[timeIndex][freqIndex]
            } else {
                magnitude = 0.0  // データなし
            }

            let color = magnitudeToColor(magnitude)

            // キャンバス座標でセルを描画
            let rect = CGRect(x: x, y: yTop, width: cellWidth, height: cellHeight)
            context.fill(Path(rect), with: .color(color))
        }
    }
}
```

**重要**: セルの高さ `cellHeight` は**キャンバス全体の高さを基準**に計算される。ビューポートサイズは無関係。

### Step 5: スクロールの実装

```swift
// AnalysisView.swift
@State private var canvasOffsetY: CGFloat = 0

var body: some View {
    GeometryReader { geometry in
        Canvas { context, size in
            // size = ビューポートサイズ
            let canvasHeight = calculateCanvasHeight(maxFreq: getMaxFrequency())

            // キャンバス座標で描画
            drawFrequencyLabels(context: context, canvasHeight: canvasHeight, maxFreq: getMaxFrequency())
            drawSpectrogram(context: context, canvasHeight: canvasHeight, maxFreq: getMaxFrequency())
        }
        .frame(height: calculateCanvasHeight(maxFreq: getMaxFrequency()))
        .offset(y: canvasOffsetY)  // スクロール
        .clipped()                  // ビューポートで切り抜き
        .gesture(
            DragGesture()
                .onChanged { value in
                    // 縦方向のスクロール
                    canvasOffsetY += value.translation.height

                    // スクロール範囲の制限
                    let maxOffset: CGFloat = 0
                    let minOffset = geometry.size.height - calculateCanvasHeight(maxFreq: getMaxFrequency())
                    canvasOffsetY = max(minOffset, min(maxOffset, canvasOffsetY))
                }
        )
    }
}
```

**重要**: スクロールは`.offset(y:)`のみで実装。描画コードは変更不要。

## 5. ビューポート vs キャンバスの対比表

| 項目 | ❌ 誤った実装（以前） | ✅ 正しい実装 |
|------|-------------------|-------------|
| 座標系 | ビューポート座標 | キャンバス座標 |
| 描画対象 | ビューポート内にフィット | キャンバス全体 |
| セル高さ | ビューポートサイズ基準 | キャンバス高さ基準 |
| ラベル間隔 | 可視範囲に応じて動的 | キャンバス全体で固定 |
| スクロール | 可視範囲を再計算 | キャンバスをoffset |
| ラベル表示 | 可視範囲内のみ生成 | キャンバス全体に配置 |

## 6. デバッグ用の確認項目

実装後、以下を確認：

1. **キャンバス高さ**: `canvasHeight` が適切か（例: 100～200pt）
2. **ラベル位置**: `frequencyToCanvasY(1000Hz)` が期待値になるか
3. **セル高さ**: 各周波数ビンの `cellHeight` が妥当か
4. **スクロール範囲**: `minOffset` と `maxOffset` が正しいか
5. **ビューポート切り抜き**: `.clipped()` が機能しているか

## 7. 実装時の注意事項

### ❌ やってはいけないこと

1. ビューポートサイズを描画計算に使う
2. 可視範囲に応じてラベル間隔を変える
3. `viewportHeight` でセル高さを計算する
4. 「ビューポートを埋める」という発想

### ✅ やるべきこと

1. すべての描画をキャンバス座標で行う
2. ラベル間隔はキャンバス全体で固定
3. `canvasHeight` でセル高さを計算する
4. 「キャンバスを覗く」という発想

## 8. 実装上の重要な追加考慮事項

### 8.1 優先度：高（必須対応）

#### A. canvasHeight のスケール設計

**問題**: `pixelsPerKHz = 10.0` では `10kHz → 100px` と小さすぎる。ビューポート高さ（300～800pt）に対してスクロール感が出ない。

**対策**:
```swift
private func calculateCanvasHeight(maxFreq: Double, viewportHeight: CGFloat) -> CGFloat {
    let basePixelsPerKHz: CGFloat = isExpanded ? 20 : 10
    let desiredHeight = CGFloat(maxFreq / 1000.0) * basePixelsPerKHz

    // 最小でもビューポートの1.2倍は確保（スクロール領域）
    return max(desiredHeight, viewportHeight * 1.2)
}
```

**将来的**: ズーム係数を導入してユーザー調整可能にする。

#### B. 周波数ビン幅の厳密な扱い

**重要**: FFTビンは不均一な場合がある。各ビンの`binFreqLow`と`binFreqHigh`を正確に取得すること。

```swift
// 各ビンの周波数範囲を厳密に計算
let binFreqLow = Double(data.frequencyBins[freqIndex])
let binFreqHigh: Double
if freqIndex + 1 < data.frequencyBins.count {
    binFreqHigh = Double(data.frequencyBins[freqIndex + 1])
} else {
    // 最後のビン: 前のビン幅を使用
    let prevBinWidth = freqIndex > 0 ?
        binFreqLow - Double(data.frequencyBins[freqIndex - 1]) : 100.0
    binFreqHigh = binFreqLow + prevBinWidth
}
```

#### C. 表示パフォーマンス（必須）

**問題**: 巨大キャンバス全体を毎回描画すると重い。

**対策1: Visible-only描画**
```swift
// 可視範囲のみ描画
let visibleTop = -canvasOffsetY
let visibleBottom = -canvasOffsetY + viewportHeight

// 周波数ビンのフィルタリング
if yBottom < visibleTop || yTop > visibleBottom {
    continue  // このビンはスキップ
}

// 時間列のフィルタリング
let visibleTimeStart = currentTime - timeWindow / 2
let visibleTimeEnd = currentTime + timeWindow / 2
if timestamp < visibleTimeStart || timestamp > visibleTimeEnd {
    continue  // この列はスキップ
}
```

**対策2: 描画グループ化**
```swift
Canvas { context, size in
    // 描画処理
}
.drawingGroup()  // GPU加速
```

**対策3: 実機プロファイリング必須**
- Instruments で FPS と GPU 使用率を測定
- 60fps を維持できない場合はタイル化やキャッシュを検討

#### D. ラベル生成戦略

**方針**: 生成はキャンバス基準、描画時に密度調整。

```swift
private func drawFrequencyLabels(
    context: GraphicsContext,
    canvasHeight: CGFloat,
    maxFreq: Double,
    visibleRange: ClosedRange<CGFloat>  // 可視Y範囲
) {
    let labelInterval: Double = 1000.0  // キャンバス基準で固定

    var freq: Double = 0
    var lastDrawnY: CGFloat? = nil
    let minLabelSpacing: CGFloat = 30.0  // ラベル間の最小ピクセル

    while freq <= maxFreq {
        let canvasY = frequencyToCanvasY(
            frequency: freq,
            canvasHeight: canvasHeight,
            maxFreq: maxFreq
        )

        // 可視範囲外ならスキップ
        guard visibleRange.contains(canvasY) else {
            freq += labelInterval
            continue
        }

        // 前のラベルと近すぎる場合はスキップ（重なり回避）
        if let lastY = lastDrawnY, abs(canvasY - lastY) < minLabelSpacing {
            freq += labelInterval
            continue
        }

        // ラベル描画
        drawLabel(context: context, frequency: freq, y: canvasY)
        lastDrawnY = canvasY

        freq += labelInterval
    }
}
```

### 8.2 優先度：中（推奨対応）

#### E. 線形 vs 対数スケールの明示

**現在**: 線形スケール（Hz比例）
**将来**: 対数スケール（オクターブ表示）のオプション追加

```swift
// 対数スケール版（将来実装）
private func frequencyToCanvasYLog(frequency: Double, canvasHeight: CGFloat, minFreq: Double, maxFreq: Double) -> CGFloat {
    let logMin = log10(max(minFreq, 1.0))
    let logMax = log10(maxFreq)
    let logFreq = log10(max(frequency, 1.0))
    let ratio = (logMax - logFreq) / (logMax - logMin)
    return CGFloat(ratio) * canvasHeight
}
```

#### F. ピクセル丸めとサブピクセルぼけ防止

```swift
private func roundToPixel(_ value: CGFloat) -> CGFloat {
    let scale = UIScreen.main.scale
    return round(value * scale) / scale
}

// 使用例
let canvasY = roundToPixel(frequencyToCanvasY(freq, canvasHeight, maxFreq))
```

#### G. スクロール制御の安定化

```swift
@State private var canvasOffsetY: CGFloat = 0
@State private var lastDragValue: CGFloat = 0

.gesture(
    DragGesture()
        .onChanged { value in
            let newOffset = lastDragValue + value.translation.height

            // 範囲制限
            let maxOffset: CGFloat = 0
            let minOffset = viewportHeight - canvasHeight
            canvasOffsetY = max(minOffset, min(maxOffset, newOffset))
        }
        .onEnded { _ in
            lastDragValue = canvasOffsetY
        }
)
```

#### H. 描画順（Z順）の明確化

```swift
Canvas { context, size in
    // 1. スペクトル（最背面）
    drawSpectrogram(context: context, canvasHeight: canvasHeight, maxFreq: maxFreq)

    // 2. ガイドライン（中間層）
    drawGridLines(context: context, canvasHeight: canvasHeight, maxFreq: maxFreq)

    // 3. Y軸ラベル（最前面）
    drawFrequencyLabels(context: context, canvasHeight: canvasHeight, maxFreq: maxFreq)

    // 4. 再生位置インジケーター（最前面）
    drawPlaybackPosition(context: context, size: size)
}
```

**重要**: すべてCanvas内で描画。`.overlay()`は使用しない。

### 8.3 優先度：低（余裕があれば）

#### I. アクセシビリティ／テストフック

```swift
Canvas { context, size in
    // 描画処理
}
.accessibilityIdentifier("SpectrogramCanvas")

// ラベルには個別ID
// 描画時に記録してアクセシビリティ要素として公開
```

#### J. 色マッピングの改善

```swift
private func magnitudeToColor(_ magnitude: Float, maxMagnitude: Float) -> Color {
    let normalized = CGFloat(magnitude / maxMagnitude)

    // Gamma補正（視覚的に均一に見える）
    let gamma: CGFloat = 0.5
    let corrected = pow(normalized, gamma)

    // 色相マッピング（青→緑→黄→赤）
    let hue = 0.6 - corrected * 0.6
    return Color(hue: hue, saturation: 0.8, brightness: 0.9 * corrected + 0.1)
}
```

#### K. メモリ制限

```swift
private func calculateCanvasHeight(maxFreq: Double, viewportHeight: CGFloat) -> CGFloat {
    let basePixelsPerKHz: CGFloat = isExpanded ? 20 : 10
    let desiredHeight = CGFloat(maxFreq / 1000.0) * basePixelsPerKHz
    let minHeight = viewportHeight * 1.2
    let maxHeight: CGFloat = 10000.0  // 上限設定

    return min(maxHeight, max(desiredHeight, minHeight))
}
```

## 9. 実装チェックリスト

実装完了前に以下を確認：

- [ ] `frequencyToCanvasY()` を一箇所で実装し、すべての描画で使用
- [ ] `canvasHeight` が `viewportHeight` の最小1.2倍を確保
- [ ] 描画ループが可視範囲（時間・周波数）のみを処理
- [ ] スクロールが `.offset(y: canvasOffsetY)` で実装されている
- [ ] Y軸ラベルが Canvas 内に描画されている（overlayではない）
- [ ] テスト用の `accessibilityIdentifier` を追加
- [ ] 実機で Instruments を実行し FPS を測定（目標: 60fps維持）
- [ ] visible-only 描画が機能している（プロファイラで確認）

## 10. 段階的実装順序（推奨）

### Phase 1: 基本キャンバス実装
1. `calculateCanvasHeight()` の実装（最小高さ確保）
2. `frequencyToCanvasY()` の実装
3. キャンバス座標でのY軸ラベル描画
4. `.offset()` + `.clipped()` でのスクロール

### Phase 2: スペクトル描画
5. キャンバス座標でのスペクトル描画
6. 周波数ビン幅の厳密な計算

### Phase 3: パフォーマンス最適化
7. Visible-only 描画の実装
8. `.drawingGroup()` の追加
9. 実機プロファイリング

### Phase 4: UI改善
10. ラベル重なり回避
11. ピクセル丸め
12. スクロール制御の安定化

### Phase 5: テスト
13. UIテストの作成・実行
14. アクセシビリティIDの追加
15. スクロール動作の検証

## 11. 次のステップ

1. ✅ この計画書を確認
2. ⬜ Phase 1: 基本キャンバス実装
3. ⬜ Phase 2: スペクトル描画
4. ⬜ Phase 3: パフォーマンス最適化
5. ⬜ UIテストで検証

---

**重要**:
- ビューポート基準の計算は一切行わない
- 必ず Phase 1 → Phase 3 の順で実装（最適化は後回しにしない）
- 実機プロファイリングは必須（シミュレータのみでは不十分）
