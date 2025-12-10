# 時間ラベルとスペクトログラムの位置対応仕様

**作成日**: 2025-11-14
**目的**: 時間軸ラベルがスペクトログラムの各時刻と正確に対応した位置に表示されることを明確化

---

## 1. 核心要件

### 1.1 位置対応の原則

**時間ラベルは、スペクトログラムの対応する時刻の列と完全に同じX座標に配置される**

```
スペクトログラム:
  0s位置の列  1s位置の列  2s位置の列  3s位置の列
     ↓          ↓          ↓          ↓
  [■■■]     [■■■]     [■■■]     [■■■]
     ↓          ↓          ↓          ↓
時間ラベル:
     0s         1s         2s         3s
```

### 1.2 座標系の統一

**スペクトログラムと時間ラベルは、同じCanvas座標系で描画される**

- スペクトログラムのセル: `x = timestamp × pixelsPerSecond`
- 時間ラベル: `x = timestamp × pixelsPerSecond`
- **結果**: 完全に同じX座標 → 位置が一致 ✅

---

## 2. 視覚的イメージ

### 2.1 初期状態（currentTime = 0）

```
┌─────────────────────────────────────────────────────┐
│ ビューポート                                          │
│                                                      │
│ [余白]      0s   1s   2s   3s   4s   5s            │
│             ↑↑   ↑↑   ↑↑   ↑↑   ↑↑   ↑↑            │
│             ││   ││   ││   ││   ││   ││            │
│          スペクトログラムの列                          │
│             ││   ││   ││   ││   ││   ││            │
│             ↓↓   ↓↓   ↓↓   ↓↓   ↓↓   ↓↓            │
│             0s   1s   2s   3s   4s   5s   ← 時間ラベル │
│                                                      │
│      画面中央（赤線）                                 │
│         ↓                                            │
│         ┃ ← 赤線は0s列とラベルの位置に一致            │
└─────────────────────────────────────────────────────┘
```

### 2.2 再生中（currentTime = 3s）

```
┌─────────────────────────────────────────────────────┐
│ ビューポート                                          │
│                                                      │
│ 0s   1s   2s   3s   4s   5s   6s   7s             │
│ ↑↑   ↑↑   ↑↑   ↑↑   ↑↑   ↑↑   ↑↑   ↑↑             │
│ ││   ││   ││   ││   ││   ││   ││   ││             │
│ スペクトログラムの列                                  │
│ ││   ││   ││   ││   ││   ││   ││   ││             │
│ ↓↓   ↓↓   ↓↓   ↓↓   ↓↓   ↓↓   ↓↓   ↓↓             │
│ 0s   1s   2s   3s   4s   5s   6s   7s  ← 時間ラベル │
│                                                      │
│         画面中央（赤線）                              │
│             ↓                                        │
│             ┃ ← 赤線は3s列とラベルの位置に一致        │
└─────────────────────────────────────────────────────┘
```

**重要**: スペクトログラム全体が左へシフトし、時間ラベルも同じ量だけシフトする。
→ 各列とラベルの相対位置関係は常に保たれる ✅

---

## 3. 実装における位置対応の保証

### 3.1 Canvas座標系での描画

#### スペクトログラムのセル描画

```swift
// AnalysisView.swift - drawSpectrogramOnCanvas()
for (timeIndex, timestamp) in data.timeStamps.enumerated() {
    // Canvas X座標（絶対位置）
    let x = CGFloat(timestamp) * pixelsPerSecond

    // セルを描画
    let rect = CGRect(x: x, y: yTop, width: cellWidth, height: cellHeight)
    context.fill(Path(rect), with: .color(color))
}
```

#### 時間ラベル描画

```swift
// AnalysisView.swift - drawSpectrogramTimeAxis()
var time: Double = 0
let labelInterval: Double = 1.0  // 1秒間隔

while time <= durationSec {
    // Canvas X座標（絶対位置）- スペクトログラムと同じ計算式
    let x = CGFloat(time) * pixelsPerSecond

    let text = Text(String(format: "%.0fs", time))
        .font(.caption2)
        .foregroundColor(.white)

    // セルと同じX座標に描画
    context.draw(text, at: CGPoint(x: x, y: y), anchor: .center)

    time += labelInterval
}
```

**ポイント**:
- スペクトログラムセル: `x = timestamp × pps`
- 時間ラベル: `x = time × pps`
- **同じ計算式** → 同じX座標 → 完全一致 ✅

### 3.2 スクロールによる同期

#### Canvas全体のスクロール

```swift
Canvas { context, size in
    // 1. スペクトログラム描画（Canvas座標系）
    drawSpectrogramOnCanvas(...)

    // 2. 時間ラベル描画（Canvas座標系）
    // ⚠️ Y軸スクロールの影響を受けないように補正
    var timeAxisContext = context
    timeAxisContext.translateBy(x: 0, y: -paperTop)
    drawSpectrogramTimeAxis(context: timeAxisContext, ...)
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(x: -offsetX, y: paperTop)  // ✅ Canvas全体を移動
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

**結果**:
- `.offset(x: -offsetX)`でCanvas全体が左右に移動
- スペクトログラムと時間ラベルが**一体となって**移動
- 相対位置関係は保たれる ✅

---

## 4. 横スクロール実装の詳細

### 4.1 offsetXの計算

```swift
// 現在時刻のCanvas上の位置
let currentTimeCanvasX = CGFloat(currentTime) * pixelsPerSecond

// ビューポート中央（赤線位置）
let playheadX = viewportWidth / 2

// スクロールオフセット（Canvasを左にシフトする量）
let offsetX = currentTimeCanvasX - playheadX
```

**例**: currentTime = 3s, pixelsPerSecond = 50, viewportWidth = 300

```
currentTimeCanvasX = 3 × 50 = 150px
playheadX = 300 / 2 = 150px
offsetX = 150 - 150 = 0px
```

**解釈**:
- Canvas X=150px（3s位置）がビューポート X=150px（中央）に来る
- **0s位置**（Canvas X=0）はビューポート X=0に来る
- **0sラベル**（Canvas X=0）もビューポート X=0に来る
- → 0s列と0sラベルが一致 ✅

### 4.2 .offset()の適用

```swift
.offset(x: -offsetX, y: paperTop)
```

**符号の理由**:
- `offsetX = 150`: Canvasを左に150pxシフトしたい
- `.offset(x: -150)`: Canvasを左に150px移動（負の値で左）
- 結果: Canvas X=150が画面左端（X=0）に移動
- しかし、ビューポートは幅300pxなので、Canvas X=150〜450が表示される
- **待って、計算が違う！**

**正しい理解**:
- `offsetX = currentTimeCanvasX - playheadX`
- `.offset(x: -offsetX)` = `.offset(x: -(currentTimeCanvasX - playheadX))`
- = `.offset(x: playheadX - currentTimeCanvasX)`

**例**: currentTime = 3s

```
offsetX = 150 - 150 = 0
.offset(x: -0) = .offset(x: 0)
```

→ Canvas X=0〜300が表示される
→ 0s列（Canvas X=0）がビューポート左端（X=0）
→ **あれ、これだと0sが画面中央にこない**

**再計算が必要！**

---

## 5. offsetX計算の再検証

### 5.1 目標の再定義

**目標**: Canvas上の`currentTime`位置がビューポート中央（`playheadX`）に来る

### 5.2 .offset()の動作

```swift
.offset(x: value)
```

- 正の値: Canvasを**右**へ移動
- 負の値: Canvasを**左**へ移動

**ビューポートとの関係**:
```
Canvas X=0が ビューポート X=value の位置に表示される
```

### 5.3 正しい計算

**目標**: Canvas X=`currentTimeCanvasX` がビューポート X=`playheadX` に表示される

**方法1**: `.offset()`の観点から

```
Canvas X=0 が ビューポート X=offsetValue に表示される
→ Canvas X=currentTimeCanvasX が ビューポート X=(offsetValue + currentTimeCanvasX) に表示される

目標: offsetValue + currentTimeCanvasX = playheadX
→ offsetValue = playheadX - currentTimeCanvasX
```

**例**: currentTime = 3s

```
currentTimeCanvasX = 150px
playheadX = 150px
offsetValue = 150 - 150 = 0px

.offset(x: 0)
→ Canvas X=0 が ビューポート X=0
→ Canvas X=150 が ビューポート X=150（画面中央） ✅
```

**例2**: currentTime = 0s（初期状態）

```
currentTimeCanvasX = 0px
playheadX = 150px
offsetValue = 150 - 0 = 150px

.offset(x: 150)
→ Canvas X=0 が ビューポート X=150（画面中央） ✅
→ ビューポート左側（X=0〜149）は空白（グレー余白） ✅
```

### 5.4 変数名の整理

**混乱を避けるため、変数名を変更**:

```swift
// Canvas上の現在時刻位置
let currentTimeCanvasX = CGFloat(currentTime) * pixelsPerSecond

// ビューポート中央（赤線位置）
let playheadX = viewportWidth / 2

// Canvasのオフセット（.offset()に渡す値）
let canvasOffsetX = playheadX - currentTimeCanvasX

// 適用
.offset(x: canvasOffsetX, y: paperTop)
```

**初期状態**:
```swift
canvasOffsetX = playheadX - 0 = playheadX  // 正の値
```

**再生中**:
```swift
canvasOffsetX = playheadX - (currentTime × pps)
```
- currentTimeが増加すると`canvasOffsetX`が減少
- → Canvasが左へ移動（再生が進む） ✅

---

## 6. 修正された実装計画

### 6.1 Phase 1: canvasOffsetX状態管理

```swift
@State private var canvasOffsetX: CGFloat = 0

// 初期化
.onAppear {
    if isExpanded {
        let playheadX = viewportWidth / 2
        canvasOffsetX = playheadX  // 初期状態: 0sが赤線下
    }
}

// currentTime変更時の更新
.onChange(of: currentTime) { _, newTime in
    let playheadX = viewportWidth / 2
    let currentTimeCanvasX = CGFloat(newTime) * pixelsPerSecond
    canvasOffsetX = playheadX - currentTimeCanvasX
}
```

### 6.2 Phase 4: .offset()適用

```swift
.offset(x: canvasOffsetX, y: paperTop)  // ✅ そのまま適用
```

### 6.3 各要素のtranslateBy()

#### 周波数ラベル（X固定、Y追従）

```swift
var yOnlyContext = context
yOnlyContext.translateBy(x: -canvasOffsetX, y: 0)  // X軸スクロールをキャンセル
```

#### 時間ラベル（X追従、Y固定）

```swift
var timeAxisContext = context
timeAxisContext.translateBy(x: 0, y: -paperTop)  // Y軸スクロールをキャンセル
// X方向は補正なし → Canvas全体と一緒に移動
```

#### 再生カーソー（XY固定）

```swift
var fixedContext = context
fixedContext.translateBy(x: -canvasOffsetX, y: -paperTop)  // 両方キャンセル

let playheadX_viewport = viewportWidth / 2
// ビューポート座標系で描画
```

---

## 7. 位置対応の検証方法

### 7.1 視覚的検証

1. **初期状態**:
   - [ ] 0sラベルが赤線下（画面中央）
   - [ ] スペクトログラムの0s列が赤線下

2. **再生中（例: 3s）**:
   - [ ] 3sラベルが赤線下（画面中央）
   - [ ] スペクトログラムの3s列が赤線下
   - [ ] 0sラベルと0s列が左側に見える（同じ相対位置）

3. **スクロール確認**:
   - [ ] 再生が進むと、ラベルと列が一体となって左へ流れる
   - [ ] どの時刻でも、ラベルと対応する列が同じX座標

### 7.2 ログベース検証

```swift
FileLogger.shared.log(level: "DEBUG", category: "alignment_verification",
    message: """
    📐 ALIGNMENT VERIFICATION:
    - currentTime: \(currentTime)
    - pixelsPerSecond: \(pixelsPerSecond)
    - viewportWidth: \(viewportWidth)
    - playheadX: \(playheadX)
    - currentTimeCanvasX: \(CGFloat(currentTime) * pixelsPerSecond)
    - canvasOffsetX: \(canvasOffsetX)

    ✅ Verification:
    - 0s canvas X: 0
    - 0s viewport X: 0 + canvasOffsetX = \(canvasOffsetX)
    - Should 0s be at playheadX (initial)? \(currentTime == 0 && abs(canvasOffsetX - playheadX) < 1.0)

    - currentTime canvas X: \(CGFloat(currentTime) * pixelsPerSecond)
    - currentTime viewport X: \(CGFloat(currentTime) * pixelsPerSecond) + canvasOffsetX = \(CGFloat(currentTime) * pixelsPerSecond + canvasOffsetX)
    - Should currentTime be at playheadX? \(abs((CGFloat(currentTime) * pixelsPerSecond + canvasOffsetX) - playheadX) < 1.0)
    """)
```

### 7.3 数式検証

**検証式**:
```
Canvas X座標 + canvasOffsetX = ビューポート X座標
```

**例**: currentTime = 3s

```
3s列のCanvas X = 3 × 50 = 150px
canvasOffsetX = 150 - 150 = 0px
3s列のビューポート X = 150 + 0 = 150px = playheadX ✅

0s列のCanvas X = 0 × 50 = 0px
0s列のビューポート X = 0 + 0 = 0px（画面左端）
```

---

## 8. まとめ

### 8.1 位置対応の保証メカニズム

1. ✅ **同じ座標系**: スペクトログラムと時間ラベルを同じCanvas座標系で描画
2. ✅ **同じ計算式**: `x = timestamp × pixelsPerSecond`
3. ✅ **一体スクロール**: `.offset(x: canvasOffsetX)`でCanvas全体を移動
4. ✅ **時間ラベルは補正なし**: X方向のtranslateBy()を適用しない

### 8.2 難しさのポイント

横スクロール自体は難しくありません。**座標系の統一と符号の正確性**が重要です。

- ❌ 誤解しやすい: Canvas中央とビューポート中央の混同
- ❌ 符号ミス: `offsetX`の定義と`.offset()`の適用
- ✅ 解決策: Canvas座標系で一貫して描画、`.offset()`で全体移動

### 8.3 実装の核心

```swift
// Canvas座標系での描画
let x_spectrogram = CGFloat(timestamp) * pixelsPerSecond
let x_label = CGFloat(time) * pixelsPerSecond
// → 同じ値 → 位置一致 ✅

// Canvas全体のスクロール
let canvasOffsetX = playheadX - currentTimeCanvasX
.offset(x: canvasOffsetX, y: paperTop)
// → スペクトログラムと時間ラベルが一体で移動 ✅
```

これで、時間ラベルとスペクトログラムの列が常に対応した位置に表示されます。
