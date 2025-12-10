# 時間軸スクロール - ギャップ分析と実装計画

**作成日**: 2025-11-14
**目的**: 現在の実装と要件のギャップを分析し、正しい実装計画を策定

---

## 1. 要件の再確認

**要件書**: `spectrogram_time_axis_requirements.md`

### 1.1 核心要件

1. **赤線は画面中央に固定** - 動かない
2. **0sが初期状態で赤線下** - 紙の左端が赤線に接する
3. **再生中、現在時刻が赤線下** - スペクトログラムが左へ流れる
4. **終端時、録音終端が赤線下** - 紙の右端が赤線に接する
5. **時間ラベルはスペクトログラムと同期** - 横方向に流れる
6. **時間ラベル帯は画面下端固定** - 縦スクロールの影響を受けない

---

## 2. 現在の実装の分析

### 2.1 スペクトログラム描画（Line 789-793）

```swift
for (timeIndex, timestamp) in data.timeStamps.enumerated() {
    let timeOffset = timestamp - currentTime
    guard abs(timeOffset) <= timeWindow / 2 else { continue }

    let x = centerX + CGFloat(timeOffset) * pixelsPerSecond
    // ...
}
```

**問題点**:

#### ❌ 問題1: `centerX = canvasWidth / 2`

- **現在**: Canvasの中央を基準にしている
- **要件**: ビューポートの中央（赤線位置）を基準にすべき

**なぜ問題か**:
- `canvasWidth`はデータ全体の幅であり、ビューポート幅とは無関係
- 例: 10秒の録音 → `canvasWidth = 500px`、ビューポート幅 = `300px`
- `centerX = 250px`だが、ビューポート中央は`150px`
- 結果: 描画位置がずれる

#### ❌ 問題2: `timeOffset = timestamp - currentTime`

**現在のロジック**:
- `currentTime = 3s`のとき
- `timestamp = 0s` → `timeOffset = -3s` → `x = centerX - 150px`
- `timestamp = 3s` → `timeOffset = 0s` → `x = centerX`

**問題**:
- `centerX`がCanvas中央（不正確）なので、ビューポート中央とずれる
- 要件: `timestamp = currentTime`のセルがビューポート中央（赤線下）に来るべき

### 2.2 時間ラベル描画（Line 851-870）

```swift
private func drawSpectrogramTimeAxis(context: GraphicsContext, size: CGSize) {
    let pixelsPerSecond: CGFloat = 50
    let timeWindow = Double(size.width / pixelsPerSecond)
    let halfWindow = timeWindow / 2

    let timeOffsets: [Double] = [-halfWindow, 0, halfWindow]
    let positions: [CGFloat] = [0.1, 0.5, 0.9]

    for (offset, position) in zip(timeOffsets, positions) {
        let time = currentTime + offset
        guard time >= 0 else { continue }

        let x = size.width * position
        let y = size.height - 5
        let text = Text(String(format: "%.1fs", time)).font(.caption2).foregroundColor(.white)
        context.draw(text, at: CGPoint(x: x, y: y))
    }
}
```

**問題点**:

#### ❌ 問題3: 固定位置にラベル配置

- **現在**: `positions: [0.1, 0.5, 0.9]` - ビューポートの固定位置
- **要件**: スペクトログラムと同期して流れるべき

**なぜ問題か**:
- 時間ラベルがスペクトログラムと独立して表示されている
- 要件: 0sラベルは常にスペクトログラムの0s位置と同じX座標にあるべき

#### ❌ 問題4: `timeOffset`ベースのラベル

- `[-halfWindow, 0, halfWindow]` - 現在時刻の前後のみ表示
- **要件**: 0s, 1s, 2s, ... と固定間隔で全範囲に配置

### 2.3 再生カーソー描画（Line 837-849）

```swift
private func drawPlaybackPosition(context: GraphicsContext, size: CGSize) {
    let centerX = size.width / 2

    context.stroke(
        Path { path in
            path.move(to: CGPoint(x: centerX, y: 0))
            path.addLine(to: CGPoint(x: centerX, y: size.height))
        },
        with: .color(.white),
        lineWidth: 2
    )
}
```

**問題点**:

#### ❌ 問題5: `size.width`がCanvas幅

- **現在**: `size.width` = `canvasWidth`（データ全体の幅）
- **問題**: Canvas内で描画しているため、`size`はCanvas座標系
- **要件**: ビューポート幅の中央に固定

**解決策の方向性**:
- Canvas内で描画する場合、`translateBy()`で補正が必要
- または、Canvas外にOverlayで描画

### 2.4 Canvas構造（Line 491-526）

```swift
Canvas { context, size in
    // size = canvasWidth × canvasHeight

    // 1. スペクトログラム描画
    drawSpectrogramOnCanvas(...)

    // 2. 周波数ラベル描画
    drawFrequencyLabelsOnCanvas(...)

    // 3. 時間軸・再生カーソー描画（translateBy補正付き）
    var fixedContext = context
    fixedContext.translateBy(x: 0, y: -paperTop)

    drawSpectrogramTimeAxis(context: fixedContext, ...)
    drawPlaybackPosition(context: fixedContext, ...)
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(y: paperTop)
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

**現状**:
- ✅ 単一Canvas構造（良い）
- ✅ `translateBy(y: -paperTop)`で縦スクロール補正（良い）
- ❌ X方向のスクロールがない
- ❌ Canvas座標系とビューポート座標系の混同

---

## 3. ギャップ分析まとめ

| 要件 | 現在の実装 | ギャップ |
|------|-----------|---------|
| 赤線は画面中央固定 | Canvas中央に描画 | ビューポート中央との不一致 |
| 0sが初期状態で赤線下 | `timeOffset`方式で描画位置が不正確 | X座標計算の誤り |
| 再生中、現在時刻が赤線下 | `centerX`が間違っている | Canvas座標とビューポート座標の混同 |
| 時間ラベルが流れる | 固定位置`[0.1, 0.5, 0.9]`に配置 | スペクトログラムと非同期 |
| 時間ラベルは全範囲 | `[-halfWindow, 0, halfWindow]`のみ | ラベルが不足 |

---

## 4. 正しい実装の考え方

### 4.1 座標系の統一

**原則**: すべての描画をCanvas座標系で行う

#### Canvas座標系の定義

```
X軸: 0s位置 = 0px
     1s位置 = 50px
     2s位置 = 100px
     ...

Y軸: 最大周波数 = 0px
     0Hz = canvasHeight
```

#### ビューポート座標系との関係

```
ビューポート中央（赤線） = viewportWidth / 2
時間tのCanvas X座標 = t × pixelsPerSecond
```

**スクロールの本質**:
- 現在時刻tのCanvas位置がビューポート中央に来るように、Canvasを左右にシフト

### 4.2 スクロールオフセットの計算

#### 目標

```
Canvas上の (currentTime × pps) の位置が
ビューポート上の (viewportWidth / 2) に一致する
```

#### オフセット計算

```swift
// Canvas上のcurrentTime位置
let currentTimeCanvasX = currentTime * pixelsPerSecond

// ビューポート中央
let playheadX = viewportWidth / 2

// 必要なX方向オフセット（Canvasを左にシフトする量）
let offsetX = currentTimeCanvasX - playheadX
```

**解釈**:
- `offsetX = 0` のとき: Canvas X=0 がビューポート X=0
- `offsetX = 100` のとき: Canvas X=100 がビューポート X=0（Canvasを左に100px）
- `offsetX = -150` のとき: Canvas X=0 がビューポート X=150（Canvasを右に150px）

#### 初期状態（currentTime = 0）

```swift
offsetX = 0 × 50 - playheadX = -playheadX
```

**結果**: Canvasが右に`playheadX`分シフト
- Canvas X=0 がビューポート中央に来る ✅
- ビューポート左側に余白が生まれる ✅

### 4.3 `.offset()`の適用

```swift
Canvas { context, size in
    // Canvas座標系で描画
    // size = canvasWidth × canvasHeight
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(x: -offsetX, y: paperTop)  // ⚠️ 符号注意
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

**符号の理由**:
- `.offset(x: value)`: 正の値で**右へ**移動、負の値で**左へ**移動
- `offsetX`の定義: Canvasを**左にシフトする量**
- したがって、`.offset(x: -offsetX)` で正しく適用

### 4.4 各要素の描画

#### スペクトログラム（Canvas座標系）

```swift
for (timeIndex, timestamp) in data.timeStamps.enumerated() {
    let x = CGFloat(timestamp) * pixelsPerSecond  // Canvas X座標

    let rect = CGRect(x: x, y: yTop, width: cellWidth, height: cellHeight)
    context.fill(Path(rect), with: .color(color))
}
```

**重要**: `currentTime`や`timeOffset`は使わない。純粋にtimestampから計算。

#### 時間ラベル（Canvas座標系）

```swift
var time: Double = 0
let labelInterval: Double = 1.0  // 1秒間隔

while time <= durationSec {
    let x = CGFloat(time) * pixelsPerSecond  // Canvas X座標
    let y = canvasHeight - 20  // Canvas下部

    let text = Text(String(format: "%.0fs", time))
    context.draw(text, at: CGPoint(x: x, y: y))

    time += labelInterval
}
```

**ポイント**:
- Canvas座標系で0s, 1s, 2s, ...を固定位置に配置
- `.offset()`でスクロールすると、スペクトログラムと同期して流れる

#### 再生カーソー（ビューポート座標系）

```swift
var fixedContext = context
fixedContext.translateBy(x: offsetX, y: -paperTop)  // スクロール補正

let playheadX_viewport = viewportWidth / 2
let cursorPath = Path { path in
    path.move(to: CGPoint(x: playheadX_viewport, y: 0))
    path.addLine(to: CGPoint(x: playheadX_viewport, y: viewportHeight))
}

fixedContext.stroke(cursorPath, with: .color(.red), lineWidth: 2)
```

**ポイント**:
- `translateBy(x: offsetX)`でX方向のスクロールをキャンセル
- `translateBy(y: -paperTop)`でY方向のスクロールをキャンセル
- ビューポート座標系（`viewportWidth / 2`）で描画

#### 周波数ラベル（Canvas Y + ビューポート X）

```swift
var yOnlyContext = context
yOnlyContext.translateBy(x: offsetX, y: 0)  // X方向のスクロールをキャンセル

var freq: Double = 0
while freq <= maxFreq {
    let canvasY = frequencyToCanvasY(freq, canvasHeight, maxFreq)

    let labelText = "\(Int(freq))Hz"
    let text = Text(labelText).font(.caption2).foregroundColor(.white)

    // ビューポート左端に固定
    yOnlyContext.draw(text, at: CGPoint(x: 20, y: canvasY))

    freq += 1000
}
```

**ポイント**:
- `translateBy(x: offsetX)`でX方向のスクロールをキャンセル
- Y座標はCanvas座標系（スペクトログラムと同期）
- X座標は固定（ビューポート左端）

---

## 5. 実装計画（段階的アプローチ）

### Phase 1: `offsetX`の計算と状態管理

#### 5.1.1 状態変数の追加

```swift
@State private var offsetX: CGFloat = 0
```

#### 5.1.2 初期化

```swift
.onAppear {
    if isExpanded {
        let playheadX = viewportWidth / 2
        offsetX = -playheadX  // 初期状態: 0sが赤線下
    }
}
```

#### 5.1.3 currentTime変更時の更新

```swift
.onChange(of: currentTime) { _, newTime in
    let playheadX = viewportWidth / 2
    let currentTimeCanvasX = CGFloat(newTime) * pixelsPerSecond
    offsetX = currentTimeCanvasX - playheadX
}
```

#### 5.1.4 検証

- ログ出力で`offsetX`の値を確認
- 初期状態: `offsetX = -viewportWidth / 2` (負の値)
- 再生中: `offsetX`が増加
- 視覚的変化なし（まだ`.offset()`に適用していない）

---

### Phase 2: スペクトログラム描画の修正

#### 5.2.1 X座標計算の変更

**Before**:
```swift
let timeOffset = timestamp - currentTime
let x = centerX + CGFloat(timeOffset) * pixelsPerSecond
```

**After**:
```swift
let x = CGFloat(timestamp) * pixelsPerSecond  // Canvas座標系
```

#### 5.2.2 可視範囲フィルタリングの追加（最適化）

```swift
// ビューポート可視範囲（Canvas座標系）
let visibleLeft = offsetX
let visibleRight = offsetX + viewportWidth

for (timeIndex, timestamp) in data.timeStamps.enumerated() {
    let x = CGFloat(timestamp) * pixelsPerSecond

    // 可視範囲外はスキップ
    guard x >= visibleLeft - cellWidth && x <= visibleRight + cellWidth else {
        continue
    }

    // 描画処理
    // ...
}
```

#### 5.2.3 検証

- ログ出力で描画範囲を確認
- 視覚的変化なし（まだ`.offset()`に適用していない）

---

### Phase 3: 時間ラベル描画の修正

#### 5.3.1 Canvas座標系での固定配置

**Before**:
```swift
let timeOffsets: [Double] = [-halfWindow, 0, halfWindow]
let positions: [CGFloat] = [0.1, 0.5, 0.9]

for (offset, position) in zip(timeOffsets, positions) {
    let time = currentTime + offset
    let x = size.width * position
    // ...
}
```

**After**:
```swift
var time: Double = 0
let labelInterval: Double = 1.0  // 1秒間隔

while time <= durationSec {
    let x = CGFloat(time) * pixelsPerSecond  // Canvas X座標
    let y = canvasHeight - 20  // Canvas下部

    let text = Text(String(format: "%.0fs", time))
        .font(.caption2)
        .foregroundColor(.white)

    context.draw(text, at: CGPoint(x: x, y: y), anchor: .center)

    time += labelInterval
}
```

#### 5.3.2 検証

- ログ出力でラベル位置を確認
- 視覚的変化なし（まだ`.offset()`に適用していない）

---

### Phase 4: `.offset()`の適用とCanvas構造の調整

#### 5.4.1 Canvas構造の変更

**重要な判断**: 時間ラベルをCanvas内に統合

**Before**:
```swift
Canvas { context, size in
    // スペクトログラム
    drawSpectrogramOnCanvas(...)

    // 周波数ラベル
    drawFrequencyLabelsOnCanvas(...)

    // 時間軸（別描画、translateBy補正）
    var fixedContext = context
    fixedContext.translateBy(x: 0, y: -paperTop)
    drawSpectrogramTimeAxis(context: fixedContext, ...)
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(y: paperTop)
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

**After**:
```swift
Canvas { context, size in
    // 1. スペクトログラム（SCROLLABLE - XY両方）
    drawSpectrogramOnCanvas(
        context: context,
        canvasWidth: size.width,
        canvasHeight: size.height,  // ⚠️ 時間ラベル帯を含む高さ
        maxFreq: maxFreq,
        data: data
    )

    // 2. 時間ラベル（SCROLLABLE - X方向のみ、Y固定）
    drawSpectrogramTimeAxis(
        context: context,
        canvasWidth: size.width,
        canvasHeight: size.height,
        durationSec: durationSec
    )

    // 3. 周波数ラベル（SCROLLABLE - Y方向のみ、X固定）
    var yOnlyContext = context
    yOnlyContext.translateBy(x: offsetX, y: 0)  // X軸スクロールをキャンセル
    drawFrequencyLabelsOnCanvas(
        context: yOnlyContext,
        canvasHeight: size.height,
        maxFreq: maxFreq
    )

    // 4. 再生カーソー（FIXED - XY両方固定）
    var fixedContext = context
    fixedContext.translateBy(x: offsetX, y: -paperTop)  // 両方キャンセル
    drawPlaybackPosition(
        context: fixedContext,
        viewportWidth: viewportWidth,
        viewportHeight: viewportHeight
    )
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(x: -offsetX, y: paperTop)  // ✅ 2Dスクロール
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

#### 5.4.2 Canvas高さの調整

**問題**: 時間ラベル帯をCanvas内に含めるため、Canvas高さを調整

**対策**:
- 時間ラベル帯の高さ（例: 30px）をCanvas高さに含める
- または、時間ラベルをCanvas最下部に描画（周波数範囲外）

**推奨アプローチ**:
```swift
// Canvas高さに時間ラベル帯を含めない（現状維持）
// 時間ラベルはCanvas最下部（canvasHeight付近）に描画
// translateBy()は不要（常にCanvas最下部なので、Y方向スクロールの影響を受けない）
```

**修正案**:
```swift
// 時間ラベル描画
var timeAxisContext = context
timeAxisContext.translateBy(x: 0, y: -paperTop)  // Y軸スクロールをキャンセル

var time: Double = 0
while time <= durationSec {
    let x = CGFloat(time) * pixelsPerSecond
    let y = canvasHeight - 10  // Canvas最下部（固定位置）

    timeAxisContext.draw(text, at: CGPoint(x: x, y: y))
    time += 1.0
}
```

#### 5.4.3 検証

- 視覚的確認: スペクトログラムが左へ流れる
- 0sラベルが初期状態で赤線下
- 再生中、赤線下に現在時刻ラベル
- 縦スクロールしても時間ラベルは固定

---

### Phase 5: UIテストと受け入れ基準検証

#### 5.5.1 検証項目

1. **初期表示**:
   - [ ] 0sラベルが赤線下（viewportWidth / 2 の位置）
   - [ ] 赤線より左側に余白がある

2. **再生中**:
   - [ ] 赤線は画面中央で静止
   - [ ] スペクトログラムが左へ流れる
   - [ ] 赤線下に常に現在時刻ラベル

3. **録音終端**:
   - [ ] 録音終了時、赤線下に終了時刻ラベル
   - [ ] スペクトログラムの右端が赤線に一致

4. **縦スクロール**:
   - [ ] 時間ラベル帯は上下に動かない
   - [ ] 周波数ラベルは上下に追従

5. **フルスクリーン切替**:
   - [ ] 時間スケール不変（1秒=50px）
   - [ ] 再生位置不変

#### 5.5.2 ログベース検証

```swift
FileLogger.shared.log(level: "DEBUG", category: "time_axis_verification",
    message: """
    📐 TIME AXIS VERIFICATION:
    - offsetX: \(offsetX)
    - playheadX: \(viewportWidth / 2)
    - currentTime: \(currentTime)
    - 0s label X (canvas): 0 × 50 = 0
    - 0s label X (viewport): 0 - offsetX = \(-offsetX)
    - Should 0s be at playheadX (initial)? \(abs(-offsetX - viewportWidth / 2) < 1.0)
    - currentTime label X (canvas): \(currentTime) × 50 = \(currentTime * 50)
    - currentTime label X (viewport): \(currentTime * 50) - offsetX = \(currentTime * 50 - offsetX)
    - Should currentTime be at playheadX? \(abs((currentTime * 50 - offsetX) - viewportWidth / 2) < 1.0)
    """)
```

---

## 6. リスクと対策

### 6.1 リスク1: 符号ミスによるスクロール反転

**リスク**: `offsetX`と`.offset(x:)`の符号を間違えると、逆方向にスクロール

**対策**:
- ログ出力で`offsetX`の値を常時監視
- 初期状態で`offsetX < 0`（負）を確認
- 再生中、`offsetX`が増加することを確認

### 6.2 リスク2: Canvas高さの計算

**リスク**: 時間ラベル帯をCanvas内に含めると、周波数スクロール範囲が変わる

**対策**:
- Canvas高さは周波数範囲のみで計算（時間ラベル帯を含めない）
- 時間ラベルはCanvas最下部に描画し、`translateBy(y: -paperTop)`で固定

### 6.3 リスク3: パフォーマンス

**リスク**: すべての時間ラベル（0s〜durationSec）を描画すると重い

**対策**:
- 可視範囲外のラベルはスキップ
- `drawingGroup()`でGPU加速
- Instrumentsでプロファイリング

---

## 7. 実装チェックリスト

### 7.1 Phase 1
- [ ] `@State private var offsetX: CGFloat = 0` 追加
- [ ] `.onAppear`で初期化
- [ ] `.onChange(of: currentTime)`で更新
- [ ] ログ出力で値を確認

### 7.2 Phase 2
- [ ] スペクトログラムのX座標計算を変更
- [ ] `let x = CGFloat(timestamp) * pixelsPerSecond`
- [ ] 可視範囲フィルタリング追加
- [ ] ログ出力で描画範囲を確認

### 7.3 Phase 3
- [ ] 時間ラベルをCanvas座標系で固定配置
- [ ] `while time <= durationSec`ループ
- [ ] ログ出力でラベル位置を確認

### 7.4 Phase 4
- [ ] Canvas構造を調整
- [ ] 周波数ラベルに`translateBy(x: offsetX, y: 0)`追加
- [ ] 再生カーソーに`translateBy(x: offsetX, y: -paperTop)`追加
- [ ] 時間ラベルに`translateBy(x: 0, y: -paperTop)`追加
- [ ] `.offset(x: -offsetX, y: paperTop)`に変更

### 7.5 Phase 5
- [ ] UIテストで全受け入れ基準を検証
- [ ] ログベース検証を実施
- [ ] パフォーマンステスト（Instruments）

---

## 8. まとめ

### 8.1 現在の実装の問題

1. ❌ Canvas中央（`canvasWidth / 2`）を基準にしている
2. ❌ `timeOffset`方式でX座標を計算している
3. ❌ 時間ラベルが固定位置`[0.1, 0.5, 0.9]`に配置されている
4. ❌ X方向のスクロールがない

### 8.2 正しい実装の核心

1. ✅ すべての描画をCanvas座標系で行う
2. ✅ `offsetX = currentTime × pps - viewportWidth / 2`でスクロールオフセットを計算
3. ✅ `.offset(x: -offsetX, y: paperTop)`で2Dスクロール
4. ✅ `translateBy()`で各要素のスクロール追従を制御

### 8.3 段階的実装の重要性

各Phaseで縦軸スクロールが壊れていないことを確認しながら進める。
特にPhase 4の`.offset()`適用時に注意。
