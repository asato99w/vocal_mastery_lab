# 縦軸スクロール調査と時間軸スクロール再実装計画

**作成日**: 2025-11-14
**目的**: 縦軸（周波数）スクロールの要件と実装を確認し、それを壊さずに時間軸スクロールを実装するプランを策定

---

## 1. 縦軸（周波数）スクロール - 現在の実装

### 1.1 要件

**Commit 2d4d438, e98c509 で実装済み**

#### アーキテクチャ原則（`spectrogram_canvas_architecture_plan.md`より）

1. **巨大なキャンバスを作成** - 0Hz～MaxHzまでの全高さ
2. **すべてキャンバス座標系で描画** - スペクトルもラベルも
3. **ビューポートは`.clipped()`で覗くだけ** - 描画しない
4. **スクロールはキャンバスの`.offset(y:)`のみ**
5. **セル高さ・ラベル間隔はキャンバス基準で固定**

#### 動作

- **paperTop**: ビューポート上端から見たキャンバス上端のY座標
  - `paperTop = 0` (maxPaperTop): キャンバス上端がビューポート上端に揃う（これ以上上に押せない）
  - `paperTop = viewportH - canvasH` (minPaperTop): キャンバス下端がビューポート下端に揃う（これ以上下に押せない）
  - **初期値**: `minPaperTop` (下揃え、低周波が見える)

### 1.2 現在の実装（Commit 4bf76af - revert後）

```swift
// AnalysisView.swift Line 463-530
GeometryReader { geometry in
    let viewportWidth = geometry.size.width
    let viewportHeight = geometry.size.height
    let maxFreq = getMaxFrequency()
    let canvasHeight = calculateCanvasHeight(maxFreq: maxFreq, viewportHeight: viewportHeight)

    // Canvas: Contains the entire frequency range (0Hz ~ maxFreq)
    Canvas { context, size in
        if let data = spectrogramData, !data.timeStamps.isEmpty {
            // 1. Draw spectrogram (background) - SCROLLABLE
            drawSpectrogramOnCanvas(context: context, canvasWidth: size.width,
                                   canvasHeight: canvasHeight, maxFreq: maxFreq, data: data)

            // 2. Draw Y-axis labels (foreground) - SCROLLABLE
            drawFrequencyLabelsOnCanvas(context: context, canvasHeight: canvasHeight, maxFreq: maxFreq)

            // 3. Draw time axis and playback position - FIXED (viewport coordinates)
            var fixedContext = context
            fixedContext.translateBy(x: 0, y: -paperTop)  // ✅ 重要: スクロール補正

            drawSpectrogramTimeAxis(context: fixedContext, size: CGSize(width: size.width, height: viewportHeight))
            drawPlaybackPosition(context: fixedContext, size: CGSize(width: size.width, height: viewportHeight))
        }
    }
    .frame(width: canvasWidth, height: canvasHeight)     // キャンバスサイズ固定
    .offset(y: paperTop)                                  // ✅ スクロール（縦のみ）
    .frame(width: viewportWidth, height: viewportHeight)  // ビューポート
    .clipped()                                            // クリップ
}
```

#### ✅ 正しく動作している部分

1. **キャンバスサイズ**: `calculateCanvasHeight()` で周波数範囲全体の高さを計算
2. **周波数→Y座標変換**: `frequencyToCanvasY()` でキャンバス座標系に変換
3. **スクロール**: `.offset(y: paperTop)` でキャンバス全体を移動
4. **時間軸の固定**: `translateBy(x: 0, y: -paperTop)` でスクロール補正
5. **ドラッグジェスチャー**: 縦方向優位の判定（`angle > .pi / 4`）

---

## 2. 前回の時間軸スクロール実装で壊れた理由

### 2.1 問題のあった実装（Commit 1623846）

**Canvas構造を大幅に変更** - これが縦軸スクロールを破壊

```swift
// HStack: Separate label column (left lane) and spectrogram area (right lane)
HStack(spacing: 0) {
    // Left lane: Frequency labels column (fixed 72px width)
    Canvas { /* 周波数ラベル */ }
        .frame(width: labelColW, height: canvasHeight)
        .offset(x: 0, y: -paperTop)  // Y-only tracking
        .frame(width: labelColW, height: viewportHeight)
        .clipped()

    // Right lane: VStack containing spectrogram area and time label band
    VStack(spacing: 0) {
        // Upper: Spectrogram canvas
        ZStack(alignment: .topLeading) {
            Canvas { /* スペクトログラム */ }
                .frame(width: canvasWidth, height: canvasHeight)
                .offset(x: -paperLeft, y: -paperTop)  // 2D scroll
                .frame(width: spectroViewportW, height: viewportHeight)
                .clipped()

            Canvas { /* 赤いカーソル */ }  // Foreground overlay
        }

        // Lower: Time label band (separate lane)
        Canvas { /* 時間軸ラベル */ }
            .frame(width: spectroViewportW, height: timeLabelHeight)
    }
}
```

### 2.2 何が壊れたか

#### ❌ 問題1: Canvas分割による座標系の分離

**以前（正常）**:
- 1つのCanvasに全要素（スペクトル、周波数ラベル、時間軸）を描画
- `translateBy(x: 0, y: -paperTop)` で時間軸をスクロール補正

**変更後（破壊）**:
- 3つの独立したCanvas（周波数ラベル、スペクトル、時間軸）
- 各Canvasが独自の座標系を持つ
- **`translateBy()`が機能しない** - 別Canvasなので補正不可

#### ❌ 問題2: 周波数ラベルの独立化

```swift
Canvas { /* 周波数ラベル */ }
    .offset(x: 0, y: -paperTop)  // Y-only tracking
```

- **問題**: `.offset(y: -paperTop)` は「紙を上に動かす」（負の値）
- **正しい動作**: `paperTop` が正なら下にスクロール、負なら上にスクロール
- **しかし**: この実装では符号が逆（`-paperTop`）で、スクロール方向が反転している可能性

#### ❌ 問題3: スペクトログラムの2Dオフセット

```swift
.offset(x: -paperLeft, y: -paperTop)  // 2D scroll
```

- X軸スクロール（`-paperLeft`）とY軸スクロール（`-paperTop`）を同時適用
- **問題**: 周波数ラベルCanvasとスペクトログラムCanvasで`paperTop`の適用方法が異なる
  - 周波数ラベル: `.offset(y: -paperTop)`
  - スペクトログラム: `.offset(x: -paperLeft, y: -paperTop)`
- この不一致により、スクロール時にズレが発生

#### ❌ 問題4: ビューポート高さの分割

```swift
let viewportHeightTotal = geometry.size.height
let timeLabelHeight: CGFloat = 30
let viewportHeight = max(0, viewportHeightTotal - timeLabelHeight)
```

- スペクトログラムのビューポート高さから時間軸の高さを引く
- **問題**: `canvasHeight`の計算が`viewportHeight`を使用していたため、キャンバス高さが不正確
- 結果: スクロール範囲の計算がずれる

### 2.3 なぜ時間軸ラベルは正しく動いたか

時間軸ラベルは**独立したCanvas**で、スペクトログラムと同じ`.offset(x: -paperLeft)`を適用していたため、X方向のスクロールは正しく追従した。

しかし、Y方向（縦軸）のスクロールは**別Canvas**なので、スペクトログラムとのY座標同期が崩れた。

---

## 3. 縦軸スクロールを壊さないための制約

### 3.1 絶対に守るべき原則

1. ✅ **単一Canvas構造を維持**
   - スペクトログラム、周波数ラベル、時間軸、再生カーソルを**1つのCanvas内**に描画
   - HStack/VStack/ZStackでCanvasを分割しない

2. ✅ **`.offset(y: paperTop)` を維持**
   - Y軸スクロールは**キャンバス全体**に対して`paperTop`で制御
   - 符号: 正の値（下にスクロール）、負の値（上にスクロール）

3. ✅ **`translateBy()`による補正を維持**
   - 固定要素（時間軸、再生カーソー）は`translateBy(x: 0, y: -paperTop)`で補正
   - これにより、Y軸スクロールの影響を打ち消す

4. ✅ **ドラッグジェスチャーの縦方向判定を維持**
   ```swift
   let angle = atan2(abs(translation.height), abs(translation.width))
   if angle > .pi / 4 {  // 縦方向優位
       paperTop = max(minPaperTop, min(maxPaperTop, candidate))
   }
   ```

### 3.2 避けるべき実装

1. ❌ Canvas を HStack/VStack/ZStack で分割
2. ❌ 周波数ラベルを独立したCanvasに分離
3. ❌ `.offset(y: -paperTop)` のような符号反転
4. ❌ ビューポート高さの分割計算（時間軸の高さを引く）

---

## 4. 時間軸スクロール再実装プラン（縦軸を壊さない）

### 4.1 設計方針

**原則**: 現在の単一Canvas構造を維持し、その中で時間軸スクロールを実装

#### 4.1.1 Canvas構造（変更なし）

```swift
Canvas { context, size in
    // size = canvasWidth × canvasHeight (データ全体のサイズ)

    // 1. スペクトログラム描画（SCROLLABLE - XY両方）
    drawSpectrogramOnCanvas(...)

    // 2. 周波数ラベル描画（SCROLLABLE - Y方向のみ）
    var yOnlyContext = context
    yOnlyContext.translateBy(x: -paperLeft, y: 0)  // X軸スクロールをキャンセル
    drawFrequencyLabelsOnCanvas(context: yOnlyContext, ...)

    // 3. 時間軸ラベル描画（SCROLLABLE - X方向のみ）
    var xOnlyContext = context
    xOnlyContext.translateBy(x: 0, y: -paperTop)  // Y軸スクロールをキャンセル
    drawSpectrogramTimeAxis(context: xOnlyContext, ...)

    // 4. 再生カーソー描画（FIXED - XY両方固定）
    var fixedContext = context
    fixedContext.translateBy(x: -paperLeft, y: -paperTop)  // 両方キャンセル
    drawPlaybackPosition(context: fixedContext, ...)
}
.frame(width: canvasWidth, height: canvasHeight)
.offset(x: paperLeft, y: paperTop)  // ✅ 2Dスクロール（符号注意）
.frame(width: viewportWidth, height: viewportHeight)
.clipped()
```

#### 4.1.2 `paperLeft` の管理

**仕様**: `spectrogram_time_axis_specification.md` に準拠

```swift
@State private var paperLeft: CGFloat = 0
@State private var lastPaperLeft: CGFloat = 0

// 初期化（currentTime = 0）
let playheadX = viewportWidth / 2
paperLeft = -playheadX  // ✅ 負の値を許容

// 再生中の更新
.onChange(of: currentTime) { _, newTime in
    let playheadX = viewportWidth / 2
    let paperLeft_target = CGFloat(newTime) * pixelsPerSecond - playheadX
    let paperLeft_max = canvasWidth - playheadX
    paperLeft = min(paperLeft_target, paperLeft_max)
}
```

#### 4.1.3 2Dドラッグジェスチャー（NEW）

```swift
.gesture(
    DragGesture()
        .onChanged { value in
            let translation = value.translation
            let angle = atan2(abs(translation.height), abs(translation.width))

            if angle > .pi / 4 {
                // 縦方向優位 - Y軸スクロール（既存の実装）
                let viewportH = viewportHeight
                let canvasH = canvasHeight
                let maxPaperTop: CGFloat = 0
                let minPaperTop = viewportH - canvasH
                let candidate = lastPaperTop + translation.height
                paperTop = max(minPaperTop, min(maxPaperTop, candidate))
            } else {
                // 横方向優位 - X軸スクロール（NEW）
                // ⚠️ 再生中は自動スクロールが優先されるため、手動スクロールは無効化
                // または、ドラッグ開始時に再生を一時停止する
            }
        }
        .onEnded { _ in
            lastPaperTop = paperTop
            lastPaperLeft = paperLeft
        }
)
```

**重要判断**: 手動X軸スクロールの扱い

- **Option A**: 再生中は手動X軸スクロールを無効化（`currentTime`連動のみ）
- **Option B**: ドラッグ開始時に再生を一時停止
- **Option C**: 手動スクロール後、`currentTime`との連動を一時的に解除（複雑）

**推奨**: **Option A** - 再生中は手動X軸スクロール無効化（仕様に準拠）

### 4.2 符号の整理（重要）

#### 4.2.1 `paperTop` の符号

**定義**: ビューポート上端から見たキャンバス上端のY座標

- **正の値**: キャンバスが**下に**移動 → 上部（高周波）が見える
- **負の値**: キャンバスが**上に**移動 → 下部（低周波）が見える
- **初期値**: `minPaperTop = viewportH - canvasH` (通常は負) → 下揃え

**`.offset(y: paperTop)`**: 正しい（変更なし）

#### 4.2.2 `paperLeft` の符号

**定義**: ビューポート左端から見たキャンバス左端のX座標

- **正の値**: キャンバスが**右に**移動 → 左側（過去）が見える
- **負の値**: キャンバスが**左に**移動 → 右側（未来）が見える
- **初期値**: `-playheadX` (負) → 0s位置が画面中央

**`.offset(x: paperLeft)`**: 正しい（符号そのまま）

#### 4.2.3 `translateBy()` の符号

**目的**: スクロールの影響を打ち消す

- **周波数ラベル（Y方向のみスクロール）**: `translateBy(x: -paperLeft, y: 0)`
  - X軸スクロールをキャンセル（左列固定）
- **時間軸ラベル（X方向のみスクロール）**: `translateBy(x: 0, y: -paperTop)`
  - Y軸スクロールをキャンセル（下部固定）
- **再生カーソー（XY両方固定）**: `translateBy(x: -paperLeft, y: -paperTop)`
  - 両方キャンセル（画面中央固定）

### 4.3 実装ステップ（段階的）

#### Phase 1: `paperLeft` 状態管理の追加（スクロールなし）

1. `@State private var paperLeft: CGFloat = 0` を追加
2. 初期化: `paperLeft = -viewportWidth / 2`
3. `onChange(of: currentTime)` で更新式を実装
4. ログ出力で値を確認

**検証**: `paperLeft`が正しく計算されることを確認（視覚的変化なし）

#### Phase 2: Canvas内で`translateBy()`を実装（2D補正）

1. 周波数ラベルに `translateBy(x: -paperLeft, y: 0)` 追加
2. 時間軸ラベルに `translateBy(x: 0, y: -paperTop)` 追加
3. 再生カーソーに `translateBy(x: -paperLeft, y: -paperTop)` 追加

**検証**: 縦軸スクロールが壊れていないことを確認

#### Phase 3: `.offset(x: paperLeft, y: paperTop)` に変更

1. `.offset(y: paperTop)` を `.offset(x: paperLeft, y: paperTop)` に変更

**検証**:
- 再生すると時間軸がスクロールする
- 縦軸スクロールが引き続き動作する

#### Phase 4: UIテストと受け入れ基準検証

`spectrogram_time_axis_specification.md` の検証項目:
1. 初期表示: 0sラベルが画面中央の赤線真下
2. 再生中: 赤線は画面中央で静止、スペクトログラムが左へ流れる
3. 縦スクロール: 周波数を上下スクロールしても時間ラベルは動かない
4. 録音終端: 紙の右端が赤線に到達したら停止

---

## 5. 検証計画

### 5.1 縦軸スクロールの継続動作確認

**テストケース**:
1. スペクトログラムを上下にドラッグ
2. 周波数ラベルが正しくスクロール
3. 時間軸ラベルと再生カーソーは固定
4. スクロール範囲の制限が正しい（`minPaperTop`～`maxPaperTop`）

### 5.2 時間軸スクロールの動作確認

**テストケース**:
1. 再生開始
2. スペクトログラムが左へ流れる
3. 赤い再生カーソーは画面中央で固定
4. 時間軸ラベルがスペクトログラムと同期
5. 周波数ラベルは固定（左列）

### 5.3 2Dスクロールの組み合わせ確認

**テストケース**:
1. 再生中に縦軸スクロール
2. 周波数ラベルと時間軸ラベルが独立して動作
3. 再生カーソーは常に画面中央

---

## 6. リスクと対策

### 6.1 リスク1: `translateBy()`のパフォーマンス

**リスク**: Canvas内で複数回`translateBy()`を呼ぶと描画コストが増加

**対策**:
- `.drawingGroup()` を追加してGPU加速
- Instrumentsでプロファイリング
- 60fps維持を確認

### 6.2 リスク2: 符号ミスによるスクロール反転

**リスク**: `paperLeft`と`paperTop`の符号を間違えると、スクロール方向が逆転

**対策**:
- ログ出力で値を常時監視
- UIテストでスクロール方向を明示的に検証

### 6.3 リスク3: ビューポートサイズ変更時の再計算

**リスク**: フルスクリーン切替時に`playheadX`が変化し、`paperLeft`の再計算が必要

**対策**:
- `.onChange(of: isExpanded)` で `paperLeft`を再計算
- 既存の`paperTop`再計算と同様の処理を追加

---

## 7. 次のステップ

### 7.1 即座に実行

- [x] 縦軸スクロールの要件と実装を確認
- [x] 前回の壊れた状態との比較調査
- [x] 調査結果をドキュメント化（このファイル）

### 7.2 実装フェーズ

- [ ] Phase 1: `paperLeft` 状態管理の追加
- [ ] Phase 2: Canvas内で`translateBy()`を実装
- [ ] Phase 3: `.offset(x: paperLeft, y: paperTop)` に変更
- [ ] Phase 4: UIテストと受け入れ基準検証

### 7.3 検証フェーズ

- [ ] 縦軸スクロールの継続動作確認
- [ ] 時間軸スクロールの動作確認
- [ ] 2Dスクロールの組み合わせ確認
- [ ] パフォーマンステスト（Instruments）

---

## 8. まとめ

### 8.1 前回失敗した理由

**Canvas分割による座標系の分離** が根本原因。HStack/VStack/ZStackで複数Canvasに分けたため、`translateBy()`による補正が機能しなくなり、縦軸スクロールが破壊された。

### 8.2 今回の方針

**単一Canvas構造を維持**し、その中で`translateBy()`を使い分けることで、縦軸と時間軸の両方のスクロールを実現する。

### 8.3 成功の鍵

1. ✅ Canvas構造を変更しない
2. ✅ `translateBy()`で各要素のスクロール追従を制御
3. ✅ `.offset(x: paperLeft, y: paperTop)` で2Dスクロール
4. ✅ 段階的実装とテスト
