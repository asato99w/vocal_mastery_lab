# ピッチグラフ ギャップ検出設計書

## 概要

ピッチグラフにおいて、検出がされていない箇所でも線が引かれてしまう問題を解決するための設計書。

## 問題の詳細

### 現状の挙動
`PitchGraphRenderer.drawPitchData` メソッドでは、すべてのピッチデータポイントを連続した線で接続している。

```swift
// PitchGraphRenderer.swift:42-64
public func drawPitchData(...) {
    var path = Path()
    var isFirstPoint = true

    for point in pitchData {
        let x = coordinateSystem.timeToCanvasX(time: point.time, leftPadding: leftPadding)
        let y = coordinateSystem.frequencyToCanvasY(frequency: point.frequency, canvasHeight: canvasHeight)

        if isFirstPoint {
            path.move(to: CGPoint(x: x, y: y))
            isFirstPoint = false
        } else {
            path.addLine(to: CGPoint(x: x, y: y))  // ← すべての点を連続して接続
        }
    }
    // ...
}
```

### 問題点
1. `AnalysisView.preparePitchData` で周波数範囲外のデータはフィルタリングされる
2. しかし、残ったデータポイント間で時間的なギャップがあっても連続した線で接続される
3. これにより、検出がない区間にも斜めの線が引かれ、視認性が悪化する

### 影響を受けるファイル
- `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphRenderer.swift`

## 解決策

### 方式: 時間ギャップ検出

連続するポイント間の時間差が閾値を超えた場合、新しいパスセグメントを開始する。

### 閾値の設定

```swift
// 提案する定数（PitchGraphConstants.swift に追加）
static let gapThreshold: Double = 0.1  // 100ms
```

**理由**:
- ピッチ検出は通常10-50msごとにサンプリングされる
- 100ms以上のギャップは「検出なし」と判断して良い
- 短すぎると正常なデータも分断される可能性がある

### 実装案

```swift
public func drawPitchData(
    context: GraphicsContext,
    canvasHeight: CGFloat,
    pitchData: [(time: Double, frequency: Double, confidence: Float)],
    leftPadding: CGFloat
) {
    guard !pitchData.isEmpty else { return }

    var path = Path()
    var previousTime: Double?

    for point in pitchData {
        let x = coordinateSystem.timeToCanvasX(time: point.time, leftPadding: leftPadding)
        let y = coordinateSystem.frequencyToCanvasY(frequency: point.frequency, canvasHeight: canvasHeight)

        // ギャップ検出: 前のポイントとの時間差が閾値を超えたら新しいセグメントを開始
        let shouldStartNewSegment: Bool
        if let prevTime = previousTime {
            shouldStartNewSegment = (point.time - prevTime) > PitchGraphConstants.gapThreshold
        } else {
            shouldStartNewSegment = true  // 最初のポイント
        }

        if shouldStartNewSegment {
            path.move(to: CGPoint(x: x, y: y))
        } else {
            path.addLine(to: CGPoint(x: x, y: y))
        }

        previousTime = point.time
    }

    context.stroke(
        path,
        with: .color(PitchGraphConstants.pitchLineColor),
        lineWidth: PitchGraphConstants.pitchLineWidth
    )

    // ドットの描画は変更なし（既存のまま）
    // ...
}
```

## 代替案の検討

### 案A: 信頼度（confidence）による分断
- 低い信頼度のポイントで線を分断
- **却下理由**: 信頼度が低くても連続している場合は線を引きたい場合がある

### 案B: 周波数の急激な変化による分断
- 周波数が大きく変化した場合に分断
- **却下理由**: 実際の歌唱では急激な周波数変化（ジャンプ）があり得る

### 案C: 時間ギャップ検出（採用）
- シンプルで予測可能な動作
- ユーザーの期待に沿う（検出がない=線がない）

## テスト計画

### 単体テスト（TDD）

1. **testDrawPitchData_withContinuousData_drawsSinglePath**
   - 連続したデータ（ギャップなし）で1本の線が描画される

2. **testDrawPitchData_withGap_startsNewPathSegment**
   - 100ms以上のギャップがある場合、新しいセグメントが開始される

3. **testDrawPitchData_withSmallGap_continuesPath**
   - 100ms未満のギャップでは線が継続される

4. **testDrawPitchData_withMultipleGaps_createsMultipleSegments**
   - 複数のギャップがある場合、複数のセグメントが作成される

5. **testDrawPitchData_emptyData_drawsNothing**
   - 空データの場合、何も描画されない（既存テストで対応済み）

### 統合テスト（手動確認）

1. 実際の録音データでピッチグラフを表示
2. 検出がない区間で線が引かれていないことを確認
3. 検出がある区間では連続した線が引かれていることを確認

## 実装手順

1. **Phase 1: 定数追加**（Red-Green）
   - `PitchGraphConstants.swift` に `gapThreshold` を追加
   - テストで定数の存在を確認

2. **Phase 2: ギャップ検出実装**（Red-Green-Refactor）
   - テストを先に書く
   - `drawPitchData` メソッドを修正
   - リファクタリング

3. **Phase 3: 統合確認**
   - 実際のアプリで動作確認
   - 必要に応じて閾値を調整

## リスクと対策

| リスク | 対策 |
|--------|------|
| 閾値が不適切で線が分断されすぎる | 閾値を定数化し、調整可能にする |
| パフォーマンスへの影響 | 単純な比較のみなので影響は軽微 |
| 既存テストの破壊 | 既存テストは `frequencyToNoteName` のみで影響なし |

## 参照

- `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphRenderer.swift`
- `VocalisStudio/Presentation/Components/PitchGraph/PitchGraphConstants.swift`
- `VocalisStudio/Presentation/Views/Analysis/AnalysisView.swift`
