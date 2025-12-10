# グラフ拡張機能 - 期待値とのずれ調査報告

## 調査日時
2025-11-11

## 問題概要
グラフ拡張機能のピクセル密度修正(80→30)を実施し、動的時間軸ラベルも実装したが、実際の改善率(1.92倍)が期待値(3.64倍)と大きく乖離している。

## 実測値

### スクリーンショット分析結果
- **通常表示**: 0.0s → 3.6s (3.6秒幅)
- **拡張表示**: 0.0s → 6.9s (6.9秒幅)
- **実際の改善率**: 1.92倍

### 逆算した画面幅
- **通常表示**: 3.6秒 × 50 px/s = **180 pt**
- **拡張表示**: 6.9秒 × 30 px/s = **207 pt**

## 期待値

### iPhone 16スペック
- **全画面幅**: 393 pt (論理解像度)
- **物理解像度**: 1179 px (3xスケール)

### 本来期待される値
- **拡張表示(フルスクリーン使用時)**: 393 pt / 30 px/s = **13.1秒**
- **期待される改善率**: 13.1 / 3.6 = **3.64倍**

## 問題の本質

### 画面幅使用率
```
拡張表示の実際の幅: 207 pt
全画面幅:           393 pt
使用率:             207 / 393 = 52.7%
未使用領域:         186 pt (47.3%)
```

**拡張表示が画面全体を使えておらず、約半分しか使用できていません。**

## 技術的分析

### 現在の実装 (AnalysisView.swift)

#### 拡張表示レイアウト (Line 210-261)
```swift
private func expandedGraphOverlay(for type: ExpandedGraphType) -> some View {
    ZStack(alignment: .topTrailing) {
        // Background
        ColorPalette.background
            .ignoresSafeArea()

        // Graph content
        VStack(spacing: 0) {
            // Graph area (maximized)
            switch type {
            case .spectrogram:
                SpectrogramView(
                    currentTime: viewModel.currentTime,
                    spectrogramData: viewModel.analysisResult?.spectrogramData,
                    isExpanded: true
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)  // ✓ フルスクリーン指定
```

`.frame(maxWidth: .infinity, maxHeight: .infinity)` により、理論上は全画面を使用するはずですが、実際には207ptに制限されています。

### 制約の可能性

1. **CompactPlaybackControl の影響**
   - 下部に配置されるプレイバックコントロールがスペースを取っている可能性
   - しかし、これは高さ方向の制約であり、幅方向には影響しないはず

2. **Safe Area の影響**
   - `.ignoresSafeArea()` を背景にのみ適用
   - グラフコンテンツ自体には適用されていない可能性

3. **ZStackのalignment制約**
   - `ZStack(alignment: .topTrailing)` による配置制約の可能性

4. **親Viewからの制約**
   - overlayの実装方法による制約の可能性

## 推奨される修正案

### Option 1: Safe Areaの完全無視
```swift
SpectrogramView(...)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .ignoresSafeArea(.all)  // グラフ自体もSafe Areaを無視
```

### Option 2: GeometryReaderによる明示的サイズ指定
```swift
GeometryReader { geometry in
    SpectrogramView(...)
        .frame(width: geometry.size.width, height: geometry.size.height)
}
.ignoresSafeArea(.all)
```

### Option 3: フルスクリーンモディファイア
```swift
SpectrogramView(...)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .edgesIgnoringSafeArea(.all)
```

## 次のステップ

1. 拡張表示のレイアウト実装を詳細に検証
2. Safe Area、パディング、マージンの影響を調査
3. GeometryReaderを使用して実際のサイズを計測
4. 適切な修正を実施し、スクリーンショットで検証

## 参考情報

### ピクセル密度の逆関係
- pixels/second が低い → 時間範囲が広い
- pixels/second が高い → 時間範囲が狭い
- 30 px/s は 50 px/s の 60% → 時間範囲は 1.67倍になるべき(同じ画面幅の場合)

### 現在のコード状態
- ピクセル密度: 正しく 30 px/s に設定済み
- 時間軸ラベル: 動的計算に修正済み
- 問題: 拡張表示の画面幅が制限されている
