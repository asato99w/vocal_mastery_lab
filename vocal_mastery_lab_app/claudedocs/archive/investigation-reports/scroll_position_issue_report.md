# スペクトログラム フルスクリーン展開時のスクロール位置問題レポート

**作成日時**: 2025-11-12 15:50
**対象機能**: スペクトログラム表示のフルスクリーン展開

## 問題の概要

フルスクリーン展開時に「ピコっと画面が切り替わる」挙動が発生している。
ユーザーの期待: 常に下端（低周波数側、0Hz付近）が表示される

## 観察された挙動

### スクリーンショット分析

**スクリーンショット1** (15:45:42):
- 縦長フルスクリーン表示
- 周波数ラベル: 8k, 7k, 6k, 5k, 4k, 3k, 2k, 1k, 0Hz
- **下端（0Hz）が画面下部に表示されている** ✓

**スクリーンショット2** (15:46:08):
- 横長（より大きい）フルスクリーン表示
- 周波数ラベル: 8k, 7k, 6k, 5k, 4k, 3k, 2k, 1k, 0Hz
- **下端（0Hz）が画面下部に表示されている** ✓

### ユーザー報告

> 「拡大した際に上端に合わせて表示され、クリックすると下端に移動するという謎の挙動」
> 「スクロールでもなく、ピコっと画面が切り替わります」

### 矛盾点

- スクリーンショット: 両方とも下端表示されている
- ユーザー報告: 上端表示 → クリックで下端に移動

## 実装した修正内容

### 修正1: Canvas高さの固定化
**ファイル**: `AnalysisView.swift` Line 559-568
```swift
private func calculateCanvasHeight(maxFreq: Double, viewportHeight: CGFloat) -> CGFloat {
    let basePixelsPerKHz: CGFloat = 60.0
    let canvasHeight = CGFloat(maxFreq / 1000.0) * basePixelsPerKHz
    let maxHeight: CGFloat = 10000.0
    return min(maxHeight, canvasHeight)
}
```
- ❌ 削除: `let minHeight = viewportHeight * 2.0` (viewport依存の最小高さ)
- ✅ 結果: canvasHが480pxで固定（通常/フルスクリーン両方）

### 修正2: スクロール位置の初期化
**ファイル**: `AnalysisView.swift` Line 521-530
```swift
.onChange(of: isExpanded) { _, newValue in
    if newValue {
        let minOffset = viewportHeight - canvasHeight
        canvasOffsetY = min(0, minOffset)
        lastDragValue = canvasOffsetY
    }
}
```
- `isExpanded`が`true`になった時に下端配置を実行

### テスト結果

**UIテスト**: `testSpectrogramExpandDisplay` - ✅ PASSED (33.933秒)

**ログ出力**:
```
scrollableRange=-197.7
canvasOffsetY=0.0
minOffset=197.7
```

**解釈**:
- `canvasH=480 < viewportH=677.7` → キャンバスがビューポートより小さい
- スクロール不要な状態（全体が表示可能）

## 問題の本質（未解決）

### 仮説1: データによる挙動の違い
- テストデータ: 短い録音 → canvasが小さい → スクロール不要
- 実際のデータ: 長い録音 → canvasが大きい → スクロール発生
- **検証が必要**

### 仮説2: 画面遷移時の状態リセット
- `fullScreenCover`での画面遷移時に`canvasOffsetY`が0にリセットされる？
- `.onChange(of: isExpanded)`が発火しない？
- **State管理の問題の可能性**

### 仮説3: GeometryReader再計算タイミング
- ビューポートサイズ計算 → Canvas描画 → スクロール位置設定
- このタイミングのずれで一瞬上端が見える？
- **描画順序の問題の可能性**

## 再現手順が不明確な点

1. **どのタイミングで上端表示されるのか？**
   - フルスクリーン展開直後？
   - 画面回転時？
   - データ長による？

2. **「クリック」とは何を指すのか？**
   - 画面タップ？
   - 閉じるボタン？
   - 特定のUI要素？

3. **「ピコっと切り替わる」の詳細**
   - アニメーションなし？
   - 一瞬で位置が変わる？
   - どの時点で発生？

## 必要な追加情報

### デバッグログの追加箇所
1. `expandedGraphFullScreen()`の実行時
2. `SpectrogramView`の`init`時
3. `canvasOffsetY`の変更時すべて
4. `GeometryReader`のサイズ計算時

### 実機での確認項目
1. 長い録音データでの挙動
2. 画面回転時の挙動
3. 実際の「クリック」操作の内容
4. タイミングの詳細な観察

## 次のステップ

### 優先度1: 再現条件の特定
- [ ] より長い録音データでのテスト
- [ ] 画面回転時の挙動確認
- [ ] 具体的な操作手順の確認

### 優先度2: State管理の見直し
- [ ] `@State private var canvasOffsetY`のライフサイクル確認
- [ ] `fullScreenCover`遷移時のState継承確認
- [ ] `isExpanded`変更タイミングの確認

### 優先度3: 描画順序の最適化
- [ ] GeometryReaderの計算完了待機
- [ ] `.onAppear`と`.onChange`の併用検討
- [ ] 初期オフセット設定の別アプローチ

## 結論

**現状**: 修正は実装したが、実際の問題が再現できていない
**原因**: テストケースと実際の使用状況の差異
**必要**: ユーザーからの詳細な再現手順と具体的な操作内容の確認
