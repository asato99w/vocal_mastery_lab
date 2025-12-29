# AnalysisUITests シナリオ設計書

## 概要

分析画面のUIテストを2つの包括的なテストケースに統合し、実際のユーザーフローに沿った堅牢な検証を行う。

---

## テスト実行方法

```bash
cd vocal_mastery_lab_app

# 全AnalysisUITests実行
./scripts/test-runner.sh ui AnalysisUITests

# 個別テスト実行
./scripts/test-runner.sh ui AnalysisUITests/testAnalysisComprehensiveFlow
./scripts/test-runner.sh ui AnalysisUITests/testGraphVisualizationAndStatistics
```

---

## テスト構成

| テスト名 | 目的 | 想定時間 |
|---------|------|---------|
| `testAnalysisComprehensiveFlow` | 分析開始・進捗表示・再生・シークの完全フロー | ~50秒 |
| `testGraphVisualizationAndStatistics` | グラフタブ切り替え・拡大表示・統計シートの検証 | ~60秒 |

---

## 1. testAnalysisComprehensiveFlow

### ユーザーストーリー
ユーザーが録音を分析し、進捗を確認し、分析完了後に再生・シーク操作を行う

### 前提条件
- 抽出済み録音が1件以上存在
- 録音一覧画面から開始

### フェーズ詳細

#### フェーズ1: 準備（抽出済み録音の作成）
- 録音A作成（2秒）
- ボーカル抽出実行
- 抽出完了後、録音一覧へ遷移

| アサート | 内容 |
|---------|------|
| ✓ | 録音が正常に完了 |
| ✓ | 抽出が正常に完了 |
| ✓ | 録音一覧画面に遷移 |

#### フェーズ2: 分析画面へ遷移
- 録音Aのメニュー → 「ボーカル分析」タップ

| アサート | 内容 |
|---------|------|
| ✓ | 分析画面に遷移（ナビゲーションタイトル「分析」） |
| ✓ | 進捗表示が開始（「分析中...」テキスト） |

#### フェーズ3: 分析進捗の検証
- 進捗バーの表示を確認

| アサート | 内容 |
|---------|------|
| ✓ | 進捗表示が存在 |
| ✓ | パーセンテージ表示が存在 |

#### フェーズ4: 分析完了後の初期状態
- 分析完了待機（タイムアウト60秒）

| アサート | 内容 |
|---------|------|
| ✓ | `RecordingInfoPanel exists` または `RecordingInfoCompact exists` |
| ✓ | `GraphTabPicker exists` |
| ✓ | `AnalysisPlayPauseButton exists` |
| ✓ | `AnalysisProgressSlider exists` |
| ✓ | `AnalysisSeekBackButton exists` |
| ✓ | `AnalysisSeekForwardButton exists` |
| ✓ | `PitchAnalysisView exists`（デフォルトタブ） |

#### フェーズ5: 再生操作
- AnalysisPlayPauseButton タップ（再生開始）

| アサート | 内容 |
|---------|------|
| ✓ | 再生が開始（ボタンアイコン変化） |
| ✓ | 0.5秒後、進捗スライダー値が進行 |

- AnalysisPlayPauseButton タップ（一時停止）

| アサート | 内容 |
|---------|------|
| ✓ | 一時停止（ボタンアイコン変化） |

- AnalysisPlayPauseButton タップ（再開）

| アサート | 内容 |
|---------|------|
| ✓ | 再生再開（ボタンアイコン変化） |

#### フェーズ6: シーク操作
- AnalysisSeekBackButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 位置が後退（5秒戻る） |

- AnalysisSeekForwardButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 位置が前進（5秒進む） |

#### フェーズ7: スライダーによるシーク
- AnalysisProgressSlider を中間位置にドラッグ

| アサート | 内容 |
|---------|------|
| ✓ | 位置が変化 |
| ✓ | 再生が継続 |

#### フェーズ8: 戻る操作
- 戻るボタンタップ

| アサート | 内容 |
|---------|------|
| ✓ | 録音一覧画面に戻る |

---

## 2. testGraphVisualizationAndStatistics

### ユーザーストーリー
分析完了後にグラフタブを切り替え、拡大表示を確認し、統計情報を閲覧する

### 前提条件
- 抽出済み録音が1件以上存在
- 分析が完了している状態

### フェーズ詳細

#### フェーズ1: 準備（分析完了まで）
- 録音A作成（2秒）
- ボーカル抽出実行
- 一覧からボーカル分析へ遷移
- 分析完了待機

| アサート | 内容 |
|---------|------|
| ✓ | 録音・抽出が正常に完了 |
| ✓ | 分析が正常に完了 |

#### フェーズ2: 初期タブ確認（ピッチ分析）
- デフォルトタブの確認

| アサート | 内容 |
|---------|------|
| ✓ | `GraphTabPicker exists` |
| ✓ | `PitchAnalysisView exists` |
| ✓ | `AutoFollowToggle exists` |
| ✓ | `PitchGraphExpandButton exists` |

#### フェーズ3: ピッチグラフ拡大表示
- PitchGraphExpandButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `ExpandedPitchGraphView exists` |
| ✓ | `PitchGraphCollapseButton exists` |
| ✓ | `ExpandedAnalysisPlayPauseButton exists` |

- PitchGraphCollapseButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 通常表示に戻る |
| ✓ | `PitchAnalysisView exists` |

#### フェーズ4: スペクトログラムタブへ切り替え
- GraphTabPicker でスペクトログラムを選択

| アサート | 内容 |
|---------|------|
| ✓ | `SpectrogramView exists` |
| ✓ | `SpectrogramCanvas exists` |
| ✓ | `SpectrogramExpandButton exists` |
| ✓ | `PitchAnalysisView !exists` |

#### フェーズ5: スペクトログラム拡大表示
- SpectrogramExpandButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | `ExpandedSpectrogramView exists` |
| ✓ | `SpectrogramCollapseButton exists` |
| ✓ | `ExpandedAnalysisPlayPauseButton exists` |

- 拡大表示での再生操作
- ExpandedAnalysisPlayPauseButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 再生が開始 |

- SpectrogramCollapseButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | 通常表示に戻る |
| ✓ | `SpectrogramView exists` |

#### フェーズ6: ピッチ分析タブに戻る
- GraphTabPicker でピッチ分析を選択

| アサート | 内容 |
|---------|------|
| ✓ | `PitchAnalysisView exists` |
| ✓ | `SpectrogramView !exists` |

#### フェーズ7: 統計シートの表示
- StatisticsButton タップ（Portrait: `StatisticsButtonCompact`、Landscape: `StatisticsButton`）

| アサート | 内容 |
|---------|------|
| ✓ | `StatisticsSheetView exists` |
| ✓ | `PitchAnalysisSection exists` |
| ✓ | `SpectrumAnalysisSection exists` |
| ✓ | `StatisticsSheetCloseButton exists` |

#### フェーズ8: 統計セクションの展開/折りたたみ
- PitchAnalysisSectionToggleButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | ピッチ分析セクションが展開/折りたたみ |

- PositionSectionToggleButton タップ（存在する場合）

| アサート | 内容 |
|---------|------|
| ✓ | ポジションセクションが展開 |
| ✓ | `PositionSectionContent exists` |

- PitchSectionToggleButton タップ（存在する場合）

| アサート | 内容 |
|---------|------|
| ✓ | ピッチセクションが展開 |
| ✓ | `PitchSectionContent exists` |

- VibratoSectionToggleButton タップ（存在する場合）

| アサート | 内容 |
|---------|------|
| ✓ | ビブラートセクションが展開 |
| ✓ | `VibratoSectionContent exists` または `VibratoSectionNoData exists` |

#### フェーズ9: 統計シートを閉じる
- StatisticsSheetCloseButton タップ

| アサート | 内容 |
|---------|------|
| ✓ | シートが閉じる |
| ✓ | `StatisticsSheetView !exists` |
| ✓ | 分析画面に戻る |

---

## アサート総数

| テスト | フェーズ数 | アサート数（概算） |
|-------|-----------|------------------|
| testAnalysisComprehensiveFlow | 8 | ~35 |
| testGraphVisualizationAndStatistics | 9 | ~45 |
| **合計** | 17 | **~80** |

---

## 検証カバレッジ

- ✅ 分析開始と進捗表示
- ✅ 分析完了後の初期状態
- ✅ 再生・一時停止・再開
- ✅ シーク操作（ボタン・スライダー）
- ✅ グラフタブ切り替え（ピッチ分析⇄スペクトログラム）
- ✅ 拡大表示と縮小
- ✅ 拡大表示での再生操作
- ✅ 統計シートの表示
- ✅ 統計セクションの展開/折りたたみ
- ✅ オートフォロートグル

---

## 主要なaccessibilityIdentifier

### 分析画面共通
| Identifier | 要素 |
|-----------|------|
| `GraphTabPicker` | グラフタブ切り替えピッカー |
| `RecordingInfoPanel` | 録音情報パネル（Landscape） |
| `RecordingInfoCompact` | 録音情報パネル（Portrait） |
| `StatisticsButton` | 統計ボタン（Landscape） |
| `StatisticsButtonCompact` | 統計ボタン（Portrait） |

### 再生コントロール
| Identifier | 要素 |
|-----------|------|
| `AnalysisPlayPauseButton` | 再生/一時停止ボタン |
| `AnalysisSeekBackButton` | 5秒戻るボタン |
| `AnalysisSeekForwardButton` | 5秒進むボタン |
| `AnalysisProgressSlider` | 進捗スライダー |
| `ExpandedAnalysisPlayPauseButton` | 拡大表示時の再生ボタン |

### ピッチ分析
| Identifier | 要素 |
|-----------|------|
| `PitchAnalysisView` | ピッチ分析ビュー |
| `AutoFollowToggle` | オートフォロートグル |
| `PitchGraphExpandButton` | 拡大ボタン |
| `PitchGraphCollapseButton` | 縮小ボタン |
| `ExpandedPitchGraphView` | 拡大表示ビュー |

### スペクトログラム
| Identifier | 要素 |
|-----------|------|
| `SpectrogramView` | スペクトログラムビュー |
| `SpectrogramCanvas` | スペクトログラムキャンバス |
| `SpectrogramExpandButton` | 拡大ボタン |
| `SpectrogramCollapseButton` | 縮小ボタン |
| `ExpandedSpectrogramView` | 拡大表示ビュー |

### 統計シート
| Identifier | 要素 |
|-----------|------|
| `StatisticsSheetView` | 統計シート |
| `StatisticsSheetCloseButton` | 閉じるボタン |
| `PitchAnalysisSection` | ピッチ分析セクション |
| `PitchAnalysisSectionToggleButton` | ピッチ分析トグル |
| `SpectrumAnalysisSection` | スペクトル分析セクション |
| `SpectrumAnalysisSectionToggleButton` | スペクトル分析トグル |
| `PositionSectionToggleButton` | ポジショントグル |
| `PositionSectionContent` | ポジションコンテンツ |
| `PitchSectionToggleButton` | ピッチトグル |
| `PitchSectionContent` | ピッチコンテンツ |
| `VibratoSectionToggleButton` | ビブラートトグル |
| `VibratoSectionContent` | ビブラートコンテンツ |
| `VibratoSectionNoData` | ビブラートデータなし表示 |

---

## 注意事項

### 画面回転への対応
- Portrait/Landscapeで異なるレイアウトを使用
- 統計ボタンのIdentifierが異なる（`StatisticsButton` vs `StatisticsButtonCompact`）
- テストでは両方を確認するか、いずれかの存在を確認

### 分析時間
- 分析処理には時間がかかる（録音時間に依存）
- タイムアウトを適切に設定（60秒推奨）

### 拡大表示
- fullScreenCoverで表示
- 拡大表示中は別のビュー階層になる

### 統計セクション
- セクションの存在は分析結果に依存
- ビブラートデータがない場合は `VibratoSectionNoData` が表示される
