# 縦軸(周波数軸)拡張機能 - 調査報告

## 調査日時
2025-11-11

## 現状分析

### SpectrogramView の問題点

#### 固定された周波数ラベル (AnalysisView.swift Line 474-504)
```swift
Text("2000Hz")  // 常に固定
Text("1100Hz")  // 常に固定
Text("200Hz")   // 常に固定
```

**問題**: `isExpanded` の状態に関わらず、常に 200Hz〜2000Hz と表示される。

### PitchAnalysisView の実装

#### 動的な周波数範囲計算 (Line 643-648)
```swift
// Expanded view: show wider frequency range
let minFreq = isExpanded ? max(100.0, baseMinFreq - 100) : baseMinFreq
let maxFreq = isExpanded ? min(2000.0, baseMaxFreq + 200) : baseMaxFreq
```

**参考**: PitchAnalysisView は既に動的計算を実装済み。

## 目標

### 通常表示
- データに基づいた周波数範囲を表示
- 例: 実際のデータが 300Hz〜1500Hz なら、その範囲を表示

### 拡張表示
- より広い周波数範囲を表示
- 選択肢1: 固定範囲 (例: 100Hz〜3000Hz)
- 選択肢2: データベース範囲の拡張 (例: ±300Hz)

## 実装方針

### 1. データベースの周波数範囲を取得
```swift
// SpectrogramData から実際の周波数範囲を計算
let frequencyRange = data.frequencyBins.max() ?? 2000.0
let minDisplayFreq = isExpanded ? 100.0 : 200.0
let maxDisplayFreq = isExpanded ? 3000.0 : min(2000.0, frequencyRange)
```

### 2. 動的ラベル生成
```swift
// 3つのラベルを動的に計算
let topLabel = "\(Int(maxDisplayFreq))Hz"
let midLabel = "\(Int((maxDisplayFreq + minDisplayFreq) / 2))Hz"
let bottomLabel = "\(Int(minDisplayFreq))Hz"
```

### 3. フルスクリーン表示での最適化
- iPhone 16 画面高さ: 852pt (論理解像度)
- より広い周波数範囲を表示可能
- 拡張表示で 100Hz〜3000Hz を提案

## 次のステップ

1. SpectrogramView の周波数ラベルを動的計算に変更
2. isExpanded 時の周波数範囲を拡大
3. UIテストで動作確認
4. スクリーンショットで視覚的に検証
