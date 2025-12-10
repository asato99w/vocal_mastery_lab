# スペクトログラム解像度改善調査報告書

**作成日**: 2025-11-15
**対象**: VocalisStudio スペクトログラム表示機能

---

## エグゼクティブサマリー

### 現状の問題点
- **周波数解像度が極めて粗い**: わずか20個の周波数ビンで0-2000Hzをカバー（1ビン = 100Hz幅）
- **ボーカル分析として不十分**: 倍音構造が見えず、視覚的な分析精度が低い
- **時間解像度も粗い**: 100msサンプリング間隔、オーバーラップなし

### 推奨される改善案
**Phase 1（即時実装推奨）**: 周波数ビン数を 20 → 100 に変更
- **効果**: 1ビンあたり 100Hz → 20Hz に改善（5倍）
- **リスク**: 低（定数変更のみ）
- **データ量**: 48KB → 240KB（依然として軽量）
- **実装時間**: < 5分

**Phase 2（Phase 1確認後）**: 時間解像度の向上
- **効果**: 100ms → 23ms（4.3倍）、75%オーバーラップ導入
- **業界標準**: 音声分析の一般的な設定に準拠

### 期待される効果
- 倍音構造が明確に可視化される
- ビブラートやピッチの細かい変化が捉えられる
- ボーカル分析アプリとしての品質が大幅に向上

---

## 1. 現状分析

### 1.1 現在のパラメータ

**AudioFileAnalyzer.swift の設定**:
```swift
// 時間解像度
private let spectrogramSamplingInterval = 0.1  // 100ms

// 周波数解像度
private let spectrogramFreqBins = 20  // わずか20個
private let spectrogramMaxFreq = 2000.0  // 0-2000Hz

// FFT設定
private let spectrogramBufferSize = 4096  // FFTウィンドウサイズ
private let sampleRate = 44100.0
```

**計算結果**:
| パラメータ | 現在の値 | 評価 |
|-----------|---------|------|
| 周波数ビン数 | 20個 | ❌ 非常に粗い |
| 1ビンあたりの幅 | 100Hz | ❌ ボーカル分析には不十分 |
| 時間フレーム数（60秒） | 600フレーム | ❌ 粗い |
| 時間解像度 | 100ms | ❌ ビブラート追跡不可 |
| オーバーラップ率 | 0%（なし） | ❌ 業界標準外 |
| データサイズ（60秒） | 約48KB | ✅ 軽量 |

### 1.2 FFT理論に基づく評価

**FFTの理論的周波数解像度**:
- FFTバッファサイズ N = 4096サンプル
- サンプリングレート fs = 44100Hz
- 周波数解像度 Δf = fs / N = **10.77Hz**
- FFT出力ビン数 = N/2 = **2048個**

**問題の本質**:
FFTは10.77Hzの高い周波数解像度（2048ビン）を提供しているが、それを**わずか20個のビン**にまとめている。
- **利用率**: 20/2048 ≈ **0.98%**
- **捨てているデータ**: FFT計算結果の99%以上を破棄している

### 1.3 ボーカル分析の要件との比較

**必要な周波数解像度**:
| 対象 | 周波数範囲 | 必要な解像度 | 現在 | 評価 |
|------|-----------|-------------|------|------|
| 基本周波数（男声） | 80-180Hz | 5-10Hz | 100Hz | ❌ |
| 基本周波数（女声） | 150-300Hz | 5-10Hz | 100Hz | ❌ |
| 第1倍音 | 160-600Hz | 10-20Hz | 100Hz | ❌ |
| 第2倍音 | 240-900Hz | 10-20Hz | 100Hz | ❌ |
| フォルマント | 300-3000Hz | 20-50Hz | 100Hz | ⚠️ |

**現在の100Hz/binでは、倍音を個別に識別できない**。

---

## 2. 技術的背景

### 2.1 周波数解像度と時間解像度のトレードオフ

**Gaborの不確定性原理**:
```
Δt × Δf ≥ 定数
```

- **窓長を長くする**: 周波数解像度↑、時間解像度↓
- **窓長を短くする**: 時間解像度↑、周波数解像度↓

**現在の設定（窓長 = 4096サンプル ≈ 93ms）**:
- 理論的周波数解像度: 10.77Hz ✅ 十分
- 理論的時間解像度: 93ms ⚠️ やや粗い

### 2.2 業界標準とベストプラクティス

**音声分析の標準設定**（調査結果）:
- **窓長**: 20-30ms（音声の基本単位）
- **オーバーラップ**: 50-75%（4x-8xオーバーサンプリング）
- **周波数解像度**: 5-10Hz（倍音分離に必要）
- **時間解像度**: 10-25ms（ビブラート追跡に必要）

**参考文献**:
- Avisoft Bioacoustics: "Selecting appropriate spectrogram parameters"
- Speech Processing (Aalto University): "Spectrogram and the STFT"
- DSP Stack Exchange: 専門家による推奨設定

**ボーカル分析特有の要件**:
- 倍音を個別に識別するため、基本周波数（~100Hz）よりも細かい解像度
- ピッチ変化（ビブラート: ~150-200ms周期）を追跡するため、十分な時間解像度
- フォルマント（母音の特徴）を捉えるため、広い周波数範囲

---

## 3. 改善案の比較

### 案1: 周波数解像度のみ向上（保守的アプローチ）

**変更内容**:
```swift
private let spectrogramFreqBins = 100  // 20 → 100
```

**パラメータ**:
| 項目 | 現在 | 改善後 | 変化 |
|------|------|--------|------|
| 周波数ビン数 | 20 | 100 | 5倍 |
| 1ビンあたりの幅 | 100Hz | 20Hz | 1/5 |
| 時間解像度 | 100ms | 100ms | 変更なし |
| オーバーラップ | 0% | 0% | 変更なし |
| データサイズ（60秒） | 48KB | 240KB | 5倍 |
| 描画矩形数（60秒） | 12,000 | 60,000 | 5倍 |

**メリット**:
- ✅ 倍音構造が視覚的に明確になる
- ✅ 実装が極めて簡単（定数1つ変更）
- ✅ リスクが最小
- ✅ データ量は依然として軽量
- ✅ 描画負荷も許容範囲

**デメリット**:
- ❌ 時間解像度は粗いまま
- ❌ ビブラートの細かい変化は捉えられない

**推奨度**: ★★★★★（最優先で実装すべき）

---

### 案2: 時間解像度も向上（標準アプローチ）

**変更内容**:
```swift
private let spectrogramFreqBins = 100  // 20 → 100
private let spectrogramSamplingInterval = 0.023  // 100ms → 23ms
// 内部的に hop size = 1024サンプル（75%オーバーラップ）
```

**パラメータ**:
| 項目 | 現在 | 改善後 | 変化 |
|------|------|--------|------|
| 周波数ビン数 | 20 | 100 | 5倍 |
| 1ビンあたりの幅 | 100Hz | 20Hz | 1/5 |
| 時間解像度 | 100ms | 23ms | 4.3倍 |
| オーバーラップ | 0% | 75% | 業界標準 |
| 時間フレーム数（60秒） | 600 | 2,600 | 4.3倍 |
| データサイズ（60秒） | 48KB | 約2MB | 42倍 |
| 描画矩形数（60秒） | 12,000 | 260,000 | 22倍 |

**メリット**:
- ✅ 業界標準に準拠
- ✅ 倍音とビブラートの両方を捉えられる
- ✅ 時間的な滑らかさが向上
- ✅ データサイズは依然として許容範囲（2MB）

**デメリット**:
- ⚠️ 描画負荷が高い（260,000矩形）
- ⚠️ 描画最適化が必要になる可能性
- ⚠️ 実装がやや複雑

**推奨度**: ★★★★☆（案1の効果確認後に実装）

---

### 案3: 最高品質（理想アプローチ）

**変更内容**:
```swift
private let spectrogramFreqBins = 200  // 20 → 200
private let spectrogramSamplingInterval = 0.012  // 100ms → 12ms
// 内部的に hop size = 512サンプル（87.5%オーバーラップ）
```

**パラメータ**:
| 項目 | 現在 | 改善後 | 変化 |
|------|------|--------|------|
| 周波数ビン数 | 20 | 200 | 10倍 |
| 1ビンあたりの幅 | 100Hz | 10Hz | 1/10 |
| 時間解像度 | 100ms | 12ms | 8.3倍 |
| オーバーラップ | 0% | 87.5% | 高精度 |
| 時間フレーム数（60秒） | 600 | 5,000 | 8.3倍 |
| データサイズ（60秒） | 48KB | 約4MB | 83倍 |
| 描画矩形数（60秒） | 12,000 | 1,000,000 | 83倍 |

**メリット**:
- ✅ 研究レベルの分析品質
- ✅ 最高の視覚化精度
- ✅ すべての詳細を捉えられる

**デメリット**:
- ❌ 描画負荷が非常に高い（100万矩形超）
- ❌ Metal最適化が必須
- ❌ 実装が複雑
- ⚠️ データサイズがやや大きい（4MB）

**推奨度**: ★★☆☆☆（将来的な選択肢、現時点では過剰）

---

## 4. 実装戦略

### 4.1 推奨アプローチ: 段階的改善

**Phase 1: 周波数解像度の即時改善**（優先度: 最高）

**目的**: 最小限の変更で最大の効果

**変更箇所**:
```swift
// AudioFileAnalyzer.swift: L23
private let spectrogramFreqBins = 100  // 20 → 100
```

**期待される効果**:
- 倍音構造が明確に見える
- 視覚的品質が大幅に向上
- ユーザー体験の即座の改善

**リスク評価**: ⭐ 極めて低
- 変更範囲が最小（定数1つ）
- データ量は軽量（240KB）
- 描画負荷も許容範囲（60,000矩形）

**実装時間**: < 5分
**テスト時間**: 数分（実際の録音で確認）

---

**Phase 2: 時間解像度の改善**（優先度: 中）

**前提条件**: Phase 1の効果を確認し、ユーザーがさらなる改善を望む場合

**目的**: 業界標準に準拠し、ビブラート追跡を可能にする

**変更箇所**:
```swift
// AudioFileAnalyzer.swift: L18
private let spectrogramSamplingInterval = 0.023  // 0.1 → 0.023 (23ms)
```

内部的に以下の計算が変更される:
```swift
// analyzeSpectrogram() 内部
let hopSamples = Int(sampleRate * spectrogramSamplingInterval)
// 4410サンプル(100ms) → 1014サンプル(23ms)
// オーバーラップ率: (4096 - 1014) / 4096 = 75%
```

**期待される効果**:
- ビブラートやピッチ変化を滑らかに追跡
- 時間的な詳細が見える
- 業界標準の品質達成

**リスク評価**: ⭐⭐ 低-中
- 描画負荷が増加（要監視）
- 描画最適化が必要になる可能性

**実装時間**: 10-15分
**テスト時間**: 15-30分（パフォーマンス確認含む）

---

**Phase 3: 描画最適化**（優先度: 低-中）

**前提条件**: Phase 2でパフォーマンス問題が発生した場合のみ

**目的**: 高解像度データを効率的に描画

**実装方針**:

1. **LOD (Level of Detail) 戦略**:
   ```swift
   // 画面解像度に応じてデータをサンプリング
   let visibleFrameCount = Int(viewportWidth / cellWidth)
   let dataFrameCount = spectrogramData.timeStamps.count
   let stride = max(1, dataFrameCount / visibleFrameCount)

   for i in stride(from: 0, to: dataFrameCount, by: stride) {
       // stride間隔でデータを間引いて描画
   }
   ```

2. **drawingGroup() の活用**:
   ```swift
   Canvas { ... }
       .drawingGroup()  // Metal経由でオフスクリーンレンダリング
   ```

3. **AsyncCanvas への移行**（必要に応じて）:
   ```swift
   // 非同期レンダリングでUIをブロックしない
   AsyncCanvas { context, size in
       await renderSpectrogram(...)
   }
   ```

4. **Metal Shader への移行**（必要に応じて）:
   - SwiftUI Canvasの限界に達した場合
   - Metal ShaderでGPU加速

**実装時間**: 数時間～数日（複雑度による）

---

### 4.2 リスク評価と対策

| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|--------|------|
| Phase 1でのメモリ不足 | 極めて低 | 低 | 240KBは軽量、問題なし |
| Phase 1での描画遅延 | 低 | 低 | 60,000矩形は許容範囲 |
| Phase 2でのメモリ不足 | 低 | 中 | 2MBは許容範囲だが監視 |
| Phase 2での描画遅延 | 中 | 中 | LOD実装で対応 |
| 既存機能への影響 | 極めて低 | 中 | 単体テスト実施 |

---

## 5. 技術詳細

### 5.1 変更対象ファイル

**主要ファイル**:
```
VocalisStudio/Infrastructure/Analysis/AudioFileAnalyzer.swift
```

**関連ファイル**（変更不要だが影響を受ける）:
```
Packages/VocalisDomain/Sources/VocalisDomain/ValueObjects/SpectrogramData.swift
VocalisStudio/Presentation/Views/AnalysisView.swift
```

### 5.2 データフロー

```
録音ファイル (m4a)
    ↓
AudioFileAnalyzer.analyze()
    ↓
AudioFileAnalyzer.analyzeSpectrogram()
    ↓ FFT実行（4096サンプルずつ）
    ↓ hop間隔でスライド（現在4410サンプル）
    ↓ 周波数ビンにグルーピング（現在20個）
    ↓
SpectrogramData (timeStamps, frequencyBins, magnitudes)
    ↓
AnalysisView.SpectrogramView
    ↓ Canvas描画
    ↓
画面表示
```

### 5.3 メモリとパフォーマンスの詳細計算

**Phase 1（周波数解像度向上）**:

計算時間:
```
FFT実行回数: 600回（変更なし）
1回のFFT: 4096サンプル、約<1ms（vDSP高速）
総FFT時間: <600ms
後処理: 数百ms
合計: 約1秒（60秒の録音に対して）
```

メモリ:
```
SpectrogramData構造体サイズ:
- timeStamps: [Double] × 600 = 4.8KB
- frequencyBins: [Float] × 100 = 400バイト
- magnitudes: [[Float]] × 600 × 100 = 240KB
合計: 約245KB
```

描画:
```
矩形数: 600 × 100 = 60,000個
SwiftUI Canvas: 数万個の矩形は問題なく描画可能
1フレーム描画時間: 推定10-30ms（60fps維持可能）
```

**Phase 2（時間解像度向上）**:

計算時間:
```
FFT実行回数: 2,600回（4.3倍増加）
総FFT時間: <2.6秒
合計: 約3-4秒（60秒の録音に対して）
非同期処理のため、UI影響なし
```

メモリ:
```
SpectrogramData構造体サイズ:
- timeStamps: [Double] × 2,600 = 21KB
- frequencyBins: [Float] × 100 = 400バイト
- magnitudes: [[Float]] × 2,600 × 100 = 1.04MB
合計: 約1.06MB（5分録音で約5.3MB）
```

描画:
```
矩形数: 2,600 × 100 = 260,000個
要注意: LOD実装が推奨される
画面幅が500pxなら、実際には500×100=50,000個だけ描画
```

---

## 6. 描画最適化手法の詳細

### 6.1 現在の描画実装

```swift
// AnalysisView.swift: drawSpectrogramOnCanvas()
for binIndex in 0..<totalBinsNeeded {  // 周波数ビン
    for (timeIndex, timestamp) in data.timeStamps.enumerated() {  // 時間フレーム
        // 各セルを個別に矩形描画
        let rect = CGRect(x: x, y: yTop, width: cellWidth, height: cellHeight)
        context.fill(Path(rect), with: .color(color))
    }
}
```

**問題点**: 全データを毎回個別に描画（Phase 2で260,000矩形）

### 6.2 LOD (Level of Detail) 実装例

```swift
private func drawSpectrogramOnCanvas(...) {
    // 画面に表示される実際の時間フレーム数を計算
    let visibleTimeRange = canvasWidth / pixelsPerSecond  // 表示秒数
    let visibleFrameCount = Int(canvasWidth / cellWidth)  // 画面に収まるフレーム数
    let dataFrameCount = data.timeStamps.count  // データのフレーム数

    // データを間引く間隔を計算
    let stride = max(1, dataFrameCount / visibleFrameCount)

    for binIndex in 0..<totalBinsNeeded {
        // stride間隔でデータをサンプリング
        for timeIndex in stride(from: 0, to: dataFrameCount, by: stride) {
            let timestamp = data.timeStamps[timeIndex]
            // 描画処理
            ...
        }
    }
}
```

**効果**: 2,600フレーム → 500フレーム（画面幅による）に削減

### 6.3 SwiftUI Canvas最適化

**drawingGroup() の活用**:
```swift
Canvas { context, size in
    // スペクトログラム描画
    ...
}
.drawingGroup()  // Metal経由でオフスクリーンレンダリング
```

**注意点**:
- 単純な描画では逆効果の可能性
- パフォーマンス問題が実際に発生してから使用
- オフスクリーンレンダリングのオーバーヘッドあり

### 6.4 AsyncCanvas への移行（必要時）

```swift
// 非同期レンダリングでUIをブロックしない
import AsyncCanvas

AsyncCanvas { context, size in
    await Task.detached {
        // 重い描画処理
        return renderedImage
    }.value
}
```

**利点**:
- メインスレッドをブロックしない
- UIの応答性を維持
- 複雑な描画に対応

---

## 7. 参考資料

### 7.1 学術論文・技術資料

1. **Gabor, D. (1946)**: "Theory of communication" - 時間-周波数不確定性原理
2. **de Cheveigné, A. & Kawahara, H. (2002)**: "YIN, a fundamental frequency estimator for speech and music" - ピッチ検出
3. **Avisoft Bioacoustics**: "Selecting appropriate spectrogram parameters" - パラメータ選択ガイド
4. **Aalto University**: "Introduction to Speech Processing - Spectrogram and the STFT"

### 7.2 業界標準

- **音声分析標準**: 20-30ms窓長、50-75%オーバーラップ
- **FFT設定**: 2048-4096サンプル、Hanning/Hamming窓
- **周波数解像度**: 5-10Hz（音声分析）

### 7.3 実装参考

- **Audacity**: オープンソース音声編集ソフト、スペクトログラム実装
- **Praat**: 音声分析ソフトウェア、業界標準
- **SwiftUI Canvas**: Apple公式ドキュメント "Add rich graphics to your SwiftUI app" (WWDC21)
- **Metal Performance Shaders**: 高速描画

---

## 8. 結論と推奨事項

### 8.1 即座に実施すべき改善（Phase 1）

**変更内容**: 周波数ビン数を 20 → 100 に変更

**理由**:
1. 最小限の変更で最大の効果
2. リスクが極めて低い
3. 実装が簡単（定数1つ変更）
4. ユーザー体験の即座の改善

**実装**:
```swift
// AudioFileAnalyzer.swift: Line 23
private let spectrogramFreqBins = 100  // 20 → 100
```

**期待される改善**:
- 倍音構造が5倍詳細に見える
- ボーカル分析として実用的なレベルに到達
- データ量・描画負荷ともに許容範囲

### 8.2 将来的な改善（Phase 2）

**タイミング**: Phase 1の効果確認後、ユーザーの要望に応じて

**変更内容**: 時間解像度の向上（75%オーバーラップ導入）

**準備事項**:
- パフォーマンス監視体制
- LOD実装の検討
- ユーザーフィードバック収集

### 8.3 技術的実現可能性

**結論**: ✅ すべての改善案が技術的に実現可能

**根拠**:
- メモリ: Phase 2でも2MB程度、現代のiOSデバイスで全く問題なし
- CPU: FFT計算は高速（vDSP使用）、非同期処理でUI影響なし
- 描画: LOD実装により高解像度データも効率的に描画可能

### 8.4 推奨実装スケジュール

| フェーズ | 内容 | 工数 | タイミング |
|---------|------|------|-----------|
| Phase 1 | 周波数ビン数100 | < 5分 | 即座 |
| 検証 | 実録音で効果確認 | 10分 | Phase 1直後 |
| Phase 2 | 時間解像度向上 | 15分 | Phase 1確認後 |
| Phase 3 | 描画最適化 | 数時間 | 必要に応じて |

---

**調査担当**: Claude Code
**調査方法**: コード分析、FFT理論計算、業界標準調査、Web検索
**信頼性**: 高（理論計算と実装分析に基づく）
