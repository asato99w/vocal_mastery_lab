# Voice Analysis Features - 音声分析機能仕様

## 概要

VocalisStudioで実装可能な高度な音声分析機能についての技術仕様書です。
既存のスペクトル分析・ピッチ検出インフラを活用して、ボイストレーニングに有用なフィードバックを提供します。

---

## 1. 母音判別（Vowel Detection）

### 原理

母音は声道の形状によって決まる**フォルマント周波数（F1, F2）**で特徴づけられます。

### 日本語母音のフォルマント特性

| 母音 | F1 (Hz) | F2 (Hz) | 特徴 |
|------|---------|---------|------|
| あ (a) | 800-1000 | 1200-1500 | F1高、F2中 |
| い (i) | 300-400 | 2200-2800 | F1低、F2高 |
| う (u) | 300-400 | 800-1200 | F1低、F2低 |
| え (e) | 500-700 | 1800-2200 | F1中、F2高 |
| お (o) | 500-600 | 800-1100 | F1中、F2低 |

### 実装アプローチ

| アプローチ | 難易度 | 精度 | 説明 |
|-----------|--------|------|------|
| シンプルなピーク検出 | 中 | 70-80% | スペクトルの局所最大値を検出 |
| LPC（線形予測符号化） | 高 | 85-90% | より精密なフォルマント推定 |
| 機械学習モデル | 高 | 90%+ | 事前学習済みモデルを使用 |

### 処理フロー

```
FFTスペクトル
    ↓
ピーク検出（局所最大値）
    ↓
F1, F2候補の特定
    ↓
母音分類（最近傍法など）
    ↓
判定結果: あ/い/う/え/お
```

### 用途

- 発音の明瞭さの確認
- 高音域での母音維持の練習
- 外国語発音トレーニング

---

## 2. シンガーズフォルマント（Singer's Formant）

### 概要

訓練されたクラシック歌手に特徴的な音響現象。**2500〜3500Hz付近**に現れる強いエネルギーの集中で、オーケストラの音を突き抜けて声が聴衆に届く要因となります。

### 音響特性

| 項目 | 値 |
|------|-----|
| 周波数帯域 | 2500〜3500 Hz（中心約3000 Hz） |
| 帯域幅 | 約500〜1000 Hz |
| 強度 | 周囲の倍音より10〜20 dB高い |

### 発声メカニズム

- **喉頭の低下**: 声道が長くなる
- **咽頭腔の拡大**: F3, F4, F5が近接してクラスター化
- **狭い喉頭口**: 音響インピーダンスの変化

### ピッチによる影響

| ピッチ域 | SF検出 | 注意点 |
|----------|--------|--------|
| 低〜中音域（〜A4） | 容易 | 倍音が豊富、SF明確 |
| 高音域（A4〜C5） | やや困難 | 倍音間隔が広がる |
| 超高音域（C5以上） | 困難 | 倍音がSF帯域に少ない場合あり |

**結論**: シンガーズフォルマントは声道形状で決まるため、ピッチとは比較的独立して安定しています。ただし高音域では検出精度が下がる可能性があります。

### 検出指標

| 指標 | 計算方法 | 用途 |
|------|----------|------|
| SF比率（%） | SF帯域エネルギー / 全体エネルギー | 相対的な強さ |
| SF強度（dB） | SF帯域 - 周囲帯域の平均 | 絶対的なレベル差 |

### 実装例

```swift
struct SingersFormantAnalysis {
    let ratio: Float      // SF帯域のエネルギー比率（0-1）
    let intensity: Float  // 周囲帯域との差（dB）
    let isPresent: Bool   // 閾値を超えているか
    let confidence: Float // 検出信頼度（高音域で低下）
}

func analyzeSingersFormant(spectrum: [Float], pitchHz: Float) -> SingersFormantAnalysis {
    let sfBand = 2500...3500  // Hz
    let sfEnergy = calculateBandEnergy(spectrum, band: sfBand)
    let totalEnergy = calculateTotalEnergy(spectrum)
    let surroundingEnergy = calculateSurroundingEnergy(spectrum, excludeBand: sfBand)

    let ratio = sfEnergy / totalEnergy
    let intensity = 10 * log10(sfEnergy / surroundingEnergy)

    // 高音域では信頼度を下げる
    let confidence = pitchHz < 440 ? 1.0 : max(0.5, 1.0 - (pitchHz - 440) / 500)

    return SingersFormantAnalysis(
        ratio: ratio,
        intensity: intensity,
        isPresent: ratio > 0.1 && intensity > 5,
        confidence: confidence
    )
}
```

---

## 3. 音声スペクトル分析モデル（3層構造）

### 概要

分析対象を**3層構造**に分離することで、ピッチ依存性・物理的起源・音色上位成分を整理して扱います。

| 層 | 主な成分 | 主な目的 | ピッチ依存性 |
|----|----------|----------|-------------|
| **Source層** | F₀、H1–H2、Body、Tilt | 声帯振動・声門閉鎖・厚み | 強い |
| **Vowel層** | F1, F2, Formant Salience | 共鳴焦点、母音の明瞭さ | 中程度（F₀比で補正） |
| **Supra-Vowel層** | Singer's Formant, Brightness, Air, Sibilance | スタイル・抜け・輝き・息成分 | 弱い／ほぼ非依存 |

簡潔に言えば：
- **Source** = "声帯の力学"
- **Vowel** = "口の形"
- **Supra-Vowel** = "空気感と輝き"

---

### 3.1 Source層 — 声帯起源（低域中心）

発声そのもののエネルギー源。「声の厚み」「ブレシー／タイト」「支え」を定量化。

| 指標 | 周波数帯 | 説明 |
|------|----------|------|
| **F₀（ピッチ）** | - | 基本周波数 |
| **H1–H2** | 第1・第2倍音 | 声門閉鎖強度（Open Quotient） |
| **Body** | 100-500 Hz | 胴鳴り、温かさ（F₀倍音に対して正規化） |
| **Spectral Tilt** | 全域 | 高域/低域比（傾き） |

#### H1–H2について

- **H1 > H2**: 声門が緩く閉じる → ブレシー、柔らかい
- **H1 < H2**: 声門がしっかり閉じる → タイト、力強い

```swift
struct SourceFeatures {
    let f0: Float           // Hz（基本周波数）
    let h1h2Ratio: Float    // dB（H1-H2差、正=ブレシー、負=タイト）
    let body: Float         // 正規化された低域エネルギー
    let spectralTilt: Float // dB/octave（傾き）
}
```

---

### 3.2 Vowel層 — 母音共鳴（中域）

声道形状と共鳴位置を表す。「母音焦点」「明瞭さ」「開口/閉口位置」を追う。

| 指標 | 説明 |
|------|------|
| **F1** | 第1フォルマント（口の開き） |
| **F2** | 第2フォルマント（舌の前後位置） |
| **F1/F₀, F2/F₀** | 母音チューニング比（ピッチ補正） |
| **Formant Salience** | フォルマントピークの顕著度 |
| **Formant Stability** | フォルマントの時間的揺れ |

#### 母音とフォルマントの関係（再掲）

| 母音 | F1 (Hz) | F2 (Hz) | F1特性 | F2特性 |
|------|---------|---------|--------|--------|
| あ (a) | 800-1000 | 1200-1500 | 高（開口大） | 中 |
| い (i) | 300-400 | 2200-2800 | 低（開口小） | 高（舌が前） |
| う (u) | 300-400 | 800-1200 | 低 | 低（舌が後） |
| え (e) | 500-700 | 1800-2200 | 中 | 高 |
| お (o) | 500-600 | 800-1100 | 中 | 低 |

```swift
struct VowelFeatures {
    let f1: Float              // Hz
    let f2: Float              // Hz
    let f1NormalizedByF0: Float // F1/F0比
    let f2NormalizedByF0: Float // F2/F0比
    let formantSalience: Float  // ピーク顕著度（0-1）
    let formantStability: Float // 時間的安定度（0-1）
    let detectedVowel: Vowel?   // 推定母音
}

enum Vowel {
    case a, i, u, e, o
}
```

---

### 3.3 Supra-Vowel層 — 高域共鳴（上位倍音群）

スタイル・抜け・輝き・息成分など、音色的印象を左右。ピッチ非依存の「声の抜け」や「録音映え」を定量化。

| 指標 | 周波数帯 | 説明 |
|------|----------|------|
| **Singer's Formant** | 2.5-4 kHz | クラシック声楽の「芯」 |
| **Brightness** | 4-6 kHz | 声の輝き、明るさ |
| **Air / Presence** | 6-9 kHz | 息成分、存在感 |
| **Sibilance** | 8-12 kHz | 歯擦音（任意） |

```swift
struct SupraVowelFeatures {
    let singersFormant: Float  // 2.5-4 kHz帯域エネルギー比
    let brightness: Float      // 4-6 kHz帯域エネルギー比
    let air: Float             // 6-9 kHz帯域エネルギー比
    let sibilance: Float       // 8-12 kHz帯域エネルギー比（任意）
}
```

#### 計測方法: 全帯域エネルギー比

各Supra-Vowel成分は**全帯域エネルギーに対する比率（%）**で表現します。

```swift
// 全帯域エネルギーに対する各成分の比率
let totalEnergy = bandEnergy(100...9000)  // 全帯域（100Hz〜9kHz）

let sfRatio = bandEnergy(2500...4000) / totalEnergy      // 例: 8%
let brightRatio = bandEnergy(4000...6000) / totalEnergy  // 例: 5%
let airRatio = bandEnergy(6000...9000) / totalEnergy     // 例: 3%
```

**意味**: 「声のエネルギー全体のうち、何%がその帯域に分布しているか」

| 成分 | 計算 | 解釈例 |
|------|------|--------|
| SF 8% | 2.5-4kHz / 全体 | 声全体の8%がSF帯域にある |
| Bright 5% | 4-6kHz / 全体 | 声全体の5%がBrightness帯域にある |
| Air 3% | 6-9kHz / 全体 | 声全体の3%がAir帯域にある |

**利点**:
- 意味が明確（「声のX%がこの帯域」）
- 音量非依存（比率なので正規化される）
- 各値を独立して解釈可能
- 計算・実装がシンプル

#### ジャンル別の目安

| ジャンル | SF | Bright | Air |
|----------|-----|--------|-----|
| クラシック（オペラ） | 10-15% | 5-8% | 1-3% |
| ミュージカル | 8-12% | 5-7% | 2-4% |
| ポップス | 5-8% | 4-6% | 3-5% |
| R&B/ソウル | 6-10% | 4-6% | 3-5% |
| ウィスパーボイス | 2-4% | 2-4% | 8-12% |

#### UI表示例

```
┌──────────────────────────────────────┐
│  高域共鳴分析                         │
│                                      │
│  SF      ████████░░   8%             │
│  Bright  █████░░░░░   5%             │
│  Air     ███░░░░░░░   3%             │
└──────────────────────────────────────┘
```

---

### 3.4 Voice Descriptor — 統合構造

```swift
/// 音声スペクトルの3層分析結果
struct VoiceDescriptor {
    let source: SourceFeatures      // 声帯由来
    let vowel: VowelFeatures        // 母音共鳴
    let supra: SupraVowelFeatures   // 高域共鳴
    let timestamp: TimeInterval     // 分析時刻
}
```

---

### 3.5 ジャンル別の傾向

| ジャンル | Source (Body/Tilt) | Vowel (Salience) | Supra (SF/Bright/Air) |
|----------|-------------------|------------------|----------------------|
| クラシック（オペラ） | Body強、Tilt緩 | 高い | SF強、Bright強、Air弱 |
| ミュージカル | Body中、Tilt中 | 高い | SF中〜強、Bright強、Air弱〜中 |
| ポップス | Body中、Tilt中 | 中程度 | SF弱〜中、Bright中、Air中 |
| R&B/ソウル | Body強、Tilt緩 | 中程度 | SF中、Bright中、Air中 |
| ウィスパーボイス | Body弱、Tilt急 | 低い | SF弱、Bright弱、Air強 |

---

### 3.6 分析フロー

```
音声信号
    │
    ├─→ [ピッチ検出] → F₀
    │         │
    │         ↓
    │   [Source層算出]
    │     - H1-H2計算
    │     - Body正規化
    │     - Spectral Tilt
    │
    ├─→ [LPC/包絡抽出] → スペクトル包絡
    │         │
    │         ↓
    │   [Vowel層算出]
    │     - F1, F2検出
    │     - F₀比正規化
    │     - Salience/Stability
    │
    └─→ [帯域積分] → 帯域別エネルギー
              │
              ↓
        [Supra-Vowel層算出]
          - SF, Brightness
          - Air, Sibilance
              │
              ↓
        [VoiceDescriptor統合]
```

---

### 3.7 この構造の利点

| 利点 | 説明 |
|------|------|
| **ピッチ変化に頑強** | F₀依存と非依存を明確分離 |
| **解釈性が高い** | どの層の変化が声質に寄与したか一目で分かる |
| **拡張容易** | Supra層にジャンル別特徴を追加可能 |
| **リアルタイム解析に適合** | 各層の演算コストが独立 |

---

### 3.8 注意点

- 高周波数帯域は**マイク特性**や**録音環境**の影響を受けやすい
- iPhoneマイクでも8000Hz程度までは取得可能
- **絶対値より相対的な変化**を見る方が実用的
- 各層は**独立正規化・時間平滑化**を行う
- 必要に応じて「Source×Vowel」「Vowel×Supra」などの**相互指標**を派生可能

---

## 4. ビブラート分析（Vibrato Analysis）

### ビブラートの音響特性

| パラメータ | 典型的な値 | 説明 |
|------------|-----------|------|
| **Rate（速度）** | 5-7 Hz | 振動の速さ |
| **Extent（深さ）** | ±30-100 cents | ピッチ変動幅（半音=100cents） |
| **Regularity（規則性）** | 0-1 | 周期の安定度 |

### ジャンル別の傾向

| ジャンル | Rate | Extent | 特徴 |
|----------|------|--------|------|
| クラシック（オペラ） | 5-6 Hz | ±50-100 cents | 深く安定 |
| クラシック（リート） | 5-7 Hz | ±30-50 cents | 控えめ |
| ポップス | 5-7 Hz | ±30-50 cents | 軽め、曲調による |
| 演歌 | 4-5 Hz | ±100+ cents | 深く遅め（こぶし） |
| ロック | 6-8 Hz | ±20-40 cents | 速め、浅め |
| ストレートトーン | - | - | ビブラートなし |

### 検出アルゴリズム

```
ピッチ時系列データ（既存機能）
    ↓
ピッチ変動の抽出（中心ピッチからの偏差）
    ↓
周期性分析（自己相関 or FFT）
    ↓
Rate, Extent, Regularity算出
```

### 実装例

```swift
struct VibratoAnalysis {
    let rate: Float       // Hz（振動速度）
    let extent: Float     // cents（振動幅）
    let regularity: Float // 0-1（規則性）
    let isPresent: Bool   // ビブラートあり/なし
}

func analyzeVibrato(pitchHistory: [Float], sampleRate: Float) -> VibratoAnalysis {
    // 1. ピッチ変動を抽出
    let meanPitch = pitchHistory.reduce(0, +) / Float(pitchHistory.count)
    let deviations = pitchHistory.map { $0 - meanPitch }

    // 2. 変動のFFTで周期性を分析
    let fftResult = performFFT(deviations)

    // 3. 4-8Hz帯域でピークを探す（ビブラートの典型的な速度）
    let vibratoRange = 4...8  // Hz
    let peakInRange = findPeakInRange(fftResult, range: vibratoRange, sampleRate: sampleRate)

    // 4. Rate（ピーク周波数）
    let rate = peakInRange.frequency

    // 5. Extent（ピッチ変動の振幅をcentsに変換）
    let maxDeviation = deviations.max() ?? 0
    let minDeviation = deviations.min() ?? 0
    let extentHz = (maxDeviation - minDeviation) / 2
    let extentCents = 1200 * log2((meanPitch + extentHz) / meanPitch)

    // 6. Regularity（ピークの鋭さ）
    let regularity = peakInRange.amplitude / fftResult.averageAmplitude

    return VibratoAnalysis(
        rate: rate,
        extent: extentCents,
        regularity: min(1.0, regularity / 10),
        isPresent: peakInRange.amplitude > threshold && rate > 4 && rate < 8
    )
}
```

### 実装の難易度

| 要素 | 難易度 | 理由 |
|------|--------|------|
| Rate検出 | 低 | ピッチ変動のFFTで可能 |
| Extent検出 | 低 | ピッチの最大-最小差 |
| Regularity検出 | 中 | 自己相関の分析が必要 |
| リアルタイム表示 | 中 | 500ms-1s程度のバッファが必要 |

### トレーニング活用例

| 練習目標 | フィードバック |
|----------|---------------|
| ビブラートを安定させる | Regularityを上げる |
| 速度をコントロール | 目標Rateを設定して練習 |
| 深さを調整 | Extentの目標値を表示 |
| ストレートトーンを練習 | 「ビブラートなし」を維持 |

---

## 5. 実装優先度の提案

### 既存インフラとの親和性

| 機能 | 必要な追加処理 | 難易度 | 即効性 |
|------|---------------|--------|--------|
| ビブラート分析 | ピッチ時系列分析のみ | 低〜中 | 高 |
| シンガーズフォルマント | 帯域エネルギー計算 | 低 | 高 |
| スペクトルバランス | 帯域分割計算 | 低 | 中 |
| 母音判別 | フォルマント検出 | 中〜高 | 中 |

### 推奨実装順序

1. **ビブラート分析**（既存ピッチ検出を活用、即効性高い）
2. **シンガーズフォルマント**（単純な帯域計算）
3. **スペクトルバランス**（SF検出の拡張）
4. **母音判別**（より複雑な処理が必要）

---

## 6. UI表示案

### ビブラート表示

```
┌─────────────────────────────┐
│  ビブラート分析              │
│                             │
│  速度: ████████░░ 6.2 Hz    │
│  深さ: ██████░░░░ 45 cents  │
│  安定: ████████░░ 85%       │
│                             │
│  [良好なビブラート]          │
└─────────────────────────────┘
```

### スペクトルバランス表示

```
┌─────────────────────────────┐
│  声質分析                    │
│                             │
│  Body      ████████░░       │
│  Clarity   ██████████       │
│  SF        ██████░░░░       │
│  Bright    ████░░░░░░       │
│  Air       ██░░░░░░░░       │
│                             │
│  プロファイル: クラシック向き │
└─────────────────────────────┘
```

### 総合ダッシュボード案

```
┌─────────────────────────────────────┐
│  音声分析                            │
├─────────────────────────────────────┤
│  ピッチ: A4 (440Hz)  ズレ: +5 cents  │
│  母音: [あ]                          │
├─────────────────────────────────────┤
│  SF:     ███████░░░ 12%             │
│  Bright: █████░░░░░                  │
├─────────────────────────────────────┤
│  ビブラート: ON                      │
│  Rate: 5.8Hz  Depth: 42cents        │
└─────────────────────────────────────┘
```

---

## 7. 技術的注意事項

### マイク特性

- iPhoneのマイクは8000Hz程度まで信頼性あり
- 高周波数帯域は環境ノイズの影響を受けやすい
- 相対値（変化量）を重視した設計が望ましい

### リアルタイム処理

| 機能 | 必要バッファ | 更新頻度 |
|------|-------------|---------|
| ピッチ検出 | 50-100ms | 20Hz |
| SF検出 | 100-200ms | 10Hz |
| ビブラート | 500-1000ms | 2-5Hz |

### パフォーマンス考慮

- 複数の分析を並列実行する場合はCPU負荷に注意
- 録音後の分析（オフライン）とリアルタイム分析を分離
- 設定でユーザーが有効/無効を切り替えられるように

---

## 改訂履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-12-02 | 1.0 | 初版作成 |
