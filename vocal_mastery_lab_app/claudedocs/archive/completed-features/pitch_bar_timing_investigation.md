# ピッチグラフ音程バーのタイミングずれ調査報告書

## 問題概要

ピッチグラフの音程バーが、実際にスケールが鳴っているタイミングより**早く表示**されている。
このずれは環境依存があると報告されている。

## 調査結果サマリー

### 計測された値

| 測定方法 | 値 | 説明 |
|----------|-----|------|
| **波形分析（最終的なずれ）** | **~53ms** | タイムスタンプと録音ファイル内の音声開始の差 |
| オーディオタップ（内部遅延） | ~105ms | startNote()からsamplerバッファ出力まで |
| AVAudioSession ioBuffer | 10.88ms | システム報告値 |
| AVAudioSession outputLatency | 0.10ms | システム報告値（シミュレータ） |
| コード実行時間 | 0.01ms | recordNoteStart + startNote |

### 原因の特定

**主原因：AVAudioUnitSampler + AVAudioEngineの内部レイテンシ**

現在のコードフロー:
```swift
recordNoteStart(note)           // ← ここでタイムスタンプ記録
sampler.startNote(...)          // ← MIDIコマンド発行
// ↓ 以下が「見えない遅延」
// - SF2サウンドバンクからのサンプル読み込み
// - AVAudioUnitSamplerの内部処理（エンベロープ等）
// - AVAudioEngineのバッファリング（sampler → mixer → output）
// - オーディオバッファのDAC出力
// = 合計約50-100ms（環境依存）
```

タイムスタンプは「MIDIコマンドを発行した時刻」を記録しているが、
実際の音声出力は「コマンド発行 + 内部レイテンシ」後に発生。

この差分がピッチグラフで「音程バーが早く表示される」原因。

## 環境依存性の説明

以下の要因でレイテンシが変動:

1. **デバイスモデル**: IOバッファサイズが異なる
2. **シミュレータ vs 実機**: シミュレータはmacOSのオーディオスタック経由で処理
3. **オーディオルート**: スピーカー vs ヘッドホンでレイテンシが異なる
4. **システム負荷**: バッファリングが変動

## 詳細計測データ

### AVAudioSession報告値（シミュレータ iPhone 16）

```
ioBuffer = 10.88ms
outputLatency = 0.10ms
inputLatency = 0.10ms
sampleRate = 44100Hz
bufferSamples = 479
```

### コード実行時間

```
recordNoteStart() = 0.003-0.013ms
sampler.startNote() = 0.004-0.005ms
合計 = 0.007-0.018ms
→ 無視可能
```

### オーディオタップによる内部遅延計測

```
startNote()呼び出しからsamplerノードに音声バッファが現れるまで:
actualLatency = 104.57ms
```

### 波形分析による外部遅延計測

```
ログ記録タイムスタンプ: 1.247s
波形で音声確認: ~1.300s
差分: ~53ms
```

## 構造的な理解

```
時間軸:
t=0        : recordNoteStart() → タイムスタンプ記録
t=0.01ms   : startNote() → MIDIコマンド発行
t=???      : [SF2サンプル読み込み + エンベロープ処理]
t=???      : [AVAudioEngineバッファリング]
t=~50ms    : 実際の音声がスピーカーから出力
```

この「~50ms」がAVAudioSessionのレイテンシ（~11ms）だけでは説明できない。
残りの~40msは**AVAudioUnitSamplerの内部処理時間**と推測される。

---

## 実装済み対応: 戦略パターンによるタイムスタンプ記録

### 設計方針

タイムスタンプ記録方式を**戦略パターン**で切り替え可能にした。
これにより、異なるアプローチを容易にテスト・比較できる。

### 実装ファイル

- `VocalisStudio/Infrastructure/Audio/ScaleTimestampStrategy.swift` (新規)
- `VocalisStudio/Infrastructure/Audio/AVAudioEngineScalePlayer.swift` (修正)

### 利用可能な戦略

#### 1. ImmediateTimestampStrategy（デフォルト）

```swift
// 即時記録: startNote()呼び出し時にタイムスタンプを記録
// シンプルだが ~50ms の遅延オフセットあり
let strategy = ImmediateTimestampStrategy()
```

#### 2. TapBasedTimestampStrategy（D案）

```swift
// タップベース: 実際のオーディオバッファ検出時にタイムスタンプを記録
// より正確だが、リアルタイムスレッド制約に注意が必要
let strategy = TapBasedTimestampStrategy()
```

### 使用方法

```swift
// デフォルト（即時記録）で初期化
let scalePlayer = AVAudioEngineScalePlayer(settingsRepository: repo)

// タップベースに切り替え
let tapStrategy = TapBasedTimestampStrategy()
scalePlayer.setTimestampStrategy(tapStrategy)

// 元に戻す
scalePlayer.setTimestampStrategy(ImmediateTimestampStrategy())
```

### 戦略パターンのメリット

| 項目 | 効果 |
|------|------|
| **切り替えコスト** | 1行で切り替え可能 |
| **既存コード影響** | ScalePlayer内部のみ変更 |
| **テスト** | 各戦略を独立してテスト可能 |
| **ロールバック** | 問題があれば即座に元の方式に戻せる |
| **A/Bテスト** | 環境や条件で戦略を切り替え可能 |

### TapBasedTimestampStrategyの動作

```
1. prepareForNoteStart(note) → pendingNoteを設定、audioDetected=false
2. sampler.startNote() → MIDIコマンド発行
3. [タップコールバック] → バッファに音声検出時にaudioDetected=true、時刻記録
4. waitAndRecordNoteStart() → 検出されたタイムスタンプでイベント記録
```

### 停止タイムスタンプについて

**開始のみタップベース、停止は即時記録**のハイブリッドアプローチを採用:

| 操作 | 方式 | 理由 |
|------|------|------|
| noteStart | タップベース | 内部遅延(~50ms)が大きいため正確な検出が必要 |
| noteEnd | 即時記録 | stopNote()のレイテンシは小さく、即時で十分 |

---

## 今後の課題

### 1. 実機での検証

シミュレータと実機では異なるレイテンシ特性を持つ可能性がある。
実機でタップベース戦略の効果を検証する必要がある。

### 2. 将来の追加戦略（検討中）

- **CompensatedTimestampStrategy**: AVAudioSession報告値 + 経験的補正値
- **AudioTimeBasedStrategy**: AVAudioTime/hostTimeを使用した高精度方式

### 3. DIコンテナへの組み込み

現在はデフォルトでImmediateTimestampStrategyを使用。
設定や環境に応じて戦略を選択する仕組みの検討。

---

## 添付: 調査に使用した計測コード

デバッグ用のログ出力は以下のファイルに含まれている:
- `VocalisStudio/Infrastructure/Audio/ScaleTimestampStrategy.swift`
  - TapBasedTimestampStrategy内のFileLogger呼び出し

本番リリース前にはログ出力を無効化することを推奨。
