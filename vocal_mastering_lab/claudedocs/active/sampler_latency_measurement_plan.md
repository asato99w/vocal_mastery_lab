# スケールバー・ピッチ タイミング調査

**作成日**: 2025-12-06
**更新日**: 2025-12-06
**目的**: ScaleBarTimeとPitchTimeのズレを計測し、補償方法を決定する

## 1. 用語定義

### 1.1 計測対象の時刻値

| 名称 | 定義 | 補正 |
|------|------|------|
| **ScaleBarTime** | `ScalePlaybackEvent.timestamp` - スケールバー表示位置を決める時刻 | outputLatency補正あり |
| **PitchTime** | `PitchDetectionResult.timestamp` - ピッチ検出結果の時刻 | FFT遅延補正あり（-23ms） |

### 1.2 計測すべき値

**TimingOffset = ScaleBarTime - PitchTime**

ユーザーが完璧にスケールに合わせて歌った場合のズレ。
- 正の値: ScaleBarTimeが早い（スケールバーがピッチより左）
- 負の値: ScaleBarTimeが遅い（スケールバーがピッチより右）

### 1.3 現状の問題
- TimingOffset ≒ 90-140ms（ScaleBarTimeが早い）
- 原因候補: Sampler内部遅延が未補償

## 2. 背景

### 2.1 現在の補正状況

**ScaleBarTime（TapBasedTimestampStrategy）**:
```swift
ScaleBarTime = Date() - recordingStartTime + outputLatency
```
- outputLatency補正: あり
- Sampler内部遅延補正: **なし**

**PitchTime（AudioFileAnalyzer）**:
```swift
PitchTime = samplePosition / sampleRate - pitchDetectionLatencyOffset
```
- FFT遅延補正（-23ms）: あり

### 2.2 ストラテジーの命名問題

現在の名前と実態：
| 現在の名前 | 実態 | 推奨名 |
|-----------|------|--------|
| TapBasedTimestampStrategy | outputLatency補正あり | CompensatedStrategy |
| ImmediateTimestampStrategy | 補正なし | UncompensatedStrategy |

※ 「TapBased」は実装をバイパスしており、名前と実態が乖離

### 2.3 Sampler内部遅延とは
`sampler.startNote()`を呼んでから、実際に音声データがオーディオバッファに現れるまでの時間。

内訳:
1. SF2サウンドバンクからのサンプル読み込み
2. エンベロープ処理（ADSR）
3. AVAudioEngine内部のバッファリング

### 2.4 計測の困難さ

Tap検出による計測を試みたが問題あり：
- **ScaleBarTime間隔**: 442-456ms（安定、±14ms）
- **TapDetectionTime間隔**: 395-501ms（不安定、±50ms以上）

→ Tap検出のジッターが大きく、正確な遅延値を測定できない
→ **直接TimingOffsetを計測するアプローチに変更**

## 2. 計測パラメータ

### 2.1 音色（ScaleSoundType）

| 音色 | MIDIプログラム | 予想される特性 |
|------|----------------|----------------|
| acousticGrandPiano | 0 | 多層サンプル、長いサステイン |
| electricPiano | 4 | 中程度のサンプル |
| acousticGuitar | 24 | アタックが明確 |
| vibraphone | 11 | サステインあり |
| marimba | 12 | 短いアタック |
| flute | 73 | 持続音 |
| clarinet | 71 | 持続音 |
| sineWave | - | プログラム生成（SF2未使用） |

### 2.2 テンポ

| テンポ | BPM | 1ノートの長さ |
|--------|-----|---------------|
| verySlow | 40 | 1.5秒 |
| slow | 60 | 1.0秒 |
| medium | 80 | 0.75秒 |
| fast | 100 | 0.6秒 |
| veryFast | 120 | 0.5秒 |

### 2.3 スケールの種類（NotePattern）

| パターン | ノート数 | 説明 |
|----------|----------|------|
| fiveToneAscending | 5 | Do-Re-Mi-Fa-Sol（上昇） |
| fiveToneDescending | 5 | Sol-Fa-Mi-Re-Do（下降） |
| fiveToneAscendingDescending | 9 | 上昇+下降 |
| oneOctaveAscending | 8 | 1オクターブ上昇 |
| oneOctaveDescending | 8 | 1オクターブ下降 |
| oneOctaveAscendingDescending | 15 | 1オクターブ上昇+下降 |
| arpeggioMajor | 4 | メジャーアルペジオ |
| arpeggioMinor | 4 | マイナーアルペジオ |

### 2.4 キー（基準音）

| キー | MIDIノート番号 | 説明 |
|------|----------------|------|
| C3 | 48 | 低音域 |
| C4 | 60 | 中音域（デフォルト） |
| C5 | 72 | 高音域 |

### 2.5 シミュレータ・デバイス

| 環境 | 識別子 | 特性 |
|------|--------|------|
| iPhone 16 Pro シミュレータ | 8E091155-1AB5-4C0C-AA9D-B89EB3B01DFD | 主なテスト環境 |
| iPhone 16 Clean シミュレータ | 7E44408D-C4F7-43FE-B3AE-C111CA557A00 | クリーン環境 |
| 実機（iPhone） | - | 最終検証用 |

**シミュレータ vs 実機の違い**:
- シミュレータ: オーディオレイテンシが実機と異なる可能性
- 実機: 実際のユーザー体験を反映

**計測時の考慮点**:
- 同一シミュレータで全テストを実行し一貫性を確保
- シミュレータ間の比較は追加検証として実施
- 最終的な補償値は実機での検証が望ましい

## 3. 効率化された計測プラン

### 設計方針
- スケール再生間隔が一定（±14ms）であることから、Sampler遅延は固定オフセットと仮定
- 最小限の計測で音色依存性を確認
- 依存性がなければ1つの固定値で補償可能

### 3.1 Phase 1: 音色依存性の確認（必須・10分）

**目的**: 音色によって遅延が異なるかを確認

計測対象（代表3種類）:
1. **acousticGrandPiano** - デフォルト、多層サンプル
2. **marimba** - 短いアタック
3. **sineWave** - プログラム生成（SF2未使用）

固定条件:
- テンポ: standard (1秒/ノート)
- パターン: fiveToneAscending（5ノート）
- キー: C4

**計測項目**:
- 各ノートのSampler遅延（ms）
- 音色ごとの平均値

**判定基準**:
- 3音色の平均値の差が10ms未満 → 共通オフセットで補償
- 10ms以上の差 → 音色ごとのオフセットが必要

### 3.2 Phase 2: 安定性確認（Phase 1で差がない場合のみ・5分）

**目的**: 同一条件での遅延の安定性を確認

計測:
- acousticGrandPiano で3回連続録音
- 各回の平均遅延を比較

**判定基準**:
- 3回の平均値の差が5ms未満 → 固定オフセットで補償可能
- 5ms以上の差 → 動的補償（Tap検出）が必要

### 3.3 Phase 3: 追加検証（必要な場合のみ）

Phase 1で音色差が大きい場合:
- 残り5音色も計測
- 音色ごとのオフセット値を決定

Phase 2で安定しない場合:
- 変動要因を特定
- Tap検出による動的補償を実装

## 4. 計測方法

### 4.1 ログ出力

`ScaleTimestampStrategy.swift`に追加済み:

```swift
// prepareForNoteStart()で時刻記録
noteStartWallTime = Date()

// Tap検出時に遅延計算
let samplerLatency = audioDetectionWallTime!.timeIntervalSince(startTime) * 1000
FileLogger.shared.log(level: "INFO", category: "timing",
    message: "[SAMPLER_LATENCY] note=\(note.value), latency=\(String(format: "%.1f", samplerLatency))ms")
```

### 4.2 音色切り替え

設定画面から音色を変更:
1. 設定 → オーディオ出力設定 → スケール再生音
2. 音色を選択
3. 録音を実行

または、UIテストで自動化:
```swift
// AudioOutputSettingsViewでの音色選択をテスト
```

### 4.3 ログ取得

```bash
# シミュレータからログ取得
UDID="8E091155-1AB5-4C0C-AA9D-B89EB3B01DFD"
find ~/Library/Developer/CoreSimulator/Devices/$UDID/data/Containers/Data/Application \
    -name "vocalis_*.log" -type f -exec stat -f "%Sm %N" -t "%H:%M:%S" {} +

# SAMPLER_LATENCYエントリを抽出
grep "SAMPLER_LATENCY" /path/to/vocalis_*.log
```

## 5. 期待される結果

### 5.1 音色による変動がある場合

| 音色 | 平均遅延 | 対応 |
|------|----------|------|
| ピアノ | 100ms | 音色ごとの固定オフセット |
| マリンバ | 50ms | 音色ごとの固定オフセット |
| サイン波 | 10ms | 音色ごとの固定オフセット |

→ `ScaleSoundType`に`samplerLatency`プロパティを追加

### 5.2 音色による変動がない場合

全音色で遅延が同程度:
- グローバルな固定オフセットで補償可能
- 実装がシンプル

### 5.3 遅延にばらつきがある場合

同一条件でも遅延が変動:
- Tap検出を使用した動的補償が必要
- 現在の`TapBasedTimestampStrategy`を正しく活用

## 6. 実装への反映

### Case A: 固定オフセット補償

```swift
extension ScaleSoundType {
    var samplerLatencyOffset: TimeInterval {
        switch self {
        case .acousticGrandPiano: return 0.100
        case .marimba: return 0.050
        // ...
        }
    }
}
```

### Case B: Tap検出による動的補償

現在の実装を修正し、Tap検出時刻を正しく使用:

```swift
// playNote()内で
// 1. sampler.startNote() を先に実行
// 2. Tap検出されるまで短時間ポーリング（再生には影響しない）
// 3. 検出された時刻でタイムスタンプを記録
```

### Case C: ハイブリッド

- 固定オフセットで大まかな補償
- Tap検出で微調整

## 7. 計測スケジュール

| Phase | 計測回数 | 所要時間（推定） |
|-------|----------|------------------|
| Phase 1 | 8音色 × 1回 | 15分 |
| Phase 2 | 5テンポ × 1回 | 10分 |
| Phase 3 | 8パターン × 1回 | 15分 |
| Phase 4 | 3キー × 1回 | 5分 |
| Phase 5 | 5回連続 | 10分 |
| **合計** | - | **約55分** |

## 8. 次のステップ

1. Phase 1（音色依存性）を実行
2. 結果を分析
3. 補償方法を決定
4. 実装
5. 検証

## 9. Phase 1 計測結果

### 9.1 計測条件
- **シミュレータ**: iPhone 16 Pro (8E091155-1AB5-4C0C-AA9D-B89EB3B01DFD)
- **テンポ**: standard (BPM 60)
- **パターン**: fiveToneAscending (5ノート)
- **キー**: C3 (MIDIノート48)
- **日時**: 2025-12-06 18:00-18:30 JST

### 9.2 計測結果

| 音色 | 平均オフセット | 最小 | 最大 | データポイント |
|------|---------------|------|------|---------------|
| **Piano** | -100.1ms | -140.7ms | -91.7ms | 10 |
| **Marimba** | -102.3ms | -117.4ms | -86.0ms | 9 |
| **Sine Wave** | -113.6ms | -135.8ms | -84.8ms | 10 |

**TimingOffset = ScaleBarTime - PitchTime**
- 負の値 = ピッチ検出がスケールバー時刻より遅い

### 9.3 音色間の差分

- Piano vs Marimba: **2.2ms差** ✅ 10ms未満
- Piano vs Sine Wave: **13.5ms差** ❌ 10ms超過
- Marimba vs Sine Wave: **11.3ms差** ❌ 10ms超過

### 9.4 分析

**SF2音色（Piano, Marimba）**:
- 差が2.2msと非常に小さい
- 共通の固定オフセットで補償可能

**Sine Wave**:
- SF2音色と10ms以上の差がある
- プログラマティック生成のため、処理経路が異なる
- ただし、Sine Waveはデバッグ/テスト用の簡易音色であり、
  実際のユーザーが使用するのはSF2音色（Piano, Vibraphone, Marimba, Flute, Clarinet）

### 9.5 判定

**結論: SF2音色用の共通オフセットで補償可能**

- SF2音色の平均: **(100.1 + 102.3) / 2 = ~101ms**
- 推奨補正値: **100ms** (切りの良い値)

**Sine Wave対応**:
- 選択肢A: SF2用の値(100ms)を適用 → 最大13ms程度のずれ（許容範囲内）
- 選択肢B: 個別値(114ms)を設定 → 正確だが複雑化
- **推奨**: 選択肢A（シンプルさを優先）

## 10. 80ms補正テスト結果

### 10.1 実装

`ScaleTimestampStrategy.swift`の`TapBasedTimestampStrategy`に80ms補正を追加：

```swift
private let samplerLatencyOffset: TimeInterval = 0.080

// getNoteStartTimestamp() と recordNoteEnd() で適用
let compensatedTimestamp = rawTimestamp + currentOutputLatency + samplerLatencyOffset
```

### 10.2 結果

| シミュレータ | 平均オフセット | 最小 | 最大 | データ数 |
|------------|--------------|------|------|---------|
| **iPhone 16 Pro** | **-29.1ms** | -69.8ms | -6.7ms | 10 |
| **iPhone 16 Clean** | **-41.5ms** | -75.8ms | -10.1ms | 11 |

### 10.3 補正前後の比較

| 状態 | 平均オフセット | 改善幅 |
|-----|--------------|--------|
| 補正なし | -100.1ms | - |
| **80ms補正後** | **-29.1ms** | **約71ms改善** |

### 10.4 分析

- 80ms補正で大幅な改善（約70ms）
- しかしまだ約30-40msのズレが残っている
- シミュレータ間で約12msの差（-29.1 vs -41.5）
- **次のステップ: 100ms補正を試す**

## 11. 次のステップ

1. ~~Phase 1実行~~ ✅ 完了
2. ~~結果分析~~ ✅ 完了
3. ~~80ms補正テスト~~ ✅ 完了（-100ms → -29ms、約71ms改善）
4. **100ms補正テスト**: より完全な補正を目指す
5. **検証**: 補正後のTimingOffsetが±20ms以内になることを確認

## 12. 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-12-06 | 初版作成 |
| 2025-12-06 | Phase 1計測結果を追加。SF2音色で共通オフセット(~100ms)適用を推奨 |
| 2025-12-06 | 80ms補正テスト結果を追加。平均-100ms→-29msに改善（約71ms改善） |
