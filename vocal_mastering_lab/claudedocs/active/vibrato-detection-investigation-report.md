# ビブラート検出調査レポート

## 概要

**問題**: 実機でビブラートが全く検出されない
**調査日**: 2025-12-10
**ステータス**: ✅ 修正完了

## 発見された問題と修正

### 問題1: maxVibratoRate=8.0がハードコード

**原因**: `VibratoAnalyzer.swift`でmaxVibratoRateが8.0に固定されていた。

**影響**: lag整数除算により、計算されたrateが8.0を僅かに超えるケースが発生：
```
minLag = Int(172.3 / 8.0) = 21
rate = 172.3 / 21 = 8.2 Hz > 8.0 → 検出失敗
```

**修正**: maxVibratoRateを10.0に変更
- 一般的なビブラートは5-7Hz
- 10Hzまで許容しても誤検出リスクは低い

### 問題2: minLag計算の境界値問題

**原因**: `Int(sampleRate / maxVibratoRate)` が切り捨て除算のため、実際のrateが上限を超える

**例**:
```
Int(172.3 / 10.0) = 17
rate = 172.3 / 17 = 10.13 Hz > 10.0 → 検出失敗
```

**修正**: `ceil()`を使用してminLagを計算
```swift
// 修正前
let minLag = max(1, Int(sampleRate / maxVibratoRate))

// 修正後
let minLag = max(1, Int(ceil(sampleRate / maxVibratoRate)))
```

これにより:
```
ceil(172.3 / 10.0) = 18
rate = 172.3 / 18 = 9.57 Hz < 10.0 ✓
```

## 修正後のテスト結果

### vocadito_1データ（172Hz F0アノテーション）

| パラメータ | 検出率 | 備考 |
|-----------|--------|------|
| デフォルト (minRegularity=0.3) | **25.6%** | FCPE相当 |
| YIN相当 (minRegularity=0.15) | **35.9%** | YIN/pYINで使用 |

### YINシミュレーション（20Hzダウンサンプリング）

| パラメータ | 検出率 | 備考 |
|-----------|--------|------|
| YIN相当 (minRegularity=0.15) | **56.2%** | 実機YINの期待値 |

## 修正したファイル

1. **VocalisDomain/Services/VibratoAnalyzer.swift**
   - maxVibratoRate: 8.0 → 10.0
   - minLag計算: `Int()` → `Int(ceil())`

2. **PitchDetectionPOC/Sources/PitchDetectionPOC/Vibrato/VibratoAnalyzer.swift**
   - 同様の修正

## 既存の実装（確認済み・変更なし）

`PitchDetectionAlgorithm.swift`のアルゴリズム別パラメータは正しく実装されていた：

| パラメータ | FCPE | YIN/pYIN |
|-----------|------|----------|
| vibratoMinConfidence | 0.5 | 0.3 |
| vibratoMinRegularity | 0.3 | 0.15 |

`RecordingStatisticsCalculator`はこれらを正しく使用している。

## 次のステップ

1. [x] maxVibratoRateを10.0に変更
2. [x] minLag計算でceil()を使用
3. [ ] 実機でテストして効果を確認
4. [ ] 必要に応じてユニットテストを更新

## POCツールの使用方法

```bash
# F0データでビブラート検出テスト
.build/debug/pitch-poc-cli vibrato --f0 vocadito_1_f0.csv --verbose

# ダウンサンプリングでYINをシミュレート
.build/debug/pitch-poc-cli vibrato --f0 vocadito_1_f0.csv --downsample 20 --min-regularity 0.15

# デフォルトパラメータ（FCPE相当）
.build/debug/pitch-poc-cli vibrato --f0 vocadito_1_f0.csv

# YINパラメータ
.build/debug/pitch-poc-cli vibrato --f0 vocadito_1_f0.csv --min-regularity 0.15
```

## 参考ファイル

- 本体VibratoAnalyzer: `VocalisDomain/Sources/VocalisDomain/Services/VibratoAnalyzer.swift`
- POC VibratoAnalyzer: `PitchDetectionPOC/Sources/PitchDetectionPOC/Vibrato/VibratoAnalyzer.swift`
- アルゴリズムパラメータ: `VocalisDomain/Sources/VocalisDomain/ValueObjects/PitchDetectionAlgorithm.swift`
- 統計計算: `VocalisDomain/Sources/VocalisDomain/Services/RecordingStatisticsCalculator.swift`
