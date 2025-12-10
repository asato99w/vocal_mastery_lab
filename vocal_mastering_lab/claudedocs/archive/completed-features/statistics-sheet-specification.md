# Statistics Sheet 仕様書

## 概要

統計シートは、録音の分析結果をスケール設定に基づいて表示する機能。
表示内容はスケール設定から**計算可能**であり、実際の検出結果に依存しない。

## セクション構成

### 1. Overall（全体統計）
- Avg Deviation: 平均偏差（セント）
- Variability: ばらつき（セント）
- Median: 中央値（セント）
- Detection Rate: 検出率（%）
- Vocal Range: 音域

### 2. By Scale Position（ポジション別統計）
- **スケールパターンのポジション数だけ常に表示**
- 検出の有無に関係なく全ポジションを表示
- 折りたたみUIが必要（ポジション数が多いため）

### 3. By Pitch（ピッチ別統計）
- **スケール設定から計算される全ユニーク音名を表示**
- 検出の有無に関係なく全音名を表示
- 折りたたみUIが必要

---

## 計算ロジック

### By Scale Position の計算

**ポジション数 = notePattern.playbackPattern.count**

| NotePattern | playbackPattern | ポジション数 |
|-------------|-----------------|-------------|
| fiveToneScale | [0, 2, 4, 5, 7, 5, 4, 2, 0] | **9** |
| octaveRepeat | [0, 4, 7, 12, 12, 12, 12, 7, 4, 0] | **10** |

表示例（fiveToneScale）:
```
1st: +10.5 cents
2st: -5.2 cents
3st: +3.1 cents
...
9st: -2.0 cents
```

### By Pitch の計算

**全キーにわたるユニーク音名の集合**

計算手順:
1. `generateKeyRoots()` でキーの列を取得
2. 各キーで `notePattern.intervals` の音を計算
3. 全キーの音を集めてユニーク化

#### MVPデフォルト設定の例

**ScaleSettings:**
```swift
startNote: C4 (MIDI 60)
notePattern: .fiveToneScale  // intervals = [0, 2, 4, 5, 7]
keyProgressionPattern: .ascendingThenDescending
ascendingKeyCount: 3
descendingKeyCount: 3
keyStepInterval: 1 (半音)
```

**Step 1: generateKeyRoots()**
```
ascendingThenDescending の場合:
- 上昇: [60, 61, 62, 63] = [C4, C#4, D4, D#4]
- 下降: [62, 61, 60] = [D4, C#4, C4] (ピーク除外)
- 合計キー列: [C4, C#4, D4, D#4, D4, C#4, C4]
```

**Step 2: 各キーの音を計算**
```
intervals = [0, 2, 4, 5, 7]

キーC4 (60):  60+0, 60+2, 60+4, 60+5, 60+7 = C4, D4, E4, F4, G4
キーC#4 (61): 61+0, 61+2, 61+4, 61+5, 61+7 = C#4, D#4, F4, F#4, G#4
キーD4 (62):  62+0, 62+2, 62+4, 62+5, 62+7 = D4, E4, F#4, G4, A4
キーD#4 (63): 63+0, 63+2, 63+4, 63+5, 63+7 = D#4, F4, G4, G#4, A#4
```

**Step 3: ユニーク音名**
```
MIDI値: 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70
音名:   C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4, A4, A#4
合計: 11個のユニーク音名
```

---

## UIテストで検証すべき内容

### testPositionStatistics_FiveToneScale
- **9ポジション全て**（1st〜9st）が表示されること
- `XCTAssertEqual(positionCount, 9)`

### testPositionStatistics_OctaveRepeatScale
- **10ポジション全て**（1st〜10st）が表示されること
- `XCTAssertEqual(positionCount, 10)`

### testPitchStatistics_FiveToneScale_NoteNames
- MVPデフォルト設定で**11個のユニーク音名**が表示されること
- 音名: C4, C#4, D4, D#4, E4, F4, F#4, G4, G#4, A4, A#4
- `XCTAssertEqual(pitchCount, 11)`
- 各音名が存在することを検証

### testPitchStatistics_OctaveRepeatScale_NoteNames
- オクターブリピート設定でのユニーク音名数を検証
- intervals = [0, 4, 7, 12] から計算

---

## 現在のバグ（修正が必要）

1. **ポジション数不足**: 検出されたポジションのみ表示している
   - 期待: 全9ポジション（fiveToneScale）
   - 実際: 検出されたポジションのみ

2. **折りたたみ未実装**: ポジションセクションに折りたたみがない

3. **ピッチ表示がキー変更を考慮していない**:
   - 期待: 11個のユニーク音名（MVPデフォルト）
   - 実際: 検出された音のみ、またはスケールパターンの5音のみ

---

## 関連ファイル

- `ScaleSettings.swift`: スケール設定エンティティ
- `NotePattern.swift`: スケールパターン定義
- `KeyProgressionPattern.swift`: キー進行パターン
- `RecordingStatisticsCalculator.swift`: 統計計算サービス
- `StatisticsSheetView.swift`: 統計シートUI（未確認）
