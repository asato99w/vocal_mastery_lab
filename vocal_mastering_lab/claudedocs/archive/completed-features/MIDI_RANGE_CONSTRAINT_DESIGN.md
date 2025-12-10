# MIDI範囲制約設計書

## 1. 現状分析

### 1.1 計算式の詳細解析

現在の実装では、`ScaleSettings.generateKeyRoots()` が以下のロジックでキーのルート音を計算しています：

#### キー進行パターン別の計算式

**1. 上昇のみ (ascendingOnly)**
```
roots = [start, start + interval, start + 2×interval, ..., start + count×interval]
最高音 = start + count × ascendingInterval + scaleTopOffset
最低音 = start
```

**2. 下降のみ (descendingOnly)**
```
roots = [start, start - interval, start - 2×interval, ..., start - count×interval]
最高音 = start + scaleTopOffset
最低音 = start - count × descendingInterval
```

**3. 上昇→下降 (ascendingThenDescending)**
```
上昇部: [start, ..., start + ascCount×ascInterval]
下降部: [peak - descInterval, ..., peak - descCount×descInterval]
  where peak = start + ascCount×ascInterval

最高音 = peak + scaleTopOffset
最低音 = min(start, peak - descCount × descendingInterval)
```

**4. 下降→上昇 (descendingThenAscending)**
```
下降部: [start, ..., start - descCount×descInterval]
上昇部: [valley + ascInterval, ..., valley + ascCount×ascInterval]
  where valley = start - descCount×descInterval

最高音 = max(start, valley + ascCount × ascendingInterval) + scaleTopOffset
最低音 = valley
```

#### スケールパターンのオフセット

- **fiveToneScale**: `playbackPattern = [0, 2, 4, 5, 7, 5, 4, 2, 0]`
  - 最高オフセット: `+7` (完全5度)

- **octaveRepeat**: `playbackPattern = [0, 4, 7, 12, 12, 12, 12, 7, 4, 0]`
  - 最高オフセット: `+12` (1オクターブ)

### 1.2 問題ケースの列挙

#### 最悪ケース（MIDI範囲外）

**上限超過の例**:
```
パラメータ:
- 開始音: C6 (MIDI 84)
- キー進行: 上昇のみ
- 上昇キー数: 12
- 上昇間隔: 4 (長3度)
- スケールパターン: octaveRepeat

計算:
最高音 = 84 + (12 × 4) + 12 = 84 + 48 + 12 = 144
結果: MIDI 127超過（範囲外）
```

**下限超過の例**:
```
パラメータ:
- 開始音: C2 (MIDI 36)
- キー進行: 下降のみ
- 下降キー数: 12
- 下降間隔: 4 (長3度)

計算:
最低音 = 36 - (12 × 4) = 36 - 48 = -12
結果: MIDI 0未満（範囲外）
```

#### 境界ケースの検証

**上限境界 (MIDI 127)**:
```
安全な最大設定:
- fiveToneScale: start ≤ 127 - maxAscending - 7
- octaveRepeat: start ≤ 127 - maxAscending - 12

例 (octaveRepeat, 上昇のみ):
- start = 84 (C6)
- ascCount = 10
- ascInterval = 1
→ 最高音 = 84 + 10 + 12 = 106 (安全)

- start = 84 (C6)
- ascCount = 12
- ascInterval = 3
→ 最高音 = 84 + 36 + 12 = 132 (範囲外)
```

**下限境界 (MIDI 0)**:
```
安全な最小設定:
- 下降のみ: start ≥ maxDescending
- 下降→上昇: start ≥ maxDescending

例 (下降のみ):
- start = 36 (C2)
- descCount = 10
- descInterval = 3
→ 最低音 = 36 - 30 = 6 (安全)

- start = 36 (C2)
- descCount = 12
- descInterval = 4
→ 最低音 = 36 - 48 = -12 (範囲外)
```

### 1.3 MIDI範囲チェック関数の必要性

現在の実装では：
- `MIDINote.init(_:)` が 0-127 の範囲チェックを実施
- `ScaleSettings.generateKeyRoots()` や `majorTriad()` 内で `try? MIDINote(value)` を使用
- 範囲外の音は `compactMap` で無視される

**問題点**:
1. ユーザーに事前警告がない（録音開始時に初めて気づく）
2. 無効な設定でも録音開始可能（一部の音が欠落）
3. 設定画面でのリアルタイムフィードバックがない

---

## 2. 設計アプローチの比較

### アプローチA: 開始音の動的制限

**概要**: 他のパラメータ（キー数、間隔、スケールパターン）に基づいて、選択可能な開始音を動的に制限する。

#### メリット
- ✅ ユーザーが無効な設定を作れない（事前防止）
- ✅ 実装がシンプル（制限ロジックはViewModel側のみ）
- ✅ 設定変更時のフィードバックが直感的

#### デメリット
- ❌ 開始音の選択肢が動的に変わり、混乱する可能性
- ❌ 複雑な計算（4つのパラメータ × 4つのパターン × 2つのスケール）
- ❌ パフォーマンス懸念（設定変更のたびに再計算）

#### 実装イメージ
```swift
// RecordingSettingsViewModel
var availableStartPitchIndices: [Int] {
    let maxAscending = ascendingKeyCount * ascendingKeyStepInterval
    let maxDescending = descendingKeyCount * descendingKeyStepInterval
    let scaleOffset = scaleType == .octaveRepeat ? 12 : 7

    return (0..<availablePitches.count).filter { index in
        let midiStart = 36 + index
        let (highest, lowest) = calculateRange(
            start: midiStart,
            pattern: keyProgressionPattern,
            maxAsc: maxAscending,
            maxDesc: maxDescending,
            offset: scaleOffset
        )
        return highest <= 127 && lowest >= 0
    }
}
```

---

### アプローチB: キー数・間隔の動的制限

**概要**: 開始音とスケールパターンに基づいて、選択可能なキー数と間隔を動的に制限する。

#### メリット
- ✅ 開始音は自由に選べる（ユーザーの意図を尊重）
- ✅ 制限の理由が明確（開始音から計算可能な最大値を提示）
- ✅ 事前防止により無効設定を回避

#### デメリット
- ❌ キー数と間隔の両方を制限する必要があり、複雑
- ❌ ユーザーが期待する設定ができない場合がある
- ❌ UIが複雑化（動的に変わるPicker選択肢）

#### 実装イメージ
```swift
var maxAscendingKeyCount: Int {
    let midiStart = 36 + startPitchIndex
    let scaleOffset = scaleType == .octaveRepeat ? 12 : 7
    let remaining = 127 - midiStart - scaleOffset
    return min(12, remaining / ascendingKeyStepInterval)
}

var availableAscendingIntervals: [Int] {
    let midiStart = 36 + startPitchIndex
    let scaleOffset = scaleType == .octaveRepeat ? 12 : 7
    let remaining = 127 - midiStart - scaleOffset
    let maxInterval = remaining / ascendingKeyCount
    return [1, 2, 3, 4].filter { $0 <= maxInterval }
}
```

---

### アプローチC: バリデーション + 警告表示

**概要**: すべてのパラメータを自由に設定可能にし、範囲外になる場合はリアルタイムで警告を表示。録音開始をブロックする。

#### メリット
- ✅ ユーザーの自由度が最も高い
- ✅ 制限ロジックがシンプル（検証のみ）
- ✅ 段階的な実装が可能（まず警告、後で録音ブロック）
- ✅ 保守性が高い（新しいパラメータ追加時の影響が小さい）

#### デメリット
- ❌ 無効な設定を作成できてしまう
- ❌ 録音開始時に初めてエラーが出る可能性
- ❌ 警告を無視して録音ボタンを押せる（UI制御が必要）

#### 実装イメージ
```swift
// ScaleSettings extension
func validateMIDIRange() -> MIDIRangeValidationResult {
    let roots = generateKeyRoots()
    let scaleOffset = notePattern.playbackPattern.max() ?? 0

    guard let minRoot = roots.min(), let maxRoot = roots.max() else {
        return .valid
    }

    let lowestNote = minRoot
    let highestNote = maxRoot + UInt8(scaleOffset)

    if highestNote > 127 {
        return .exceedsUpperLimit(highestNote)
    }
    if lowestNote < 0 {
        return .exceedsLowerLimit(lowestNote)
    }
    return .valid
}

// RecordingSettingsPanel
@State private var validationError: MIDIRangeValidationResult = .valid

var isRecordingEnabled: Bool {
    validationError == .valid
}
```

---

### アプローチD: 自動調整

**概要**: 範囲外になる場合は、自動的にパラメータを調整して有効な設定にする。

#### メリット
- ✅ ユーザーが設定に悩まない
- ✅ 常に有効な設定が保証される
- ✅ 実装がシンプル（調整ロジックのみ）

#### デメリット
- ❌ ユーザーの意図と異なる設定になる可能性
- ❌ 何が調整されたか分かりにくい
- ❌ 予測不可能な動作（どのパラメータを優先調整するか？）
- ❌ ユーザー体験が低下（自動調整への不信感）

#### 実装イメージ
```swift
mutating func adjustToValidRange() {
    while !validateMIDIRange().isValid {
        // 優先順位: interval > count > startNote
        if ascendingKeyStepInterval > 1 {
            ascendingKeyStepInterval -= 1
        } else if ascendingKeyCount > 1 {
            ascendingKeyCount -= 1
        } else if descendingKeyCount > 1 {
            descendingKeyCount -= 1
        } else {
            // 最終手段: 開始音を調整
            startPitchIndex = max(0, startPitchIndex - 1)
        }
    }
}
```

---

### アプローチE: ハイブリッド（推奨）

**概要**: アプローチCの「リアルタイムバリデーション」をベースに、アプローチDの「軽微な自動調整」を組み合わせる。

#### 戦略
1. **リアルタイムバリデーション**: 設定変更時に常にMIDI範囲をチェック
2. **視覚的フィードバック**: 範囲外の場合は警告アイコンと説明メッセージを表示
3. **軽微な自動調整**: Picker変更時に、わずかに範囲外（±5以内）なら最寄りの有効値に自動調整
4. **録音ブロック**: 範囲外の場合は録音開始ボタンを無効化

#### メリット
- ✅ ユーザーの自由度を保ちつつ、無効設定を防止
- ✅ リアルタイムフィードバックで学習効果
- ✅ 軽微なミスは自動修正、大きなミスは明確に警告
- ✅ 段階的実装が可能

#### デメリット
- ⚠️ 実装が最も複雑
- ⚠️ 自動調整の閾値設定が難しい

#### 実装イメージ
```swift
// ViewModelにバリデーション状態を追加
@Published private(set) var midiRangeValidation: MIDIRangeValidationResult = .valid

// 設定変更時に自動検証
private func validateCurrentSettings() {
    guard let settings = generateScaleSettings() else {
        midiRangeValidation = .valid
        return
    }
    midiRangeValidation = settings.validateMIDIRange()

    // 軽微な範囲外（±5）の場合は自動調整
    if case .exceedsUpperLimit(let value) = midiRangeValidation,
       value - 127 <= 5 {
        autoAdjustToValidRange()
    }
}

// UI: 警告表示 + 録音ボタン無効化
if case .exceedsUpperLimit(let value) = viewModel.midiRangeValidation {
    HStack {
        Image(systemName: "exclamationmark.triangle.fill")
            .foregroundColor(.orange)
        Text("最高音 MIDI \(value) が範囲外です（最大: 127）")
            .font(.caption)
    }
}

Button("録音開始") { ... }
    .disabled(viewModel.midiRangeValidation != .valid)
```

---

## 3. アプローチ比較表

| 観点 | A: 開始音制限 | B: キー数/間隔制限 | C: バリデーション | D: 自動調整 | **E: ハイブリッド** |
|-----|------------|-----------------|---------------|----------|----------------|
| **ユーザー自由度** | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️ | ⭐️⭐️⭐️⭐️ |
| **事前防止** | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ |
| **実装複雑度** | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️⭐️ |
| **保守性** | ⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ |
| **UX直感性** | ⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ |
| **テスト容易性** | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ |
| **パフォーマンス** | ⭐️⭐️⭐️ | ⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ | ⭐️⭐️⭐️⭐️ |

**推奨**: **アプローチE（ハイブリッド）**

---

## 4. 推奨アプローチの詳細設計

### 4.1 アーキテクチャ概要

```
┌─────────────────────────────────────────────────┐
│ Presentation Layer (SwiftUI)                    │
│ ┌─────────────────────────────────────────────┐ │
│ │ RecordingSettingsPanel                      │ │
│ │ - 警告表示                                    │ │
│ │ - 録音ボタン無効化                             │ │
│ └─────────────────────────────────────────────┘ │
│                      ↕️                          │
│ ┌─────────────────────────────────────────────┐ │
│ │ RecordingSettingsViewModel                  │ │
│ │ - @Published midiRangeValidation            │ │
│ │ - validateCurrentSettings()                 │ │
│ │ - autoAdjustToValidRange() [optional]       │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
                      ↕️
┌─────────────────────────────────────────────────┐
│ Domain Layer (VocalisDomain)                    │
│ ┌─────────────────────────────────────────────┐ │
│ │ ScaleSettings                               │ │
│ │ + validateMIDIRange() -> ValidationResult   │ │
│ │ + calculateHighestNote() -> UInt8?          │ │
│ │ + calculateLowestNote() -> UInt8?           │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────┐ │
│ │ MIDIRangeValidationResult (enum)            │ │
│ │ - valid                                     │ │
│ │ - exceedsUpperLimit(UInt8)                  │ │
│ │ - exceedsLowerLimit(Int)                    │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 4.2 Domain Layer実装

#### 4.2.1 MIDIRangeValidationResult

```swift
// VocalisDomain/Entities/MIDIRangeValidationResult.swift
public enum MIDIRangeValidationResult: Equatable {
    case valid
    case exceedsUpperLimit(actualValue: Int)
    case exceedsLowerLimit(actualValue: Int)

    public var isValid: Bool {
        if case .valid = self {
            return true
        }
        return false
    }

    /// User-facing error message (localization key)
    public var errorMessageKey: String? {
        switch self {
        case .valid:
            return nil
        case .exceedsUpperLimit(let value):
            return "error.midi_range.exceeds_upper_limit" // "最高音 MIDI \(value) が範囲外です（最大: 127）"
        case .exceedsLowerLimit(let value):
            return "error.midi_range.exceeds_lower_limit" // "最低音 MIDI \(value) が範囲外です（最小: 0）"
        }
    }

    /// Detailed error description for debugging
    public var debugDescription: String {
        switch self {
        case .valid:
            return "MIDI range is valid"
        case .exceedsUpperLimit(let value):
            return "Highest note MIDI \(value) exceeds upper limit 127"
        case .exceedsLowerLimit(let value):
            return "Lowest note MIDI \(value) is below lower limit 0"
        }
    }
}
```

#### 4.2.2 ScaleSettings拡張

```swift
// VocalisDomain/Entities/ScaleSettings.swift
extension ScaleSettings {
    /// Calculate the highest MIDI note that will be generated
    public func calculateHighestNote() -> Int {
        let roots = generateKeyRoots()
        guard let maxRoot = roots.max() else { return Int(startNote.value) }

        let scaleTopOffset = notePattern.playbackPattern.max() ?? 0
        return Int(maxRoot) + scaleTopOffset
    }

    /// Calculate the lowest MIDI note that will be generated
    public func calculateLowestNote() -> Int {
        let roots = generateKeyRoots()
        guard let minRoot = roots.min() else { return Int(startNote.value) }

        // Scale pattern always includes root (0), so no negative offset
        return Int(minRoot)
    }

    /// Validate that all generated notes are within MIDI range (0-127)
    public func validateMIDIRange() -> MIDIRangeValidationResult {
        let highest = calculateHighestNote()
        let lowest = calculateLowestNote()

        if highest > 127 {
            return .exceedsUpperLimit(actualValue: highest)
        }

        if lowest < 0 {
            return .exceedsLowerLimit(actualValue: lowest)
        }

        return .valid
    }
}
```

### 4.3 Presentation Layer実装

#### 4.3.1 RecordingSettingsViewModel拡張

```swift
// Presentation/ViewModels/RecordingSettingsViewModel.swift
extension RecordingSettingsViewModel {
    /// Current MIDI range validation result
    @Published private(set) var midiRangeValidation: MIDIRangeValidationResult = .valid

    /// Whether recording can start (valid MIDI range required)
    public var canStartRecording: Bool {
        midiRangeValidation.isValid
    }

    /// Validate current settings and update validation state
    public func validateCurrentSettings() {
        guard let settings = generateScaleSettings() else {
            midiRangeValidation = .valid // Scale off - no validation needed
            return
        }

        midiRangeValidation = settings.validateMIDIRange()
    }

    /// Attempt to auto-adjust settings to valid range (if within threshold)
    /// Returns true if successfully adjusted, false if manual intervention needed
    @discardableResult
    public func autoAdjustToValidRange(threshold: Int = 5) -> Bool {
        guard !midiRangeValidation.isValid else { return true }

        switch midiRangeValidation {
        case .exceedsUpperLimit(let value):
            let excess = value - 127
            if excess <= threshold {
                // Try reducing interval first (least disruptive)
                if ascendingKeyStepInterval > 1 {
                    ascendingKeyStepInterval -= 1
                    validateCurrentSettings()
                    return midiRangeValidation.isValid
                }
                // Then try reducing count
                if ascendingKeyCount > 1 {
                    ascendingKeyCount -= 1
                    validateCurrentSettings()
                    return midiRangeValidation.isValid
                }
            }
            return false

        case .exceedsLowerLimit(let value):
            let excess = abs(value)
            if excess <= threshold {
                // Try reducing interval first
                if descendingKeyStepInterval > 1 {
                    descendingKeyStepInterval -= 1
                    validateCurrentSettings()
                    return midiRangeValidation.isValid
                }
                // Then try reducing count
                if descendingKeyCount > 1 {
                    descendingKeyCount -= 1
                    validateCurrentSettings()
                    return midiRangeValidation.isValid
                }
            }
            return false

        case .valid:
            return true
        }
    }

    // Observe changes and auto-validate
    public init() {
        // Existing init code...

        // Add validation observers
        $scaleType.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $startPitchIndex.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $keyProgressionPattern.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $ascendingKeyCount.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $descendingKeyCount.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $ascendingKeyStepInterval.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)

        $descendingKeyStepInterval.sink { [weak self] _ in
            self?.validateCurrentSettings()
        }.store(in: &cancellables)
    }

    private var cancellables = Set<AnyCancellable>()
}
```

#### 4.3.2 RecordingSettingsPanel UI拡張

```swift
// Presentation/Views/Recording/RecordingSettingsPanel.swift
struct RecordingSettingsPanel: View {
    // ... existing code ...

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // ... existing settings controls ...

                // MIDI Range Validation Warning
                if !viewModel.midiRangeValidation.isValid {
                    MIDIRangeWarningView(validationResult: viewModel.midiRangeValidation)
                        .padding(.top, 8)
                }
            }
            .padding(12)
        }
        .background(ColorPalette.secondary)
    }
}

/// Warning view for MIDI range validation errors
struct MIDIRangeWarningView: View {
    let validationResult: MIDIRangeValidationResult

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
                .font(.title3)

            VStack(alignment: .leading, spacing: 4) {
                Text("warning.title".localized) // "設定エラー"
                    .font(.subheadline)
                    .fontWeight(.semibold)

                if let errorKey = validationResult.errorMessageKey {
                    Text(localizedError(for: validationResult))
                        .font(.caption)
                        .foregroundColor(ColorPalette.text.opacity(0.8))
                }

                Text("warning.midi_range.hint".localized) // "キー数や間隔を調整してください"
                    .font(.caption2)
                    .foregroundColor(ColorPalette.text.opacity(0.6))
            }
        }
        .padding(12)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(8)
        .accessibilityIdentifier("MIDIRangeWarning")
    }

    private func localizedError(for result: MIDIRangeValidationResult) -> String {
        switch result {
        case .exceedsUpperLimit(let value):
            return String(format: "error.midi_range.exceeds_upper_limit".localized, value)
        case .exceedsLowerLimit(let value):
            return String(format: "error.midi_range.exceeds_lower_limit".localized, abs(value))
        case .valid:
            return ""
        }
    }
}
```

#### 4.3.3 録音ボタン無効化

```swift
// RecordingView.swift (録音開始ボタンがある場所)
Button(action: {
    // Start recording
}) {
    Text("recording.start_button".localized)
}
.disabled(!settingsViewModel.canStartRecording)
.opacity(settingsViewModel.canStartRecording ? 1.0 : 0.5)
.accessibilityIdentifier("StartRecordingButton")
.accessibilityHint(settingsViewModel.canStartRecording ?
    "" : "error.midi_range.recording_disabled".localized)
```

### 4.4 ローカライゼーション

```swift
// Localizable.strings (ja)
"warning.title" = "設定エラー";
"error.midi_range.exceeds_upper_limit" = "最高音 MIDI %d が範囲外です（最大: 127）";
"error.midi_range.exceeds_lower_limit" = "最低音 MIDI %d が範囲外です（最小: 0）";
"warning.midi_range.hint" = "キー数や間隔を調整してください";
"error.midi_range.recording_disabled" = "MIDI範囲エラーのため録音できません";

// Localizable.strings (en)
"warning.title" = "Configuration Error";
"error.midi_range.exceeds_upper_limit" = "Highest note MIDI %d exceeds limit (max: 127)";
"error.midi_range.exceeds_lower_limit" = "Lowest note MIDI %d is below limit (min: 0)";
"warning.midi_range.hint" = "Please adjust key count or interval";
"error.midi_range.recording_disabled" = "Recording disabled due to MIDI range error";
```

---

## 5. UX観点での考慮事項

### 5.1 ユーザーの直感的な操作性

**原則**: ユーザーが自由に設定を試行錯誤できるようにし、範囲外になる場合のみ警告を表示する。

**設計決定**:
- ✅ すべてのパラメータを自由に選択可能
- ✅ 範囲外になった瞬間に視覚的フィードバック（オレンジ色の警告）
- ✅ 具体的な数値を表示（「MIDI 144が範囲外」など）
- ✅ 録音ボタンを無効化（誤操作防止）

### 5.2 設定変更時のフィードバック

**リアルタイム検証**:
- Pickerやスライダー変更時に即座にバリデーション実行
- Combineの`$published`を活用して自動検証
- 計算コストは低い（O(n)でnは最大12）ため、パフォーマンス問題なし

**段階的なフィードバック**:
1. **有効な設定**: 警告なし、録音ボタン有効
2. **軽微な範囲外（±5）**: 自動調整試行 → 成功なら警告なし
3. **大幅な範囲外**: 警告表示、録音ボタン無効

### 5.3 制限の理由をユーザーに伝える方法

**警告メッセージの設計**:
```
┌────────────────────────────────────────┐
│ ⚠️ 設定エラー                            │
│                                        │
│ 最高音 MIDI 144 が範囲外です（最大: 127） │
│ キー数や間隔を調整してください            │
└────────────────────────────────────────┘
```

**教育的要素**:
- MIDI番号を明示（ユーザーが理解を深められる）
- 調整のヒントを提供（「キー数や間隔を調整」）
- アクセシビリティ対応（VoiceOverで警告を読み上げ）

### 5.4 既存の設定が無効になる場合の対応

**シナリオ**: プリセットを読み込んだが、範囲外の設定だった場合

**対応策**:
1. プリセット読み込み時に自動バリデーション実行
2. 範囲外の場合は警告表示
3. ユーザーに手動調整を促す（自動調整は行わない）
4. プリセット自体は保持（ユーザーが他のパラメータを調整可能）

**実装**:
```swift
func loadPreset(_ preset: ScalePreset) {
    // パラメータを設定
    scaleType = preset.scaleType
    startPitchIndex = preset.startPitchIndex
    // ... other params ...

    // 自動バリデーション（Combineで自動実行される）
    // 範囲外の場合はUIに警告が表示される
}
```

---

## 6. 実装複雑度の評価

### 6.1 実装コスト

| 作業項目 | 工数見積もり | 理由 |
|--------|-----------|------|
| Domain Layer実装 | 2時間 | シンプルな計算ロジック |
| ViewModel拡張 | 3時間 | Combine統合、自動調整ロジック |
| UI実装 | 2時間 | 警告View、ボタン無効化 |
| ローカライゼーション | 1時間 | 日英2言語 |
| **Unit Tests** | **4時間** | **境界値テスト、パターン網羅** |
| **UI Tests** | **2時間** | **警告表示、ボタン無効化の検証** |
| **合計** | **14時間** | **約2営業日** |

### 6.2 テスト容易性

**Unit Testsの観点**:
- ✅ Domain Layerは純粋関数（副作用なし）でテスト容易
- ✅ 境界値テストケースが明確（0, 127, 128など）
- ✅ ViewModelのバリデーションロジックもモック不要

**テストケース例**:
```swift
class ScaleSettingsMIDIRangeTests: XCTestCase {
    func testValidRange_fiveToneScale_ascendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(60), // C4
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 10,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // 最高音: 60 + 10 + 7 = 77 (valid)
        XCTAssertEqual(settings.validateMIDIRange(), .valid)
    }

    func testExceedsUpperLimit_octaveRepeat_ascendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(84), // C6
            endNote: try! MIDINote(96),
            notePattern: .octaveRepeat,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 12,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 4 // Major third
        )

        // 最高音: 84 + (12*4) + 12 = 144 (exceeds 127)
        if case .exceedsUpperLimit(let value) = settings.validateMIDIRange() {
            XCTAssertEqual(value, 144)
        } else {
            XCTFail("Expected exceedsUpperLimit")
        }
    }

    func testExceedsLowerLimit_descendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(36), // C2
            endNote: try! MIDINote(48),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 12,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 4 // Major third
        )

        // 最低音: 36 - (12*4) = -12 (below 0)
        if case .exceedsLowerLimit(let value) = settings.validateMIDIRange() {
            XCTAssertEqual(value, -12)
        } else {
            XCTFail("Expected exceedsLowerLimit")
        }
    }
}
```

### 6.3 保守性

**変更容易性**:
- ✅ 新しいスケールパターン追加時: `NotePattern.playbackPattern`のみ変更
- ✅ 新しいキー進行パターン追加時: `generateKeyRoots()`のswitch文に追加
- ✅ バリデーションロジックは独立（他のロジックへの影響なし）

**依存関係**:
- Domain Layer ← Presentation Layer（単方向依存）
- 他のモジュールへの影響なし

---

## 7. 実装計画

### フェーズ1: Domain Layer実装（TDD）

**目標**: MIDI範囲検証ロジックの実装

**タスク**:
1. ✅ `MIDIRangeValidationResult` enum作成
2. ✅ `ScaleSettings.calculateHighestNote()` 実装 + テスト
3. ✅ `ScaleSettings.calculateLowestNote()` 実装 + テスト
4. ✅ `ScaleSettings.validateMIDIRange()` 実装 + テスト
5. ✅ 境界値テストケース網羅（各パターン × 各スケール）

**成果物**:
- `VocalisDomain/Entities/MIDIRangeValidationResult.swift`
- `VocalisDomain/Entities/ScaleSettings+MIDIRange.swift`
- `VocalisDomainTests/Entities/ScaleSettingsMIDIRangeTests.swift`

**工数**: 6時間

---

### フェーズ2: Presentation Layer実装

**目標**: リアルタイムバリデーションとUI警告表示

**タスク**:
1. ✅ `RecordingSettingsViewModel.midiRangeValidation` プロパティ追加
2. ✅ `RecordingSettingsViewModel.validateCurrentSettings()` 実装
3. ✅ Combine observersで自動バリデーション実装
4. ✅ `MIDIRangeWarningView` コンポーネント作成
5. ✅ `RecordingSettingsPanel`に警告View統合
6. ✅ 録音ボタン無効化ロジック追加
7. ✅ ViewModelのUnit Tests追加

**成果物**:
- `Presentation/ViewModels/RecordingSettingsViewModel+Validation.swift`
- `Presentation/Views/Recording/MIDIRangeWarningView.swift`
- `Presentation/Views/Recording/RecordingSettingsPanel.swift` (更新)
- `VocalisStudioTests/ViewModels/RecordingSettingsViewModelValidationTests.swift`

**工数**: 5時間

---

### フェーズ3: ローカライゼーションとUI Tests

**目標**: 多言語対応とE2Eテスト

**タスク**:
1. ✅ 日本語ローカライゼーション追加
2. ✅ 英語ローカライゼーション追加
3. ✅ UI Testsで警告表示を検証
4. ✅ UI Testsで録音ボタン無効化を検証
5. ✅ アクセシビリティテスト（VoiceOver）

**成果物**:
- `Resources/ja.lproj/Localizable.strings` (更新)
- `Resources/en.lproj/Localizable.strings` (更新)
- `VocalisStudioUITests/RecordingSettingsMIDIRangeUITests.swift`

**工数**: 3時間

---

### フェーズ4: オプショナル - 自動調整機能

**目標**: 軽微な範囲外を自動修正

**タスク**:
1. ✅ `RecordingSettingsViewModel.autoAdjustToValidRange()` 実装
2. ✅ 自動調整のUnit Tests追加
3. ✅ UIでの自動調整フィードバック（トースト通知など）

**成果物**:
- `Presentation/ViewModels/RecordingSettingsViewModel+AutoAdjust.swift`
- `VocalisStudioTests/ViewModels/RecordingSettingsViewModelAutoAdjustTests.swift`

**工数**: 4時間（オプショナル）

**決定**: フェーズ1-3を優先実装、フェーズ4は必要に応じて後回し

---

### 総工数見積もり

- **必須フェーズ（1-3）**: 14時間（約2営業日）
- **オプショナル（フェーズ4）**: 4時間（0.5営業日）

---

## 8. リスクと緩和策

### リスク1: パフォーマンス低下

**リスク**: 設定変更のたびにバリデーション実行でUIが遅延

**緩和策**:
- ✅ バリデーション計算は軽量（O(n)、n≤12）
- ✅ Combineの`debounce`で連続変更時の実行回数削減
- ✅ 必要に応じて非同期処理化

**実装例**:
```swift
$ascendingKeyCount
    .debounce(for: 0.1, scheduler: RunLoop.main)
    .sink { [weak self] _ in
        self?.validateCurrentSettings()
    }
    .store(in: &cancellables)
```

---

### リスク2: ユーザー混乱（なぜ警告が出るか分からない）

**リスク**: 技術的なエラーメッセージで初心者ユーザーが困惑

**緩和策**:
- ✅ 平易な日本語メッセージ
- ✅ 具体的な調整方法を提示（「キー数や間隔を減らしてください」）
- ✅ ヘルプボタンで詳細説明を表示（将来拡張）

---

### リスク3: 既存プリセットとの互換性

**リスク**: 保存済みプリセットが範囲外になる可能性

**緩和策**:
- ✅ プリセット読み込み時に自動バリデーション
- ✅ 範囲外の場合は警告表示（プリセット自体は保持）
- ✅ プリセット保存時にもバリデーション実施

**実装**:
```swift
func savePreset(name: String) -> Result<Void, PresetError> {
    guard viewModel.canStartRecording else {
        return .failure(.invalidMIDIRange)
    }
    // プリセット保存処理...
}
```

---

### リスク4: テストケースの網羅漏れ

**リスク**: 境界値の組み合わせが多く、テストケース漏れ

**緩和策**:
- ✅ パラメータ化テストで組み合わせ網羅
- ✅ 境界値分析を系統的に実施
- ✅ TDDで設計とテストを同時開発

**テストマトリックス**:
```
| ScalePattern | KeyPattern | KeyCount | Interval | Expected |
|--------------|------------|----------|----------|----------|
| fiveTone     | ascending  | 12       | 4        | exceeds  |
| fiveTone     | ascending  | 10       | 4        | valid    |
| octaveRepeat | ascending  | 8        | 4        | exceeds  |
| octaveRepeat | descending | 12       | 4        | exceeds  |
| ...          | ...        | ...      | ...      | ...      |
```

---

## 9. 実装のベストプラクティス

### 9.1 TDDサイクル遵守

**Red → Green → Refactor**:
1. 失敗するテストを書く（例: `testExceedsUpperLimit`）
2. 最小限の実装で通す
3. リファクタリングで品質向上

### 9.2 Clean Architectureの厳守

**依存方向**:
- ❌ Domain Layer が Presentation Layer に依存してはいけない
- ✅ Presentation Layer が Domain Layer に依存する

**実装確認**:
```swift
// ❌ NG: Domain Layerに@Published
public struct ScaleSettings {
    @Published var validationResult: MIDIRangeValidationResult // NG!
}

// ✅ OK: Presentation Layerで@Published
public class RecordingSettingsViewModel {
    @Published private(set) var midiRangeValidation: MIDIRangeValidationResult // OK
}
```

### 9.3 アクセシビリティの考慮

**VoiceOver対応**:
```swift
.accessibilityLabel("warning.midi_range.title".localized)
.accessibilityValue(validationResult.debugDescription)
.accessibilityHint("warning.midi_range.hint".localized)
```

---

## 10. まとめ

### 推奨アプローチ: ハイブリッド（C + 軽微なD）

**理由**:
1. ✅ ユーザーの自由度を最大化
2. ✅ リアルタイムフィードバックで学習効果
3. ✅ 事前防止により無効設定を回避
4. ✅ 実装とテストが容易
5. ✅ 保守性が高い

**実装ステップ**:
1. **フェーズ1**: Domain Layer（6時間）
2. **フェーズ2**: Presentation Layer（5時間）
3. **フェーズ3**: ローカライゼーション & UI Tests（3時間）
4. **フェーズ4（オプショナル）**: 自動調整（4時間）

**総工数**: 14時間（必須）+ 4時間（オプショナル）= **約2-3営業日**

**次のアクション**:
1. 本設計書のレビュー
2. フェーズ1の実装開始（TDDで進める）
3. 各フェーズ完了後にデモとフィードバック

---

## 付録A: コード実装サンプル

### A.1 完全なテストケース例

```swift
// VocalisDomainTests/Entities/ScaleSettingsMIDIRangeTests.swift
import XCTest
@testable import VocalisDomain

class ScaleSettingsMIDIRangeTests: XCTestCase {

    // MARK: - calculateHighestNote Tests

    func testCalculateHighestNote_fiveToneScale_ascendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(60), // C4
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 5,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // 最高音: 60 + 5 + 7 = 72
        XCTAssertEqual(settings.calculateHighestNote(), 72)
    }

    func testCalculateHighestNote_octaveRepeat_ascendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(60), // C4
            endNote: try! MIDINote(72),
            notePattern: .octaveRepeat,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 5,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 2
        )

        // 最高音: 60 + (5*2) + 12 = 82
        XCTAssertEqual(settings.calculateHighestNote(), 82)
    }

    // MARK: - calculateLowestNote Tests

    func testCalculateLowestNote_descendingOnly() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(60), // C4
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 5,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 2
        )

        // 最低音: 60 - (5*2) = 50
        XCTAssertEqual(settings.calculateLowestNote(), 50)
    }

    // MARK: - validateMIDIRange Tests

    func testValidateMIDIRange_validCase() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(60),
            endNote: try! MIDINote(72),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 10,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        XCTAssertEqual(settings.validateMIDIRange(), .valid)
    }

    func testValidateMIDIRange_exceedsUpperLimit() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(84), // C6
            endNote: try! MIDINote(96),
            notePattern: .octaveRepeat,
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 12,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 4
        )

        // 最高音: 84 + (12*4) + 12 = 144
        if case .exceedsUpperLimit(let value) = settings.validateMIDIRange() {
            XCTAssertEqual(value, 144)
        } else {
            XCTFail("Expected exceedsUpperLimit")
        }
    }

    func testValidateMIDIRange_exceedsLowerLimit() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(36), // C2
            endNote: try! MIDINote(48),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 12,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 4
        )

        // 最低音: 36 - (12*4) = -12
        if case .exceedsLowerLimit(let value) = settings.validateMIDIRange() {
            XCTAssertEqual(value, -12)
        } else {
            XCTFail("Expected exceedsLowerLimit")
        }
    }

    // MARK: - Edge Cases

    func testValidateMIDIRange_exactUpperBoundary() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(120),
            endNote: try! MIDINote(127),
            notePattern: .fiveToneScale, // +7
            tempo: .standard,
            keyProgressionPattern: .ascendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1
        )

        // 最高音: 120 + 0 + 7 = 127 (exactly valid)
        XCTAssertEqual(settings.validateMIDIRange(), .valid)
    }

    func testValidateMIDIRange_exactLowerBoundary() {
        let settings = ScaleSettings(
            startNote: try! MIDINote(7), // Minimum for fiveToneScale
            endNote: try! MIDINote(19),
            notePattern: .fiveToneScale,
            tempo: .standard,
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 0,
            descendingKeyCount: 0,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )

        // 最低音: 7 - 0 = 7 (valid, but close to 0)
        XCTAssertEqual(settings.validateMIDIRange(), .valid)
    }
}
```

### A.2 UI Test例

```swift
// VocalisStudioUITests/RecordingSettingsMIDIRangeUITests.swift
import XCTest

class RecordingSettingsMIDIRangeUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    func testMIDIRangeWarning_appearsWhenExceedingUpperLimit() {
        // 設定画面へ移動
        // ... navigation code ...

        // C6を選択
        app.pickers["StartPitchPicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "C6")

        // 上昇のみ
        app.pickers["KeyProgressionPatternPicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "上昇のみ")

        // キー数12
        app.pickers["AscendingKeyCountPicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "12 回")

        // 間隔: 長3度
        app.pickers["AscendingKeyStepIntervalPicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "長3度")

        // octaveRepeatを選択
        app.pickers["ScaleTypePicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "オクターブリピート")

        // 警告が表示される
        XCTAssertTrue(app.staticTexts["MIDIRangeWarning"].exists)
        XCTAssertTrue(app.staticTexts.containing(NSPredicate(format: "label CONTAINS '144'")).firstMatch.exists)

        // 録音ボタンが無効化される
        XCTAssertFalse(app.buttons["StartRecordingButton"].isEnabled)
    }

    func testMIDIRangeWarning_disappearsWhenAdjusted() {
        // 警告が出る状態を作る
        testMIDIRangeWarning_appearsWhenExceedingUpperLimit()

        // キー数を5に減らす
        app.pickers["AscendingKeyCountPicker"].tap()
        app.pickerWheels.element.adjust(toPickerWheelValue: "5 回")

        // 警告が消える
        XCTAssertFalse(app.staticTexts["MIDIRangeWarning"].exists)

        // 録音ボタンが有効化される
        XCTAssertTrue(app.buttons["StartRecordingButton"].isEnabled)
    }
}
```

---

## 付録B: シーケンス図

```
┌──────┐         ┌──────────────┐         ┌──────────────┐         ┌─────────────┐
│ User │         │SettingsPanel │         │  ViewModel   │         │ScaleSettings│
└──┬───┘         └──────┬───────┘         └──────┬───────┘         └──────┬──────┘
   │                    │                        │                        │
   │ Change Picker      │                        │                        │
   ├───────────────────>│                        │                        │
   │                    │                        │                        │
   │                    │ @Published updated     │                        │
   │                    ├───────────────────────>│                        │
   │                    │                        │                        │
   │                    │                        │ Combine observer fires │
   │                    │                        │ validateCurrentSettings()
   │                    │                        ├───┐                    │
   │                    │                        │   │                    │
   │                    │                        │<──┘                    │
   │                    │                        │                        │
   │                    │                        │ generateScaleSettings()│
   │                    │                        ├───────────────────────>│
   │                    │                        │                        │
   │                    │                        │ validateMIDIRange()    │
   │                    │                        │<───────────────────────│
   │                    │                        │                        │
   │                    │                        │ MIDIRangeValidationResult
   │                    │                        │<───────────────────────│
   │                    │                        │                        │
   │                    │ UI update (warning)    │                        │
   │                    │<───────────────────────┤                        │
   │                    │                        │                        │
   │ See warning        │                        │                        │
   │<───────────────────┤                        │                        │
   │                    │                        │                        │
```

---

**設計書終わり**
