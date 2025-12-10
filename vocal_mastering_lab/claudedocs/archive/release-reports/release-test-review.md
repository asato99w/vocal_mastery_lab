# リリース前テストレビュー

**作成日**: 2025-11-27
**目的**: リリース前のバグ洗い出しとテスト強化計画
**関連ドキュメント**: `code-analysis-report-2025-11-24.md` (コード解析レポート)

---

## 0. 11-24レポートとの整合性確認

### 11-24で修正済みの項目 ✅

| 項目 | 内容 | 状態 |
|------|------|------|
| P0 | ScaleSettings.swift majorTriad() の `try!` → `try?` + compactMap | ✅ **修正済み** |
| P1 | デバッグprint文の削除 (8箇所) | ✅ **修正済み** |

### 11-24で指摘された未対応項目

| 優先度 | 項目 | 状態 |
|--------|------|------|
| P2 | AnalysisView.swift 分割 (1,114行) | 🔄 未対応 |
| P3 | Keychain移行 (サブスクリプション状態) | 🔄 未対応 |
| P3 | メモリ管理強化 (`[weak self]` 追加) | 🔄 未対応 |

### 今回新たに発見した問題

11-24レポートでは言及されていない追加の問題点を本レポートで特定：
- generateKeyRoots() の UInt8 アンダーフロー問題
- ScalePreset UseCase のテスト不足
- RecordingLimit 境界値テスト不足
- RecordingStateViewModel 状態遷移テスト不足
- ScaleSettings Codable 後方互換性

---

## 1. 現状のテストカバレッジ概要

### テストファイル数
- **ユニットテスト**: 67ファイル
- **UIテスト**: 9ファイル
- **ソースファイル**: 77ファイル（Packages除く）

### テスト済み領域 ✅
| 領域 | テストファイル | カバー状況 |
|------|----------------|------------|
| Domain/Entities | ScaleSettingsTests, RecordingTests, AnalysisResultTests | 良好 |
| Domain/ValueObjects | MIDINoteTests, RecordingIdTests, KeyProgressionPatternTests | 良好 |
| Application/UseCases | StartRecording, StopRecording, AnalyzeRecording, Purchase, Restore | 良好 |
| Application/Services | RecordingPolicyServiceTests, ScalePlaybackCoordinatorTests | 良好 |
| Infrastructure/Audio | RealtimePitchDetector, AVAudioEngineScalePlayer, AudioSessionManager | 基本テスト |
| Infrastructure/Repositories | FileRecordingRepository, UserDefaultsScalePreset, AudioSettings | 良好 |
| Presentation/ViewModels | RecordingState, RecordingSettings, RecordingList, Paywall, Subscription | 良好 |
| UI Tests | RecordingFlow, Analysis, Settings, Navigation, Playback | 基本フロー |

---

## 2. 発見された問題点と潜在的バグ

### 🔴 Critical（クラッシュの可能性）

#### 2.1 majorTriad() - MIDI範囲オーバーフロー
**ファイル**: `ScaleSettings.swift:280-286`

```swift
private func majorTriad(_ root: UInt8) -> [MIDINote] {
    return [
        try? MIDINote(root),      // root = 127 → OK
        try? MIDINote(root + 4),  // 127 + 4 = 131 → UInt8オーバーフロー!
        try? MIDINote(root + 7)   // 127 + 7 = 134 → UInt8オーバーフロー!
    ].compactMap { $0 }
}
```

**問題**:
- `root + 4` や `root + 7` が UInt8 の最大値 (255) を超えると、オーバーフローして予期しない小さな値になる
- 例: root=125 の場合、125+7=132 だが、UInt8では 132 (範囲内) → OK
- 例: root=252 の場合、252+4=256 → UInt8で 0 にラップ → 予期しないMIDIノート

**影響**: 高音域（MIDI 121以上）でスケール生成時に不正なコード生成

**推奨テスト**:
```swift
func testMajorTriad_HighRootNote_HandlesOverflowGracefully()
func testMajorTriad_MaxMIDINote127_ReturnsPartialTriad()
```

---

#### 2.2 generateKeyRoots() - UInt8 アンダーフロー
**ファイル**: `ScaleSettings.swift:229-233`

```swift
case .descendingOnly:
    var roots: [UInt8] = []
    for i in 0...descendingKeyCount {
        roots.append(start - UInt8(i) * descInterval)  // アンダーフロー危険!
    }
    return roots
```

**問題**:
- `start - UInt8(i) * descInterval` が負の値になるとアンダーフロー
- 例: start=48 (C3), descendingKeyCount=10, descInterval=5
  - i=10 の時: 48 - 50 = -2 → UInt8 では 254 にラップ

**影響**: 低音域から大きく下降するスケール設定でクラッシュまたは不正な音

**推奨テスト**:
```swift
func testGenerateKeyRoots_DescendingBelowZero_DoesNotCrash()
func testGenerateKeyRoots_LargeDescendingInterval_HandlesUnderflow()
```

---

#### 2.3 MIDINote 静的プロパティの try!
**ファイル**: `MIDINote.swift:15-16`

```swift
public static let middleC = try! MIDINote(60)  // C4
public static let hiC = try! MIDINote(72)      // C5
```

**問題**:
- `try!` は値が正しい限り問題ないが、コードレビューで見落としやすい
- 将来の変更で不正な値が入った場合、起動時クラッシュ

**影響**: 現状は問題なし（60, 72は有効値）

**推奨テスト**:
```swift
func testStaticMIDINotes_MiddleC_IsValid()
func testStaticMIDINotes_HiC_IsValid()
```

---

### 🟡 High（機能不全の可能性）

#### 2.4 SaveScalePresetUseCase - テストなし
**ファイル**: `SaveScalePresetUseCase.swift`（存在確認必要）

**問題**:
- スケールプリセット保存のUseCaseテストが存在しない
- 空の名前、重複名、特殊文字を含む名前での動作が未検証

**推奨テスト**:
```swift
func testSaveScalePreset_ValidName_SavesSuccessfully()
func testSaveScalePreset_EmptyName_ThrowsError()
func testSaveScalePreset_DuplicateName_OverwritesOrThrows()
func testSaveScalePreset_SpecialCharacters_HandlesGracefully()
```

---

#### 2.5 DeleteScalePresetUseCase - テストなし
**ファイル**: `DeleteScalePresetUseCase.swift`（存在確認必要）

**問題**:
- スケールプリセット削除のUseCaseテストが存在しない
- 存在しないプリセット削除時の動作が未検証

**推奨テスト**:
```swift
func testDeleteScalePreset_ExistingPreset_DeletesSuccessfully()
func testDeleteScalePreset_NonExistingPreset_ThrowsOrIgnores()
```

---

#### 2.6 LoadScalePresetsUseCase - テストなし
**ファイル**: `LoadScalePresetsUseCase.swift`（存在確認必要）

**問題**:
- 破損したJSONの読み込み時の動作が未検証
- 空のプリセットリストの処理が未検証

**推奨テスト**:
```swift
func testLoadScalePresets_EmptyRepository_ReturnsEmptyArray()
func testLoadScalePresets_CorruptedData_HandlesGracefully()
```

---

#### 2.7 RecordingLimit 境界値
**ファイル**: `RecordingLimit.swift`

```swift
public func isCountWithinLimit(_ count: Int) -> Bool {
    guard let limit = dailyCount else { return true }
    return count < limit  // ← < であり <= ではない
}
```

**問題**:
- `count == limit` の場合、`false` を返す（制限に達した）
- `remainingCount()` で `current > limit` の場合の表示

**推奨テスト**:
```swift
func testIsCountWithinLimit_AtExactLimit_ReturnsFalse()
func testIsCountWithinLimit_OneBeforeLimit_ReturnsTrue()
func testRemainingCount_ExceedsLimit_ReturnsZero()
```

---

#### 2.8 RecordingStateViewModel 状態遷移
**ファイル**: `RecordingStateViewModel.swift`

**問題**:
- 無効な状態遷移（例: idle → stopped）のテストなし
- 二重呼び出し（startRecording中にstartRecording）のテストなし
- エラー発生後の状態復帰テストなし

**推奨テスト**:
```swift
func testStartRecording_WhenAlreadyRecording_ThrowsOrIgnores()
func testStopRecording_WhenNotRecording_ThrowsOrIgnores()
func testRecordingState_AfterError_ReturnsToIdle()
```

---

### 🟢 Medium（エッジケース）

#### 2.9 ScaleSettings Codable 後方互換性
**ファイル**: `ScaleSettings.swift`

**問題**:
- `ascendingKeyStepInterval` / `descendingKeyStepInterval` は後から追加されたプロパティ
- 古いバージョンで保存されたJSONにこれらのフィールドがない場合のデコード

**推奨テスト**:
```swift
func testCodable_OldVersionWithoutStepInterval_DecodesWithDefaults()
```

---

#### 2.10 durationDescription フォーマット境界
**ファイル**: `RecordingLimit.swift`

```swift
public var durationDescription: String {
    guard let duration = maxDuration else { return "無制限" }
    if duration >= 60 {
        let minutes = Int(duration / 60)
        return "\(minutes)分"
    } else {
        return "\(Int(duration))秒"
    }
}
```

**問題**:
- 59秒と60秒の境界での表示切り替え
- 30.9秒 → "30秒" への切り捨て

**推奨テスト**:
```swift
func testDurationDescription_59Seconds_ShowsSeconds()
func testDurationDescription_60Seconds_ShowsMinutes()
func testDurationDescription_DecimalSeconds_Truncates()
```

---

## 3. テスト追加優先度

### Phase 1: Critical（リリースブロッカー）
| # | テスト対象 | 理由 |
|---|-----------|------|
| 1 | majorTriad MIDI境界値 | 高音域でクラッシュの可能性 |
| 2 | generateKeyRoots アンダーフロー | 低音域でクラッシュの可能性 |
| 3 | MIDINote 静的プロパティ検証 | 起動時クラッシュ防止 |

### Phase 2: High（機能品質）
| # | テスト対象 | 理由 |
|---|-----------|------|
| 4 | SaveScalePresetUseCase | プリセット保存機能の品質保証 |
| 5 | DeleteScalePresetUseCase | プリセット削除機能の品質保証 |
| 6 | LoadScalePresetsUseCase | プリセット読み込みの堅牢性 |
| 7 | RecordingLimit 境界値 | 課金機能の正確性 |
| 8 | RecordingStateViewModel 状態遷移 | 録音フローの安定性 |

### Phase 3: Medium（エッジケース）
| # | テスト対象 | 理由 |
|---|-----------|------|
| 9 | ScaleSettings Codable互換性 | アップデート後の設定引き継ぎ |
| 10 | durationDescription 境界値 | UI表示の正確性 |

---

## 4. 推奨アクション

### 即座に実施（リリース前必須）
1. **Phase 1のテスト追加** - クラッシュ防止
2. **既存テストの全実行** - 回帰確認
3. **実機での手動テスト** - シミュレータでは発見できない問題

### 時間があれば実施
1. **Phase 2のテスト追加** - 機能品質向上
2. **Phase 3のテスト追加** - エッジケース対応

### 将来の改善
1. **コードカバレッジ測定** - 数値での可視化
2. **CI/CDでのテスト自動実行** - 品質ゲート
3. **プロパティベーステスト** - 境界値の網羅的検証

---

## 5. 付録: テストケース詳細仕様

### A. majorTriad 境界値テスト

```swift
// ScaleSettingsTests.swift に追加

// MARK: - MajorTriad Boundary Tests

/// root=120 (MIDI上限付近) でトライアドが正しく生成されることを確認
/// 120 + 4 = 124, 120 + 7 = 127 → すべて有効範囲
func testMajorTriad_Root120_ReturnsFullTriad() throws {
    let settings = ScaleSettings(
        startNote: try MIDINote(120),
        endNote: try MIDINote(127),
        notePattern: .fiveToneScale,
        tempo: .standard,
        keyProgressionPattern: .ascendingOnly,
        ascendingKeyCount: 1,
        descendingKeyCount: 0
    )

    let elements = settings.generateScaleWithKeyChange()
    let chords = elements.compactMap { element -> [MIDINote]? in
        if case .chordLong(let notes) = element { return notes }
        return nil
    }

    // root=120 のトライアドは 120, 124, 127
    XCTAssertEqual(chords.first?.count, 3)
    XCTAssertEqual(chords.first?[0].value, 120)
    XCTAssertEqual(chords.first?[1].value, 124)
    XCTAssertEqual(chords.first?[2].value, 127)
}

/// root=124 でトライアドの一部が範囲外になることを確認
/// 124 + 4 = 128 → 範囲外, 124 + 7 = 131 → 範囲外
/// compactMap で nil が除外されるため、ルートのみのトライアド
func testMajorTriad_Root124_ReturnsPartialTriad() throws {
    // 実装の動作確認: UInt8オーバーフローが発生するか、
    // MIDINote初期化で例外がスローされてcompactMapで除外されるか
    let settings = ScaleSettings(
        startNote: try MIDINote(124),
        endNote: try MIDINote(127),
        notePattern: .fiveToneScale,
        tempo: .standard,
        keyProgressionPattern: .ascendingOnly,
        ascendingKeyCount: 1,
        descendingKeyCount: 0
    )

    let elements = settings.generateScaleWithKeyChange()
    let chords = elements.compactMap { element -> [MIDINote]? in
        if case .chordLong(let notes) = element { return notes }
        return nil
    }

    // 現在の実装では UInt8 オーバーフローが発生するため、
    // このテストで実際の動作を確認する
    XCTAssertNotNil(chords.first)
    // 期待: トライアドが不完全（1-2音）または予期しない音
}
```

### B. generateKeyRoots アンダーフローテスト

```swift
// ScaleSettingsTests.swift に追加

// MARK: - GenerateKeyRoots Boundary Tests

/// 低い開始音から大きく下降しても、アンダーフローしないことを確認
func testGenerateKeyRoots_DescendingFromLowNote_DoesNotUnderflow() throws {
    // C3 (48) から 10ステップ × 5セミトーン = 50 下降 → -2 になるはず
    let settings = ScaleSettings(
        startNote: try MIDINote(48),  // C3
        endNote: try MIDINote(60),
        notePattern: .fiveToneScale,
        tempo: .standard,
        keyProgressionPattern: .descendingOnly,
        ascendingKeyCount: 0,
        descendingKeyCount: 10,
        ascendingKeyStepInterval: 1,
        descendingKeyStepInterval: 5  // 5セミトーン × 10 = 50
    )

    // このテストで実際にアンダーフローが発生するか確認
    // 期待: クラッシュしない、または適切なエラーハンドリング
    let roots = settings.generateKeyRoots()

    // すべての roots が有効な MIDI 範囲内であることを確認
    for root in roots {
        XCTAssertLessThanOrEqual(root, 127, "Root \(root) exceeds MIDI max")
        // UInt8 なので 0 以下にはならないが、ラップアラウンドの確認
    }
}

/// 境界ちょうどの下降（アンダーフローしない最大値）
func testGenerateKeyRoots_DescendingToExactlyZero_Succeeds() throws {
    // MIDI 48 から 48ステップ × 1セミトーン = ちょうど 0
    let settings = ScaleSettings(
        startNote: try MIDINote(48),
        endNote: try MIDINote(60),
        notePattern: .fiveToneScale,
        tempo: .standard,
        keyProgressionPattern: .descendingOnly,
        ascendingKeyCount: 0,
        descendingKeyCount: 48,
        ascendingKeyStepInterval: 1,
        descendingKeyStepInterval: 1
    )

    let roots = settings.generateKeyRoots()

    // 最後の root は 0 であるべき
    XCTAssertEqual(roots.last, 0)
}
```

### C. RecordingLimit 境界値テスト

```swift
// RecordingLimitTests.swift に追加

// MARK: - Boundary Value Tests

func testIsCountWithinLimit_AtExactLimit_ReturnsFalse() {
    // Given: Free tier with limit of 5
    let limit = RecordingLimit.free

    // When: count equals the limit
    let result = limit.isCountWithinLimit(5)

    // Then: Should return false (at limit, cannot record more)
    XCTAssertFalse(result)
}

func testIsCountWithinLimit_OneBeforeLimit_ReturnsTrue() {
    // Given: Free tier with limit of 5
    let limit = RecordingLimit.free

    // When: count is one less than limit
    let result = limit.isCountWithinLimit(4)

    // Then: Should return true (can still record one more)
    XCTAssertTrue(result)
}

func testRemainingCount_AtExactLimit_ReturnsZero() {
    // Given: Free tier with limit of 5
    let limit = RecordingLimit.free

    // When: current equals limit
    let remaining = limit.remainingCount(5)

    // Then: Should show 0 remaining
    XCTAssertEqual(remaining, "0/5")
}

func testRemainingCount_ExceedsLimit_ReturnsZeroNotNegative() {
    // Given: Free tier with limit of 5
    let limit = RecordingLimit.free

    // When: current exceeds limit (edge case)
    let remaining = limit.remainingCount(10)

    // Then: Should show 0, not negative
    XCTAssertEqual(remaining, "0/5")
}

func testDurationDescription_59Seconds_ShowsSeconds() {
    // Given: Limit with 59 second duration
    let limit = RecordingLimit(dailyCount: 5, maxDuration: 59.0)

    // Then: Should display in seconds
    XCTAssertEqual(limit.durationDescription, "59秒")
}

func testDurationDescription_60Seconds_ShowsMinutes() {
    // Given: Limit with exactly 60 second duration
    let limit = RecordingLimit(dailyCount: 5, maxDuration: 60.0)

    // Then: Should display in minutes
    XCTAssertEqual(limit.durationDescription, "1分")
}

func testDurationDescription_DecimalSeconds_Truncates() {
    // Given: Limit with decimal duration
    let limit = RecordingLimit(dailyCount: 5, maxDuration: 30.9)

    // Then: Should truncate to integer
    XCTAssertEqual(limit.durationDescription, "30秒")
}
```

---

## 6. 次のステップ

このドキュメントの内容を確認の上、以下の順序で進めることを推奨します：

1. **ドキュメントレビュー** - 追加テストの妥当性確認
2. **Phase 1 テスト実装** - クリティカルな境界値テスト
3. **テスト実行・バグ修正** - 発見された問題の修正
4. **Phase 2-3 テスト実装** - 時間があれば

ご確認いただき、進め方についてフィードバックをお願いします。
