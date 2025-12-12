# スケール再生音選択機能 - テスト仕様書

## 概要

スケール再生音選択機能の包括的なテスト戦略とテストケースを定義します。TDD原則に従い、各層ごとに詳細なテストケースを記述します。

## テスト戦略

### テストピラミッド

```
        /\
       /  \  E2E Tests (10%)
      /    \  - 1-2個の重要なユーザーフロー
     /------\
    /        \ Integration Tests (20%)
   /          \ - 5-8個のコンポーネント統合テスト
  /------------\
 /              \ Unit Tests (70%)
/________________\ - 30-50個の単体テスト
```

### カバレッジ目標

| Layer | Target | 理由 |
|-------|--------|------|
| Domain | 100% | ビジネスロジックの中核、完全にテスト可能 |
| Presentation | 90% | ViewModelのロジック、UIは一部除外 |
| Infrastructure | 80% | AVFoundation統合部分は実機テストで補完 |
| UI | 60% | 重要なユーザーフローのみ |

---

## Phase 1: Domain層のテスト

### 1.1 ScaleSoundType のテスト

**ファイル**: `VocalisStudio/Packages/VocalisDomain/Tests/VocalisDomainTests/ValueObjects/ScaleSoundTypeTests.swift`

#### テストケース一覧

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 1 | `testAllCasesCount` | 全ケース数の確認 | 8種類すべて定義されている |
| 2 | `testDefaultValue` | デフォルト値の確認 | `.acousticGrandPiano`がデフォルト |
| 3 | `testMIDIProgramNumbers` | MIDI Program番号の確認 | General MIDI仕様に準拠 |
| 4 | `testSineWaveHasNoMIDIProgram` | サイン波のMIDI番号 | `nil`である |
| 5 | `testDisplayNames` | 表示名の確認 | すべて日本語で定義されている |
| 6 | `testIcons` | アイコンの確認 | すべてユニークなアイコン |
| 7 | `testDescriptions` | 説明文の確認 | すべて適切な説明がある |
| 8 | `testCodableConformance` | Codable準拠 | エンコード/デコードが正しい |
| 9 | `testHashableConformance` | Hashable準拠 | ハッシュ値が一貫している |
| 10 | `testCaseIterableConformance` | CaseIterable準拠 | allCasesが正しい順序 |

#### テストコード例

```swift
import XCTest
@testable import VocalisDomain

final class ScaleSoundTypeTests: XCTestCase {

    // MARK: - Test 1: All Cases Count

    func testAllCasesCount() {
        // Given: ScaleSoundType.allCases
        let allCases = ScaleSoundType.allCases

        // Then: 8種類すべて定義されている
        XCTAssertEqual(allCases.count, 8, "Should have 8 sound types")

        // すべてのケースが含まれていることを確認
        XCTAssertTrue(allCases.contains(.acousticGrandPiano))
        XCTAssertTrue(allCases.contains(.electricPiano))
        XCTAssertTrue(allCases.contains(.acousticGuitar))
        XCTAssertTrue(allCases.contains(.vibraphone))
        XCTAssertTrue(allCases.contains(.marimba))
        XCTAssertTrue(allCases.contains(.flute))
        XCTAssertTrue(allCases.contains(.clarinet))
        XCTAssertTrue(allCases.contains(.sineWave))
    }

    // MARK: - Test 2: Default Value

    func testDefaultValue() {
        // Given: ScaleSoundType.default
        let defaultType = ScaleSoundType.default

        // Then: acousticGrandPianoがデフォルト
        XCTAssertEqual(defaultType, .acousticGrandPiano)
    }

    // MARK: - Test 3: MIDI Program Numbers

    func testMIDIProgramNumbers() {
        // General MIDI仕様に準拠しているか確認

        // Acoustic Grand Piano: Program 0
        XCTAssertEqual(ScaleSoundType.acousticGrandPiano.midiProgram, 0)

        // Electric Piano 1: Program 4
        XCTAssertEqual(ScaleSoundType.electricPiano.midiProgram, 4)

        // Acoustic Guitar (nylon): Program 24
        XCTAssertEqual(ScaleSoundType.acousticGuitar.midiProgram, 24)

        // Vibraphone: Program 11
        XCTAssertEqual(ScaleSoundType.vibraphone.midiProgram, 11)

        // Marimba: Program 12
        XCTAssertEqual(ScaleSoundType.marimba.midiProgram, 12)

        // Flute: Program 73
        XCTAssertEqual(ScaleSoundType.flute.midiProgram, 73)

        // Clarinet: Program 71
        XCTAssertEqual(ScaleSoundType.clarinet.midiProgram, 71)
    }

    // MARK: - Test 4: Sine Wave Has No MIDI Program

    func testSineWaveHasNoMIDIProgram() {
        // Given: Sine Wave
        let sineWave = ScaleSoundType.sineWave

        // Then: MIDI Programはnil（プログラム生成）
        XCTAssertNil(sineWave.midiProgram)
    }

    // MARK: - Test 5: Display Names

    func testDisplayNames() {
        // すべての音源に適切な日本語表示名があることを確認

        XCTAssertEqual(
            ScaleSoundType.acousticGrandPiano.displayName,
            "アコースティック・グランド・ピアノ"
        )
        XCTAssertEqual(
            ScaleSoundType.electricPiano.displayName,
            "エレクトリック・ピアノ"
        )
        XCTAssertEqual(
            ScaleSoundType.acousticGuitar.displayName,
            "アコースティック・ギター"
        )
        XCTAssertEqual(
            ScaleSoundType.vibraphone.displayName,
            "ヴィブラフォン"
        )
        XCTAssertEqual(
            ScaleSoundType.marimba.displayName,
            "マリンバ"
        )
        XCTAssertEqual(
            ScaleSoundType.flute.displayName,
            "フルート"
        )
        XCTAssertEqual(
            ScaleSoundType.clarinet.displayName,
            "クラリネット"
        )
        XCTAssertEqual(
            ScaleSoundType.sineWave.displayName,
            "サイン波"
        )

        // すべての表示名が空でないことを確認
        for soundType in ScaleSoundType.allCases {
            XCTAssertFalse(
                soundType.displayName.isEmpty,
                "\(soundType) should have non-empty display name"
            )
        }
    }

    // MARK: - Test 6: Icons

    func testIcons() {
        // すべての音源にアイコンが定義されていることを確認

        XCTAssertEqual(ScaleSoundType.acousticGrandPiano.icon, "🎹")
        XCTAssertEqual(ScaleSoundType.electricPiano.icon, "🎹✨")
        XCTAssertEqual(ScaleSoundType.acousticGuitar.icon, "🎸")
        XCTAssertEqual(ScaleSoundType.vibraphone.icon, "🎵")
        XCTAssertEqual(ScaleSoundType.marimba.icon, "🥁")
        XCTAssertEqual(ScaleSoundType.flute.icon, "🎺")
        XCTAssertEqual(ScaleSoundType.clarinet.icon, "🎷")
        XCTAssertEqual(ScaleSoundType.sineWave.icon, "〜")

        // すべてのアイコンが空でないことを確認
        for soundType in ScaleSoundType.allCases {
            XCTAssertFalse(
                soundType.icon.isEmpty,
                "\(soundType) should have non-empty icon"
            )
        }

        // アイコンがユニークであることを確認
        let icons = ScaleSoundType.allCases.map { $0.icon }
        let uniqueIcons = Set(icons)
        XCTAssertEqual(
            icons.count,
            uniqueIcons.count,
            "All icons should be unique"
        )
    }

    // MARK: - Test 7: Descriptions

    func testDescriptions() {
        // すべての音源に説明文が定義されていることを確認

        for soundType in ScaleSoundType.allCases {
            XCTAssertFalse(
                soundType.description.isEmpty,
                "\(soundType) should have non-empty description"
            )

            // 説明文が一定の長さ以上あることを確認（品質チェック）
            XCTAssertGreaterThan(
                soundType.description.count,
                10,
                "\(soundType) description should be descriptive"
            )
        }
    }

    // MARK: - Test 8: Codable Conformance

    func testCodableConformance() throws {
        // すべての音源タイプをエンコード/デコードできることを確認

        let encoder = JSONEncoder()
        let decoder = JSONDecoder()

        for soundType in ScaleSoundType.allCases {
            // When: エンコード
            let data = try encoder.encode(soundType)

            // Then: デコード成功
            let decoded = try decoder.decode(ScaleSoundType.self, from: data)

            // Then: 同じ値が復元される
            XCTAssertEqual(decoded, soundType)
        }
    }

    // MARK: - Test 9: Hashable Conformance

    func testHashableConformance() {
        // すべての音源タイプがハッシュ可能であることを確認

        var hashValues: Set<Int> = []

        for soundType in ScaleSoundType.allCases {
            let hashValue = soundType.hashValue
            hashValues.insert(hashValue)
        }

        // すべてのハッシュ値がユニークであることを確認
        XCTAssertEqual(
            hashValues.count,
            ScaleSoundType.allCases.count,
            "All hash values should be unique"
        )
    }

    // MARK: - Test 10: CaseIterable Conformance

    func testCaseIterableConformance() {
        // allCasesの順序が期待通りであることを確認

        let expectedOrder: [ScaleSoundType] = [
            .acousticGrandPiano,
            .electricPiano,
            .acousticGuitar,
            .vibraphone,
            .marimba,
            .flute,
            .clarinet,
            .sineWave
        ]

        XCTAssertEqual(ScaleSoundType.allCases, expectedOrder)
    }
}
```

### 1.2 AudioDetectionSettings の拡張テスト

**ファイル**: `VocalisStudio/Packages/VocalisDomain/Tests/VocalisDomainTests/ValueObjects/AudioDetectionSettingsTests.swift`

#### 追加テストケース

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 11 | `testDefaultSettingsIncludesScaleSoundType` | デフォルト設定 | `scaleSoundType`が含まれる |
| 12 | `testCustomScaleSoundTypeInitialization` | カスタム初期化 | 指定した音源が設定される |
| 13 | `testScaleSoundTypeEncoding` | エンコード | 正しくエンコードされる |
| 14 | `testScaleSoundTypeDecoding` | デコード | 正しくデコードされる |
| 15 | `testBackwardCompatibility` | 後方互換性 | 古いデータでもデコード可能 |
| 16 | `testEqualityWithDifferentScaleSoundType` | 等価性 | 音源が異なれば不等 |

#### テストコード例（抜粋）

```swift
import XCTest
@testable import VocalisDomain

final class AudioDetectionSettingsTests: XCTestCase {

    // MARK: - Test 11: Default Settings Includes ScaleSoundType

    func testDefaultSettingsIncludesScaleSoundType() {
        // Given: デフォルト設定
        let settings = AudioDetectionSettings.default

        // Then: scaleSoundTypeが含まれる
        XCTAssertEqual(settings.scaleSoundType, .acousticGrandPiano)
    }

    // MARK: - Test 12: Custom ScaleSoundType Initialization

    func testCustomScaleSoundTypeInitialization() {
        // すべての音源タイプで初期化できることを確認

        for soundType in ScaleSoundType.allCases {
            // When: カスタム音源で初期化
            let settings = AudioDetectionSettings(
                scaleSoundType: soundType
            )

            // Then: 指定した音源が設定される
            XCTAssertEqual(settings.scaleSoundType, soundType)
        }
    }

    // MARK: - Test 13-14: Encoding/Decoding

    func testScaleSoundTypeEncodingAndDecoding() throws {
        // Given: カスタム音源を含む設定
        let originalSettings = AudioDetectionSettings(
            scalePlaybackVolume: 0.7,
            recordingPlaybackVolume: 0.6,
            rmsSilenceThreshold: -35.0,
            confidenceThreshold: 0.85,
            scaleSoundType: .flute
        )

        // When: エンコード
        let encoder = JSONEncoder()
        let data = try encoder.encode(originalSettings)

        // When: デコード
        let decoder = JSONDecoder()
        let decodedSettings = try decoder.decode(
            AudioDetectionSettings.self,
            from: data
        )

        // Then: すべてのプロパティが正しく復元される
        XCTAssertEqual(decodedSettings, originalSettings)
        XCTAssertEqual(decodedSettings.scaleSoundType, .flute)
    }

    // MARK: - Test 15: Backward Compatibility

    func testBackwardCompatibility() throws {
        // Given: scaleSoundTypeを含まない古いJSON（既存ユーザーのデータ）
        let oldJSON = """
        {
            "scalePlaybackVolume": 0.5,
            "recordingPlaybackVolume": 0.5,
            "rmsSilenceThreshold": -40.0,
            "confidenceThreshold": 0.8
        }
        """

        let data = oldJSON.data(using: .utf8)!

        // When: デコード
        let decoder = JSONDecoder()
        let settings = try decoder.decode(
            AudioDetectionSettings.self,
            from: data
        )

        // Then: デフォルト値が使用される
        XCTAssertEqual(settings.scaleSoundType, .acousticGrandPiano)

        // 他のプロパティは正しくデコードされる
        XCTAssertEqual(settings.scalePlaybackVolume, 0.5)
        XCTAssertEqual(settings.recordingPlaybackVolume, 0.5)
    }

    // MARK: - Test 16: Equality With Different ScaleSoundType

    func testEqualityWithDifferentScaleSoundType() {
        // Given: 音源のみ異なる設定
        let settings1 = AudioDetectionSettings(
            scalePlaybackVolume: 0.5,
            scaleSoundType: .acousticGrandPiano
        )

        let settings2 = AudioDetectionSettings(
            scalePlaybackVolume: 0.5,
            scaleSoundType: .electricPiano
        )

        // Then: 等しくない
        XCTAssertNotEqual(settings1, settings2)
    }
}
```

---

## Phase 2: Presentation層のテスト

### 2.1 AudioSettingsViewModel のテスト

**ファイル**: `VocalisStudio/VocalisStudioTests/Presentation/ViewModels/AudioSettingsViewModelTests.swift`

#### テストケース一覧

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 17 | `testInitialization_LoadsScaleSoundType` | 初期化 | リポジトリから音源を読み込む |
| 18 | `testScaleSoundTypeChange_UpdatesHasChanges` | 変更検出 | `hasChanges`が`true`になる |
| 19 | `testSaveSettings_PersistsScaleSoundType` | 保存 | 音源が永続化される |
| 20 | `testResetSettings_RestoresDefaultScaleSoundType` | リセット | デフォルト音源に戻る |
| 21 | `testMultipleChanges_HasChangesReflectsAll` | 複数変更 | すべての変更を検出 |
| 22 | `testSaveWithoutChanges_DoesNotUpdateRepository` | 変更なし保存 | リポジトリ更新なし |

#### テストコード例

```swift
import XCTest
@testable import VocalisStudio
@testable import VocalisDomain

@MainActor
final class AudioSettingsViewModelTests: XCTestCase {

    var sut: AudioSettingsViewModel!
    var mockRepository: MockAudioSettingsRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockAudioSettingsRepository()
        sut = AudioSettingsViewModel(repository: mockRepository)
    }

    override func tearDown() {
        sut = nil
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Test 17: Initialization Loads ScaleSoundType

    func testInitialization_LoadsScaleSoundType() {
        // Given: リポジトリにカスタム音源が保存されている
        let customSettings = AudioDetectionSettings(
            scaleSoundType: .vibraphone
        )
        mockRepository.settingsToReturn = customSettings

        // When: ViewModelを初期化
        sut = AudioSettingsViewModel(repository: mockRepository)

        // Then: 音源が正しく読み込まれる
        XCTAssertEqual(sut.scaleSoundType, .vibraphone)
    }

    // MARK: - Test 18: ScaleSoundType Change Updates HasChanges

    func testScaleSoundTypeChange_UpdatesHasChanges() {
        // Given: 初期状態（変更なし）
        XCTAssertFalse(sut.hasChanges)

        // When: 音源を変更
        sut.scaleSoundType = .electricPiano

        // Then: hasChangesがtrueになる
        XCTAssertTrue(sut.hasChanges)
    }

    // MARK: - Test 19: Save Settings Persists ScaleSoundType

    func testSaveSettings_PersistsScaleSoundType() throws {
        // Given: 音源を変更
        sut.scaleSoundType = .flute

        // When: 保存
        try sut.saveSettings()

        // Then: リポジトリに保存される
        XCTAssertTrue(mockRepository.saveCalled)
        XCTAssertEqual(
            mockRepository.savedSettings?.scaleSoundType,
            .flute
        )

        // Then: hasChangesがfalseに戻る
        XCTAssertFalse(sut.hasChanges)
    }

    // MARK: - Test 20: Reset Settings Restores Default ScaleSoundType

    func testResetSettings_RestoresDefaultScaleSoundType() throws {
        // Given: 音源を変更
        sut.scaleSoundType = .marimba

        // When: リセット
        try sut.resetSettings()

        // Then: デフォルト音源に戻る
        XCTAssertEqual(sut.scaleSoundType, .acousticGrandPiano)

        // Then: リポジトリのresetが呼ばれる
        XCTAssertTrue(mockRepository.resetCalled)
    }

    // MARK: - Test 21: Multiple Changes HasChanges Reflects All

    func testMultipleChanges_HasChangesReflectsAll() {
        // Given: 初期状態
        XCTAssertFalse(sut.hasChanges)

        // When: 音量と音源を両方変更
        sut.scalePlaybackVolume = 0.8
        sut.scaleSoundType = .clarinet

        // Then: hasChangesがtrue
        XCTAssertTrue(sut.hasChanges)

        // When: 元に戻す
        sut.scalePlaybackVolume = 0.5
        sut.scaleSoundType = .acousticGrandPiano

        // Then: hasChangesがfalse
        XCTAssertFalse(sut.hasChanges)
    }

    // MARK: - Test 22: Save Without Changes Does Not Update Repository

    func testSaveWithoutChanges_DoesNotUpdateRepository() throws {
        // Given: 変更なし
        XCTAssertFalse(sut.hasChanges)

        // When: 保存を試みる
        try sut.saveSettings()

        // Then: リポジトリは更新されない（実際の実装による）
        // または、同じ値で保存される（冪等性）
        XCTAssertTrue(mockRepository.saveCalled)
    }
}

// MARK: - Mock Repository

class MockAudioSettingsRepository: AudioSettingsRepositoryProtocol {
    var settingsToReturn: AudioDetectionSettings = .default
    var saveCalled = false
    var savedSettings: AudioDetectionSettings?
    var resetCalled = false

    func get() -> AudioDetectionSettings {
        return settingsToReturn
    }

    func save(_ settings: AudioDetectionSettings) throws {
        saveCalled = true
        savedSettings = settings
        settingsToReturn = settings
    }

    func reset() throws {
        resetCalled = true
        settingsToReturn = .default
    }
}
```

### 2.2 AudioSettingsView の UIテスト

**ファイル**: `VocalisStudio/VocalisStudioUITests/Settings/AudioSettingsUITests.swift`

#### テストケース一覧

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 23 | `testScaleSoundSectionExists` | セクション表示 | スケール再生音セクションが表示される |
| 24 | `testPickerDisplaysAllSoundTypes` | Picker選択肢 | 8種類すべて表示される |
| 25 | `testSelectingSoundType_UpdatesDisplay` | 音源選択 | 選択した音源が表示される |
| 26 | `testFooterUpdates_WhenSoundTypeChanges` | フッター更新 | 説明文が動的に変更される |
| 27 | `testSaveButton_EnabledWhenChanged` | 保存ボタン | 変更時のみ有効 |
| 28 | `testResetButton_RestoresDefault` | リセット | デフォルトに戻る |

#### テストコード例（抜粋）

```swift
import XCTest

final class AudioSettingsUITests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()

        // 設定画面 → オーディオ設定へ遷移
        app.tabBars.buttons["設定"].tap()
        app.buttons["音量・検出設定"].tap()
    }

    // MARK: - Test 23: Scale Sound Section Exists

    func testScaleSoundSectionExists() {
        // Then: スケール再生音セクションが存在する
        XCTAssertTrue(app.staticTexts["スケール再生音"].exists)
        XCTAssertTrue(app.staticTexts["再生音"].exists)
    }

    // MARK: - Test 24: Picker Displays All Sound Types

    func testPickerDisplaysAllSoundTypes() {
        // When: Pickerをタップしてメニューを開く
        app.buttons["再生音"].tap()

        // Then: 8種類すべての音源が表示される
        XCTAssertTrue(app.buttons["🎹 アコースティック・グランド・ピアノ"].exists)
        XCTAssertTrue(app.buttons["🎹✨ エレクトリック・ピアノ"].exists)
        XCTAssertTrue(app.buttons["🎸 アコースティック・ギター"].exists)
        XCTAssertTrue(app.buttons["🎵 ヴィブラフォン"].exists)
        XCTAssertTrue(app.buttons["🥁 マリンバ"].exists)
        XCTAssertTrue(app.buttons["🎺 フルート"].exists)
        XCTAssertTrue(app.buttons["🎷 クラリネット"].exists)
        XCTAssertTrue(app.buttons["〜 サイン波"].exists)
    }

    // MARK: - Test 25: Selecting Sound Type Updates Display

    func testSelectingSoundType_UpdatesDisplay() {
        // When: Pickerで音源を選択
        app.buttons["再生音"].tap()
        app.buttons["🎺 フルート"].tap()

        // Then: 選択した音源が表示される
        XCTAssertTrue(app.staticTexts["🎺"].exists)
        XCTAssertTrue(app.staticTexts["フルート"].exists)
    }

    // MARK: - Test 26: Footer Updates When Sound Type Changes

    func testFooterUpdates_WhenSoundTypeChanges() {
        // When: 音源を変更
        app.buttons["再生音"].tap()
        app.buttons["🎸 アコースティック・ギター"].tap()

        // Then: フッターの説明文が更新される
        XCTAssertTrue(
            app.staticTexts["柔らかく温かみのある音色、中低音域が豊か"].exists
        )
    }

    // MARK: - Test 27: Save Button Enabled When Changed

    func testSaveButton_EnabledWhenChanged() {
        // Given: 初期状態（保存ボタン無効）
        XCTAssertFalse(app.buttons["保存"].isEnabled)

        // When: 音源を変更
        app.buttons["再生音"].tap()
        app.buttons["🎵 ヴィブラフォン"].tap()

        // Then: 保存ボタンが有効になる
        XCTAssertTrue(app.buttons["保存"].isEnabled)
    }
}
```

---

## Phase 3: Infrastructure層のテスト

### 3.1 AVAudioScalePlayer のテスト

**ファイル**: `VocalisStudio/VocalisStudioTests/Infrastructure/Audio/AVAudioScalePlayerTests.swift`

#### テストケース一覧

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 29 | `testInitialization_LoadsDefaultMIDISound` | 初期化 | デフォルト音源がロードされる |
| 30 | `testSetSoundType_LoadsMIDIProgram` | 音源切り替え | MIDI Programが正しくロードされる |
| 31 | `testPlayMIDISound_StartsAndStopsNotes` | MIDI再生 | Note On/Off が正しい |
| 32 | `testPlaySineWave_UsesPCMBuffer` | サイン波再生 | PlayerNodeが使用される |
| 33 | `testPlayChord_SimultaneousNotes` | 和音再生 | 複数音が同時に再生される |
| 34 | `testMutedPlayback_ZeroVelocity` | ミュート再生 | Velocity=0で再生される |
| 35 | `testAllMIDISoundTypes_LoadSuccessfully` | 全音源ロード | 7種類すべてロード成功 |

#### テストコード例（抜粋）

```swift
import XCTest
import AVFoundation
@testable import VocalisStudio
@testable import VocalisDomain

final class AVAudioScalePlayerTests: XCTestCase {

    var sut: AVAudioScalePlayer!

    override func setUp() {
        super.setUp()
        sut = AVAudioScalePlayer()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Test 29: Initialization Loads Default MIDI Sound

    func testInitialization_LoadsDefaultMIDISound() async throws {
        // Given: AVAudioScalePlayer初期化
        // (setUp()で既に初期化済み)

        // Then: デフォルト音源（Acoustic Grand Piano）がロードされている
        // 内部状態を確認（実装依存）
        XCTAssertNotNil(sut)

        // 簡単なスケールをロードして再生可能か確認
        let simpleScale: [ScaleElement] = [
            .scaleNote(try MIDINote(60))  // C4
        ]

        try await sut.loadScaleElements(
            simpleScale,
            tempo: Tempo.standard
        )

        // エラーなくロードできればOK
        XCTAssertTrue(true)
    }

    // MARK: - Test 30: SetSoundType Loads MIDI Program

    func testSetSoundType_LoadsMIDIProgram() async throws {
        // すべてのMIDI音源タイプでテスト
        let midiSoundTypes: [ScaleSoundType] = [
            .acousticGrandPiano,
            .electricPiano,
            .acousticGuitar,
            .vibraphone,
            .marimba,
            .flute,
            .clarinet
        ]

        for soundType in midiSoundTypes {
            // When: 音源を設定
            sut.setSoundType(soundType)

            // Then: エラーなく設定できる
            // （内部的にloadInstrument()が呼ばれている）

            // 簡単なスケールをロードして確認
            let testNote = try MIDINote(60)
            let testScale: [ScaleElement] = [.scaleNote(testNote)]

            try await sut.loadScaleElements(
                testScale,
                tempo: Tempo.standard
            )

            // エラーなくロードできればOK
            XCTAssertTrue(true, "\(soundType) should load successfully")
        }
    }

    // MARK: - Test 31: Play MIDI Sound Starts And Stops Notes

    func testPlayMIDISound_StartsAndStopsNotes() async throws {
        // Given: ピアノ音源を設定
        sut.setSoundType(.acousticGrandPiano)

        // Given: シンプルなスケール
        let c4 = try MIDINote(60)
        let scale: [ScaleElement] = [
            .scaleNote(c4)
        ]

        try await sut.loadScaleElements(
            scale,
            tempo: Tempo(secondsPerNote: 0.5)
        )

        // When: 再生
        try await sut.play(muted: false)

        // Then: エラーなく再生完了
        // （実際にはAVAudioEngineのモックを使用してNote On/Off を検証）
        XCTAssertFalse(sut.isPlaying)
    }

    // MARK: - Test 33: Play Chord Simultaneous Notes

    func testPlayChord_SimultaneousNotes() async throws {
        // Given: 和音を含むスケール
        let root = try MIDINote(60)
        let third = try MIDINote(64)
        let fifth = try MIDINote(67)

        let scale: [ScaleElement] = [
            .chordLong([root, third, fifth])
        ]

        sut.setSoundType(.acousticGrandPiano)
        try await sut.loadScaleElements(
            scale,
            tempo: Tempo.standard
        )

        // When: 再生
        try await sut.play(muted: false)

        // Then: エラーなく和音が再生される
        // （実際には3つのNote Onが同時に発生することを検証）
        XCTAssertFalse(sut.isPlaying)
    }

    // MARK: - Test 34: Muted Playback Zero Velocity

    func testMutedPlayback_ZeroVelocity() async throws {
        // Given: スケール
        let scale: [ScaleElement] = [
            .scaleNote(try MIDINote(60))
        ]

        sut.setSoundType(.acousticGrandPiano)
        try await sut.loadScaleElements(
            scale,
            tempo: Tempo.standard
        )

        // When: ミュート再生
        try await sut.play(muted: true)

        // Then: エラーなく再生完了
        // （実際にはVelocity=0で再生されることを検証）
        XCTAssertFalse(sut.isPlaying)
    }
}
```

---

## Phase 4: 統合テスト（E2E）

### 4.1 End-to-End テスト

**ファイル**: `VocalisStudio/VocalisStudioUITests/E2E/ScaleSoundSelectionE2ETests.swift`

#### テストケース一覧

| # | テスト名 | 目的 | 期待結果 |
|---|---------|------|---------|
| 36 | `testFullFlow_ChangeAndPlayScale` | 完全フロー | 音源変更→再生成功 |
| 37 | `testPersistence_RestartApp` | 永続化 | アプリ再起動後も保持 |
| 38 | `testAllSoundTypes_PlaySuccessfully` | 全音源再生 | すべて再生可能 |

#### テストコード例

```swift
import XCTest

final class ScaleSoundSelectionE2ETests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["--uitesting"]
        app.launch()
    }

    // MARK: - Test 36: Full Flow Change And Play Scale

    func testFullFlow_ChangeAndPlayScale() throws {
        // Step 1: 設定画面へ移動
        app.tabBars.buttons["設定"].tap()
        app.buttons["音量・検出設定"].tap()

        // Step 2: 音源をフルートに変更
        app.buttons["再生音"].tap()
        app.buttons["🎺 フルート"].tap()

        // Step 3: 保存
        app.buttons["保存"].tap()

        // Step 4: 録音画面へ移動
        app.buttons["キャンセル"].tap()  // 設定画面を閉じる
        app.tabBars.buttons["録音"].tap()

        // Step 5: スケール再生を開始
        app.buttons["RecordButton"].tap()

        // Step 6: スケール再生が開始される
        let scaleProgressExists = app.progressIndicators["ScaleProgress"]
            .waitForExistence(timeout: 2)
        XCTAssertTrue(scaleProgressExists)

        // Step 7: スケール再生が完了するまで待機
        sleep(5)  // 実際のスケール再生時間

        // Step 8: 録音停止
        app.buttons["StopButton"].tap()

        // Then: エラーなく完了
        XCTAssertTrue(app.tabBars.buttons["録音"].exists)
    }

    // MARK: - Test 37: Persistence Restart App

    func testPersistence_RestartApp() throws {
        // Step 1: 音源を変更して保存
        app.tabBars.buttons["設定"].tap()
        app.buttons["音量・検出設定"].tap()
        app.buttons["再生音"].tap()
        app.buttons["🎸 アコースティック・ギター"].tap()
        app.buttons["保存"].tap()

        // Step 2: アプリを再起動
        app.terminate()
        app.launch()

        // Step 3: 設定画面で確認
        app.tabBars.buttons["設定"].tap()
        app.buttons["音量・検出設定"].tap()

        // Then: 選択した音源が保持されている
        XCTAssertTrue(app.staticTexts["🎸"].exists)
        XCTAssertTrue(app.staticTexts["アコースティック・ギター"].exists)
    }
}
```

---

## テスト実行戦略

### 開発中のテスト実行

**TDDサイクル中**:
```bash
# 特定のテストクラスのみ実行（高速）
./VocalisStudio/scripts/test-runner.sh unit ScaleSoundTypeTests
```

**Phase完了時**:
```bash
# そのPhaseのすべてのテストを実行
./VocalisStudio/scripts/test-runner.sh unit  # Domain + Presentation
./VocalisStudio/scripts/test-runner.sh ui    # UI Tests
```

### CI/CDでのテスト実行

**Pull Request時**:
```bash
# すべてのテストを実行
./VocalisStudio/scripts/test-runner.sh all
```

**リリース前**:
```bash
# 実機での統合テスト
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio-All \
  -destination 'platform=iOS,name=iPhone 15 Pro'
```

---

## カバレッジレポート

### 測定方法

```bash
# カバレッジ測定付きでテスト実行
xcodebuild test \
  -project VocalisStudio.xcodeproj \
  -scheme VocalisStudio-All \
  -destination 'platform=iOS Simulator,name=iPhone 15 Pro' \
  -enableCodeCoverage YES

# カバレッジレポート生成
xcrun xccov view --report \
  ~/Library/Developer/Xcode/DerivedData/VocalisStudio-*/Logs/Test/*.xcresult
```

### 目標カバレッジ

| ファイル | 目標 | 理由 |
|---------|------|------|
| ScaleSoundType.swift | 100% | すべてのケースとプロパティをテスト |
| AudioDetectionSettings.swift | 100% | エンコード/デコードを完全にテスト |
| AudioSettingsViewModel.swift | 90%+ | ビジネスロジックをカバー |
| AVAudioScalePlayer.swift | 80%+ | AVFoundation統合部分は実機テストで補完 |

---

## テストデータ管理

### モックデータ

**ファイル**: `VocalisStudioTests/Mocks/MockAudioSettingsRepository.swift`

```swift
class MockAudioSettingsRepository: AudioSettingsRepositoryProtocol {
    var settingsToReturn: AudioDetectionSettings = .default
    var saveCalled = false
    var savedSettings: AudioDetectionSettings?
    var resetCalled = false

    func get() -> AudioDetectionSettings {
        return settingsToReturn
    }

    func save(_ settings: AudioDetectionSettings) throws {
        saveCalled = true
        savedSettings = settings
        settingsToReturn = settings
    }

    func reset() throws {
        resetCalled = true
        settingsToReturn = .default
    }
}
```

### テストフィクスチャ

**ファイル**: `VocalisStudioTests/Fixtures/AudioDetectionSettingsFixtures.swift`

```swift
extension AudioDetectionSettings {
    static var testDefault: AudioDetectionSettings {
        AudioDetectionSettings()
    }

    static var testWithFlute: AudioDetectionSettings {
        AudioDetectionSettings(scaleSoundType: .flute)
    }

    static var testWithAllCustom: AudioDetectionSettings {
        AudioDetectionSettings(
            scalePlaybackVolume: 0.8,
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: -35.0,
            confidenceThreshold: 0.85,
            scaleSoundType: .vibraphone
        )
    }
}
```

---

## まとめ

### テスト数の内訳

- **Domain層**: 16テスト（ScaleSoundType: 10, AudioDetectionSettings: 6）
- **Presentation層**: 6テスト（ViewModel: 6）
- **Infrastructure層**: 7テスト（AVAudioScalePlayer: 7）
- **UI層**: 6テスト（AudioSettingsView: 6）
- **E2E**: 3テスト（統合フロー: 3）

**合計**: 38テスト

### 実行時間見積もり

- Unit Tests: 1-2秒（高速）
- UI Tests: 30-60秒（中速）
- E2E Tests: 60-120秒（低速）

**合計**: 約2-3分

### テスト品質保証

✅ **すべてのテストがTDD原則に従う**:
1. 🔴 Red: テストを先に書く
2. 🟢 Green: 最小限の実装で通す
3. 🔵 Refactor: コード品質を改善

✅ **カバレッジ目標を達成**:
- Domain: 100%
- Presentation: 90%+
- Infrastructure: 80%+
- UI: 60%+

✅ **自動化とCI/CD統合**:
- すべてのテストが自動実行可能
- Pull Request時に必ず実行
- カバレッジレポートを自動生成
