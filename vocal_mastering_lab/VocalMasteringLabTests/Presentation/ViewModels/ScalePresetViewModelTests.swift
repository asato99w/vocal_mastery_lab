import XCTest
@testable import VocalisDomain
@testable import VocalMasteringLab

/// Tests for ScalePresetViewModel
final class ScalePresetViewModelTests: XCTestCase {

    var sut: ScalePresetViewModel!
    private var mockRepository: MockScalePresetRepository!
    var saveUseCase: SaveScalePresetUseCase!
    var loadUseCase: LoadScalePresetsUseCase!
    var deleteUseCase: DeleteScalePresetUseCase!

    override func setUp() {
        super.setUp()
        mockRepository = MockScalePresetRepository()
        saveUseCase = SaveScalePresetUseCase(repository: mockRepository)
        loadUseCase = LoadScalePresetsUseCase(repository: mockRepository)
        deleteUseCase = DeleteScalePresetUseCase(repository: mockRepository)
        sut = ScalePresetViewModel(
            saveUseCase: saveUseCase,
            loadUseCase: loadUseCase,
            deleteUseCase: deleteUseCase
        )
    }

    override func tearDown() {
        sut = nil
        mockRepository = nil
        saveUseCase = nil
        loadUseCase = nil
        deleteUseCase = nil
        super.tearDown()
    }

    // MARK: - Helper

    private func createTestSettings() -> ScalePresetSettings {
        ScalePresetSettings(
            scaleType: .fiveTone,
            startPitchIndex: 12,
            tempo: 120,
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 3,
            descendingKeyCount: 3,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )
    }

    private func createSettingsViewModel() -> RecordingSettingsViewModel {
        let vm = RecordingSettingsViewModel()
        vm.scaleType = .fiveTone
        vm.startPitchIndex = 12
        vm.tempo = 120
        vm.keyProgressionPattern = .ascendingThenDescending
        vm.ascendingKeyCount = 3
        vm.descendingKeyCount = 3
        vm.ascendingKeyStepInterval = 1
        vm.descendingKeyStepInterval = 1
        return vm
    }

    // MARK: - Init Tests

    func testInit_shouldLoadPresets() {
        // Given
        let preset = ScalePreset(name: "Existing", settings: createTestSettings())
        mockRepository.presets = [preset]

        // When
        let viewModel = ScalePresetViewModel(
            saveUseCase: saveUseCase,
            loadUseCase: loadUseCase,
            deleteUseCase: deleteUseCase
        )

        // Then
        XCTAssertEqual(viewModel.presets.count, 1)
        XCTAssertEqual(viewModel.presets.first?.name, "Existing")
    }

    // MARK: - SavePreset Tests

    func testSavePreset_shouldAddToPresets() {
        // Given
        let settingsVM = createSettingsViewModel()

        // When
        sut.savePreset(name: "New Preset", from: settingsVM)

        // Then
        XCTAssertEqual(sut.presets.count, 1)
        XCTAssertEqual(sut.presets.first?.name, "New Preset")
    }

    func testSavePreset_shouldClearNewPresetName() {
        // Given
        let settingsVM = createSettingsViewModel()
        sut.newPresetName = "Test"

        // When
        sut.savePreset(name: "New Preset", from: settingsVM)

        // Then
        XCTAssertEqual(sut.newPresetName, "")
    }

    func testSavePreset_shouldDismissSaveDialog() {
        // Given
        let settingsVM = createSettingsViewModel()
        sut.isShowingSaveDialog = true

        // When
        sut.savePreset(name: "New Preset", from: settingsVM)

        // Then
        XCTAssertFalse(sut.isShowingSaveDialog)
    }

    func testSavePreset_shouldPreserveSettings() {
        // Given
        let settingsVM = RecordingSettingsViewModel()
        settingsVM.scaleType = .octaveRepeat
        settingsVM.startPitchIndex = 24
        settingsVM.tempo = 90
        settingsVM.keyProgressionPattern = .descendingOnly
        settingsVM.ascendingKeyCount = 5
        settingsVM.descendingKeyCount = 4
        settingsVM.ascendingKeyStepInterval = 2
        settingsVM.descendingKeyStepInterval = 3

        // When
        sut.savePreset(name: "Full Settings", from: settingsVM)

        // Then
        let saved = sut.presets.first
        XCTAssertEqual(saved?.settings.scaleType, .octaveRepeat)
        XCTAssertEqual(saved?.settings.startPitchIndex, 24)
        XCTAssertEqual(saved?.settings.tempo, 90)
        XCTAssertEqual(saved?.settings.keyProgressionPattern, .descendingOnly)
        XCTAssertEqual(saved?.settings.ascendingKeyCount, 5)
        XCTAssertEqual(saved?.settings.descendingKeyCount, 4)
        XCTAssertEqual(saved?.settings.ascendingKeyStepInterval, 2)
        XCTAssertEqual(saved?.settings.descendingKeyStepInterval, 3)
    }

    // MARK: - DeletePreset Tests

    func testDeletePreset_shouldRemoveFromPresets() {
        // Given
        let preset = ScalePreset(name: "To Delete", settings: createTestSettings())
        mockRepository.presets = [preset]
        sut.loadPresets()

        // When
        sut.deletePreset(id: preset.id)

        // Then
        XCTAssertTrue(sut.presets.isEmpty)
    }

    func testDeletePreset_withNonExistingId_shouldSetErrorMessage() {
        // Given
        let nonExistingId = UUID()

        // When
        sut.deletePreset(id: nonExistingId)

        // Then
        XCTAssertNotNil(sut.errorMessage)
    }

    // MARK: - ApplyPreset Tests

    func testApplyPreset_shouldUpdateSettingsViewModel() {
        // Given
        let settings = ScalePresetSettings(
            scaleType: .octaveRepeat,
            startPitchIndex: 24,
            tempo: 90,
            keyProgressionPattern: .descendingOnly,
            ascendingKeyCount: 5,
            descendingKeyCount: 4,
            ascendingKeyStepInterval: 2,
            descendingKeyStepInterval: 3
        )
        let preset = ScalePreset(name: "Apply Me", settings: settings)
        let settingsVM = RecordingSettingsViewModel()

        // When
        sut.applyPreset(preset, to: settingsVM)

        // Then
        XCTAssertEqual(settingsVM.scaleType, .octaveRepeat)
        XCTAssertEqual(settingsVM.startPitchIndex, 24)
        XCTAssertEqual(settingsVM.tempo, 90)
        XCTAssertEqual(settingsVM.keyProgressionPattern, .descendingOnly)
        XCTAssertEqual(settingsVM.ascendingKeyCount, 5)
        XCTAssertEqual(settingsVM.descendingKeyCount, 4)
        XCTAssertEqual(settingsVM.ascendingKeyStepInterval, 2)
        XCTAssertEqual(settingsVM.descendingKeyStepInterval, 3)
    }

    func testApplyPreset_shouldDismissPresetList() {
        // Given
        let preset = ScalePreset(name: "Test", settings: createTestSettings())
        let settingsVM = RecordingSettingsViewModel()
        sut.isShowingPresetList = true

        // When
        sut.applyPreset(preset, to: settingsVM)

        // Then
        XCTAssertFalse(sut.isShowingPresetList)
    }

    func testApplyPreset_withFiveToneType_shouldSetCorrectScaleType() {
        // Given
        let settings = ScalePresetSettings(
            scaleType: .fiveTone,
            startPitchIndex: 12,
            tempo: 120,
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 3,
            descendingKeyCount: 3,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )
        let preset = ScalePreset(name: "Five Tone", settings: settings)
        let settingsVM = RecordingSettingsViewModel()

        // When
        sut.applyPreset(preset, to: settingsVM)

        // Then
        XCTAssertEqual(settingsVM.scaleType, .fiveTone)
    }

    func testApplyPreset_withOffType_shouldSetCorrectScaleType() {
        // Given
        let settings = ScalePresetSettings(
            scaleType: .off,
            startPitchIndex: 12,
            tempo: 120,
            keyProgressionPattern: .ascendingThenDescending,
            ascendingKeyCount: 3,
            descendingKeyCount: 3,
            ascendingKeyStepInterval: 1,
            descendingKeyStepInterval: 1
        )
        let preset = ScalePreset(name: "Off", settings: settings)
        let settingsVM = RecordingSettingsViewModel()

        // When
        sut.applyPreset(preset, to: settingsVM)

        // Then
        XCTAssertEqual(settingsVM.scaleType, .off)
    }

    // MARK: - IsValidPresetName Tests

    func testIsValidPresetName_withEmptyString_shouldReturnFalse() {
        // When/Then
        XCTAssertFalse(sut.isValidPresetName(""))
    }

    func testIsValidPresetName_withWhitespaceOnly_shouldReturnFalse() {
        // When/Then
        XCTAssertFalse(sut.isValidPresetName("   "))
    }

    func testIsValidPresetName_withValidName_shouldReturnTrue() {
        // When/Then
        XCTAssertTrue(sut.isValidPresetName("Valid Name"))
    }

    func testIsValidPresetName_withDuplicateName_shouldReturnFalse() {
        // Given
        let preset = ScalePreset(name: "Existing", settings: createTestSettings())
        mockRepository.presets = [preset]
        sut.loadPresets()

        // When/Then
        XCTAssertFalse(sut.isValidPresetName("Existing"))
    }

    func testIsValidPresetName_withDifferentName_shouldReturnTrue() {
        // Given
        let preset = ScalePreset(name: "Existing", settings: createTestSettings())
        mockRepository.presets = [preset]
        sut.loadPresets()

        // When/Then
        XCTAssertTrue(sut.isValidPresetName("Different"))
    }

    // MARK: - Round-trip Tests (Save → Load → Verify)
    // These tests ensure that settings are preserved through the save/load cycle
    // This is the key test that would have caught the original bug

    func testSaveAndApply_shouldPreserveAllSettings_roundTrip() {
        // Given: A RecordingSettingsViewModel with specific settings
        let originalVM = RecordingSettingsViewModel()
        originalVM.scaleType = .brokenScale
        originalVM.startPitchIndex = 18
        originalVM.tempo = 85
        originalVM.keyProgressionPattern = .descendingThenAscending
        originalVM.ascendingKeyCount = 7
        originalVM.descendingKeyCount = 6
        originalVM.ascendingKeyStepInterval = 3
        originalVM.descendingKeyStepInterval = 2

        // When: Save as preset and apply to a new ViewModel
        sut.savePreset(name: "Round Trip Test", from: originalVM)
        let savedPreset = sut.presets.first!
        let loadedVM = RecordingSettingsViewModel()
        sut.applyPreset(savedPreset, to: loadedVM)

        // Then: All settings should match exactly
        XCTAssertEqual(loadedVM.scaleType, originalVM.scaleType, "ScaleType mismatch after round-trip")
        XCTAssertEqual(loadedVM.startPitchIndex, originalVM.startPitchIndex, "startPitchIndex mismatch")
        XCTAssertEqual(loadedVM.tempo, originalVM.tempo, "tempo mismatch")
        XCTAssertEqual(loadedVM.keyProgressionPattern, originalVM.keyProgressionPattern, "keyProgressionPattern mismatch")
        XCTAssertEqual(loadedVM.ascendingKeyCount, originalVM.ascendingKeyCount, "ascendingKeyCount mismatch")
        XCTAssertEqual(loadedVM.descendingKeyCount, originalVM.descendingKeyCount, "descendingKeyCount mismatch")
        XCTAssertEqual(loadedVM.ascendingKeyStepInterval, originalVM.ascendingKeyStepInterval, "ascendingKeyStepInterval mismatch")
        XCTAssertEqual(loadedVM.descendingKeyStepInterval, originalVM.descendingKeyStepInterval, "descendingKeyStepInterval mismatch")
    }

    func testSaveAndApply_withAllScaleTypes_shouldPreserveScaleType() {
        // This test verifies all ScaleType values can round-trip correctly
        let allScaleTypes: [ScaleType] = [
            .fiveTone,
            .fiveToneDown,
            .octaveRepeat,
            .brokenScale,
            .brokenScaleDouble,
            .rossiniScale,
            .off
        ]

        for scaleType in allScaleTypes {
            // Given
            let originalVM = RecordingSettingsViewModel()
            originalVM.scaleType = scaleType

            // When
            sut.savePreset(name: "Test \(scaleType.rawValue)", from: originalVM)
            let savedPreset = sut.presets.first!
            let loadedVM = RecordingSettingsViewModel()
            sut.applyPreset(savedPreset, to: loadedVM)

            // Then
            XCTAssertEqual(
                loadedVM.scaleType,
                scaleType,
                "ScaleType \(scaleType.rawValue) not preserved in round-trip"
            )

            // Cleanup for next iteration
            sut.deletePreset(id: savedPreset.id)
        }
    }

    func testSingleSourceOfTruth_settingsPropertyShouldMatch() {
        // This test verifies the Single Source of Truth pattern:
        // The 'settings' property should always reflect the current state
        let vm = RecordingSettingsViewModel()
        vm.scaleType = .rossiniScale
        vm.startPitchIndex = 30
        vm.tempo = 100

        // The settings struct should contain the same values
        XCTAssertEqual(vm.settings.scaleType, .rossiniScale)
        XCTAssertEqual(vm.settings.startPitchIndex, 30)
        XCTAssertEqual(vm.settings.tempo, 100)

        // Modifying through settings should update the computed properties
        vm.settings.scaleType = .fiveToneDown
        XCTAssertEqual(vm.scaleType, .fiveToneDown)
    }
}

// MARK: - Mock Repository

private class MockScalePresetRepository: ScalePresetRepositoryProtocol {
    var presets: [ScalePreset] = []
    var shouldThrowOnSave = false
    var shouldThrowOnDelete = false

    func loadAll() -> [ScalePreset] {
        return presets.sorted { $0.updatedAt > $1.updatedAt }
    }

    func save(_ preset: ScalePreset) throws {
        if shouldThrowOnSave {
            throw ScalePresetRepositoryError.saveFailed("Mock error")
        }
        if let index = presets.firstIndex(where: { $0.id == preset.id }) {
            presets[index] = preset
        } else {
            presets.append(preset)
        }
    }

    func delete(id: UUID) throws {
        if shouldThrowOnDelete {
            throw ScalePresetRepositoryError.deleteFailed("Mock error")
        }
        guard let index = presets.firstIndex(where: { $0.id == id }) else {
            throw ScalePresetRepositoryError.notFound
        }
        presets.remove(at: index)
    }

    func find(id: UUID) -> ScalePreset? {
        return presets.first { $0.id == id }
    }
}
