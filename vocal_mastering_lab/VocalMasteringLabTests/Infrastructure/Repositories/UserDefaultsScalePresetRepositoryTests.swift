import XCTest
@testable import VocalisDomain
@testable import VocalMasteringLab

/// Tests for UserDefaultsScalePresetRepository
final class UserDefaultsScalePresetRepositoryTests: XCTestCase {

    var sut: UserDefaultsScalePresetRepository!
    var testUserDefaults: UserDefaults!
    var testSuiteName: String!

    override func setUp() {
        super.setUp()
        testSuiteName = "test.scale.presets.\(UUID().uuidString)"
        testUserDefaults = UserDefaults(suiteName: testSuiteName)!
        sut = UserDefaultsScalePresetRepository(userDefaults: testUserDefaults)
    }

    override func tearDown() {
        testUserDefaults.removePersistentDomain(forName: testSuiteName)
        testUserDefaults = nil
        sut = nil
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

    // MARK: - LoadAll Tests

    func testLoadAll_whenEmpty_shouldReturnEmptyArray() {
        // When
        let presets = sut.loadAll()

        // Then
        XCTAssertTrue(presets.isEmpty)
    }

    func testLoadAll_withSavedPresets_shouldReturnAllPresets() throws {
        // Given
        let preset1 = ScalePreset(name: "Preset 1", settings: createTestSettings())
        let preset2 = ScalePreset(name: "Preset 2", settings: createTestSettings())
        try sut.save(preset1)
        try sut.save(preset2)

        // When
        let presets = sut.loadAll()

        // Then
        XCTAssertEqual(presets.count, 2)
    }

    func testLoadAll_shouldReturnPresetsSortedByUpdatedAtDescending() throws {
        // Given
        let preset1 = ScalePreset(
            name: "Old Preset",
            settings: createTestSettings(),
            updatedAt: Date().addingTimeInterval(-3600)
        )
        let preset2 = ScalePreset(
            name: "New Preset",
            settings: createTestSettings(),
            updatedAt: Date()
        )
        try sut.save(preset1)
        try sut.save(preset2)

        // When
        let presets = sut.loadAll()

        // Then
        XCTAssertEqual(presets.first?.name, "New Preset")
        XCTAssertEqual(presets.last?.name, "Old Preset")
    }

    // MARK: - Save Tests

    func testSave_shouldPersistPreset() throws {
        // Given
        let preset = ScalePreset(name: "Test Preset", settings: createTestSettings())

        // When
        try sut.save(preset)

        // Then
        let presets = sut.loadAll()
        XCTAssertEqual(presets.count, 1)
        XCTAssertEqual(presets.first?.name, "Test Preset")
        XCTAssertEqual(presets.first?.id, preset.id)
    }

    func testSave_withExistingId_shouldUpdatePreset() throws {
        // Given
        let preset = ScalePreset(name: "Original Name", settings: createTestSettings())
        try sut.save(preset)

        // When
        let updatedPreset = preset.withUpdatedName("Updated Name")
        try sut.save(updatedPreset)

        // Then
        let presets = sut.loadAll()
        XCTAssertEqual(presets.count, 1)
        XCTAssertEqual(presets.first?.name, "Updated Name")
        XCTAssertEqual(presets.first?.id, preset.id)
    }

    func testSave_multiplePresets_shouldPersistAll() throws {
        // Given
        let preset1 = ScalePreset(name: "Preset 1", settings: createTestSettings())
        let preset2 = ScalePreset(name: "Preset 2", settings: createTestSettings())
        let preset3 = ScalePreset(name: "Preset 3", settings: createTestSettings())

        // When
        try sut.save(preset1)
        try sut.save(preset2)
        try sut.save(preset3)

        // Then
        let presets = sut.loadAll()
        XCTAssertEqual(presets.count, 3)
    }

    // MARK: - Delete Tests

    func testDelete_withExistingId_shouldRemovePreset() throws {
        // Given
        let preset = ScalePreset(name: "To Delete", settings: createTestSettings())
        try sut.save(preset)

        // When
        try sut.delete(id: preset.id)

        // Then
        let presets = sut.loadAll()
        XCTAssertTrue(presets.isEmpty)
    }

    func testDelete_withNonExistingId_shouldThrowNotFoundError() {
        // Given
        let nonExistingId = UUID()

        // When/Then
        XCTAssertThrowsError(try sut.delete(id: nonExistingId)) { error in
            XCTAssertEqual(error as? ScalePresetRepositoryError, .notFound)
        }
    }

    func testDelete_shouldOnlyRemoveSpecifiedPreset() throws {
        // Given
        let preset1 = ScalePreset(name: "Keep", settings: createTestSettings())
        let preset2 = ScalePreset(name: "Delete", settings: createTestSettings())
        try sut.save(preset1)
        try sut.save(preset2)

        // When
        try sut.delete(id: preset2.id)

        // Then
        let presets = sut.loadAll()
        XCTAssertEqual(presets.count, 1)
        XCTAssertEqual(presets.first?.name, "Keep")
    }

    // MARK: - Find Tests

    func testFind_withExistingId_shouldReturnPreset() throws {
        // Given
        let preset = ScalePreset(name: "Find Me", settings: createTestSettings())
        try sut.save(preset)

        // When
        let found = sut.find(id: preset.id)

        // Then
        XCTAssertNotNil(found)
        XCTAssertEqual(found?.name, "Find Me")
        XCTAssertEqual(found?.id, preset.id)
    }

    func testFind_withNonExistingId_shouldReturnNil() {
        // Given
        let nonExistingId = UUID()

        // When
        let found = sut.find(id: nonExistingId)

        // Then
        XCTAssertNil(found)
    }

    // MARK: - Persistence Tests

    func testPersistence_shouldSurviveNewRepositoryInstance() throws {
        // Given
        let preset = ScalePreset(name: "Persistent", settings: createTestSettings())
        try sut.save(preset)

        // When
        let newRepository = UserDefaultsScalePresetRepository(userDefaults: testUserDefaults)
        let presets = newRepository.loadAll()

        // Then
        XCTAssertEqual(presets.count, 1)
        XCTAssertEqual(presets.first?.name, "Persistent")
    }

    // MARK: - Settings Preservation Tests

    func testSave_shouldPreserveAllSettings() throws {
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
        let preset = ScalePreset(name: "Full Settings", settings: settings)

        // When
        try sut.save(preset)
        let found = sut.find(id: preset.id)

        // Then
        XCTAssertNotNil(found)
        XCTAssertEqual(found?.settings.scaleType, .octaveRepeat)
        XCTAssertEqual(found?.settings.startPitchIndex, 24)
        XCTAssertEqual(found?.settings.tempo, 90)
        XCTAssertEqual(found?.settings.keyProgressionPattern, .descendingOnly)
        XCTAssertEqual(found?.settings.ascendingKeyCount, 5)
        XCTAssertEqual(found?.settings.descendingKeyCount, 4)
        XCTAssertEqual(found?.settings.ascendingKeyStepInterval, 2)
        XCTAssertEqual(found?.settings.descendingKeyStepInterval, 3)
    }

    func testSave_shouldPreserveAllScaleTypes() throws {
        // This test ensures all ScaleType enum values can be saved and loaded correctly
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
            let settings = ScalePresetSettings(scaleType: scaleType)
            let preset = ScalePreset(name: "Test \(scaleType.rawValue)", settings: settings)

            // When
            try sut.save(preset)
            let found = sut.find(id: preset.id)

            // Then
            XCTAssertEqual(
                found?.settings.scaleType,
                scaleType,
                "ScaleType \(scaleType.rawValue) not preserved after save/load"
            )

            // Cleanup
            try sut.delete(id: preset.id)
        }
    }
}
