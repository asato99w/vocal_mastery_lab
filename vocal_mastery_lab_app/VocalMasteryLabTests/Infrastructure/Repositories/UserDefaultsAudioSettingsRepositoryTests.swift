import XCTest
@testable import VocalisDomain
@testable import VocalMasteryLab

/// Tests for UserDefaultsAudioSettingsRepository
final class UserDefaultsAudioSettingsRepositoryTests: XCTestCase {

    var sut: UserDefaultsAudioSettingsRepository!
    var testUserDefaults: UserDefaults!
    var testSuiteName: String!

    override func setUp() {
        super.setUp()
        // Use a test suite name to isolate test data
        testSuiteName = "test.audio.settings.\(UUID().uuidString)"
        testUserDefaults = UserDefaults(suiteName: testSuiteName)!
        sut = UserDefaultsAudioSettingsRepository(userDefaults: testUserDefaults)
    }

    override func tearDown() {
        // Clean up test data
        testUserDefaults.removePersistentDomain(forName: testSuiteName)
        testUserDefaults = nil
        sut = nil
        super.tearDown()
    }

    // MARK: - Get Tests

    func testGet_whenNoSavedSettings_shouldReturnDefaultSettings() {
        // When
        let settings = sut.get()

        // Then
        #if targetEnvironment(simulator)
        XCTAssertEqual(settings, AudioDetectionSettings.simulator, "Simulator should return simulator defaults")
        #else
        XCTAssertEqual(settings, AudioDetectionSettings.default, "Real device should return default settings")
        #endif
    }

    func testGet_whenSettingsSaved_shouldReturnSavedSettings() throws {
        // Given
        let customSettings = AudioDetectionSettings(
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.015,
            confidenceThreshold: 0.35
        )
        try sut.save(customSettings)

        // When
        let retrieved = sut.get()

        // Then
        XCTAssertEqual(retrieved, customSettings)
    }

    // MARK: - Save Tests

    func testSave_shouldPersistSettings() throws {
        // Given
        let settings = AudioDetectionSettings(
            recordingPlaybackVolume: 0.6,
            rmsSilenceThreshold: 0.01,
            confidenceThreshold: 0.5
        )

        // When
        try sut.save(settings)

        // Then: Create new repository instance to verify persistence
        let newRepository = UserDefaultsAudioSettingsRepository(userDefaults: testUserDefaults)
        let retrieved = newRepository.get()
        XCTAssertEqual(retrieved, settings)
    }

    func testSave_multipleTimes_shouldOverwritePreviousSettings() throws {
        // Given
        let settings1 = AudioDetectionSettings(
            recordingPlaybackVolume: 0.5,
            rmsSilenceThreshold: 0.01,
            confidenceThreshold: 0.3
        )
        let settings2 = AudioDetectionSettings(
            recordingPlaybackVolume: 0.9,
            rmsSilenceThreshold: 0.03,
            confidenceThreshold: 0.6
        )

        // When
        try sut.save(settings1)
        try sut.save(settings2)

        // Then
        let retrieved = sut.get()
        XCTAssertEqual(retrieved, settings2, "Should retrieve the most recent settings")
        XCTAssertNotEqual(retrieved, settings1, "Should not retrieve the old settings")
    }

    // MARK: - Reset Tests

    func testReset_shouldRemoveSavedSettings() throws {
        // Given: Save custom settings
        let customSettings = AudioDetectionSettings(
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.015,
            confidenceThreshold: 0.35
        )
        try sut.save(customSettings)

        // When: Reset to defaults
        try sut.reset()

        // Then: Should return default settings
        let retrieved = sut.get()
        #if targetEnvironment(simulator)
        XCTAssertEqual(retrieved, AudioDetectionSettings.simulator)
        #else
        XCTAssertEqual(retrieved, AudioDetectionSettings.default)
        #endif
    }

    func testReset_whenNoSettingsSaved_shouldNotThrowError() throws {
        // When: Reset when no settings exist
        // Then: Should not throw error
        XCTAssertNoThrow(try sut.reset())
    }

    func testReset_shouldPersistDefaultsAfterReset() throws {
        // Given
        let customSettings = AudioDetectionSettings(
            recordingPlaybackVolume: 0.7,
            rmsSilenceThreshold: 0.015,
            confidenceThreshold: 0.35
        )
        try sut.save(customSettings)
        try sut.reset()

        // When: Create new repository instance after reset
        let newRepository = UserDefaultsAudioSettingsRepository(userDefaults: testUserDefaults)
        let retrieved = newRepository.get()

        // Then: Should still return defaults
        #if targetEnvironment(simulator)
        XCTAssertEqual(retrieved, AudioDetectionSettings.simulator)
        #else
        XCTAssertEqual(retrieved, AudioDetectionSettings.default)
        #endif
    }

    // MARK: - Edge Case Tests

    func testSave_withClampedValues_shouldPersistClampedValues() throws {
        // Given: Settings with out-of-range values (will be clamped)
        let settings = AudioDetectionSettings(
            recordingPlaybackVolume: 1.5,  // Will be clamped to 1.0
            rmsSilenceThreshold: -0.1,  // Will be clamped to 0.001
            confidenceThreshold: 2.0  // Will be clamped to 1.0
        )

        // When
        try sut.save(settings)
        let retrieved = sut.get()

        // Then: Should retrieve clamped values
        XCTAssertEqual(retrieved.recordingPlaybackVolume, 1.0)
        XCTAssertEqual(retrieved.rmsSilenceThreshold, 0.001)
        XCTAssertEqual(retrieved.confidenceThreshold, 1.0)
    }
}
