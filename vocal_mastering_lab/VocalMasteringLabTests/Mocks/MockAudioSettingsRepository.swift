import Foundation
@testable import VocalisDomain
@testable import VocalMasteringLab

/// Mock implementation of AudioSettingsRepositoryProtocol for testing
class MockAudioSettingsRepository: AudioSettingsRepositoryProtocol {

    enum TestError: Error {
        case saveFailed
        case resetFailed
    }

    var settingsToReturn = AudioDetectionSettings.default
    var getCalled = false
    var saveCalled = false
    var resetCalled = false
    var savedSettings: AudioDetectionSettings?
    var shouldThrowOnSave = false
    var shouldThrowOnReset = false

    func get() -> AudioDetectionSettings {
        getCalled = true
        return settingsToReturn
    }

    func save(_ settings: AudioDetectionSettings) throws {
        saveCalled = true
        savedSettings = settings
        if shouldThrowOnSave {
            throw TestError.saveFailed
        }
        // Update return value for next get()
        settingsToReturn = settings
    }

    func reset() throws {
        resetCalled = true
        if shouldThrowOnReset {
            throw TestError.resetFailed
        }
        // Reset to defaults
        #if targetEnvironment(simulator)
        settingsToReturn = AudioDetectionSettings.simulator
        #else
        settingsToReturn = AudioDetectionSettings.default
        #endif
    }

    func resetForTest() {
        getCalled = false
        saveCalled = false
        resetCalled = false
        savedSettings = nil
        shouldThrowOnSave = false
        shouldThrowOnReset = false
        settingsToReturn = AudioDetectionSettings.default
    }
}
