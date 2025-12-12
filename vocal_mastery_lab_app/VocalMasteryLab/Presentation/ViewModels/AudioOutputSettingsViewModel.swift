import Foundation
import Combine
import VocalisDomain

/// ViewModel for audio output settings (recording playback volume)
@MainActor
final class AudioOutputSettingsViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var recordingPlaybackVolume: Float

    // MARK: - Private Properties

    private let repository: AudioSettingsRepositoryProtocol
    private var originalSettings: AudioDetectionSettings

    // MARK: - Computed Properties

    /// Whether the current settings differ from saved settings
    var hasChanges: Bool {
        let settings = repository.get()
        return recordingPlaybackVolume != settings.recordingPlaybackVolume
    }

    // MARK: - Initialization

    init(repository: AudioSettingsRepositoryProtocol) {
        self.repository = repository

        // Load current settings from repository
        let settings = repository.get()
        self.originalSettings = settings

        // Initialize published properties
        self.recordingPlaybackVolume = settings.recordingPlaybackVolume
    }

    // MARK: - Public Methods

    /// Save current settings to repository
    func saveSettings() throws {
        // Get current full settings and update only output-related properties
        let currentSettings = repository.get()
        let settings = AudioDetectionSettings(
            recordingPlaybackVolume: recordingPlaybackVolume,
            rmsSilenceThreshold: currentSettings.rmsSilenceThreshold,
            confidenceThreshold: currentSettings.confidenceThreshold,
            pitchAlgorithm: currentSettings.pitchAlgorithm
        )
        try repository.save(settings)

        // Update original settings after successful save
        originalSettings = settings
    }

    /// Reset output settings to defaults
    func resetSettings() throws {
        // Get default settings
        let defaultSettings = AudioDetectionSettings.default

        // Get current settings and update only output-related properties
        let currentSettings = repository.get()
        let settings = AudioDetectionSettings(
            recordingPlaybackVolume: defaultSettings.recordingPlaybackVolume,
            rmsSilenceThreshold: currentSettings.rmsSilenceThreshold,
            confidenceThreshold: currentSettings.confidenceThreshold,
            pitchAlgorithm: currentSettings.pitchAlgorithm
        )
        try repository.save(settings)

        // Update UI
        recordingPlaybackVolume = defaultSettings.recordingPlaybackVolume
        originalSettings = settings
    }
}
