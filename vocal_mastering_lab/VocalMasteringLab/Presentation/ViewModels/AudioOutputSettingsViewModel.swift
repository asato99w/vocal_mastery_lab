import Foundation
import Combine
import VocalisDomain

/// ViewModel for audio output settings (volumes and scale sound type)
@MainActor
final class AudioOutputSettingsViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var scalePlaybackVolume: Float
    @Published var recordingPlaybackVolume: Float
    @Published var scaleSoundType: ScaleSoundType

    // MARK: - Private Properties

    private let repository: AudioSettingsRepositoryProtocol
    private var originalSettings: AudioDetectionSettings

    // MARK: - Computed Properties

    /// Whether the current settings differ from saved settings
    var hasChanges: Bool {
        let settings = repository.get()
        return scalePlaybackVolume != settings.scalePlaybackVolume ||
               recordingPlaybackVolume != settings.recordingPlaybackVolume ||
               scaleSoundType != settings.scaleSoundType
    }

    // MARK: - Initialization

    init(repository: AudioSettingsRepositoryProtocol) {
        self.repository = repository

        // Load current settings from repository
        let settings = repository.get()
        self.originalSettings = settings

        // Initialize published properties (output-related only)
        self.scalePlaybackVolume = settings.scalePlaybackVolume
        self.recordingPlaybackVolume = settings.recordingPlaybackVolume
        self.scaleSoundType = settings.scaleSoundType
    }

    // MARK: - Public Methods

    /// Save current settings to repository
    func saveSettings() throws {
        // Get current full settings and update only output-related properties
        var settings = repository.get()
        settings = AudioDetectionSettings(
            scalePlaybackVolume: scalePlaybackVolume,
            recordingPlaybackVolume: recordingPlaybackVolume,
            rmsSilenceThreshold: settings.rmsSilenceThreshold,
            confidenceThreshold: settings.confidenceThreshold,
            scaleSoundType: scaleSoundType
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
        var currentSettings = repository.get()
        currentSettings = AudioDetectionSettings(
            scalePlaybackVolume: defaultSettings.scalePlaybackVolume,
            recordingPlaybackVolume: defaultSettings.recordingPlaybackVolume,
            rmsSilenceThreshold: currentSettings.rmsSilenceThreshold,
            confidenceThreshold: currentSettings.confidenceThreshold,
            scaleSoundType: defaultSettings.scaleSoundType
        )
        try repository.save(currentSettings)

        // Update UI
        scalePlaybackVolume = defaultSettings.scalePlaybackVolume
        recordingPlaybackVolume = defaultSettings.recordingPlaybackVolume
        scaleSoundType = defaultSettings.scaleSoundType
        originalSettings = currentSettings
    }
}
