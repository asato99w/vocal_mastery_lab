import Foundation
import Combine
import VocalisDomain

/// ViewModel for algorithm settings (pitch detection algorithm selection)
@MainActor
final class AlgorithmSettingsViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var pitchAlgorithm: PitchDetectionAlgorithm

    // MARK: - Private Properties

    private let repository: AudioSettingsRepositoryProtocol
    private var originalSettings: AudioDetectionSettings

    // MARK: - Computed Properties

    /// Whether the current settings differ from saved settings
    var hasChanges: Bool {
        let settings = repository.get()
        return pitchAlgorithm != settings.pitchAlgorithm
    }

    // MARK: - Initialization

    init(repository: AudioSettingsRepositoryProtocol) {
        self.repository = repository

        // Load current settings from repository
        let settings = repository.get()
        self.originalSettings = settings

        // Initialize published properties
        self.pitchAlgorithm = settings.pitchAlgorithm
    }

    // MARK: - Public Methods

    /// Save current settings to repository
    func saveSettings() throws {
        // Get current full settings and update only algorithm property
        var settings = repository.get()
        settings = AudioDetectionSettings(
            scalePlaybackVolume: settings.scalePlaybackVolume,
            recordingPlaybackVolume: settings.recordingPlaybackVolume,
            rmsSilenceThreshold: settings.rmsSilenceThreshold,
            confidenceThreshold: settings.confidenceThreshold,
            scaleSoundType: settings.scaleSoundType,
            pitchAlgorithm: pitchAlgorithm
        )
        try repository.save(settings)

        // Update original settings after successful save
        originalSettings = settings
    }

    /// Reset algorithm settings to defaults
    func resetSettings() throws {
        // Get default settings
        let defaultSettings = AudioDetectionSettings.default

        // Get current settings and update only algorithm property
        var currentSettings = repository.get()
        currentSettings = AudioDetectionSettings(
            scalePlaybackVolume: currentSettings.scalePlaybackVolume,
            recordingPlaybackVolume: currentSettings.recordingPlaybackVolume,
            rmsSilenceThreshold: currentSettings.rmsSilenceThreshold,
            confidenceThreshold: currentSettings.confidenceThreshold,
            scaleSoundType: currentSettings.scaleSoundType,
            pitchAlgorithm: defaultSettings.pitchAlgorithm
        )
        try repository.save(currentSettings)

        // Update UI
        pitchAlgorithm = defaultSettings.pitchAlgorithm
        originalSettings = currentSettings
    }
}
