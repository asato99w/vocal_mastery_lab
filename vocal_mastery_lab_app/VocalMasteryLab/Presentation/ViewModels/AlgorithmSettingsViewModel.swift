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
        let settings = repository.get()
        let newSettings = AudioDetectionSettings(
            recordingPlaybackVolume: settings.recordingPlaybackVolume,
            rmsSilenceThreshold: settings.rmsSilenceThreshold,
            confidenceThreshold: settings.confidenceThreshold,
            pitchAlgorithm: pitchAlgorithm
        )
        try repository.save(newSettings)

        // Update original settings after successful save
        originalSettings = newSettings
    }

    /// Reset algorithm settings to defaults
    func resetSettings() throws {
        // Get default settings
        let defaultSettings = AudioDetectionSettings.default

        // Get current settings and update only algorithm property
        let currentSettings = repository.get()
        let newSettings = AudioDetectionSettings(
            recordingPlaybackVolume: currentSettings.recordingPlaybackVolume,
            rmsSilenceThreshold: currentSettings.rmsSilenceThreshold,
            confidenceThreshold: currentSettings.confidenceThreshold,
            pitchAlgorithm: defaultSettings.pitchAlgorithm
        )
        try repository.save(newSettings)

        // Update UI
        pitchAlgorithm = defaultSettings.pitchAlgorithm
        originalSettings = newSettings
    }
}
