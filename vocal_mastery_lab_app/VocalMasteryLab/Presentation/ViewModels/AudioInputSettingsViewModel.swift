import Foundation
import Combine
import VocalisDomain

/// ViewModel for audio input settings (detection sensitivity, confidence threshold, and pitch algorithm)
@MainActor
final class AudioInputSettingsViewModel: ObservableObject {

    // MARK: - Published Properties

    @Published var detectionSensitivity: AudioDetectionSettings.DetectionSensitivity
    @Published var confidenceThreshold: Float
    @Published var pitchAlgorithm: PitchDetectionAlgorithm

    // MARK: - Private Properties

    private let repository: AudioSettingsRepositoryProtocol
    private var originalSettings: AudioDetectionSettings

    // MARK: - Computed Properties

    /// Whether the current settings differ from saved settings
    var hasChanges: Bool {
        let settings = repository.get()
        return detectionSensitivity != settings.sensitivity ||
               confidenceThreshold != settings.confidenceThreshold ||
               pitchAlgorithm != settings.pitchAlgorithm
    }

    // MARK: - Initialization

    init(repository: AudioSettingsRepositoryProtocol) {
        self.repository = repository

        // Load current settings from repository
        let settings = repository.get()
        self.originalSettings = settings

        // Initialize published properties (input-related only)
        self.detectionSensitivity = settings.sensitivity
        self.confidenceThreshold = settings.confidenceThreshold
        self.pitchAlgorithm = settings.pitchAlgorithm
    }

    // MARK: - Public Methods

    /// Save current settings to repository
    func saveSettings() throws {
        // Get current full settings and update only input-related properties
        let settings = repository.get()
        let newSettings = AudioDetectionSettings(
            recordingPlaybackVolume: settings.recordingPlaybackVolume,
            rmsSilenceThreshold: detectionSensitivity.rmsThreshold,
            confidenceThreshold: confidenceThreshold,
            pitchAlgorithm: pitchAlgorithm
        )
        try repository.save(newSettings)

        // Update original settings after successful save
        originalSettings = newSettings
    }

    /// Reset input settings to defaults
    func resetSettings() throws {
        // Get default settings
        let defaultSettings = AudioDetectionSettings.default

        // Get current settings and update only input-related properties
        let currentSettings = repository.get()
        let newSettings = AudioDetectionSettings(
            recordingPlaybackVolume: currentSettings.recordingPlaybackVolume,
            rmsSilenceThreshold: defaultSettings.rmsSilenceThreshold,
            confidenceThreshold: defaultSettings.confidenceThreshold,
            pitchAlgorithm: defaultSettings.pitchAlgorithm
        )
        try repository.save(newSettings)

        // Update UI
        detectionSensitivity = defaultSettings.sensitivity
        confidenceThreshold = defaultSettings.confidenceThreshold
        pitchAlgorithm = defaultSettings.pitchAlgorithm
        originalSettings = newSettings
    }
}
