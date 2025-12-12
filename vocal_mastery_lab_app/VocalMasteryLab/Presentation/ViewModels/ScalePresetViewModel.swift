import Foundation
import VocalisDomain

/// ViewModel for managing scale presets
public class ScalePresetViewModel: ObservableObject {
    @Published public private(set) var presets: [ScalePreset] = []
    @Published public var isShowingSaveDialog = false
    @Published public var isShowingPresetList = false
    @Published public var newPresetName = ""
    @Published public var errorMessage: String?

    private let saveUseCase: SaveScalePresetUseCase
    private let loadUseCase: LoadScalePresetsUseCase
    private let deleteUseCase: DeleteScalePresetUseCase

    public init(
        saveUseCase: SaveScalePresetUseCase,
        loadUseCase: LoadScalePresetsUseCase,
        deleteUseCase: DeleteScalePresetUseCase
    ) {
        self.saveUseCase = saveUseCase
        self.loadUseCase = loadUseCase
        self.deleteUseCase = deleteUseCase
        loadPresets()
    }

    /// Load all presets from storage
    public func loadPresets() {
        presets = loadUseCase.execute()
    }

    /// Save current settings as a new preset
    /// Uses Single Source of Truth: settings property from RecordingSettingsViewModel
    public func savePreset(name: String, from settingsViewModel: RecordingSettingsViewModel) {
        do {
            // Direct use of settings - no manual copying needed
            let preset = try saveUseCase.execute(name: name, settings: settingsViewModel.settings)
            presets.insert(preset, at: 0) // Add to beginning (most recent)
            newPresetName = ""
            isShowingSaveDialog = false
            errorMessage = nil
        } catch {
            errorMessage = "preset.save_error".localized
        }
    }

    /// Delete a preset by ID
    public func deletePreset(id: UUID) {
        do {
            try deleteUseCase.execute(id: id)
            presets.removeAll { $0.id == id }
            errorMessage = nil
        } catch {
            errorMessage = "preset.delete_error".localized
        }
    }

    /// Apply a preset to the settings view model
    /// Uses Single Source of Truth: directly assigns settings struct
    public func applyPreset(_ preset: ScalePreset, to settingsViewModel: RecordingSettingsViewModel) {
        // Direct assignment - no manual copying needed
        // ScalePresetSettings is the Single Source of Truth
        settingsViewModel.settings = preset.settings
        isShowingPresetList = false
    }

    /// Check if a preset name is valid (not empty and not duplicate)
    public func isValidPresetName(_ name: String) -> Bool {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        return !presets.contains { $0.name == trimmed }
    }
}
