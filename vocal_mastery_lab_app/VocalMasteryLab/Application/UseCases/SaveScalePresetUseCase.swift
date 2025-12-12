import Foundation
import VocalisDomain

/// Use case for saving a scale preset
public class SaveScalePresetUseCase {
    private let repository: ScalePresetRepositoryProtocol

    public init(repository: ScalePresetRepositoryProtocol) {
        self.repository = repository
    }

    /// Save a new preset with the given name and settings
    public func execute(name: String, settings: ScalePresetSettings) throws -> ScalePreset {
        let preset = ScalePreset(
            name: name,
            settings: settings
        )
        try repository.save(preset)
        return preset
    }

    /// Update an existing preset
    public func update(_ preset: ScalePreset) throws {
        try repository.save(preset)
    }
}
