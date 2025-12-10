import Foundation
import VocalisDomain

/// Use case for loading scale presets
public class LoadScalePresetsUseCase {
    private let repository: ScalePresetRepositoryProtocol

    public init(repository: ScalePresetRepositoryProtocol) {
        self.repository = repository
    }

    /// Load all saved presets
    public func execute() -> [ScalePreset] {
        return repository.loadAll()
    }

    /// Find a specific preset by ID
    public func find(id: UUID) -> ScalePreset? {
        return repository.find(id: id)
    }
}
