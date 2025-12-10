import Foundation
import VocalisDomain

/// Use case for deleting a scale preset
public class DeleteScalePresetUseCase {
    private let repository: ScalePresetRepositoryProtocol

    public init(repository: ScalePresetRepositoryProtocol) {
        self.repository = repository
    }

    /// Delete a preset by ID
    public func execute(id: UUID) throws {
        try repository.delete(id: id)
    }
}
