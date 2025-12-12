import Foundation
import VocalisDomain

/// UserDefaults-based repository implementation for scale presets
final class UserDefaultsScalePresetRepository: ScalePresetRepositoryProtocol {

    private let userDefaults: UserDefaults
    private let presetsKey = "scalePresets"

    init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    func loadAll() -> [ScalePreset] {
        guard let data = userDefaults.data(forKey: presetsKey) else {
            return []
        }

        do {
            let presets = try JSONDecoder().decode([ScalePreset].self, from: data)
            return presets.sorted { $0.updatedAt > $1.updatedAt }
        } catch {
            return []
        }
    }

    func save(_ preset: ScalePreset) throws {
        var presets = loadAll()

        // Check if preset already exists
        if let index = presets.firstIndex(where: { $0.id == preset.id }) {
            // Update existing preset
            presets[index] = preset
        } else {
            // Add new preset
            presets.append(preset)
        }

        try saveAll(presets)
    }

    func delete(id: UUID) throws {
        var presets = loadAll()

        guard let index = presets.firstIndex(where: { $0.id == id }) else {
            throw ScalePresetRepositoryError.notFound
        }

        presets.remove(at: index)
        try saveAll(presets)
    }

    func find(id: UUID) -> ScalePreset? {
        let presets = loadAll()
        return presets.first { $0.id == id }
    }

    // MARK: - Private

    private func saveAll(_ presets: [ScalePreset]) throws {
        do {
            let data = try JSONEncoder().encode(presets)
            userDefaults.set(data, forKey: presetsKey)
        } catch {
            throw ScalePresetRepositoryError.saveFailed(error.localizedDescription)
        }
    }
}
