import Foundation
import VocalisDomain

/// Errors that can occur in ExtractedAudioRepository operations
public enum ExtractedAudioRepositoryError: Error {
    case notFound
}

/// File-based extracted audio repository using FileManager and UserDefaults
public class FileExtractedAudioRepository: ExtractedAudioRepositoryProtocol {

    private let userDefaults: UserDefaults
    private let metadataKey = "extracted_audio_metadata"

    public init(userDefaults: UserDefaults = .standard) {
        self.userDefaults = userDefaults
    }

    public func save(_ extractedAudio: ExtractedAudio) async throws {
        var items = try await loadMetadata()
        items.append(extractedAudio)
        try saveMetadata(items)
    }

    public func findAll() async throws -> [ExtractedAudio] {
        var items = try await loadMetadata()

        // Filter out items whose files no longer exist
        let validItems = items.filter { item in
            FileManager.default.fileExists(atPath: item.fileURL.path)
        }

        // If some items were invalid, clean up metadata
        if validItems.count != items.count {
            try saveMetadata(validItems)
            items = validItems
        }

        // Sort by creation date (newest first)
        return items.sorted { $0.createdAt > $1.createdAt }
    }

    public func findById(_ id: ExtractedAudioId) async throws -> ExtractedAudio? {
        let items = try await loadMetadata()
        return items.first { $0.id == id }
    }

    public func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio] {
        let items = try await loadMetadata()
        return items.filter { $0.sourceRecordingId == recordingId }
    }

    public func delete(_ id: ExtractedAudioId) async throws {
        var items = try await loadMetadata()

        guard let index = items.firstIndex(where: { $0.id == id }) else {
            return // Already deleted
        }

        let item = items[index]

        // Delete file if it exists
        if FileManager.default.fileExists(atPath: item.fileURL.path) {
            try FileManager.default.removeItem(at: item.fileURL)
        }

        items.remove(at: index)
        try saveMetadata(items)
    }

    public func deleteByRecording(_ recordingId: RecordingId) async throws {
        var items = try await loadMetadata()
        let toDelete = items.filter { $0.sourceRecordingId == recordingId }

        for item in toDelete {
            if FileManager.default.fileExists(atPath: item.fileURL.path) {
                try FileManager.default.removeItem(at: item.fileURL)
            }
        }

        items.removeAll { $0.sourceRecordingId == recordingId }
        try saveMetadata(items)
    }

    // MARK: - Private Methods

    private func loadMetadata() async throws -> [ExtractedAudio] {
        guard let data = userDefaults.data(forKey: metadataKey) else {
            return []
        }

        do {
            let decoder = JSONDecoder()
            return try decoder.decode([ExtractedAudio].self, from: data)
        } catch {
            return []
        }
    }

    private func saveMetadata(_ items: [ExtractedAudio]) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(items)
        userDefaults.set(data, forKey: metadataKey)
    }
}
