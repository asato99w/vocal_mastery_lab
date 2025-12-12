import Foundation
import VocalisDomain

/// Errors that can occur in RecordingRepository operations
public enum RecordingRepositoryError: Error {
    case notFound
}

/// File-based recording repository using FileManager and UserDefaults
public class FileRecordingRepository: RecordingRepositoryProtocol {

    private let userDefaults: UserDefaults
    private let pitchDataCache: PitchDataCacheProtocol?
    private let metadataKey = "recordings_metadata"

    public init(
        userDefaults: UserDefaults = .standard,
        pitchDataCache: PitchDataCacheProtocol? = nil
    ) {
        self.userDefaults = userDefaults
        self.pitchDataCache = pitchDataCache
    }

    public func save(_ recording: Recording) async throws {
        // Load existing metadata
        var recordings = try await loadMetadata()

        // Add new recording
        recordings.append(recording)

        // Save metadata
        try saveMetadata(recordings)
    }

    public func update(_ recording: Recording) async throws {
        // Load existing metadata
        var recordings = try await loadMetadata()

        // Find and update recording
        guard let index = recordings.firstIndex(where: { $0.id == recording.id }) else {
            throw RecordingRepositoryError.notFound
        }

        recordings[index] = recording

        // Save updated metadata
        try saveMetadata(recordings)
    }

    public func findAll() async throws -> [Recording] {
        var recordings = try await loadMetadata()

        // Filter out recordings whose files no longer exist
        let validRecordings = recordings.filter { recording in
            FileManager.default.fileExists(atPath: recording.fileURL.path)
        }

        // If some recordings were invalid, clean up metadata
        if validRecordings.count != recordings.count {
            try saveMetadata(validRecordings)
            recordings = validRecordings
        }

        // Sort by creation date (newest first)
        return recordings.sorted { $0.createdAt > $1.createdAt }
    }

    public func findById(_ id: RecordingId) async throws -> Recording? {
        let recordings = try await loadMetadata()
        return recordings.first { $0.id == id }
    }

    public func delete(_ id: RecordingId) async throws {
        // Load metadata
        var recordings = try await loadMetadata()

        // Find recording to delete
        guard let index = recordings.firstIndex(where: { $0.id == id }) else {
            return // Already deleted
        }

        let recording = recordings[index]

        // Delete file if it exists
        if FileManager.default.fileExists(atPath: recording.fileURL.path) {
            try FileManager.default.removeItem(at: recording.fileURL)
        }

        // Delete pitch data cache
        pitchDataCache?.delete(id)

        // Remove from metadata
        recordings.remove(at: index)

        // Save updated metadata
        try saveMetadata(recordings)
    }

    // MARK: - Private Methods

    private func loadMetadata() async throws -> [Recording] {
        guard let data = userDefaults.data(forKey: metadataKey) else {
            return []
        }

        do {
            let decoder = JSONDecoder()
            return try decoder.decode([Recording].self, from: data)
        } catch {
            return []
        }
    }

    private func saveMetadata(_ recordings: [Recording]) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(recordings)
        userDefaults.set(data, forKey: metadataKey)
    }
}
