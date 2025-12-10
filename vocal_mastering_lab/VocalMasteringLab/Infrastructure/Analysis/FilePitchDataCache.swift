import Foundation
import VocalisDomain

/// File-based persistent cache for pitch analysis data
/// Stores PitchAnalysisData as JSON files in the app's cache directory
public class FilePitchDataCache: PitchDataCacheProtocol {
    private let cacheDirectory: URL
    private let fileManager: FileManager
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    /// Initialize with a specific cache directory
    /// - Parameter cacheDirectory: Directory to store cache files. If nil, uses Documents/PitchCache
    public init(cacheDirectory: URL? = nil) {
        self.fileManager = FileManager.default

        if let directory = cacheDirectory {
            self.cacheDirectory = directory
        } else {
            // Default to Documents/PitchCache
            let documentsDirectory = fileManager.urls(
                for: .documentDirectory,
                in: .userDomainMask
            ).first!
            self.cacheDirectory = documentsDirectory.appendingPathComponent("PitchCache")
        }

        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()

        // Create cache directory if it doesn't exist
        createCacheDirectoryIfNeeded()
    }

    // MARK: - Public Methods

    /// Retrieve cached pitch data for a recording
    /// - Parameter id: Recording identifier
    /// - Returns: Cached PitchAnalysisData if exists, nil otherwise
    public func get(_ id: RecordingId) -> PitchAnalysisData? {
        let fileURL = cacheFileURL(for: id)

        guard fileManager.fileExists(atPath: fileURL.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: fileURL)
            return try decoder.decode(PitchAnalysisData.self, from: data)
        } catch {
            // If decoding fails, remove corrupted file
            try? fileManager.removeItem(at: fileURL)
            return nil
        }
    }

    /// Store pitch data for a recording
    /// - Parameters:
    ///   - id: Recording identifier
    ///   - pitchData: Pitch analysis data to cache
    public func set(_ id: RecordingId, pitchData: PitchAnalysisData) {
        createCacheDirectoryIfNeeded()

        let fileURL = cacheFileURL(for: id)

        do {
            let data = try encoder.encode(pitchData)
            try data.write(to: fileURL, options: .atomicWrite)
        } catch {
            // Silently fail - cache is optional performance optimization
        }
    }

    /// Delete cached data for a specific recording
    /// - Parameter id: Recording identifier
    public func delete(_ id: RecordingId) {
        let fileURL = cacheFileURL(for: id)
        try? fileManager.removeItem(at: fileURL)
    }

    /// Clear all cached pitch data
    public func clearAll() {
        try? fileManager.removeItem(at: cacheDirectory)
        createCacheDirectoryIfNeeded()
    }

    /// Check if cached data exists for a recording
    /// - Parameter id: Recording identifier
    /// - Returns: true if cache exists, false otherwise
    public func exists(_ id: RecordingId) -> Bool {
        let fileURL = cacheFileURL(for: id)
        return fileManager.fileExists(atPath: fileURL.path)
    }

    // MARK: - Private Methods

    private func cacheFileURL(for id: RecordingId) -> URL {
        return cacheDirectory.appendingPathComponent("\(id.value.uuidString).json")
    }

    private func createCacheDirectoryIfNeeded() {
        if !fileManager.fileExists(atPath: cacheDirectory.path) {
            try? fileManager.createDirectory(
                at: cacheDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }
}
