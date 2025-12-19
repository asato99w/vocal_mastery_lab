import Foundation

/// Repository interface for audio file management
public protocol AudioFileRepositoryProtocol {
    func saveAudioFile(from sourceURL: URL, recordingId: RecordingId) async throws -> URL
    func deleteAudioFile(at url: URL) async throws
    func audioFileExists(at url: URL) -> Bool
}
