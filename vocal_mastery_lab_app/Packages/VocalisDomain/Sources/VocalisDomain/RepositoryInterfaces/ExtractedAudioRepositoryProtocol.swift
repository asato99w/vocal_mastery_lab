import Foundation

/// Repository interface for ExtractedAudio aggregate
public protocol ExtractedAudioRepositoryProtocol {
    func save(_ extractedAudio: ExtractedAudio) async throws
    func findAll() async throws -> [ExtractedAudio]
    func findById(_ id: ExtractedAudioId) async throws -> ExtractedAudio?
    func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio]
    func delete(_ id: ExtractedAudioId) async throws
    func deleteByRecording(_ recordingId: RecordingId) async throws
}
