import Foundation
import VocalisDomain
@testable import VocalMasteryLab

final class MockExtractedAudioRepository: ExtractedAudioRepositoryProtocol {
    var savedAudios: [ExtractedAudio] = []
    var storedAudios: [ExtractedAudio] = []
    var shouldThrowError = false

    func save(_ extractedAudio: ExtractedAudio) async throws {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        savedAudios.append(extractedAudio)
        storedAudios.append(extractedAudio)
    }

    func findAll() async throws -> [ExtractedAudio] {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        return storedAudios
    }

    func findById(_ id: ExtractedAudioId) async throws -> ExtractedAudio? {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        return storedAudios.first { $0.id == id }
    }

    func findByRecording(_ recordingId: RecordingId) async throws -> [ExtractedAudio] {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        return storedAudios.filter { $0.sourceRecordingId == recordingId }
    }

    func delete(_ id: ExtractedAudioId) async throws {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        storedAudios.removeAll { $0.id == id }
    }

    func deleteByRecording(_ recordingId: RecordingId) async throws {
        if shouldThrowError {
            throw NSError(domain: "MockError", code: 1)
        }
        storedAudios.removeAll { $0.sourceRecordingId == recordingId }
    }
}
