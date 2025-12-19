import Foundation

/// Repository interface for Recording aggregate
public protocol RecordingRepositoryProtocol {
    func save(_ recording: Recording) async throws
    func update(_ recording: Recording) async throws
    func findAll() async throws -> [Recording]
    func findById(_ id: RecordingId) async throws -> Recording?
    func delete(_ id: RecordingId) async throws
}
