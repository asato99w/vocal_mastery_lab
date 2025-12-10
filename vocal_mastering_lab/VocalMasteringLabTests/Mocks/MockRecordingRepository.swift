import Foundation
import VocalisDomain
@testable import VocalMasteringLab

final class MockRecordingRepository: RecordingRepositoryProtocol {
    var saveCalled = false
    var updateCalled = false
    var findAllCalled = false
    var findByIdCalled = false
    var deleteCalled = false

    var savedRecordings: [Recording] = []
    var recordingsToReturn: [Recording] = []
    var saveShouldFail = false
    var updateShouldFail = false
    var findAllShouldFail = false

    func save(_ recording: Recording) async throws {
        saveCalled = true
        if saveShouldFail {
            throw NSError(domain: "MockError", code: 1)
        }
        savedRecordings.append(recording)
    }

    func update(_ recording: Recording) async throws {
        updateCalled = true
        if updateShouldFail {
            throw NSError(domain: "MockError", code: 1)
        }
        if let index = savedRecordings.firstIndex(where: { $0.id == recording.id }) {
            savedRecordings[index] = recording
        } else {
            throw RecordingRepositoryError.notFound
        }
    }

    func findAll() async throws -> [Recording] {
        findAllCalled = true
        if findAllShouldFail {
            throw NSError(domain: "MockError", code: 1)
        }
        return recordingsToReturn
    }

    func findById(_ id: RecordingId) async throws -> Recording? {
        findByIdCalled = true
        return savedRecordings.first { $0.id == id }
    }

    func delete(_ id: RecordingId) async throws {
        deleteCalled = true
        savedRecordings.removeAll { $0.id == id }
    }

    func reset() {
        saveCalled = false
        updateCalled = false
        findAllCalled = false
        findByIdCalled = false
        deleteCalled = false
        savedRecordings = []
        recordingsToReturn = []
        saveShouldFail = false
        updateShouldFail = false
        findAllShouldFail = false
    }
}
