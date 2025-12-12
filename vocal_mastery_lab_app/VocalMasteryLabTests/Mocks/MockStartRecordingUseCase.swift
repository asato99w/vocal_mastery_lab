import Foundation
import VocalisDomain
@testable import VocalMasteryLab

final class MockStartRecordingUseCase: StartRecordingUseCaseProtocol {
    var executeCalled = false
    var executeCallCount = 0
    var executeUser: User?
    var executeResult: RecordingSession?
    var executeShouldFail = false

    func execute(user: User) async throws -> RecordingSession {
        executeCalled = true
        executeCallCount += 1
        executeUser = user

        if executeShouldFail {
            throw AudioRecorderError.recordingFailed("Mock use case error")
        }

        guard let result = executeResult else {
            throw AudioRecorderError.recordingFailed("No mock result provided")
        }

        return result
    }

    func reset() {
        executeCalled = false
        executeCallCount = 0
        executeUser = nil
        executeResult = nil
        executeShouldFail = false
    }
}
