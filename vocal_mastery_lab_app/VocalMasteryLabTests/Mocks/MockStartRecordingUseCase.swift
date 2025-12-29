import Foundation
import VocalisDomain
@testable import VocalMasteryLab

final class MockStartRecordingUseCase: StartRecordingUseCaseProtocol {
    // MARK: - Tracking Properties

    var prepareCalled = false
    var prepareCallCount = 0
    var prepareUser: User?
    var prepareShouldFail = false

    var startCalled = false
    var startCallCount = 0
    var startShouldFail = false

    var executeCalled = false
    var executeCallCount = 0
    var executeUser: User?

    var executeResult: RecordingSession?
    var executeShouldFail = false

    /// URL returned from prepare()
    private var preparedURL: URL?

    // MARK: - Protocol Methods

    func prepare(user: User) async throws -> URL {
        prepareCalled = true
        prepareCallCount += 1
        prepareUser = user

        if prepareShouldFail {
            throw AudioRecorderError.recordingFailed("Mock prepare error")
        }

        guard let result = executeResult else {
            throw AudioRecorderError.recordingFailed("No mock result provided")
        }

        preparedURL = result.recordingURL
        return result.recordingURL
    }

    func start() async throws -> RecordingSession {
        startCalled = true
        startCallCount += 1

        if startShouldFail {
            throw AudioRecorderError.recordingFailed("Mock start error")
        }

        guard let result = executeResult else {
            throw AudioRecorderError.recordingFailed("No mock result provided")
        }

        preparedURL = nil
        return result
    }

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

    // MARK: - Reset

    func reset() {
        prepareCalled = false
        prepareCallCount = 0
        prepareUser = nil
        prepareShouldFail = false

        startCalled = false
        startCallCount = 0
        startShouldFail = false

        executeCalled = false
        executeCallCount = 0
        executeUser = nil

        executeResult = nil
        executeShouldFail = false
        preparedURL = nil
    }
}
