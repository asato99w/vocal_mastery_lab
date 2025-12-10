import Foundation
import VocalisDomain
@testable import VocalMasteringLab

final class MockStartRecordingWithScaleUseCase: StartRecordingWithScaleUseCaseProtocol {
    var executeCalled = false
    var executeCallCount = 0
    var executeUser: User?
    var executeSettings: ScaleSettings?
    var executeResult: RecordingSession?
    var executeShouldFail = false

    func execute(user: User, settings: ScaleSettings) async throws -> RecordingSession {
        executeCalled = true
        executeCallCount += 1
        executeUser = user
        executeSettings = settings

        if executeShouldFail {
            throw ScalePlayerError.playbackFailed("Mock use case error")
        }

        guard let result = executeResult else {
            throw ScalePlayerError.notLoaded
        }

        return result
    }

    func reset() {
        executeCalled = false
        executeCallCount = 0
        executeUser = nil
        executeSettings = nil
        executeResult = nil
        executeShouldFail = false
    }
}
