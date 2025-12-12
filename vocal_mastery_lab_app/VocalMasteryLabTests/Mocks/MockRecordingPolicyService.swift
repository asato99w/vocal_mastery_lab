import Foundation
import VocalisDomain
import SubscriptionDomain

class MockRecordingPolicyService: RecordingPolicyService {
    var canStartRecordingResult: RecordingPermission = .allowed
    var canStartRecordingCalled = false
    var lastUser: User?

    func canStartRecording(user: User) async throws -> RecordingPermission {
        canStartRecordingCalled = true
        lastUser = user
        return canStartRecordingResult
    }

    var validateDurationShouldThrow: RecordingPolicyError?
    var validateDurationCalled = false
    var lastDuration: Duration?
    var lastStatus: SubscriptionStatus?

    func validateDuration(_ duration: Duration, for status: SubscriptionStatus) throws {
        validateDurationCalled = true
        lastDuration = duration
        lastStatus = status

        if let error = validateDurationShouldThrow {
            throw error
        }
    }

    func reset() {
        canStartRecordingResult = .allowed
        canStartRecordingCalled = false
        lastUser = nil
        validateDurationShouldThrow = nil
        validateDurationCalled = false
        lastDuration = nil
        lastStatus = nil
    }
}
