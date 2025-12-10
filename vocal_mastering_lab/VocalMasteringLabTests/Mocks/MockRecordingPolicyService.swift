import Foundation
import VocalisDomain
import SubscriptionDomain

class MockRecordingPolicyService: RecordingPolicyService {
    var canStartRecordingResult: RecordingPermission = .allowed
    var canStartRecordingCalled = false
    var lastUser: User?
    var lastSettings: ScaleSettings?

    func canStartRecording(user: User, settings: ScaleSettings?) async throws -> RecordingPermission {
        canStartRecordingCalled = true
        lastUser = user
        lastSettings = settings
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
        lastSettings = nil
        validateDurationShouldThrow = nil
        validateDurationCalled = false
        lastDuration = nil
        lastStatus = nil
    }
}
