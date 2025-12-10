import Foundation
import VocalisDomain
import SubscriptionDomain

/// Implementation of RecordingPolicyService
///
/// This service implements business rules for recording permissions and validation.
/// It's in the Application layer because it orchestrates domain logic and doesn't
/// belong to any single entity.
final class RecordingPolicyServiceImpl: RecordingPolicyService {

    // MARK: - Initialization

    init() {}

    // MARK: - RecordingPolicyService

    func canStartRecording(
        user: User,
        settings: ScaleSettings?
    ) async throws -> RecordingPermission {
        // Grandfather users (v1.0 cohort) have unlimited recordings
        if user.subscriptionStatus.cohort == .v1_0 {
            return .allowed
        }

        // Check daily recording limit using RecordingLimit.forTier()
        let limit = RecordingLimit.forTier(user.subscriptionStatus.tier)
        if !limit.isCountWithinLimit(user.recordingStats.todayCount) {
            return .denied(.dailyLimitExceeded)
        }

        // Scale recording is now available for all tiers, no permission check needed

        return .allowed
    }

    func validateDuration(
        _ duration: Duration,
        for status: SubscriptionStatus
    ) throws {
        // Grandfather users (v1.0 cohort) have unlimited recording duration
        if status.cohort == .v1_0 {
            return
        }

        let limit = RecordingLimit.forTier(status.tier)

        guard limit.isWithinDurationLimit(duration.seconds) else {
            // maxDuration is guaranteed to be non-nil here because isWithinDurationLimit
            // returns true for nil (unlimited)
            let maxDuration = limit.maxDuration ?? TimeInterval.infinity
            throw RecordingPolicyError.durationLimitExceeded(
                limit: Duration(seconds: maxDuration)
            )
        }
    }
}
