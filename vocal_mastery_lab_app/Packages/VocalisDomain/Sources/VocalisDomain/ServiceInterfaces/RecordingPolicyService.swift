import Foundation
import SubscriptionDomain

/// Domain Service for recording business rules and policies
///
/// This service encapsulates recording-related business logic that:
/// - Doesn't naturally belong to any single entity
/// - Involves coordination between multiple domain concepts
/// - Represents business policies and constraints
public protocol RecordingPolicyService {
    /// Check if user can start a new recording
    ///
    /// - Parameters:
    ///   - user: The user attempting to record
    /// - Returns: Permission result indicating if recording is allowed
    /// - Throws: RecordingPolicyError for validation failures
    func canStartRecording(
        user: User
    ) async throws -> RecordingPermission

    /// Validate recording duration against subscription tier limits
    ///
    /// - Parameters:
    ///   - duration: The duration to validate
    ///   - status: User's subscription status
    /// - Throws: RecordingPolicyError if duration exceeds tier limit
    func validateDuration(
        _ duration: Duration,
        for status: SubscriptionStatus
    ) throws
}

/// Recording permission result
public enum RecordingPermission: Equatable {
    /// Recording is allowed
    case allowed

    /// Recording is denied with specific reason
    case denied(DenialReason)

    /// Reason why recording was denied
    public enum DenialReason: Equatable {
        /// User exceeded daily recording limit
        case dailyLimitExceeded

        /// Premium subscription required for this feature
        case premiumRequired

        /// Invalid recording settings
        case invalidSettings(String)
    }
}

/// Recording policy errors
public enum RecordingPolicyError: Error, Equatable {
    /// Duration exceeds subscription tier limit
    case durationLimitExceeded(limit: Duration)

    /// Invalid recording configuration
    case invalidConfiguration(String)

    /// User not found or invalid
    case invalidUser
}
