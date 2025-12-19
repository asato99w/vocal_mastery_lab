import Foundation
import SubscriptionDomain

/// User entity
///
/// Represents a user of the Vocalis Studio app with their subscription status,
/// recording history, and preferences. This is the central domain entity for
/// user-related business logic.
public struct User: Identifiable, Equatable, Codable, Sendable {
    /// Unique user identifier
    public let id: UserId

    /// Current subscription status
    public let subscriptionStatus: SubscriptionStatus

    /// Recording statistics and history
    public let recordingStats: RecordingStats

    /// User preferences (future extension point)
    public let preferences: UserPreferences

    /// Create a user
    /// - Parameters:
    ///   - id: Unique identifier
    ///   - subscriptionStatus: Current subscription status
    ///   - recordingStats: Recording statistics
    ///   - preferences: User preferences
    public init(
        id: UserId,
        subscriptionStatus: SubscriptionStatus,
        recordingStats: RecordingStats,
        preferences: UserPreferences = .default
    ) {
        self.id = id
        self.subscriptionStatus = subscriptionStatus
        self.recordingStats = recordingStats
        self.preferences = preferences
    }

    /// Create a new user with default values
    /// - Parameter cohort: User cohort (defaults to current version)
    /// - Returns: New user with free subscription
    public static func new(cohort: UserCohort = .v2_0) -> User {
        User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: cohort),
            recordingStats: .initial
        )
    }

    // MARK: - Business Logic

    /// Check if user can use scale recording feature
    /// Scale recording is now available for all tiers
    public var canUseScaleRecording: Bool {
        return true
    }

    /// Update user with new subscription status
    /// - Parameter status: New subscription status
    /// - Returns: Updated user
    public func withSubscriptionStatus(_ status: SubscriptionStatus) -> User {
        User(
            id: id,
            subscriptionStatus: status,
            recordingStats: recordingStats,
            preferences: preferences
        )
    }

    /// Update user with incremented recording count
    /// - Returns: Updated user
    public func withIncrementedRecordingCount() -> User {
        User(
            id: id,
            subscriptionStatus: subscriptionStatus,
            recordingStats: recordingStats.incrementing(),
            preferences: preferences
        )
    }

    /// Update user with reset daily recording count
    /// - Returns: Updated user
    public func withResetDailyCount() -> User {
        User(
            id: id,
            subscriptionStatus: subscriptionStatus,
            recordingStats: recordingStats.resetting(),
            preferences: preferences
        )
    }
}

/// User preferences
///
/// Encapsulates user-specific settings and preferences
/// This is a placeholder for future extension
public struct UserPreferences: Equatable, Codable, Sendable {
    /// Default preferences
    public static let `default` = UserPreferences()

    public init() {}
}
