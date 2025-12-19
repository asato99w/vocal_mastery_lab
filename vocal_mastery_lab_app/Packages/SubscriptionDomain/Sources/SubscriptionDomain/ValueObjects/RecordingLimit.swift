//
//  RecordingLimit.swift
//  VocalisStudio
//
//  Recording limits based on subscription tier
//

import Foundation

/// Recording limits for each subscription tier
public struct RecordingLimit {

    /// Daily recording count limit (nil = unlimited)
    public let dailyCount: Int?

    /// Maximum recording duration in seconds (nil = unlimited)
    public let maxDuration: TimeInterval?

    /// Public initializer
    public init(dailyCount: Int?, maxDuration: TimeInterval?) {
        self.dailyCount = dailyCount
        self.maxDuration = maxDuration
    }

    /// Configuration for recording limits (allows test overrides)
    public struct Configuration {
        public let freeDailyCount: Int
        public let freeMaxDuration: TimeInterval
        public let premiumDailyCount: Int
        public let premiumMaxDuration: TimeInterval

        public init(
            freeDailyCount: Int = 5,
            freeMaxDuration: TimeInterval = 30,
            premiumDailyCount: Int = .max,
            premiumMaxDuration: TimeInterval = 300
        ) {
            self.freeDailyCount = freeDailyCount
            self.freeMaxDuration = freeMaxDuration
            self.premiumDailyCount = premiumDailyCount
            self.premiumMaxDuration = premiumMaxDuration
        }

        /// Production configuration
        public static let production = Configuration()

        /// Test configuration with short durations and counts
        public static let test = Configuration(
            freeDailyCount: 5,
            freeMaxDuration: 2,
            premiumDailyCount: 0, // Unused - premium has unlimited count
            premiumMaxDuration: 3
        )
    }

    /// Get recording limit for subscription tier
    /// Single source of truth for both count and duration limits
    public static func forTier(_ tier: SubscriptionTier, configuration: Configuration = .production) -> RecordingLimit {
        switch tier {
        case .free:
            return RecordingLimit(dailyCount: configuration.freeDailyCount, maxDuration: configuration.freeMaxDuration)
        case .premium:
            // Premium has unlimited daily count but limited duration
            return RecordingLimit(dailyCount: nil, maxDuration: configuration.premiumMaxDuration)
        case .premiumPlus:
            return RecordingLimit(dailyCount: nil, maxDuration: nil) // Unlimited
        }
    }

    /// Check if count is within limit
    public func isCountWithinLimit(_ count: Int) -> Bool {
        guard let limit = dailyCount else {
            return true // Unlimited
        }
        return count < limit
    }

    /// Check if duration is within limit
    public func isWithinDurationLimit(_ duration: TimeInterval) -> Bool {
        guard let limit = maxDuration else {
            return true // Unlimited
        }
        return duration <= limit
    }

    /// Get remaining count for display
    public func remainingCount(_ current: Int) -> String {
        guard let limit = dailyCount else {
            return "無制限"
        }
        let remaining = max(0, limit - current)
        return "\(remaining)/\(limit)"
    }

    /// Get duration limit description
    public var durationDescription: String {
        guard let duration = maxDuration else {
            return "無制限"
        }

        if duration >= 60 {
            let minutes = Int(duration / 60)
            return "\(minutes)分"
        } else {
            return "\(Int(duration))秒"
        }
    }
}
