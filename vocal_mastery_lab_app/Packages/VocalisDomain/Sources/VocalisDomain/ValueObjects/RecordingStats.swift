import Foundation

/// Recording statistics for a user
///
/// Tracks recording activity to enforce business rules and limits
public struct RecordingStats: Equatable, Codable, Sendable {
    /// Number of recordings made today
    public let todayCount: Int

    /// Date of the last reset (typically midnight)
    public let lastResetDate: Date

    /// Total number of recordings ever made
    public let totalCount: Int

    /// Create recording statistics
    /// - Parameters:
    ///   - todayCount: Number of recordings today
    ///   - lastResetDate: When stats were last reset (defaults to today)
    ///   - totalCount: Total lifetime recordings (defaults to today count)
    public init(
        todayCount: Int,
        lastResetDate: Date = Date(),
        totalCount: Int? = nil
    ) {
        self.todayCount = todayCount
        self.lastResetDate = lastResetDate
        self.totalCount = totalCount ?? todayCount
    }

    /// Create initial statistics for new user
    public static var initial: RecordingStats {
        RecordingStats(todayCount: 0, totalCount: 0)
    }

    /// Check if statistics need daily reset
    /// - Parameter calendar: Calendar for date comparison
    /// - Returns: True if last reset was before today
    public func needsReset(calendar: Calendar = .current) -> Bool {
        !calendar.isDateInToday(lastResetDate)
    }

    /// Create new stats with incremented count
    /// - Returns: New RecordingStats with today count + 1
    public func incrementing() -> RecordingStats {
        RecordingStats(
            todayCount: todayCount + 1,
            lastResetDate: lastResetDate,
            totalCount: totalCount + 1
        )
    }

    /// Create new stats with reset daily count
    /// - Returns: New RecordingStats with today count = 0
    public func resetting() -> RecordingStats {
        RecordingStats(
            todayCount: 0,
            lastResetDate: Date(),
            totalCount: totalCount
        )
    }
}
