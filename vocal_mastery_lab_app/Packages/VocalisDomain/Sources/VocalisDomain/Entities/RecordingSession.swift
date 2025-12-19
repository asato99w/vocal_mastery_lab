import Foundation

/// Represents an active recording session
/// Tracks the state of an ongoing recording
public struct RecordingSession: Equatable {
    public let recordingURL: URL
    public let startedAt: Date

    public init(
        recordingURL: URL,
        startedAt: Date = Date()
    ) {
        self.recordingURL = recordingURL
        self.startedAt = startedAt
    }

    /// Calculate elapsed time since recording started
    public func elapsedTime(at currentTime: Date = Date()) -> TimeInterval {
        return currentTime.timeIntervalSince(startedAt)
    }
}
