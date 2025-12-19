import Foundation

public struct Duration: Equatable, Comparable, Hashable {
    public let seconds: TimeInterval

    public init(seconds: TimeInterval) {
        self.seconds = max(0, seconds)
    }

    public init(from startTime: Date, to endTime: Date) {
        self.seconds = max(0, endTime.timeIntervalSince(startTime))
    }

    public var formatted: String {
        let minutes = Int(seconds) / 60
        let remainingSeconds = Int(seconds) % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }

    // Comparable
    public static func < (lhs: Duration, rhs: Duration) -> Bool {
        lhs.seconds < rhs.seconds
    }
}