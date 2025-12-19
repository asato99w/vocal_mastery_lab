import Foundation

/// Result of stopping a recording
public struct StopRecordingResult: Equatable {
    public let duration: TimeInterval
    public let recordingId: RecordingId?  // Optional: nil when recording context was not set

    public init(duration: TimeInterval, recordingId: RecordingId? = nil) {
        self.duration = duration
        self.recordingId = recordingId
    }
}
