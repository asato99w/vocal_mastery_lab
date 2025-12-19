import Foundation

/// Protocol for audio recording functionality
/// Infrastructure layer must implement this protocol
public protocol AudioRecorderProtocol {
    /// Prepare for recording and return the file URL where audio will be saved
    func prepareRecording() async throws -> URL

    /// Start recording audio
    func startRecording() async throws

    /// Stop recording and return the duration of the recording
    func stopRecording() async throws -> TimeInterval

    /// Whether the recorder is currently recording
    var isRecording: Bool { get }
}

/// Errors that can occur during audio recording
public enum AudioRecorderError: LocalizedError, Equatable {
    case notPrepared
    case notRecording
    case recordingFailed(String)

    public static func == (lhs: AudioRecorderError, rhs: AudioRecorderError) -> Bool {
        switch (lhs, rhs) {
        case (.notPrepared, .notPrepared):
            return true
        case (.notRecording, .notRecording):
            return true
        case (.recordingFailed(let lhsMsg), .recordingFailed(let rhsMsg)):
            return lhsMsg == rhsMsg
        default:
            return false
        }
    }

    public var errorDescription: String? {
        switch self {
        case .notPrepared:
            return "Recorder not prepared. Call prepareRecording() first."
        case .notRecording:
            return "Not currently recording."
        case .recordingFailed(let message):
            return "Recording failed: \(message)"
        }
    }
}
