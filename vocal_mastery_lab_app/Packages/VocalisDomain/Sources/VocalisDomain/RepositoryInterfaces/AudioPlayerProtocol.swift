import Foundation

/// Protocol for audio playback
public protocol AudioPlayerProtocol {
    /// Play audio from URL
    /// - Parameters:
    ///   - url: Audio file URL
    ///   - withPitchDetection: If true, configures audio session for simultaneous recording (pitch detection)
    func play(url: URL, withPitchDetection: Bool) async throws
    func stop() async
    func pause()
    func resume()
    func seek(to time: TimeInterval)
    var isPlaying: Bool { get }
    var currentTime: TimeInterval { get }
    var duration: TimeInterval { get }
}

extension AudioPlayerProtocol {
    /// Default implementation without pitch detection
    public func play(url: URL) async throws {
        try await play(url: url, withPitchDetection: false)
    }
}

public enum AudioPlayerError: LocalizedError, Equatable {
    case fileNotFound
    case playbackFailed(String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound:
            return "Audio file not found"
        case .playbackFailed(let message):
            return "Playback failed: \(message)"
        }
    }
}
