import Foundation

/// A single frame of pitch detection result
/// Represents pitch information at a specific point in time
public struct PitchFrame: Equatable {
    /// Timestamp in seconds from the start of audio
    public let timestamp: Double

    /// Detected frequency in Hz, nil if unvoiced/silent
    public let frequency: Float?

    /// Confidence score for the detection (0.0 - 1.0)
    public let confidence: Float

    /// Normalized amplitude (0.0 - 1.0)
    public let amplitude: Float

    /// Whether this frame contains a voiced pitch
    public var isVoiced: Bool {
        frequency != nil
    }

    public init(
        timestamp: Double,
        frequency: Float?,
        confidence: Float,
        amplitude: Float
    ) {
        self.timestamp = timestamp
        self.frequency = frequency
        self.confidence = confidence
        self.amplitude = amplitude
    }
}
