import Foundation

/// Pitch analysis data for a recording
/// Contains detected pitch information over time with confidence scores
public struct PitchAnalysisData: Equatable, Codable {
    /// Time stamps for each data point (in seconds)
    public let timeStamps: [Double]

    /// Detected frequencies at each time stamp (in Hz)
    public let frequencies: [Float]

    /// Confidence scores for each detection (0.0 - 1.0)
    public let confidences: [Float]

    /// Target notes from the scale settings (nil if no scale was used)
    public let targetNotes: [MIDINote?]

    /// Amplitude values for each detection (RMS, 0.0 - 1.0 normalized)
    /// Used for volume-based visualization coloring
    public let amplitudes: [Float]

    /// Number of data points in this analysis
    public var dataPointCount: Int {
        return timeStamps.count
    }

    public init(
        timeStamps: [Double],
        frequencies: [Float],
        confidences: [Float],
        targetNotes: [MIDINote?],
        amplitudes: [Float] = []
    ) {
        self.timeStamps = timeStamps
        self.frequencies = frequencies
        self.confidences = confidences
        self.targetNotes = targetNotes
        // If amplitudes not provided, use empty array (backward compatibility)
        self.amplitudes = amplitudes.isEmpty ? [Float](repeating: 0, count: timeStamps.count) : amplitudes
    }

    public static func == (lhs: PitchAnalysisData, rhs: PitchAnalysisData) -> Bool {
        return lhs.timeStamps == rhs.timeStamps &&
               lhs.frequencies == rhs.frequencies &&
               lhs.confidences == rhs.confidences &&
               lhs.targetNotes == rhs.targetNotes &&
               lhs.amplitudes == rhs.amplitudes
    }
}
