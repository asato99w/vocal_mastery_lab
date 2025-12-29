import Foundation

/// Progress callback type for pitch detection
/// - Parameter progress: Current progress from 0.0 to 1.0
public typealias PitchDetectionProgressCallback = @Sendable (Double) async -> Void

/// Protocol for pitch detection algorithms
/// Allows switching between different pitch detection strategies (YIN, pYIN, etc.)
public protocol PitchDetectionStrategy {
    /// Algorithm identifier name (e.g., "YIN", "pYIN", "pYIN-balanced")
    var name: String { get }

    /// Whether this algorithm requires octave correction post-processing
    /// - YIN: true (prone to octave errors, needs correction)
    /// - pYIN: false (HMM provides temporal smoothing, low GPE)
    var requiresOctaveCorrection: Bool { get }

    /// Detect pitch from audio samples
    /// - Parameters:
    ///   - samples: Audio samples to analyze
    ///   - sampleRate: Sample rate in Hz (e.g., 44100.0)
    /// - Returns: Array of PitchFrame containing detected pitches
    func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame]

    /// Detect pitch from audio samples with progress reporting
    /// - Parameters:
    ///   - samples: Audio samples to analyze
    ///   - sampleRate: Sample rate in Hz (e.g., 44100.0)
    ///   - progress: Callback to report progress (0.0 to 1.0)
    /// - Returns: Array of PitchFrame containing detected pitches
    func detectPitch(samples: [Float], sampleRate: Double, progress: PitchDetectionProgressCallback?) async -> [PitchFrame]
}

/// Default implementation for backward compatibility
public extension PitchDetectionStrategy {
    /// Default implementation calls the synchronous version
    func detectPitch(samples: [Float], sampleRate: Double, progress: PitchDetectionProgressCallback?) async -> [PitchFrame] {
        // Report start progress
        await progress?(0.0)
        // Call synchronous version
        let result = detectPitch(samples: samples, sampleRate: sampleRate)
        // Report completion
        await progress?(1.0)
        return result
    }
}
