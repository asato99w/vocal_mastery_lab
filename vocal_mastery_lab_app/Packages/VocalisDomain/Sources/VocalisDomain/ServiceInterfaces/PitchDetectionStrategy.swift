import Foundation

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
}
