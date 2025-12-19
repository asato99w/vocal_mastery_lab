import Foundation

/// Singer's Formant analysis result
/// Analyzes the energy concentration in the 2500-3500 Hz band
/// characteristic of trained classical singers
public struct SingersFormantAnalysis: Equatable {
    /// Energy ratio in the SF band (2500-3500 Hz) relative to total energy
    /// Range: 0.0 - 1.0 (typically 0.05 - 0.15 for trained singers)
    public let ratio: Float

    /// Intensity difference between SF band and surrounding bands in dB
    /// Higher values indicate stronger formant presence
    public let intensity: Float

    /// Whether singer's formant is detected (above threshold)
    public let isPresent: Bool

    /// Detection confidence (lower for high pitched voices)
    /// Range: 0.0 - 1.0
    public let confidence: Float

    public init(ratio: Float, intensity: Float, isPresent: Bool, confidence: Float) {
        self.ratio = ratio
        self.intensity = intensity
        self.isPresent = isPresent
        self.confidence = confidence
    }

    /// No singer's formant detected
    public static var none: SingersFormantAnalysis {
        SingersFormantAnalysis(ratio: 0, intensity: 0, isPresent: false, confidence: 0)
    }
}
