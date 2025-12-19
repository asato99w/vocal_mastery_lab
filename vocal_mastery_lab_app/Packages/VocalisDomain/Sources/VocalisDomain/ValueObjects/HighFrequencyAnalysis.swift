import Foundation

/// Result of high frequency analysis for a recording or frame
/// Contains brightness (4-6 kHz) and airiness (6-9 kHz) ratios
public struct HighFrequencyAnalysis: Equatable, Sendable {

    /// Energy ratio in brightness band (4-6 kHz) relative to total
    /// Range: 0.0 to 1.0 (typically 0.01-0.10)
    public let brightnessRatio: Float

    /// Energy ratio in airiness band (6-9 kHz) relative to total
    /// Range: 0.0 to 1.0 (typically 0.001-0.05)
    public let airinessRatio: Float

    /// Brightness ratio as percentage (0-100)
    public var brightnessPercentage: Float {
        brightnessRatio * 100.0
    }

    /// Airiness ratio as percentage (0-100)
    public var airinessPercentage: Float {
        airinessRatio * 100.0
    }

    public init(brightnessRatio: Float, airinessRatio: Float) {
        self.brightnessRatio = brightnessRatio
        self.airinessRatio = airinessRatio
    }

    /// Empty analysis result when no data available
    public static let none = HighFrequencyAnalysis(
        brightnessRatio: 0,
        airinessRatio: 0
    )
}
