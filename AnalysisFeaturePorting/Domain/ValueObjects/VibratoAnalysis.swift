import Foundation

/// Vibrato analysis result
/// Contains rate, extent, regularity, and detection status
public struct VibratoAnalysis: Equatable {
    /// Vibrato rate in Hz (typical: 5-7 Hz)
    public let rate: Float

    /// Vibrato extent in cents (typical: ±30-100 cents)
    public let extent: Float

    /// Regularity of vibrato (0.0 - 1.0, higher = more regular)
    public let regularity: Float

    /// Whether vibrato is detected
    public let isPresent: Bool

    public init(rate: Float, extent: Float, regularity: Float, isPresent: Bool) {
        self.rate = rate
        self.extent = extent
        self.regularity = regularity
        self.isPresent = isPresent
    }

    /// No vibrato detected
    public static var none: VibratoAnalysis {
        VibratoAnalysis(rate: 0, extent: 0, regularity: 0, isPresent: false)
    }
}
