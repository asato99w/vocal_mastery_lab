import SwiftUI

/// Pitch graph visualization constants
/// Centralizes all hard-coded values for pitch graph rendering and coordinate system
public struct PitchGraphConstants {
    // MARK: - Frequency Range

    /// Minimum frequency for pitch graph (Hz)
    /// 55Hz (A1) - slightly below C2 (65.4Hz), the lowest playable note
    public static let minFrequency: Double = 55.0

    /// Maximum frequency for pitch graph (Hz)
    /// 1100Hz - slightly above C6 (1046.5Hz), the highest playable note
    public static let maxFrequency: Double = 1100.0

    // MARK: - Display Density

    /// Pixels per octave for frequency axis (logarithmic scale)
    /// Higher values = more vertical space per octave
    /// 55Hz to 1100Hz spans ~4.3 octaves (log2(1100/55) ≈ 4.32)
    public static let pixelsPerOctave: CGFloat = 400.0

    /// Pixels per second for time axis
    /// Matches spectrogram time axis density
    public static let pixelsPerSecond: CGFloat = 300.0

    // MARK: - Canvas Limits

    /// Maximum canvas height to prevent memory issues
    public static let maxCanvasHeight: CGFloat = 5000.0

    /// Minimum canvas width
    public static let minCanvasWidth: CGFloat = 100.0

    // MARK: - Labels

    /// Frequency label values for logarithmic scale display
    /// 100Hz intervals from 100Hz to 1100Hz
    public static let frequencyLabelValues: [Double] = [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100
    ]

    /// Time label interval in seconds
    public static let timeLabelInterval: Double = 0.5

    // MARK: - Margins

    /// Left margin for Y-axis labels
    public static let leftMargin: CGFloat = 50.0

    /// Bottom margin for X-axis labels
    public static let bottomMargin: CGFloat = 30.0

    /// Top margin
    public static let topMargin: CGFloat = 10.0

    /// Right margin
    public static let rightMargin: CGFloat = 10.0

    // MARK: - Visual Elements

    /// Pitch dot minimum radius
    public static let minDotRadius: CGFloat = 2.0

    /// Pitch dot maximum radius
    public static let maxDotRadius: CGFloat = 4.0

    /// Line width for pitch graph
    public static let pitchLineWidth: CGFloat = 1.5

    /// Target scale line width
    public static let targetLineWidth: CGFloat = 1.0

    /// Playback position line width
    public static let playbackLineWidth: CGFloat = 2.0

    // MARK: - Gap Detection

    /// Time gap threshold for pitch line segmentation (seconds)
    /// If consecutive pitch points are more than this time apart, start a new line segment
    /// 100ms threshold: typical pitch detection samples at 10-50ms intervals
    public static let gapThreshold: Double = 0.1

    // MARK: - Colors

    /// Pitch line color
    public static let pitchLineColor = Color.cyan

    /// Target scale line color
    public static let targetLineColor = Color.gray.opacity(0.3)

    /// Playback position line color
    public static let playbackLineColor = Color.white

    /// Frequency label color
    public static let frequencyLabelColor = Color.gray

    /// Time label color
    public static let timeLabelColor = Color.gray

    // MARK: - Volume-Based Color Constants (matches Spectrogram)

    /// Hue value for weakest signal (blue-purple) - same as SpectrogramConstants
    public static let weakestSignalHue: CGFloat = 0.6

    /// Hue value for strongest signal (red/yellow) - same as SpectrogramConstants
    public static let strongestSignalHue: CGFloat = 0.0

    /// Color saturation for volume-based coloring
    public static let volumeColorSaturation: CGFloat = 0.8

    /// Minimum brightness for weak signals
    public static let volumeMinBrightness: CGFloat = 0.4

    /// Maximum brightness for strong signals
    public static let volumeMaxBrightness: CGFloat = 0.95

    // MARK: - Calculated Properties

    /// Calculate canvas height based on frequency range (logarithmic scale)
    /// Returns: Canvas height in points
    public static var calculatedCanvasHeight: CGFloat {
        // Number of octaves in frequency range
        let octaves = log2(maxFrequency / minFrequency)
        let height = CGFloat(octaves) * pixelsPerOctave
        return min(maxCanvasHeight, height)
    }
}
