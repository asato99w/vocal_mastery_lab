import SwiftUI

/// Spectrogram visualization constants
/// Centralizes all hard-coded values for spectrogram rendering and coordinate system
public struct SpectrogramConstants {
    // MARK: - Coordinate System Constants

    /// Pixels per kHz for frequency axis (9.6x zoom from original 60pt/kHz)
    public static let basePixelsPerKHz: CGFloat = 576.0

    /// Maximum canvas height to prevent memory issues
    public static let maxCanvasHeight: CGFloat = 10000.0

    /// Maximum frequency for display (matches analyzed data range, extended for voice analysis)
    public static let maxFrequency: Double = 8000.0

    /// Pixels per second for time axis (6x zoom from original 50pt/s)
    public static let pixelsPerSecond: CGFloat = 300.0

    /// Minimum canvas width
    public static let minCanvasWidth: CGFloat = 100.0

    // MARK: - Renderer Constants

    /// Frequency label interval in Hz
    public static let frequencyLabelInterval: Double = 100.0

    /// Time label interval in seconds
    public static let timeLabelInterval: Double = 0.5

    /// Spectrogram cell time width multiplier
    public static let cellTimeWidthMultiplier: Double = 0.1

    // MARK: - Color Constants

    /// Hue value for weakest signal (blue-purple)
    public static let weakestSignalHue: CGFloat = 0.6

    /// Hue value for strongest signal (green-yellow)
    public static let strongestSignalHue: CGFloat = 0.0

    /// Color saturation (constant across all magnitudes)
    public static let colorSaturation: CGFloat = 0.8

    /// Minimum brightness for weakest signals
    public static let minBrightness: CGFloat = 0.3

    /// Maximum brightness for strongest signals
    public static let maxBrightness: CGFloat = 0.9

    /// Magnitude threshold for minimum brightness
    public static let minMagnitudeThreshold: CGFloat = 0.01
}
