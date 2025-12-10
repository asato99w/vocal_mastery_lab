import SwiftUI

/// Constants for PitchBar karaoke-style visualization
/// Centralizes all configuration values for pitch deviation display
public struct PitchBarConstants {
    // MARK: - Deviation Thresholds (in cents)

    /// Perfect accuracy threshold (±10 cents)
    public static let perfectThreshold: Double = 10.0

    /// Good accuracy threshold (±25 cents)
    public static let goodThreshold: Double = 25.0

    /// Acceptable accuracy threshold (±50 cents)
    public static let acceptableThreshold: Double = 50.0

    // MARK: - Deviation Colors

    /// Color for perfect accuracy (±10 cents) - Green
    public static let perfectColor = Color.green

    /// Color for good accuracy (±25 cents) - Blue
    public static let goodColor = Color.blue

    /// Color for acceptable accuracy (±50 cents) - Yellow
    public static let acceptableColor = Color.yellow

    /// Color for needs improvement (>50 cents) - Red
    public static let needsImprovementColor = Color.red

    /// Target note bar color (background)
    public static let targetBarColor = Color.gray.opacity(0.3)

    // MARK: - Display Constants

    /// Pixels per second for horizontal time axis
    public static let pixelsPerSecond: CGFloat = 100.0

    /// Height of note bar in pixels
    public static let noteBarHeight: CGFloat = 20.0

    /// Pixels per octave for vertical frequency axis (logarithmic scale)
    public static let pixelsPerOctave: CGFloat = 120.0

    /// Minimum frequency for display range (Hz)
    public static let minFrequency: Double = 55.0  // A1

    /// Maximum frequency for display range (Hz)
    public static let maxFrequency: Double = 1100.0  // ~C6

    /// Line width for detected pitch path
    public static let pitchLineWidth: CGFloat = 2.0

    /// Playback position line width
    public static let playbackLineWidth: CGFloat = 2.0

    /// Playback position line color
    public static let playbackLineColor = Color.white

    // MARK: - Functions

    /// Calculate pitch deviation in cents
    /// - Parameters:
    ///   - detected: Detected frequency in Hz
    ///   - expected: Expected (target) frequency in Hz
    /// - Returns: Deviation in cents (positive = sharp, negative = flat)
    public static func calculateDeviation(detected: Double, expected: Double) -> Double {
        guard expected > 0 && detected > 0 else { return 0 }
        return 1200.0 * log2(detected / expected)
    }

    /// Get color for a given deviation value
    /// - Parameter deviation: Deviation in cents
    /// - Returns: Color based on accuracy level
    public static func deviationColor(for deviation: Double) -> Color {
        let absDeviation = abs(deviation)

        if absDeviation <= perfectThreshold {
            return perfectColor
        } else if absDeviation <= goodThreshold {
            return goodColor
        } else if absDeviation <= acceptableThreshold {
            return acceptableColor
        } else {
            return needsImprovementColor
        }
    }

    /// Evaluate accuracy level for a given deviation
    /// - Parameter deviation: Deviation in cents
    /// - Returns: Accuracy evaluation enum
    public static func evaluateAccuracy(deviation: Double) -> AccuracyLevel {
        let absDeviation = abs(deviation)

        if absDeviation <= perfectThreshold {
            return .perfect
        } else if absDeviation <= goodThreshold {
            return .good
        } else if absDeviation <= acceptableThreshold {
            return .acceptable
        } else {
            return .needsImprovement
        }
    }

    /// Accuracy level enumeration
    public enum AccuracyLevel: String, Equatable {
        case perfect
        case good
        case acceptable
        case needsImprovement
    }

    // MARK: - Canvas Calculations

    /// Calculate canvas height based on frequency range (logarithmic scale)
    public static var calculatedCanvasHeight: CGFloat {
        let octaves = log2(maxFrequency / minFrequency)
        return CGFloat(octaves) * pixelsPerOctave
    }

    /// Convert frequency to Y coordinate on canvas (logarithmic scale)
    /// - Parameters:
    ///   - frequency: Frequency in Hz
    ///   - canvasHeight: Total canvas height
    /// - Returns: Y coordinate (0 = top = maxFrequency, canvasHeight = bottom = minFrequency)
    public static func frequencyToY(frequency: Double, canvasHeight: CGFloat) -> CGFloat {
        let clampedFreq = max(minFrequency, min(maxFrequency, frequency))
        let logRange = log2(maxFrequency / minFrequency)
        let ratio = log2(maxFrequency / clampedFreq) / logRange
        return CGFloat(ratio) * canvasHeight
    }

    /// Convert time to X coordinate on canvas
    /// - Parameters:
    ///   - time: Time in seconds
    ///   - leftPadding: Left padding for canvas
    /// - Returns: X coordinate
    public static func timeToX(time: Double, leftPadding: CGFloat = 0) -> CGFloat {
        return CGFloat(time) * pixelsPerSecond + leftPadding
    }
}
