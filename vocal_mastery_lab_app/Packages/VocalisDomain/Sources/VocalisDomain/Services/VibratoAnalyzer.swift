import Foundation

/// Analyzes pitch data for vibrato characteristics
/// Uses FFT to detect periodic pitch modulation in the 4-8 Hz range
public final class VibratoAnalyzer {

    // MARK: - Constants

    /// Minimum samples required for vibrato detection
    private let minimumSampleCount = 10

    /// Vibrato rate range (Hz)
    /// Typical vibrato is 5-7 Hz. maxVibratoRate set to 10.0 to accommodate faster vibrato.
    /// Note: minLag uses ceil() to ensure rate calculations don't exceed maxVibratoRate.
    private let minVibratoRate: Float = 4.0
    private let maxVibratoRate: Float = 10.0

    /// Minimum extent to consider as vibrato (cents)
    private let minimumExtentCents: Float = 15.0

    /// Maximum extent to consider as vibrato (cents)
    /// Typical vibrato is ±30-100 cents. Larger variations indicate pitch changes (melody), not vibrato.
    private let maximumExtentCents: Float = 200.0

    /// Minimum regularity threshold (configurable based on pitch detection algorithm)
    /// - FCPE (100Hz sampling): 0.3 (default) - high resolution allows strict threshold
    /// - YIN (20Hz sampling): 0.15 - lower resolution needs relaxed threshold
    private let minimumRegularity: Float

    /// Initialize with configurable regularity threshold
    /// - Parameter minimumRegularity: Minimum regularity threshold (0.0-1.0). Default is 0.3 for FCPE.
    public init(minimumRegularity: Float = 0.3) {
        self.minimumRegularity = minimumRegularity
    }

    // MARK: - Public API

    /// Analyze pitch data for vibrato characteristics
    /// - Parameters:
    ///   - frequencies: Array of detected frequencies in Hz
    ///   - timeStamps: Array of corresponding timestamps in seconds
    /// - Returns: VibratoAnalysis containing rate, extent, regularity, and detection status
    public func analyze(frequencies: [Float], timeStamps: [Double]) -> VibratoAnalysis {
        // Validate input
        guard frequencies.count >= minimumSampleCount,
              frequencies.count == timeStamps.count else {
            return .none
        }

        // 1. Calculate mean pitch and deviations
        let meanFrequency = frequencies.reduce(0, +) / Float(frequencies.count)
        guard meanFrequency > 0 else { return .none }

        // Convert frequencies to cents deviation from mean
        let deviations = frequencies.map { freq -> Float in
            1200.0 * log2(freq / meanFrequency)
        }

        // 2. Calculate extent (peak-to-peak / 2)
        let maxDeviation = deviations.max() ?? 0
        let minDeviation = deviations.min() ?? 0
        let extent = (maxDeviation - minDeviation) / 2.0

        // If extent is too small, no vibrato
        if extent < minimumExtentCents {
            return VibratoAnalysis(rate: 0, extent: extent, regularity: 0, isPresent: false)
        }

        // 3. Calculate sample rate from timestamps
        guard timeStamps.count >= 2 else { return .none }
        let totalDuration = timeStamps.last! - timeStamps.first!
        guard totalDuration > 0 else { return .none }
        let sampleRate = Float(frequencies.count - 1) / Float(totalDuration)

        // 4. Perform FFT on deviations to find periodic component
        let (rate, regularity) = analyzePeriodicComponent(
            deviations: deviations,
            sampleRate: sampleRate
        )

        // 5. Determine if vibrato is present
        // Note: extent must be within range (not too small, not too large)
        // Large extent indicates pitch changes (melody), not vibrato
        let isPresent = rate >= minVibratoRate &&
                        rate <= maxVibratoRate &&
                        extent >= minimumExtentCents &&
                        extent <= maximumExtentCents &&
                        regularity >= minimumRegularity

        return VibratoAnalysis(
            rate: rate,
            extent: extent,
            regularity: regularity,
            isPresent: isPresent
        )
    }

    // MARK: - Private Methods

    /// Analyze periodic component using autocorrelation
    private func analyzePeriodicComponent(deviations: [Float], sampleRate: Float) -> (rate: Float, regularity: Float) {
        let n = deviations.count
        guard n >= 4 else { return (0, 0) }

        // Calculate autocorrelation manually (more reliable than vDSP_conv for this use case)
        var autocorrelation = [Float](repeating: 0, count: n)

        // Compute autocorrelation for each lag
        for lag in 0..<n {
            var sum: Float = 0
            for i in 0..<(n - lag) {
                sum += deviations[i] * deviations[i + lag]
            }
            autocorrelation[lag] = sum
        }

        // Normalize by zero-lag autocorrelation
        if autocorrelation[0] > 0 {
            let norm = autocorrelation[0]
            for i in 0..<n {
                autocorrelation[i] /= norm
            }
        }

        // Find peak in vibrato rate range
        // Convert rate range to lag range
        // Use ceil for minLag to ensure calculated rate stays within maxVibratoRate
        // rate = sampleRate / lag, so larger lag = lower rate
        let minLag = max(1, Int(ceil(sampleRate / maxVibratoRate)))
        let maxLag = min(n - 1, Int(sampleRate / minVibratoRate))

        guard minLag < maxLag else { return (0, 0) }

        // Find the highest peak in the valid lag range
        var peakLag = minLag
        var peakValue: Float = 0

        for lag in minLag...maxLag {
            if autocorrelation[lag] > peakValue {
                peakValue = autocorrelation[lag]
                peakLag = lag
            }
        }

        // Calculate rate from lag
        let rate = sampleRate / Float(peakLag)

        // Regularity is the normalized autocorrelation value at the peak
        // Higher value means more regular periodicity
        let regularity = max(0, min(1, peakValue))

        return (rate, regularity)
    }
}
