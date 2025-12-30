import Foundation

/// Analyzes pitch data for vibrato characteristics
/// Uses autocorrelation to detect periodic pitch modulation in the 4-9 Hz range
public final class VibratoAnalyzer {

    // MARK: - Constants

    /// Minimum samples required for vibrato detection
    private let minimumSampleCount = 10

    /// Vibrato rate range (Hz)
    /// Typical vibrato is 5-7 Hz (classical), up to 7-8 Hz (pop/rock).
    /// maxVibratoRate set to 9.0 for safety margin while avoiding most false positives.
    /// Note: minLag uses ceil() to ensure rate calculations don't exceed maxVibratoRate.
    private let minVibratoRate: Float = 4.0
    private let maxVibratoRate: Float = 9.0

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
    /// - Parameter minimumRegularity: Minimum regularity threshold (0.0-1.0). Default is 0.15 (relaxed) for stable region detection.
    public init(minimumRegularity: Float = 0.15) {
        self.minimumRegularity = minimumRegularity
    }

    // MARK: - Stable Region Detection Parameters (Relaxed)

    /// Maximum pitch change between adjacent frames to be considered "stable" (in cents)
    /// Relaxed: 50.0 (default was 30.0) - better for varied pitch data
    private let stabilityThresholdCents: Double = 50.0

    /// Minimum duration of a stable region (in seconds)
    /// Relaxed: 0.05 (default was 0.1) - captures shorter stable regions
    private let minStableDurationSeconds: Double = 0.05

    // MARK: - Public API

    /// Analyze pitch data for vibrato using stable region approach
    /// First detects stable pitch regions, then analyzes vibrato within those regions
    /// More effective for filtering out pitch transitions and focusing on sustained notes
    /// - Parameters:
    ///   - frequencies: Array of detected frequencies in Hz
    ///   - timeStamps: Array of corresponding timestamps in seconds
    /// - Returns: VibratoAnalysis with the best detected vibrato (highest regularity), or none if not found
    public func analyzeWithStableRegion(
        frequencies: [Float],
        timeStamps: [Double]
    ) -> VibratoAnalysis {
        guard frequencies.count >= minimumSampleCount,
              frequencies.count == timeStamps.count,
              timeStamps.count >= 2 else {
            return .none
        }

        // 1. Detect stable regions within the data
        let stableRegions = detectStableRegions(frequencies: frequencies, timeStamps: timeStamps)

        guard !stableRegions.isEmpty else {
            return .none
        }

        // 2. Analyze vibrato in each stable region, keep the best result
        var bestAnalysis: VibratoAnalysis = .none

        for region in stableRegions {
            let regionFreqs = Array(frequencies[region.startIndex...region.endIndex])
            let regionTimes = Array(timeStamps[region.startIndex...region.endIndex])

            let analysis = analyze(frequencies: regionFreqs, timeStamps: regionTimes)

            if analysis.isPresent && analysis.regularity > bestAnalysis.regularity {
                bestAnalysis = analysis
            }
        }

        return bestAnalysis
    }

    /// Analyze pitch data for vibrato using sliding window approach (recommended)
    /// Analyzes overlapping windows to detect local vibrato patterns
    /// POC testing showed 53% detection rate vs 21% for standard analysis
    /// - Parameters:
    ///   - frequencies: Array of detected frequencies in Hz
    ///   - timeStamps: Array of corresponding timestamps in seconds
    ///   - windowDuration: Duration of analysis window in seconds (default: 0.5s for ~2-3 vibrato cycles)
    ///   - hopRatio: Window hop as ratio of window size (default: 0.5 for 50% overlap)
    /// - Returns: VibratoAnalysis with the best detected vibrato (highest regularity), or none if not found
    public func analyzeWithSlidingWindow(
        frequencies: [Float],
        timeStamps: [Double],
        windowDuration: Double = 0.5,
        hopRatio: Double = 0.5
    ) -> VibratoAnalysis {
        guard frequencies.count >= minimumSampleCount,
              frequencies.count == timeStamps.count,
              timeStamps.count >= 2 else {
            return .none
        }

        let totalDuration = timeStamps.last! - timeStamps.first!
        guard totalDuration > 0 else { return .none }

        let sampleRate = Double(frequencies.count - 1) / totalDuration
        let windowSamples = Int(windowDuration * sampleRate)
        let hopSamples = max(1, Int(Double(windowSamples) * hopRatio))

        // If data is shorter than window, fall back to standard analysis
        guard windowSamples >= minimumSampleCount && windowSamples <= frequencies.count else {
            return analyze(frequencies: frequencies, timeStamps: timeStamps)
        }

        var bestAnalysis: VibratoAnalysis = .none
        var windowStart = 0

        while windowStart + windowSamples <= frequencies.count {
            let windowEnd = windowStart + windowSamples
            let windowFreqs = Array(frequencies[windowStart..<windowEnd])
            let windowTimes = Array(timeStamps[windowStart..<windowEnd])

            let analysis = analyze(frequencies: windowFreqs, timeStamps: windowTimes)

            // Keep the best vibrato (highest regularity among detected)
            if analysis.isPresent && analysis.regularity > bestAnalysis.regularity {
                bestAnalysis = analysis
            }

            windowStart += hopSamples
        }

        return bestAnalysis
    }

    /// Analyze pitch data for vibrato using sliding window approach, returning ALL window results
    /// Used for calculating presence rate across multiple windows
    /// - Parameters:
    ///   - frequencies: Array of detected frequencies in Hz
    ///   - timeStamps: Array of corresponding timestamps in seconds
    ///   - windowDuration: Duration of analysis window in seconds (default: 0.5s for ~2-3 vibrato cycles)
    ///   - hopRatio: Window hop as ratio of window size (default: 0.5 for 50% overlap)
    /// - Returns: Array of VibratoAnalysis for each window
    public func analyzeAllWindows(
        frequencies: [Float],
        timeStamps: [Double],
        windowDuration: Double = 0.5,
        hopRatio: Double = 0.5
    ) -> [VibratoAnalysis] {
        guard frequencies.count >= minimumSampleCount,
              frequencies.count == timeStamps.count,
              timeStamps.count >= 2 else {
            return []
        }

        let totalDuration = timeStamps.last! - timeStamps.first!
        guard totalDuration > 0 else { return [] }

        let sampleRate = Double(frequencies.count - 1) / totalDuration
        let windowSamples = Int(windowDuration * sampleRate)
        let hopSamples = max(1, Int(Double(windowSamples) * hopRatio))

        // If data is shorter than window, fall back to single analysis
        guard windowSamples >= minimumSampleCount && windowSamples <= frequencies.count else {
            return [analyze(frequencies: frequencies, timeStamps: timeStamps)]
        }

        var allAnalyses: [VibratoAnalysis] = []
        var windowStart = 0

        while windowStart + windowSamples <= frequencies.count {
            let windowEnd = windowStart + windowSamples
            let windowFreqs = Array(frequencies[windowStart..<windowEnd])
            let windowTimes = Array(timeStamps[windowStart..<windowEnd])

            let analysis = analyze(frequencies: windowFreqs, timeStamps: windowTimes)
            allAnalyses.append(analysis)

            windowStart += hopSamples
        }

        return allAnalyses
    }

    /// Analyze pitch data for vibrato characteristics (standard method)
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

    // MARK: - Stable Region Detection

    /// Represents a detected stable region
    private struct DetectedRegion {
        let startIndex: Int
        let endIndex: Int
    }

    /// Detect stable regions where pitch doesn't change significantly between adjacent frames
    private func detectStableRegions(frequencies: [Float], timeStamps: [Double]) -> [DetectedRegion] {
        guard frequencies.count >= 2 else { return [] }

        var regions: [DetectedRegion] = []
        var currentStart = 0

        for i in 1..<frequencies.count {
            let prevFreq = frequencies[i - 1]
            let currFreq = frequencies[i]

            guard prevFreq > 0 && currFreq > 0 else {
                // Invalid frequency - check if we have a valid region
                if i - 1 - currentStart >= minimumSampleCount {
                    let duration = timeStamps[i - 1] - timeStamps[currentStart]
                    if duration >= minStableDurationSeconds {
                        regions.append(DetectedRegion(startIndex: currentStart, endIndex: i - 1))
                    }
                }
                currentStart = i
                continue
            }

            let centsChange = abs(1200 * log2(Double(currFreq) / Double(prevFreq)))

            if centsChange > stabilityThresholdCents {
                // Pitch changed significantly - check if current sequence is valid
                if i - 1 - currentStart >= minimumSampleCount {
                    let duration = timeStamps[i - 1] - timeStamps[currentStart]
                    if duration >= minStableDurationSeconds {
                        regions.append(DetectedRegion(startIndex: currentStart, endIndex: i - 1))
                    }
                }
                currentStart = i
            }
        }

        // Check final sequence
        let lastIndex = frequencies.count - 1
        if lastIndex - currentStart >= minimumSampleCount {
            let duration = timeStamps[lastIndex] - timeStamps[currentStart]
            if duration >= minStableDurationSeconds {
                regions.append(DetectedRegion(startIndex: currentStart, endIndex: lastIndex))
            }
        }

        return regions
    }
}
