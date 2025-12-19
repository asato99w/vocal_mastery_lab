import Foundation

/// Calculator for objective recording statistics
/// Analyzes pitch data to produce deviation metrics
public final class RecordingStatisticsCalculator {
    /// Minimum confidence for including a sample in analysis
    private let minConfidence: Float

    /// Minimum confidence for vibrato analysis (may differ from general minConfidence)
    private let vibratoMinConfidence: Float

    /// Vibrato analyzer for detecting vibrato characteristics
    private let vibratoAnalyzer: VibratoAnalyzer

    /// Singer's Formant analyzer for detecting SF presence
    private let singersFormantAnalyzer: SingersFormantAnalyzer

    /// High frequency analyzer for brightness and airiness
    private let highFrequencyAnalyzer: HighFrequencyAnalyzer

    /// Minimum samples required for vibrato analysis per segment
    private let minVibratoSamples = 10

    /// Initialize with algorithm-specific parameters
    /// - Parameter algorithm: Pitch detection algorithm used (determines vibrato detection thresholds)
    public init(algorithm: PitchDetectionAlgorithm) {
        self.minConfidence = 0.5
        self.vibratoMinConfidence = algorithm.vibratoMinConfidence
        self.vibratoAnalyzer = VibratoAnalyzer(minimumRegularity: algorithm.vibratoMinRegularity)
        self.singersFormantAnalyzer = SingersFormantAnalyzer()
        self.highFrequencyAnalyzer = HighFrequencyAnalyzer()
    }

    /// Legacy initializer for backward compatibility
    public init(minConfidence: Float = 0.5) {
        self.minConfidence = minConfidence
        self.vibratoMinConfidence = minConfidence
        self.vibratoAnalyzer = VibratoAnalyzer()
        self.singersFormantAnalyzer = SingersFormantAnalyzer()
        self.highFrequencyAnalyzer = HighFrequencyAnalyzer()
    }

    /// Calculate statistics from pitch data
    /// - Parameters:
    ///   - pitchData: Detected pitch analysis data
    /// - Returns: Calculated statistics or nil if insufficient data
    public func calculate(
        pitchData: PitchAnalysisData
    ) -> RecordingStatistics? {
        calculate(
            pitchData: pitchData,
            spectrogramData: nil
        )
    }

    /// Calculate statistics from pitch data and spectrogram
    /// - Parameters:
    ///   - pitchData: Detected pitch analysis data
    ///   - spectrogramData: Optional spectrogram for Singer's Formant analysis
    /// - Returns: Calculated statistics or nil if insufficient data
    public func calculate(
        pitchData: PitchAnalysisData,
        spectrogramData: SpectrogramData?
    ) -> RecordingStatistics? {
        let totalDuration = pitchData.timeStamps.last ?? 0

        guard !pitchData.timeStamps.isEmpty else { return nil }

        // Calculate overall statistics
        let overallStats = calculateOverallStatistics(pitchData: pitchData)

        // Calculate vibrato statistics
        let vibratoStats = calculateVibratoStatistics(pitchData: pitchData)

        // Calculate Singer's Formant statistics (if spectrogram available)
        let sfStats = calculateSingersFormantStatistics(
            spectrogramData: spectrogramData,
            pitchData: pitchData
        )

        // Calculate high frequency statistics (Brightness and Airiness)
        let hfStats = calculateHighFrequencyStatistics(spectrogramData: spectrogramData)

        return RecordingStatistics(
            overall: overallStats,
            positionStatistics: [],
            pitchStatistics: [],
            vibratoStatistics: vibratoStats,
            singersFormantStatistics: sfStats,
            highFrequencyStatistics: hfStats,
            totalDuration: totalDuration
        )
    }

    // MARK: - Private Methods

    private func calculateOverallStatistics(
        pitchData: PitchAnalysisData
    ) -> RecordingStatistics.OverallStatistics {
        var validFrequencies: [Double] = []

        for (index, _) in pitchData.timeStamps.enumerated() {
            let confidence = pitchData.confidences[index]
            guard confidence >= minConfidence else { continue }

            let frequency = Double(pitchData.frequencies[index])

            // Track all valid frequencies for range
            if frequency > 50 && frequency < 2000 {
                validFrequencies.append(frequency)
            }
        }

        let totalSamples = validFrequencies.count

        return RecordingStatistics.OverallStatistics(
            averageDeviationCents: 0, // No target to compare against
            deviationStdDev: 0,
            medianDeviationCents: 0,
            detectionRate: totalSamples > 0 ? 1.0 : 0,
            totalSamples: totalSamples,
            lowestFrequency: validFrequencies.min(),
            highestFrequency: validFrequencies.max()
        )
    }

    private func calculateVibratoStatistics(
        pitchData: PitchAnalysisData
    ) -> RecordingStatistics.VibratoStatistics? {
        var vibratoAnalyses: [VibratoAnalysis] = []

        // Always analyze entire pitch data as one unit for vibrato detection
        // Vibrato is a continuous characteristic that spans across notes
        var allFrequencies: [Float] = []
        var allTimeStamps: [Double] = []

        for (index, _) in pitchData.timeStamps.enumerated() {
            // Use vibratoMinConfidence for vibrato analysis (may be lower than general minConfidence)
            guard pitchData.confidences[index] >= vibratoMinConfidence else { continue }
            let frequency = pitchData.frequencies[index]
            // Filter out invalid frequencies
            if frequency > 50 && frequency < 2000 {
                allFrequencies.append(frequency)
                allTimeStamps.append(pitchData.timeStamps[index])
            }
        }

        if allFrequencies.count >= minVibratoSamples {
            let analysis = vibratoAnalyzer.analyze(
                frequencies: allFrequencies,
                timeStamps: allTimeStamps
            )
            vibratoAnalyses.append(analysis)
        }

        guard !vibratoAnalyses.isEmpty else { return nil }

        // Calculate aggregate statistics
        let presentAnalyses = vibratoAnalyses.filter { $0.isPresent }
        let presenceRate = Float(presentAnalyses.count) / Float(vibratoAnalyses.count)

        // Calculate averages only from segments where vibrato was detected
        let averageRate: Float
        let averageExtent: Float
        let averageRegularity: Float

        if presentAnalyses.isEmpty {
            averageRate = 0
            averageExtent = 0
            averageRegularity = 0
        } else {
            averageRate = presentAnalyses.map { $0.rate }.reduce(0, +) / Float(presentAnalyses.count)
            averageExtent = presentAnalyses.map { $0.extent }.reduce(0, +) / Float(presentAnalyses.count)
            averageRegularity = presentAnalyses.map { $0.regularity }.reduce(0, +) / Float(presentAnalyses.count)
        }

        return RecordingStatistics.VibratoStatistics(
            averageRate: averageRate,
            averageExtent: averageExtent,
            averageRegularity: averageRegularity,
            presenceRate: presenceRate,
            segmentsAnalyzed: vibratoAnalyses.count
        )
    }

    private func calculateSingersFormantStatistics(
        spectrogramData: SpectrogramData?,
        pitchData: PitchAnalysisData
    ) -> RecordingStatistics.SingersFormantStatistics? {
        guard let spectrogram = spectrogramData,
              !spectrogram.magnitudes.isEmpty else {
            return nil
        }

        // Calculate average pitch for confidence adjustment
        let validFrequencies = pitchData.frequencies.enumerated()
            .filter { pitchData.confidences[$0.offset] >= minConfidence }
            .map { $0.element }

        let averagePitch: Float? = validFrequencies.isEmpty
            ? nil
            : validFrequencies.reduce(0, +) / Float(validFrequencies.count)

        // Analyze spectrogram
        let analysis = singersFormantAnalyzer.analyzeSpectrogram(
            spectrogramData: spectrogram,
            averagePitch: averagePitch
        )

        // Return nil if no valid analysis
        guard analysis.ratio > 0 else { return nil }

        return RecordingStatistics.SingersFormantStatistics(
            averageRatio: analysis.ratio,
            averageIntensity: analysis.intensity,
            isPresent: analysis.isPresent,
            confidence: analysis.confidence
        )
    }

    private func calculateHighFrequencyStatistics(
        spectrogramData: SpectrogramData?
    ) -> RecordingStatistics.HighFrequencyStatistics? {
        guard let spectrogram = spectrogramData,
              !spectrogram.magnitudes.isEmpty else {
            return nil
        }

        // Analyze spectrogram for brightness and airiness
        let analysis = highFrequencyAnalyzer.analyzeSpectrogram(spectrogramData: spectrogram)

        // Return nil if no valid analysis
        guard analysis.brightnessRatio > 0 || analysis.airinessRatio > 0 else { return nil }

        return RecordingStatistics.HighFrequencyStatistics(
            brightnessRatio: analysis.brightnessRatio,
            airinessRatio: analysis.airinessRatio
        )
    }
}
