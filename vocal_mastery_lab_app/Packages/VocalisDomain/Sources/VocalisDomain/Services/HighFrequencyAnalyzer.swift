import Foundation

/// Analyzes spectrum data for high frequency characteristics
/// Brightness (4-6 kHz) and Airiness (6-9 kHz) bands
public final class HighFrequencyAnalyzer {

    // MARK: - Constants

    /// Brightness frequency band (Hz) - clarity and presence
    private let brightnessBandLow: Float = 4000.0
    private let brightnessBandHigh: Float = 6000.0

    /// Airiness frequency band (Hz) - breathiness and air
    private let airinessBandLow: Float = 6000.0
    private let airinessBandHigh: Float = 9000.0

    /// Analysis frequency range (Hz)
    private let analysisLow: Float = 100.0
    private let analysisHigh: Float = 10000.0

    public init() {}

    // MARK: - Public API

    /// Analyze spectrum data for high frequency characteristics
    /// - Parameters:
    ///   - spectrum: Magnitude spectrum array
    ///   - frequencyBins: Corresponding frequency values for each bin
    /// - Returns: HighFrequencyAnalysis with brightness and airiness ratios
    public func analyze(
        spectrum: [Float],
        frequencyBins: [Float]
    ) -> HighFrequencyAnalysis {
        guard spectrum.count == frequencyBins.count,
              spectrum.count > 0 else {
            return .none
        }

        // Calculate total energy for ratio calculation
        let totalEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: analysisLow,
            highFreq: analysisHigh
        )

        guard totalEnergy > 0 else {
            return .none
        }

        // Calculate brightness (4-6 kHz)
        let brightnessEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: brightnessBandLow,
            highFreq: brightnessBandHigh
        )
        let brightnessRatio = brightnessEnergy / totalEnergy

        // Calculate airiness (6-9 kHz)
        let airinessEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: airinessBandLow,
            highFreq: airinessBandHigh
        )
        let airinessRatio = airinessEnergy / totalEnergy

        return HighFrequencyAnalysis(
            brightnessRatio: brightnessRatio,
            airinessRatio: airinessRatio
        )
    }

    /// Analyze spectrogram data for average high frequency statistics
    /// - Parameter spectrogramData: Full spectrogram data from recording
    /// - Returns: HighFrequencyAnalysis averaged over all frames
    public func analyzeSpectrogram(
        spectrogramData: SpectrogramData
    ) -> HighFrequencyAnalysis {
        guard !spectrogramData.magnitudes.isEmpty else {
            return .none
        }

        var totalBrightnessRatio: Float = 0
        var totalAirinessRatio: Float = 0
        var validFrameCount = 0

        for magnitudes in spectrogramData.magnitudes {
            let frameAnalysis = analyze(
                spectrum: magnitudes,
                frequencyBins: spectrogramData.frequencyBins
            )

            if frameAnalysis.brightnessRatio > 0 || frameAnalysis.airinessRatio > 0 {
                totalBrightnessRatio += frameAnalysis.brightnessRatio
                totalAirinessRatio += frameAnalysis.airinessRatio
                validFrameCount += 1
            }
        }

        guard validFrameCount > 0 else {
            return .none
        }

        return HighFrequencyAnalysis(
            brightnessRatio: totalBrightnessRatio / Float(validFrameCount),
            airinessRatio: totalAirinessRatio / Float(validFrameCount)
        )
    }

    // MARK: - Private Methods

    /// Calculate energy in a frequency band
    private func calculateBandEnergy(
        spectrum: [Float],
        frequencyBins: [Float],
        lowFreq: Float,
        highFreq: Float
    ) -> Float {
        var energy: Float = 0

        for (index, freq) in frequencyBins.enumerated() where freq >= lowFreq && freq <= highFreq {
            // Use power (magnitude squared) for energy calculation
            energy += spectrum[index] * spectrum[index]
        }

        return energy
    }
}
