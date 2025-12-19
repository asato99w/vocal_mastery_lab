import Foundation

/// Analyzes spectrum data for Singer's Formant presence
/// Singer's Formant is an energy concentration in 2500-3500 Hz band
/// characteristic of trained classical singers
public final class SingersFormantAnalyzer {

    // MARK: - Constants

    /// Singer's Formant frequency band (Hz)
    private let sfBandLow: Float = 2500.0
    private let sfBandHigh: Float = 3500.0

    /// Surrounding band for comparison (Hz)
    private let surroundingLow: Float = 1500.0
    private let surroundingHigh: Float = 4500.0

    /// Analysis frequency range (Hz)
    private let analysisLow: Float = 100.0
    private let analysisHigh: Float = 8000.0

    /// Minimum ratio threshold for detection
    private let minRatioThreshold: Float = 0.08

    /// Minimum intensity threshold for detection (dB)
    private let minIntensityThreshold: Float = 3.0

    /// High pitch threshold for confidence reduction (Hz)
    private let highPitchThreshold: Float = 440.0

    public init() {}

    // MARK: - Public API

    /// Analyze spectrum data for singer's formant presence
    /// - Parameters:
    ///   - spectrum: Magnitude spectrum array
    ///   - frequencyBins: Corresponding frequency values for each bin
    ///   - fundamentalFrequency: Optional fundamental frequency for confidence adjustment
    /// - Returns: SingersFormantAnalysis with ratio, intensity, and detection status
    public func analyze(
        spectrum: [Float],
        frequencyBins: [Float],
        fundamentalFrequency: Float? = nil
    ) -> SingersFormantAnalysis {
        guard spectrum.count == frequencyBins.count,
              spectrum.count > 0 else {
            return .none
        }

        // Calculate band energies
        let sfEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: sfBandLow,
            highFreq: sfBandHigh
        )

        let totalEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: analysisLow,
            highFreq: analysisHigh
        )

        let surroundingEnergy = calculateSurroundingEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins
        )

        // Calculate ratio
        let ratio: Float = totalEnergy > 0 ? sfEnergy / totalEnergy : 0

        // Calculate intensity (dB difference from surrounding)
        let intensity: Float
        if surroundingEnergy > 0 {
            intensity = 10 * log10(sfEnergy / surroundingEnergy)
        } else {
            intensity = 0
        }

        // Calculate confidence based on pitch
        let confidence = calculateConfidence(fundamentalFrequency: fundamentalFrequency)

        // Determine if SF is present
        let isPresent = ratio > minRatioThreshold && intensity > minIntensityThreshold

        return SingersFormantAnalysis(
            ratio: ratio,
            intensity: intensity,
            isPresent: isPresent,
            confidence: confidence
        )
    }

    /// Analyze spectrogram data for average singer's formant statistics
    /// - Parameters:
    ///   - spectrogramData: Full spectrogram data from recording
    ///   - averagePitch: Optional average pitch for confidence calculation
    /// - Returns: SingersFormantAnalysis averaged over all frames
    public func analyzeSpectrogram(
        spectrogramData: SpectrogramData,
        averagePitch: Float? = nil
    ) -> SingersFormantAnalysis {
        guard !spectrogramData.magnitudes.isEmpty else {
            return .none
        }

        var totalRatio: Float = 0
        var totalIntensity: Float = 0
        var presentCount = 0
        var validFrameCount = 0

        for magnitudes in spectrogramData.magnitudes {
            let frameAnalysis = analyze(
                spectrum: magnitudes,
                frequencyBins: spectrogramData.frequencyBins,
                fundamentalFrequency: averagePitch
            )

            if frameAnalysis.ratio > 0 {
                totalRatio += frameAnalysis.ratio
                totalIntensity += frameAnalysis.intensity
                validFrameCount += 1

                if frameAnalysis.isPresent {
                    presentCount += 1
                }
            }
        }

        guard validFrameCount > 0 else {
            return .none
        }

        let avgRatio = totalRatio / Float(validFrameCount)
        let avgIntensity = totalIntensity / Float(validFrameCount)
        let presenceRate = Float(presentCount) / Float(validFrameCount)
        let confidence = calculateConfidence(fundamentalFrequency: averagePitch)

        // Consider present if detected in more than 30% of frames
        let isPresent = presenceRate > 0.3 && avgRatio > minRatioThreshold

        return SingersFormantAnalysis(
            ratio: avgRatio,
            intensity: avgIntensity,
            isPresent: isPresent,
            confidence: confidence
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

    /// Calculate energy in surrounding bands (excluding SF band)
    private func calculateSurroundingEnergy(
        spectrum: [Float],
        frequencyBins: [Float]
    ) -> Float {
        let lowerBandEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: surroundingLow,
            highFreq: sfBandLow
        )

        let upperBandEnergy = calculateBandEnergy(
            spectrum: spectrum,
            frequencyBins: frequencyBins,
            lowFreq: sfBandHigh,
            highFreq: surroundingHigh
        )

        // Average of surrounding bands
        return (lowerBandEnergy + upperBandEnergy) / 2.0
    }

    /// Calculate detection confidence based on pitch
    /// Higher pitches have lower confidence due to sparse harmonics
    private func calculateConfidence(fundamentalFrequency: Float?) -> Float {
        guard let pitch = fundamentalFrequency else {
            return 1.0  // Default to high confidence if pitch unknown
        }

        if pitch < highPitchThreshold {
            return 1.0
        } else {
            // Reduce confidence linearly as pitch increases above threshold
            // At 440Hz: 1.0, at 880Hz: 0.5, at 940Hz: ~0.4
            return max(0.3, 1.0 - (pitch - highPitchThreshold) / 1000.0)
        }
    }
}
