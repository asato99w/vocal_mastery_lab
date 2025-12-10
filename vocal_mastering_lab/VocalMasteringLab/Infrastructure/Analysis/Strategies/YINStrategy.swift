import Foundation
import VocalisDomain

/// YIN pitch detection algorithm implementation
/// Based on: "YIN, a fundamental frequency estimator for speech and music" (de Cheveigné & Kawahara, 2002)
public final class YINStrategy: PitchDetectionStrategy {

    // MARK: - Configuration

    public struct Configuration {
        public let bufferSize: Int
        public let hopSize: Int
        public let threshold: Float
        public let minFrequency: Double
        public let maxFrequency: Double
        public let silenceThreshold: Float

        public init(
            bufferSize: Int = 2048,
            hopSize: Int = 2205,  // 50ms at 44100Hz
            threshold: Float = 0.25,
            minFrequency: Double = 80.0,
            maxFrequency: Double = 1200.0,
            silenceThreshold: Float = 0.0001
        ) {
            self.bufferSize = bufferSize
            self.hopSize = hopSize
            self.threshold = threshold
            self.minFrequency = minFrequency
            self.maxFrequency = maxFrequency
            self.silenceThreshold = silenceThreshold
        }

        public static let `default` = Configuration()
    }

    // MARK: - Properties

    private let configuration: Configuration

    public let name: String = "YIN"

    /// YIN is prone to octave errors, requires post-processing correction
    public let requiresOctaveCorrection: Bool = true

    // MARK: - Initialization

    public init(configuration: Configuration = .default) {
        self.configuration = configuration
    }

    // MARK: - PitchDetectionStrategy

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        guard samples.count >= configuration.bufferSize else {
            return []
        }

        // Calculate global max RMS for normalization
        let globalMaxRms = calculateGlobalMaxRms(samples: samples, sampleRate: sampleRate)

        var frames: [PitchFrame] = []
        var position = 0

        while position + configuration.bufferSize <= samples.count {
            let timestamp = Double(position) / sampleRate
            let chunk = Array(samples[position..<(position + configuration.bufferSize)])

            if let (frequency, confidence, amplitude) = detectPitchUsingYIN(
                chunk,
                sampleRate: sampleRate,
                globalMaxRms: globalMaxRms
            ) {
                frames.append(PitchFrame(
                    timestamp: timestamp,
                    frequency: frequency,
                    confidence: confidence,
                    amplitude: amplitude
                ))
            }

            position += configuration.hopSize
        }

        return frames
    }

    // MARK: - Private Methods

    /// Calculate global maximum RMS for amplitude normalization
    private func calculateGlobalMaxRms(samples: [Float], sampleRate: Double) -> Float {
        var maxRms: Float = 0.0001  // Minimum to avoid division by zero
        var position = 0

        while position + configuration.bufferSize <= samples.count {
            let chunk = Array(samples[position..<(position + configuration.bufferSize)])
            let rms = sqrt(chunk.map { $0 * $0 }.reduce(0, +) / Float(chunk.count))
            maxRms = max(maxRms, rms)
            position += configuration.hopSize
        }

        return maxRms
    }

    /// YIN algorithm for pitch detection
    /// Returns (frequency, confidence, normalizedAmplitude) or nil if no pitch detected
    private func detectPitchUsingYIN(
        _ samples: [Float],
        sampleRate: Double,
        globalMaxRms: Float
    ) -> (frequency: Float, confidence: Float, amplitude: Float)? {
        let bufferSize = samples.count

        // Calculate RMS amplitude
        let rms = sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(samples.count))

        // Silence threshold
        guard rms > configuration.silenceThreshold else {
            return nil
        }

        // Normalize amplitude
        let normalizedAmplitude = min(1.0, rms / globalMaxRms)

        // Step 1: Calculate difference function
        var difference = [Float](repeating: 0, count: bufferSize / 2)
        for tau in 0..<(bufferSize / 2) {
            var sum: Float = 0
            for i in 0..<(bufferSize / 2) {
                let delta = samples[i] - samples[i + tau]
                sum += delta * delta
            }
            difference[tau] = sum
        }

        // Step 2: Calculate cumulative mean normalized difference function
        var cmndf = [Float](repeating: 0, count: bufferSize / 2)
        cmndf[0] = 1.0

        var runningSum: Float = 0
        for tau in 1..<(bufferSize / 2) {
            runningSum += difference[tau]
            if runningSum > 0 {
                cmndf[tau] = difference[tau] / (runningSum / Float(tau))
            } else {
                cmndf[tau] = 1.0
            }
        }

        // Step 3: Absolute threshold
        let tauMin = Int(sampleRate / configuration.maxFrequency)
        let tauMax = Int(sampleRate / configuration.minFrequency)

        guard tauMin < tauMax && tauMax < cmndf.count else {
            return nil
        }

        // Find first tau where CMNDF drops below threshold
        var tau = tauMin
        while tau < tauMax {
            if cmndf[tau] < configuration.threshold {
                // Found a candidate, look for local minimum
                while tau + 1 < tauMax && cmndf[tau + 1] < cmndf[tau] {
                    tau += 1
                }
                break
            }
            tau += 1
        }

        // No pitch found
        guard tau < tauMax && cmndf[tau] < configuration.threshold else {
            return nil
        }

        // Step 4: Parabolic interpolation for better precision
        var betterTau = Float(tau)
        if tau > 0 && tau < cmndf.count - 1 {
            let s0 = cmndf[tau - 1]
            let s1 = cmndf[tau]
            let s2 = cmndf[tau + 1]
            let adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
            betterTau = Float(tau) + adjustment
        }

        // Convert period to frequency
        let frequency = Float(sampleRate) / betterTau

        // Confidence is inverse of CMNDF value
        let confidence = 1.0 - min(cmndf[tau], 1.0)

        // Validate frequency range
        guard frequency >= Float(configuration.minFrequency) &&
              frequency <= Float(configuration.maxFrequency) else {
            return nil
        }

        return (frequency, confidence, normalizedAmplitude)
    }
}
