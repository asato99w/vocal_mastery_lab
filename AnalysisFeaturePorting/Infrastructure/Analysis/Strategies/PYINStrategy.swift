import Foundation
import Accelerate
import VocalisDomain

/// Probabilistic YIN (pYIN) pitch detection algorithm implementation
/// Based on: "PYIN: A fundamental frequency estimator using probabilistic threshold distributions"
/// (Mauch & Dixon, 2014)
///
/// Key improvements over YIN:
/// 1. Multiple threshold candidates instead of single threshold
/// 2. Hidden Markov Model (HMM) for temporal smoothing
/// 3. Better handling of noisy or ambiguous pitch
public final class PYINStrategy: PitchDetectionStrategy {

    // MARK: - Configuration

    public struct Configuration {
        public let bufferSize: Int
        public let hopSize: Int
        public let minFrequency: Double
        public let maxFrequency: Double
        public let silenceThreshold: Float

        // pYIN specific parameters
        public let thresholdDistribution: [Float]  // Multiple thresholds for candidate detection
        public let hmmTransitionWidth: Float       // Width of HMM transition probability in semitones
        public let voicedBias: Float               // Bias toward voiced detection (higher = more detection)

        public init(
            bufferSize: Int = 2048,
            hopSize: Int = 2205,
            minFrequency: Double = 80.0,
            maxFrequency: Double = 1200.0,
            silenceThreshold: Float = 0.0001,
            thresholdDistribution: [Float] = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            hmmTransitionWidth: Float = 12.0,
            voicedBias: Float = 0.0
        ) {
            self.bufferSize = bufferSize
            self.hopSize = hopSize
            self.minFrequency = minFrequency
            self.maxFrequency = maxFrequency
            self.silenceThreshold = silenceThreshold
            self.thresholdDistribution = thresholdDistribution
            self.hmmTransitionWidth = hmmTransitionWidth
            self.voicedBias = voicedBias
        }

        public static let `default` = Configuration()

        /// Higher detection rate - lower thresholds to catch more candidates
        public static let highDetection = Configuration(
            bufferSize: 2048,
            hopSize: 2205,
            minFrequency: 80.0,
            maxFrequency: 1200.0,
            silenceThreshold: 0.00001,
            thresholdDistribution: [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30],
            hmmTransitionWidth: 18.0,
            voicedBias: 1.5
        )

        /// Aggressive detection - maximizes detection at cost of some accuracy
        public static let aggressive = Configuration(
            bufferSize: 2048,
            hopSize: 2205,
            minFrequency: 80.0,
            maxFrequency: 1200.0,
            silenceThreshold: 0.000001,
            thresholdDistribution: [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
            hmmTransitionWidth: 24.0,
            voicedBias: 3.0
        )

        /// Balanced - better detection while maintaining reasonable accuracy
        public static let balanced = Configuration(
            bufferSize: 2048,
            hopSize: 2205,
            minFrequency: 80.0,
            maxFrequency: 1200.0,
            silenceThreshold: 0.00005,
            thresholdDistribution: [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40],
            hmmTransitionWidth: 15.0,
            voicedBias: 1.0
        )
    }

    // MARK: - Properties

    private let configuration: Configuration

    public let name: String

    /// pYIN uses HMM for temporal smoothing, doesn't need octave correction
    public let requiresOctaveCorrection: Bool = false

    // MARK: - Initialization

    public init(configuration: Configuration = .default, name: String = "pYIN") {
        self.configuration = configuration
        self.name = name
    }

    // MARK: - PitchDetectionStrategy

    public func detectPitch(samples: [Float], sampleRate: Double) -> [PitchFrame] {
        guard samples.count >= configuration.bufferSize else {
            return []
        }

        let globalMaxRms = calculateGlobalMaxRms(samples: samples, sampleRate: sampleRate)

        // Step 1: Get multiple pitch candidates per frame
        var allCandidates: [[PitchCandidate]] = []
        var position = 0

        while position + configuration.bufferSize <= samples.count {
            let chunk = Array(samples[position..<(position + configuration.bufferSize)])
            let candidates = detectPitchCandidates(chunk, globalMaxRms: globalMaxRms, sampleRate: sampleRate)
            allCandidates.append(candidates)
            position += configuration.hopSize
        }

        // Step 2: Apply Viterbi decoding for temporal smoothing
        let smoothedPitches = viterbiDecode(candidates: allCandidates, sampleRate: sampleRate)

        // Step 3: Build PitchFrame results
        var frames: [PitchFrame] = []
        for (index, pitch) in smoothedPitches.enumerated() {
            let timestamp = Double(index * configuration.hopSize) / sampleRate

            if let (freq, conf, amp) = pitch {
                frames.append(PitchFrame(
                    timestamp: timestamp,
                    frequency: freq,
                    confidence: conf,
                    amplitude: amp
                ))
            }
            // Skip unvoiced frames (no PitchFrame created)
        }

        // Fallback: If Viterbi produced poor results, use simple max-probability selection
        // This handles synthetic test signals better where HMM may not converge well
        if frames.isEmpty && !allCandidates.isEmpty {
            for (index, candidates) in allCandidates.enumerated() {
                let timestamp = Double(index * configuration.hopSize) / sampleRate
                if let best = candidates.max(by: { $0.probability < $1.probability }) {
                    frames.append(PitchFrame(
                        timestamp: timestamp,
                        frequency: best.frequency,
                        confidence: best.probability,
                        amplitude: best.amplitude
                    ))
                }
            }
        }

        return frames
    }

    // MARK: - Pitch Candidate Detection

    private struct PitchCandidate {
        let frequency: Float
        let probability: Float
        let amplitude: Float
    }

    private func detectPitchCandidates(_ samples: [Float], globalMaxRms: Float, sampleRate: Double) -> [PitchCandidate] {
        let bufferSize = samples.count
        let rms = calculateRms(samples)

        guard rms > configuration.silenceThreshold else {
            return []
        }

        let normalizedAmplitude = min(1.0, rms / globalMaxRms)

        // Calculate CMNDF (same as YIN)
        var difference = [Float](repeating: 0, count: bufferSize / 2)
        for tau in 0..<(bufferSize / 2) {
            var sum: Float = 0
            for i in 0..<(bufferSize / 2) {
                let delta = samples[i] - samples[i + tau]
                sum += delta * delta
            }
            difference[tau] = sum
        }

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

        let tauMin = Int(sampleRate / configuration.maxFrequency)
        let tauMax = Int(sampleRate / configuration.minFrequency)

        guard tauMin < tauMax && tauMax < cmndf.count else {
            return []
        }

        // Find all local minima and their probabilities at multiple thresholds
        var candidates: [PitchCandidate] = []

        for threshold in configuration.thresholdDistribution {
            var tau = tauMin
            while tau < tauMax {
                if cmndf[tau] < threshold {
                    // Look for local minimum
                    while tau + 1 < tauMax && cmndf[tau + 1] < cmndf[tau] {
                        tau += 1
                    }

                    // Parabolic interpolation for better precision
                    var betterTau = Float(tau)
                    if tau > 0 && tau < cmndf.count - 1 {
                        let s0 = cmndf[tau - 1]
                        let s1 = cmndf[tau]
                        let s2 = cmndf[tau + 1]
                        let denominator = 2 * (2 * s1 - s2 - s0)
                        if abs(denominator) > 0.0001 {
                            let adjustment = (s2 - s0) / denominator
                            betterTau = Float(tau) + adjustment
                        }
                    }

                    let frequency = Float(sampleRate) / betterTau

                    if frequency >= Float(configuration.minFrequency) && frequency <= Float(configuration.maxFrequency) {
                        // Probability is based on how far below threshold
                        let prob = (threshold - cmndf[tau]) / threshold
                        candidates.append(PitchCandidate(
                            frequency: frequency,
                            probability: max(0, min(1, prob)),
                            amplitude: normalizedAmplitude
                        ))
                    }

                    // Skip to avoid detecting same minimum at multiple thresholds
                    tau += Int(sampleRate / 2000)
                }
                tau += 1
            }
        }

        return mergeAndNormalizeCandidates(candidates)
    }

    private func mergeAndNormalizeCandidates(_ candidates: [PitchCandidate]) -> [PitchCandidate] {
        guard !candidates.isEmpty else { return [] }

        // Group candidates within 50 cents of each other
        var merged: [PitchCandidate] = []
        let sorted = candidates.sorted { $0.frequency < $1.frequency }

        var currentGroup: [PitchCandidate] = [sorted[0]]

        for i in 1..<sorted.count {
            let cents = 1200 * log2(sorted[i].frequency / currentGroup[0].frequency)
            if abs(cents) < 50 {
                currentGroup.append(sorted[i])
            } else {
                // Merge current group
                let avgFreq = currentGroup.map { $0.frequency }.reduce(0, +) / Float(currentGroup.count)
                let totalProb = currentGroup.map { $0.probability }.reduce(0, +)
                let avgAmp = currentGroup.map { $0.amplitude }.reduce(0, +) / Float(currentGroup.count)
                merged.append(PitchCandidate(frequency: avgFreq, probability: totalProb, amplitude: avgAmp))

                currentGroup = [sorted[i]]
            }
        }

        // Don't forget last group
        if !currentGroup.isEmpty {
            let avgFreq = currentGroup.map { $0.frequency }.reduce(0, +) / Float(currentGroup.count)
            let totalProb = currentGroup.map { $0.probability }.reduce(0, +)
            let avgAmp = currentGroup.map { $0.amplitude }.reduce(0, +) / Float(currentGroup.count)
            merged.append(PitchCandidate(frequency: avgFreq, probability: totalProb, amplitude: avgAmp))
        }

        // Normalize probabilities
        let totalProb = merged.map { $0.probability }.reduce(0, +)
        if totalProb > 0 {
            return merged.map {
                PitchCandidate(frequency: $0.frequency, probability: $0.probability / totalProb, amplitude: $0.amplitude)
            }
        }

        return merged
    }

    // MARK: - Viterbi Decoding (HMM)

    private func viterbiDecode(candidates: [[PitchCandidate]], sampleRate: Double) -> [(Float, Float, Float)?] {
        guard !candidates.isEmpty else { return [] }

        // State space: pitch in cents from minFreq
        let minMidi = 12 * log2(configuration.minFrequency / 440.0) + 69
        let maxMidi = 12 * log2(configuration.maxFrequency / 440.0) + 69
        let numStates = Int((maxMidi - minMidi) * 10) + 2  // 10 cents resolution + unvoiced state

        // Initialize
        var prevProb = [Float](repeating: -Float.infinity, count: numStates)
        var backpointer = [[Int]](repeating: [], count: candidates.count)

        // First frame
        if let firstCandidates = candidates.first, !firstCandidates.isEmpty {
            for candidate in firstCandidates {
                let stateIdx = midiToStateIndex(frequencyToMidi(candidate.frequency), minMidi: minMidi, numStates: numStates)
                if stateIdx >= 0 && stateIdx < numStates {
                    prevProb[stateIdx] = log(candidate.probability + 1e-10)
                }
            }
        } else {
            prevProb[numStates - 1] = 0  // Unvoiced state
        }
        backpointer[0] = Array(0..<numStates)

        // Forward pass
        for t in 1..<candidates.count {
            var currentProb = [Float](repeating: -Float.infinity, count: numStates)
            var currentBack = [Int](repeating: 0, count: numStates)

            let frameCandidates = candidates[t]

            for j in 0..<numStates {
                var bestProb: Float = -Float.infinity
                var bestPrev = 0

                for i in 0..<numStates {
                    let transitionProb = calculateTransitionProb(from: i, to: j, numStates: numStates)
                    let prob = prevProb[i] + transitionProb

                    if prob > bestProb {
                        bestProb = prob
                        bestPrev = i
                    }
                }

                // Add observation probability
                let unvoicedPenalty = max(0, 2.0 + configuration.voicedBias)

                if j == numStates - 1 {
                    // Unvoiced state
                    if frameCandidates.isEmpty {
                        currentProb[j] = bestProb - configuration.voicedBias * 0.5
                    } else {
                        currentProb[j] = bestProb - unvoicedPenalty
                    }
                } else {
                    let targetMidi = stateIndexToMidi(j, minMidi: minMidi, numStates: numStates)
                    var obsProb: Float = -Float.infinity

                    for candidate in frameCandidates {
                        let candMidi = frequencyToMidi(candidate.frequency)
                        let distance = abs(candMidi - targetMidi)
                        if distance < 0.5 {  // Within 50 cents
                            let prob = log(candidate.probability + 1e-10) - Float(distance) + configuration.voicedBias
                            obsProb = max(obsProb, prob)
                        }
                    }

                    currentProb[j] = bestProb + obsProb
                }

                currentBack[j] = bestPrev
            }

            prevProb = currentProb
            backpointer[t] = currentBack
        }

        // Backtrack
        var bestFinalState = 0
        var bestFinalProb: Float = -Float.infinity
        for i in 0..<numStates {
            if prevProb[i] > bestFinalProb {
                bestFinalProb = prevProb[i]
                bestFinalState = i
            }
        }

        var path = [Int](repeating: 0, count: candidates.count)
        path[candidates.count - 1] = bestFinalState

        for t in stride(from: candidates.count - 2, through: 0, by: -1) {
            path[t] = backpointer[t + 1][path[t + 1]]
        }

        // Convert path to frequencies
        var results: [(Float, Float, Float)?] = []
        for (t, state) in path.enumerated() {
            if state == numStates - 1 || candidates[t].isEmpty {
                results.append(nil)
            } else {
                let midi = stateIndexToMidi(state, minMidi: minMidi, numStates: numStates)
                let freq = midiToFrequency(midi)

                // Find closest candidate for confidence and amplitude
                var bestCandidate = candidates[t].first
                var minDistance: Float = Float.infinity
                for candidate in candidates[t] {
                    let distance = abs(frequencyToMidi(candidate.frequency) - midi)
                    if distance < minDistance {
                        minDistance = distance
                        bestCandidate = candidate
                    }
                }

                if let candidate = bestCandidate {
                    results.append((freq, candidate.probability, candidate.amplitude))
                } else {
                    results.append((freq, 0.5, 0.5))
                }
            }
        }

        return results
    }

    private func calculateTransitionProb(from: Int, to: Int, numStates: Int) -> Float {
        // Gaussian transition centered on same pitch
        let sigma = configuration.hmmTransitionWidth / 10.0  // In state units (10 cents each)

        if from == numStates - 1 && to == numStates - 1 {
            return 0  // Stay in unvoiced
        } else if from == numStates - 1 || to == numStates - 1 {
            return -3  // Transition to/from unvoiced has penalty
        }

        let distance = Float(abs(from - to))
        return -0.5 * (distance * distance) / (sigma * sigma)
    }

    // MARK: - Utilities

    private func frequencyToMidi(_ freq: Float) -> Float {
        return 12 * log2(freq / 440.0) + 69
    }

    private func midiToFrequency(_ midi: Float) -> Float {
        return 440.0 * pow(2, (midi - 69) / 12)
    }

    private func midiToStateIndex(_ midi: Float, minMidi: Double, numStates: Int) -> Int {
        let idx = Int((midi - Float(minMidi)) * 10)
        return max(0, min(numStates - 2, idx))
    }

    private func stateIndexToMidi(_ idx: Int, minMidi: Double, numStates: Int) -> Float {
        return Float(minMidi) + Float(idx) / 10.0
    }

    private func calculateGlobalMaxRms(samples: [Float], sampleRate: Double) -> Float {
        var maxRms: Float = 0.0001
        var position = 0

        while position + configuration.bufferSize <= samples.count {
            let chunk = Array(samples[position..<(position + configuration.bufferSize)])
            let rms = calculateRms(chunk)
            maxRms = max(maxRms, rms)
            position += configuration.hopSize
        }

        return maxRms
    }

    private func calculateRms(_ samples: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(samples, 1, &sumSquares, vDSP_Length(samples.count))
        return sqrt(sumSquares / Float(samples.count))
    }
}
