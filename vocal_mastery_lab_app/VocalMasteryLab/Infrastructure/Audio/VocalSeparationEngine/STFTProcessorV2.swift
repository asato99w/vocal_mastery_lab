import Foundation
import Accelerate

/// STFT/iSTFT implementation using vDSP_DFT (complex-to-complex DFT)
/// This version uses Apple's recommended DFT routines with librosa compatibility
final class STFTProcessorV2 {
    let fftSize: Int
    let hopSize: Int
    let window: [Float]

    private var dftSetupForward: OpaquePointer?
    private var dftSetupInverse: OpaquePointer?

    /// Number of frequency bins
    var frequencyBins: Int {
        return fftSize / 2 + 1
    }

    // MARK: - Types

    /// STFT spectrogram data
    struct SpectrogramData {
        /// Magnitude spectrogram [frequencyBins, timeFrames]
        let magnitude: [[Float]]

        /// Phase spectrogram [frequencyBins, timeFrames]
        let phase: [[Float]]

        /// Number of time frames
        var timeFrames: Int { magnitude[0].count }

        /// Number of frequency bins
        var frequencyBins: Int { magnitude.count }
    }

    init(fftSize: Int = 4096, hopSize: Int = 1024) {
        self.fftSize = fftSize
        self.hopSize = hopSize

        // Create Hann window
        var w = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&w, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        self.window = w

        // Create DFT setup for forward and inverse transforms
        self.dftSetupForward = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.FORWARD
        )

        self.dftSetupInverse = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(fftSize),
            vDSP_DFT_Direction.INVERSE
        )

        guard dftSetupForward != nil, dftSetupInverse != nil else {
            fatalError("Failed to create DFT setup")
        }
    }

    deinit {
        if let setup = dftSetupForward {
            vDSP_DFT_DestroySetup(setup)
        }
        if let setup = dftSetupInverse {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Apply reflection padding (center=True mode, librosa-compatible)
    private func reflectPad(_ audio: [Float], pad: Int) -> [Float] {
        var padded = [Float](repeating: 0, count: audio.count + 2 * pad)

        // Center: copy original audio
        for i in 0..<audio.count {
            padded[pad + i] = audio[i]
        }

        // Left: reflect
        for i in 0..<pad {
            let reflectIdx = pad - 1 - i
            padded[i] = audio[reflectIdx]
        }

        // Right: reflect
        for i in 0..<pad {
            let reflectIdx = audio.count - 1 - i
            padded[pad + audio.count + i] = audio[reflectIdx]
        }

        return padded
    }

    /// Perform STFT on input signal (librosa-compatible with center=True)
    /// Returns: (real, imag) arrays of shape [numFrames, frequencyBins]
    func stft(_ audio: [Float]) -> (real: [[Float]], imag: [[Float]]) {
        // Apply reflection padding (center=True, librosa default)
        let pad = fftSize / 2
        let paddedAudio = reflectPad(audio, pad: pad)

        let numFrames = (paddedAudio.count - fftSize) / hopSize + 1
        let frequencyBins = fftSize / 2 + 1

        var realResult: [[Float]] = []
        var imagResult: [[Float]] = []

        for frameIndex in 0..<numFrames {
            let startIndex = frameIndex * hopSize
            let endIndex = min(startIndex + fftSize, paddedAudio.count)

            // Extract frame
            var frame = [Float](repeating: 0, count: fftSize)
            let frameLength = endIndex - startIndex
            for i in 0..<frameLength {
                frame[i] = paddedAudio[startIndex + i]
            }

            // Apply window
            var windowedFrame = [Float](repeating: 0, count: fftSize)
            vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(fftSize))

            // Prepare DFT input (real input, zero imaginary)
            var realInput = windowedFrame
            var imagInput = [Float](repeating: 0, count: fftSize)
            var realOutput = [Float](repeating: 0, count: fftSize)
            var imagOutput = [Float](repeating: 0, count: fftSize)

            // Execute DFT
            vDSP_DFT_Execute(dftSetupForward!, &realInput, &imagInput, &realOutput, &imagOutput)

            // Extract positive frequencies only (0 to Nyquist)
            let realFreq = Array(realOutput[0..<frequencyBins])
            let imagFreq = Array(imagOutput[0..<frequencyBins])

            realResult.append(realFreq)
            imagResult.append(imagFreq)
        }

        return (realResult, imagResult)
    }

    /// Perform inverse STFT to reconstruct audio (librosa-compatible with center=True)
    /// Input: (real, imag) arrays of shape [numFrames, frequencyBins]
    /// Returns: reconstructed audio signal (with padding removed)
    func istft(real: [[Float]], imag: [[Float]], originalLength: Int? = nil) -> [Float] {
        guard real.count == imag.count else {
            fatalError("Real and imaginary arrays must have same number of frames")
        }

        let numFrames = real.count
        let outputLength = (numFrames - 1) * hopSize + fftSize

        var output = [Float](repeating: 0, count: outputLength)

        // 1) Calculate Wss[n] = sum_k w^2[n - kH] (librosa-compatible normalization)
        var windowSumSquared = [Float](repeating: 0, count: outputLength)
        for frameIndex in 0..<numFrames {
            let startIndex = frameIndex * hopSize
            for i in 0..<fftSize {
                if startIndex + i < outputLength {
                    windowSumSquared[startIndex + i] += window[i] * window[i]
                }
            }
        }

        // 2) Inverse transform + OLA with synthesis window g[n] = w[n] / Wss[n]
        for frameIndex in 0..<numFrames {
            // Reconstruct full spectrum (mirror negative frequencies for real signal)
            var realFull = [Float](repeating: 0, count: fftSize)
            var imagFull = [Float](repeating: 0, count: fftSize)

            let frequencyBins = real[frameIndex].count

            // Positive frequencies (0 to Nyquist)
            for i in 0..<frequencyBins {
                realFull[i] = real[frameIndex][i]
                imagFull[i] = imag[frameIndex][i]
            }

            // Negative frequencies (mirror conjugate for real signal)
            for i in 1..<(frequencyBins - 1) {
                let mirrorIndex = fftSize - i
                realFull[mirrorIndex] = realFull[i]
                imagFull[mirrorIndex] = -imagFull[i]  // Conjugate
            }

            // Execute inverse DFT
            var realOutput = [Float](repeating: 0, count: fftSize)
            var imagOutput = [Float](repeating: 0, count: fftSize)

            vDSP_DFT_Execute(dftSetupInverse!, &realFull, &imagFull, &realOutput, &imagOutput)

            // Apply inverse DFT scaling (1/N)
            var scale = 1.0 / Float(fftSize)
            vDSP_vsmul(realOutput, 1, &scale, &realOutput, 1, vDSP_Length(fftSize))

            // Overlap-add with synthesis window g[n] = w[n] / Wss[n]
            let startIndex = frameIndex * hopSize
            for i in 0..<fftSize {
                let outputIdx = startIndex + i
                if outputIdx < outputLength {
                    let wss = windowSumSquared[outputIdx]
                    let synthWindow = wss > 1e-8 ? window[i] / wss : 0.0
                    output[outputIdx] += realOutput[i] * synthWindow
                }
            }
        }

        // 3) Remove reflection padding (center=True, librosa default)
        let pad = fftSize / 2
        if let originalLength = originalLength {
            let trimStart = pad
            let trimEnd = trimStart + originalLength
            return Array(output[trimStart..<min(trimEnd, output.count)])
        } else {
            let trimStart = pad
            let trimEnd = output.count - pad
            if trimEnd > trimStart {
                return Array(output[trimStart..<trimEnd])
            } else {
                return output
            }
        }
    }

    // MARK: - Compatibility Layer

    /// Compute STFT and return as SpectrogramData
    func computeSTFT(audio: [Float]) -> SpectrogramData {
        let (real, imag) = stft(audio)

        let frequencyBins = real[0].count
        let timeFrames = real.count

        var magnitude: [[Float]] = Array(repeating: Array(repeating: 0, count: timeFrames), count: frequencyBins)
        var phase: [[Float]] = Array(repeating: Array(repeating: 0, count: timeFrames), count: frequencyBins)

        for frameIdx in 0..<timeFrames {
            for binIdx in 0..<frequencyBins {
                let re = real[frameIdx][binIdx]
                let im = imag[frameIdx][binIdx]
                magnitude[binIdx][frameIdx] = sqrtf(re * re + im * im)
                phase[binIdx][frameIdx] = atan2f(im, re)
            }
        }

        return SpectrogramData(magnitude: magnitude, phase: phase)
    }

    /// Compute STFT for stereo audio
    func computeSTFT(audioData: AudioProcessor.AudioData) -> (left: SpectrogramData, right: SpectrogramData) {
        let leftSTFT = computeSTFT(audio: audioData.samples[0])
        let rightSTFT = audioData.channelCount > 1 ?
            computeSTFT(audio: audioData.samples[1]) :
            leftSTFT

        return (leftSTFT, rightSTFT)
    }

    /// Create AudioData from iSTFT results
    func createAudioData(
        leftMagnitude: [[Float]],
        leftPhase: [[Float]],
        rightMagnitude: [[Float]],
        rightPhase: [[Float]],
        sampleRate: Double
    ) -> AudioProcessor.AudioData {
        let leftAudio = computeISTFT(magnitude: leftMagnitude, phase: leftPhase)
        let rightAudio = computeISTFT(magnitude: rightMagnitude, phase: rightPhase)

        return AudioProcessor.AudioData(
            samples: [leftAudio, rightAudio],
            sampleRate: sampleRate,
            frameCount: leftAudio.count
        )
    }

    /// Compute iSTFT from magnitude and phase
    private func computeISTFT(magnitude: [[Float]], phase: [[Float]]) -> [Float] {
        let frequencyBins = magnitude.count
        let timeFrames = magnitude[0].count

        var real: [[Float]] = []
        var imag: [[Float]] = []

        for frameIdx in 0..<timeFrames {
            var realFrame = [Float](repeating: 0, count: frequencyBins)
            var imagFrame = [Float](repeating: 0, count: frequencyBins)

            for binIdx in 0..<frequencyBins {
                let mag = magnitude[binIdx][frameIdx]
                let phi = phase[binIdx][frameIdx]
                realFrame[binIdx] = mag * cosf(phi)
                imagFrame[binIdx] = mag * sinf(phi)
            }

            real.append(realFrame)
            imag.append(imagFrame)
        }

        return istft(real: real, imag: imag)
    }
}
