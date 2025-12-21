import XCTest
@testable import VocalMasteryLab

final class STFTProcessorV2Tests: XCTestCase {

    var sut: STFTProcessorV2!

    override func setUp() {
        super.setUp()
        sut = STFTProcessorV2(fftSize: 4096, hopSize: 1024)
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInit_withDefaultParameters() {
        let processor = STFTProcessorV2()
        XCTAssertEqual(processor.fftSize, 4096)
        XCTAssertEqual(processor.hopSize, 1024)
        XCTAssertEqual(processor.frequencyBins, 2049) // fftSize/2 + 1
    }

    func testInit_withCustomParameters() {
        let processor = STFTProcessorV2(fftSize: 2048, hopSize: 512)
        XCTAssertEqual(processor.fftSize, 2048)
        XCTAssertEqual(processor.hopSize, 512)
        XCTAssertEqual(processor.frequencyBins, 1025)
    }

    // MARK: - STFT Tests

    func testSTFT_withSineWave_producesExpectedShape() {
        // Generate 1 second of 440Hz sine wave at 44100 Hz
        let sampleRate = 44100.0
        let frequency = 440.0
        let duration = 1.0
        let audio = generateSineWave(frequency: frequency, sampleRate: sampleRate, duration: duration)

        let (real, imag) = sut.stft(audio)

        // Check output dimensions
        let expectedFrames = (audio.count + sut.fftSize - 1) / sut.hopSize
        XCTAssertGreaterThan(real.count, 0)
        XCTAssertEqual(real.count, imag.count)
        XCTAssertEqual(real[0].count, sut.frequencyBins)
    }

    func testSTFT_withSilence_producesNearZeroOutput() {
        let audio = [Float](repeating: 0, count: 44100)

        let (real, imag) = sut.stft(audio)

        // All values should be near zero
        for frame in real {
            for value in frame {
                XCTAssertLessThan(abs(value), 1e-6)
            }
        }
        for frame in imag {
            for value in frame {
                XCTAssertLessThan(abs(value), 1e-6)
            }
        }
    }

    // MARK: - iSTFT Tests

    func testISTFT_afterSTFT_reconstructsSignal() {
        // Generate test signal
        let sampleRate = 44100.0
        let audio = generateSineWave(frequency: 440.0, sampleRate: sampleRate, duration: 0.5)

        // Forward STFT
        let (real, imag) = sut.stft(audio)

        // Inverse STFT
        let reconstructed = sut.istft(real: real, imag: imag)

        // Compare lengths (may differ slightly due to padding)
        let minLength = min(audio.count, reconstructed.count)
        XCTAssertGreaterThan(minLength, 0)

        // Calculate correlation coefficient
        let correlation = calculateCorrelation(
            Array(audio[0..<minLength]),
            Array(reconstructed[0..<minLength])
        )

        // Should have high correlation (> 0.99 for perfect reconstruction)
        XCTAssertGreaterThan(correlation, 0.99, "Reconstruction correlation should be > 0.99")
    }

    func testISTFT_withOriginalLength_matchesInputLength() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)
        let originalLength = audio.count

        let (real, imag) = sut.stft(audio)
        let reconstructed = sut.istft(real: real, imag: imag, originalLength: originalLength)

        XCTAssertEqual(reconstructed.count, originalLength)
    }

    // MARK: - ComputeSTFT Tests (Compatibility Layer)

    func testComputeSTFT_returnsSpectrogramData() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)

        let spectrogram = sut.computeSTFT(audio: audio)

        XCTAssertGreaterThan(spectrogram.timeFrames, 0)
        XCTAssertEqual(spectrogram.frequencyBins, sut.frequencyBins)
        XCTAssertEqual(spectrogram.magnitude.count, spectrogram.frequencyBins)
        XCTAssertEqual(spectrogram.phase.count, spectrogram.frequencyBins)
    }

    func testComputeSTFT_magnitudeIsNonNegative() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)

        let spectrogram = sut.computeSTFT(audio: audio)

        for bin in spectrogram.magnitude {
            for value in bin {
                XCTAssertGreaterThanOrEqual(value, 0, "Magnitude should be non-negative")
            }
        }
    }

    func testComputeSTFT_sineWave_hasPeakAtExpectedFrequency() {
        let sampleRate = 44100.0
        let frequency = 1000.0 // 1kHz
        let audio = generateSineWave(frequency: frequency, sampleRate: sampleRate, duration: 0.5)

        let spectrogram = sut.computeSTFT(audio: audio)

        // Find the expected bin for 1kHz
        let binFrequency = sampleRate / Double(sut.fftSize)
        let expectedBin = Int(frequency / binFrequency)

        // Get average magnitude across time for each frequency bin
        var avgMagnitudes = [Float](repeating: 0, count: spectrogram.frequencyBins)
        for binIdx in 0..<spectrogram.frequencyBins {
            let sum = spectrogram.magnitude[binIdx].reduce(0, +)
            avgMagnitudes[binIdx] = sum / Float(spectrogram.timeFrames)
        }

        // Find the peak bin
        var maxBin = 0
        var maxValue: Float = 0
        for (idx, value) in avgMagnitudes.enumerated() {
            if value > maxValue {
                maxValue = value
                maxBin = idx
            }
        }

        // Peak should be near the expected frequency bin (within 2 bins)
        XCTAssertLessThanOrEqual(abs(maxBin - expectedBin), 2,
                                  "Peak at bin \(maxBin) should be near expected bin \(expectedBin)")
    }

    // MARK: - Stereo Audio Tests

    func testComputeSTFT_withStereoAudioData_processesBothChannels() {
        let sampleRate = 44100.0
        let leftChannel = generateSineWave(frequency: 440.0, sampleRate: sampleRate, duration: 0.5)
        let rightChannel = generateSineWave(frequency: 880.0, sampleRate: sampleRate, duration: 0.5)

        let audioData = AudioProcessor.AudioData(
            samples: [leftChannel, rightChannel],
            sampleRate: sampleRate,
            frameCount: leftChannel.count
        )

        let (leftSTFT, rightSTFT) = sut.computeSTFT(audioData: audioData)

        XCTAssertEqual(leftSTFT.timeFrames, rightSTFT.timeFrames)
        XCTAssertEqual(leftSTFT.frequencyBins, rightSTFT.frequencyBins)
    }

    // MARK: - ComputeComplexSTFT Tests

    func testComputeComplexSTFT_returnsCorrectShape() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)

        let complexSTFT = sut.computeComplexSTFT(audio: audio)

        XCTAssertGreaterThan(complexSTFT.timeFrames, 0)
        XCTAssertEqual(complexSTFT.frequencyBins, sut.frequencyBins)
        XCTAssertEqual(complexSTFT.real.count, complexSTFT.frequencyBins)
        XCTAssertEqual(complexSTFT.imag.count, complexSTFT.frequencyBins)
    }

    func testComputeComplexSTFT_realImagMatchMagnitudePhase() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)

        let complexSTFT = sut.computeComplexSTFT(audio: audio)
        let spectrogram = sut.computeSTFT(audio: audio)

        // Verify: magnitude = sqrt(real² + imag²)
        // Verify: phase = atan2(imag, real)
        for binIdx in 0..<min(10, complexSTFT.frequencyBins) {
            for frameIdx in 0..<min(10, complexSTFT.timeFrames) {
                let re = complexSTFT.real[binIdx][frameIdx]
                let im = complexSTFT.imag[binIdx][frameIdx]

                let calculatedMag = sqrtf(re * re + im * im)
                let calculatedPhase = atan2f(im, re)

                let expectedMag = spectrogram.magnitude[binIdx][frameIdx]
                let expectedPhase = spectrogram.phase[binIdx][frameIdx]

                XCTAssertEqual(calculatedMag, expectedMag, accuracy: 1e-5,
                              "Magnitude mismatch at [\(binIdx)][\(frameIdx)]")
                XCTAssertEqual(calculatedPhase, expectedPhase, accuracy: 1e-5,
                              "Phase mismatch at [\(binIdx)][\(frameIdx)]")
            }
        }
    }

    func testComputeComplexSTFT_withStereoAudioData_processesBothChannels() {
        let sampleRate = 44100.0
        let leftChannel = generateSineWave(frequency: 440.0, sampleRate: sampleRate, duration: 0.5)
        let rightChannel = generateSineWave(frequency: 880.0, sampleRate: sampleRate, duration: 0.5)

        let audioData = AudioProcessor.AudioData(
            samples: [leftChannel, rightChannel],
            sampleRate: sampleRate,
            frameCount: leftChannel.count
        )

        let (leftSTFT, rightSTFT) = sut.computeComplexSTFT(audioData: audioData)

        XCTAssertEqual(leftSTFT.timeFrames, rightSTFT.timeFrames)
        XCTAssertEqual(leftSTFT.frequencyBins, rightSTFT.frequencyBins)
        XCTAssertEqual(leftSTFT.real.count, leftSTFT.frequencyBins)
        XCTAssertEqual(rightSTFT.imag.count, rightSTFT.frequencyBins)
    }

    func testComputeComplexSTFT_sineWave_hasNonZeroImaginary() {
        // A sine wave should have non-zero imaginary components
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)

        let complexSTFT = sut.computeComplexSTFT(audio: audio)

        // Find max imaginary value - should be non-zero for a sine wave
        var maxImag: Float = 0
        for bin in complexSTFT.imag {
            for value in bin {
                maxImag = max(maxImag, abs(value))
            }
        }

        XCTAssertGreaterThan(maxImag, 0.1, "Sine wave should have significant imaginary components")
    }

    // MARK: - CreateAudioData Tests

    func testCreateAudioData_producesValidOutput() {
        let audio = generateSineWave(frequency: 440.0, sampleRate: 44100.0, duration: 0.5)
        let spectrogram = sut.computeSTFT(audio: audio)

        let result = sut.createAudioData(
            leftMagnitude: spectrogram.magnitude,
            leftPhase: spectrogram.phase,
            rightMagnitude: spectrogram.magnitude,
            rightPhase: spectrogram.phase,
            sampleRate: 44100.0
        )

        XCTAssertEqual(result.channelCount, 2)
        XCTAssertEqual(result.sampleRate, 44100.0)
        XCTAssertGreaterThan(result.frameCount, 0)
    }

    // MARK: - Helper Methods

    private func generateSineWave(frequency: Double, sampleRate: Double, duration: Double) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for i in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(i) / sampleRate
            samples[i] = Float(sin(phase))
        }

        return samples
    }

    private func calculateCorrelation(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        let n = Double(a.count)
        let sumA = a.reduce(0) { $0 + Double($1) }
        let sumB = b.reduce(0) { $0 + Double($1) }
        let sumAB = zip(a, b).reduce(0) { $0 + Double($1.0) * Double($1.1) }
        let sumA2 = a.reduce(0) { $0 + Double($1) * Double($1) }
        let sumB2 = b.reduce(0) { $0 + Double($1) * Double($1) }

        let numerator = n * sumAB - sumA * sumB
        let denominator = sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB))

        guard denominator > 0 else { return 0 }
        return numerator / denominator
    }
}
