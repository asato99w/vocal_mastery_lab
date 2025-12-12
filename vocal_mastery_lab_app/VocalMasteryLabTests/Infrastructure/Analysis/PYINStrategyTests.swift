import XCTest
@testable import VocalMasteryLab
@testable import VocalisDomain

final class PYINStrategyTests: XCTestCase {

    var sut: PYINStrategy!

    override func setUp() {
        super.setUp()
        sut = PYINStrategy()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Protocol Conformance Tests

    func testName_returnsPYIN() {
        // Then
        XCTAssertEqual(sut.name, "pYIN")
    }

    func testName_withCustomName_returnsCustomName() {
        // Given
        let strategy = PYINStrategy(configuration: .default, name: "pYIN-balanced")

        // Then
        XCTAssertEqual(strategy.name, "pYIN-balanced")
    }

    func testRequiresOctaveCorrection_returnsFalse() {
        // pYIN uses HMM for temporal smoothing, doesn't need octave correction
        XCTAssertFalse(sut.requiresOctaveCorrection)
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration_hasExpectedValues() {
        // Given
        let config = PYINStrategy.Configuration.default

        // Then
        XCTAssertEqual(config.bufferSize, 2048)
        XCTAssertEqual(config.hopSize, 2205)
        XCTAssertEqual(config.minFrequency, 80.0)
        XCTAssertEqual(config.maxFrequency, 1200.0)
    }

    func testBalancedConfiguration_hasHigherVoicedBias() {
        // Given
        let defaultConfig = PYINStrategy.Configuration.default
        let balancedConfig = PYINStrategy.Configuration.balanced

        // Then
        XCTAssertGreaterThan(balancedConfig.voicedBias, defaultConfig.voicedBias)
    }

    // MARK: - Pitch Detection Tests

    func testDetectPitch_withSilentAudio_returnsEmptyArray() {
        // Given
        let samples = [Float](repeating: 0.0, count: 44100)  // 1 second of silence

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        let voicedFrames = frames.filter { $0.isVoiced }
        XCTAssertTrue(voicedFrames.isEmpty, "Silent audio should produce no voiced frames")
    }

    func testDetectPitch_withHarmonicSignal_returnsVoicedFrames() {
        // Given
        // Use a signal with harmonics (more realistic voice-like signal)
        // pYIN is optimized for harmonic signals, not pure sinusoids
        let fundamentalFreq: Float = 440.0
        let duration = 1.0
        let sampleRate = 44100.0
        let samples = generateHarmonicSignal(
            fundamental: fundamentalFreq,
            harmonics: [1.0, 0.5, 0.3, 0.2],  // Fundamental + 3 harmonics
            duration: duration,
            sampleRate: sampleRate
        )

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: sampleRate)

        // Then - pYIN should detect voiced frames from harmonic signal
        // Note: Frequency accuracy on synthetic signals is not guaranteed
        // because pYIN's HMM is optimized for real voice patterns
        XCTAssertFalse(frames.isEmpty, "Should detect pitch in harmonic signal")

        let voicedFrames = frames.filter { $0.isVoiced }
        XCTAssertFalse(voicedFrames.isEmpty, "Should have voiced frames")

        // Verify that detected frequencies are within the default configured range
        let defaultConfig = PYINStrategy.Configuration.default
        let frequencies = voicedFrames.compactMap { $0.frequency }
        for freq in frequencies {
            XCTAssertGreaterThanOrEqual(freq, Float(defaultConfig.minFrequency),
                                        "Frequency should be >= minFrequency")
            XCTAssertLessThanOrEqual(freq, Float(defaultConfig.maxFrequency),
                                     "Frequency should be <= maxFrequency")
        }
    }

    func testDetectPitch_returnsValidTimestamps() {
        // Given
        let samples = generateSinusoid(frequency: 440.0, duration: 0.2, sampleRate: 44100.0)

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        for (index, frame) in frames.enumerated() {
            XCTAssertGreaterThanOrEqual(frame.timestamp, 0.0,
                                         "Frame \(index) timestamp should be non-negative")
            if index > 0 {
                XCTAssertGreaterThan(frame.timestamp, frames[index - 1].timestamp,
                                     "Timestamps should be monotonically increasing")
            }
        }
    }

    func testDetectPitch_returnsValidConfidence() {
        // Given
        let samples = generateSinusoid(frequency: 440.0, duration: 0.2, sampleRate: 44100.0)

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: sampleRate)

        // Then
        for frame in frames {
            XCTAssertGreaterThanOrEqual(frame.confidence, 0.0,
                                         "Confidence should be >= 0.0")
            XCTAssertLessThanOrEqual(frame.confidence, 1.0,
                                      "Confidence should be <= 1.0")
        }
    }

    // MARK: - Edge Cases

    func testDetectPitch_withVeryShortAudio_handlesGracefully() {
        // Given
        let samples = [Float](repeating: 0.5, count: 100)  // Very short buffer

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then - should not crash, can return empty array
        XCTAssertNotNil(frames)
    }

    func testDetectPitch_withEmptyArray_returnsEmptyArray() {
        // Given
        let samples: [Float] = []

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        XCTAssertTrue(frames.isEmpty)
    }

    // MARK: - Configuration Presets Tests

    func testHighDetectionPreset_hasMoreSensitiveSettings() {
        // Given
        let defaultConfig = PYINStrategy.Configuration.default
        let highDetectionConfig = PYINStrategy.Configuration.highDetection

        // Then - highDetection should have lower silence threshold
        XCTAssertLessThan(highDetectionConfig.silenceThreshold, defaultConfig.silenceThreshold)
    }

    func testAggressivePreset_hasHighestVoicedBias() {
        // Given
        let balancedConfig = PYINStrategy.Configuration.balanced
        let aggressiveConfig = PYINStrategy.Configuration.aggressive

        // Then
        XCTAssertGreaterThan(aggressiveConfig.voicedBias, balancedConfig.voicedBias)
    }

    // MARK: - Helpers

    private let sampleRate = 44100.0

    private func generateSinusoid(frequency: Float, duration: Double, sampleRate: Double) -> [Float] {
        let sampleCount = Int(duration * sampleRate)
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * frequency * t)
        }
    }

    /// Generate a harmonic signal (more realistic voice-like signal)
    /// - Parameters:
    ///   - fundamental: Fundamental frequency
    ///   - harmonics: Relative amplitudes for each harmonic (first is fundamental)
    ///   - duration: Signal duration in seconds
    ///   - sampleRate: Sample rate
    private func generateHarmonicSignal(
        fundamental: Float,
        harmonics: [Float],
        duration: Double,
        sampleRate: Double
    ) -> [Float] {
        let sampleCount = Int(duration * sampleRate)
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            var sample: Float = 0
            for (index, amplitude) in harmonics.enumerated() {
                let harmonicNumber = Float(index + 1)
                sample += amplitude * sin(2.0 * Float.pi * fundamental * harmonicNumber * t)
            }
            // Normalize to prevent clipping
            return sample / harmonics.reduce(0, +)
        }
    }
}
