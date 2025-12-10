import XCTest
@testable import VocalMasteringLab
@testable import VocalisDomain

final class YINStrategyTests: XCTestCase {

    var sut: YINStrategy!

    override func setUp() {
        super.setUp()
        sut = YINStrategy()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Protocol Conformance Tests

    func testName_returnsYIN() {
        // Then
        XCTAssertEqual(sut.name, "YIN")
    }

    func testRequiresOctaveCorrection_returnsTrue() {
        // YIN is prone to octave errors, needs correction
        XCTAssertTrue(sut.requiresOctaveCorrection)
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration_hasExpectedValues() {
        // Given
        let config = YINStrategy.Configuration.default

        // Then
        XCTAssertEqual(config.bufferSize, 2048)
        XCTAssertEqual(config.hopSize, 2205)  // 50ms at 44100Hz
        XCTAssertEqual(config.threshold, 0.25)
        XCTAssertEqual(config.minFrequency, 80.0)
        XCTAssertEqual(config.maxFrequency, 1200.0)
    }

    // MARK: - Pitch Detection Tests

    func testDetectPitch_withSilentAudio_returnsEmptyArray() {
        // Given
        let samples = [Float](repeating: 0.0, count: 44100)  // 1 second of silence

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        XCTAssertTrue(frames.isEmpty)
    }

    func testDetectPitch_withSinusoid_returnsDetectedFrequency() {
        // Given
        let frequency: Float = 440.0
        let duration = 0.5  // 500ms
        let sampleRate = 44100.0
        let samples = generateSinusoid(frequency: frequency, duration: duration, sampleRate: sampleRate)

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: sampleRate)

        // Then
        XCTAssertFalse(frames.isEmpty, "Should detect pitch in sinusoid")

        // At least one frame should have frequency close to 440Hz
        let voicedFrames = frames.filter { $0.isVoiced }
        XCTAssertFalse(voicedFrames.isEmpty, "Should have voiced frames")

        if let firstVoiced = voicedFrames.first, let detectedFreq = firstVoiced.frequency {
            XCTAssertEqual(detectedFreq, frequency, accuracy: 10.0,
                           "Detected frequency should be close to 440Hz")
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
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        for frame in frames {
            XCTAssertGreaterThanOrEqual(frame.confidence, 0.0,
                                         "Confidence should be >= 0.0")
            XCTAssertLessThanOrEqual(frame.confidence, 1.0,
                                      "Confidence should be <= 1.0")
        }
    }

    func testDetectPitch_returnsValidAmplitude() {
        // Given
        let samples = generateSinusoid(frequency: 440.0, duration: 0.2, sampleRate: 44100.0)

        // When
        let frames = sut.detectPitch(samples: samples, sampleRate: 44100.0)

        // Then
        for frame in frames {
            XCTAssertGreaterThanOrEqual(frame.amplitude, 0.0,
                                         "Amplitude should be >= 0.0")
            XCTAssertLessThanOrEqual(frame.amplitude, 1.0,
                                      "Amplitude should be <= 1.0")
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

    // MARK: - Helpers

    private func generateSinusoid(frequency: Float, duration: Double, sampleRate: Double) -> [Float] {
        let sampleCount = Int(duration * sampleRate)
        return (0..<sampleCount).map { i in
            let t = Float(i) / Float(sampleRate)
            return 0.5 * sin(2.0 * Float.pi * frequency * t)
        }
    }
}
