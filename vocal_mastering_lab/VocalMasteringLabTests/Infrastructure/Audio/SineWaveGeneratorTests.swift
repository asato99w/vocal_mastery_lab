import XCTest
import AVFoundation
@testable import VocalMasteringLab
@testable import VocalisDomain

/// Tests for SineWaveGenerator
/// Generates PCM audio buffers containing sine waves for specified MIDI notes
final class SineWaveGeneratorTests: XCTestCase {

    var sut: SineWaveGenerator!

    override func setUp() {
        super.setUp()
        sut = SineWaveGenerator()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Buffer Generation Tests

    func testGenerateBuffer_returnsNonNilBuffer() throws {
        // Given
        let note = try MIDINote(60) // Middle C
        let duration: TimeInterval = 0.5
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        XCTAssertNotNil(buffer)
    }

    func testGenerateBuffer_hasCorrectFrameLength() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 1.0
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        let expectedFrameCount = AVAudioFrameCount(duration * sampleRate)
        XCTAssertEqual(buffer.frameLength, expectedFrameCount)
    }

    func testGenerateBuffer_hasCorrectFormat() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.5
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        XCTAssertEqual(buffer.format.sampleRate, sampleRate)
        XCTAssertEqual(buffer.format.channelCount, 1) // Mono
    }

    func testGenerateBuffer_containsNonZeroSamples() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.1
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        guard let channelData = buffer.floatChannelData?[0] else {
            XCTFail("Buffer should have channel data")
            return
        }

        // Check that at least some samples are non-zero (sine wave has amplitude)
        var hasNonZeroSample = false
        for i in 0..<Int(buffer.frameLength) {
            if abs(channelData[i]) > 0.001 {
                hasNonZeroSample = true
                break
            }
        }
        XCTAssertTrue(hasNonZeroSample, "Buffer should contain non-zero samples")
    }

    func testGenerateBuffer_samplesAreWithinValidRange() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.5
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        guard let channelData = buffer.floatChannelData?[0] else {
            XCTFail("Buffer should have channel data")
            return
        }

        for i in 0..<Int(buffer.frameLength) {
            let sample = channelData[i]
            XCTAssertTrue(sample >= -1.0 && sample <= 1.0, "Sample \(i) = \(sample) should be within [-1.0, 1.0]")
        }
    }

    // MARK: - Frequency Tests

    func testGenerateBuffer_producesCorrectFrequency_middleC() throws {
        // Given
        let note = try MIDINote(60) // Middle C = 261.63 Hz
        let duration: TimeInterval = 0.1
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        let detectedFrequency = detectDominantFrequency(in: buffer, sampleRate: sampleRate)
        let expectedFrequency = note.frequency // 261.63 Hz

        // Allow 5% tolerance for frequency detection
        let tolerance = expectedFrequency * 0.05
        XCTAssertEqual(detectedFrequency, expectedFrequency, accuracy: tolerance,
                       "Detected frequency \(detectedFrequency) should be close to \(expectedFrequency) Hz")
    }

    func testGenerateBuffer_producesCorrectFrequency_A440() throws {
        // Given
        let note = try MIDINote(69) // A4 = 440 Hz
        let duration: TimeInterval = 0.1
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        let detectedFrequency = detectDominantFrequency(in: buffer, sampleRate: sampleRate)
        let expectedFrequency: Double = 440.0

        let tolerance = expectedFrequency * 0.05
        XCTAssertEqual(detectedFrequency, expectedFrequency, accuracy: tolerance)
    }

    // MARK: - ADSR Envelope Tests

    func testGenerateBuffer_hasAttackPhase() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.5
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        guard let channelData = buffer.floatChannelData?[0] else {
            XCTFail("Buffer should have channel data")
            return
        }

        // First sample should be near zero (attack starts)
        XCTAssertLessThan(abs(channelData[0]), 0.1, "First sample should be near zero (attack phase)")

        // Sample after attack should have higher amplitude
        let attackSamples = Int(0.01 * sampleRate) // 10ms attack
        if attackSamples < Int(buffer.frameLength) {
            // Find peak amplitude after attack
            var maxAmplitude: Float = 0
            for i in attackSamples..<min(attackSamples + 1000, Int(buffer.frameLength)) {
                maxAmplitude = max(maxAmplitude, abs(channelData[i]))
            }
            XCTAssertGreaterThan(maxAmplitude, 0.5, "Should have significant amplitude after attack phase")
        }
    }

    func testGenerateBuffer_hasReleasePhase() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.5
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        guard let channelData = buffer.floatChannelData?[0] else {
            XCTFail("Buffer should have channel data")
            return
        }

        // Last sample should be near zero (release ends)
        let lastSample = channelData[Int(buffer.frameLength) - 1]
        XCTAssertLessThan(abs(lastSample), 0.1, "Last sample should be near zero (release phase)")
    }

    // MARK: - Edge Cases

    func testGenerateBuffer_veryShortDuration() throws {
        // Given
        let note = try MIDINote(60)
        let duration: TimeInterval = 0.01 // 10ms
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        XCTAssertEqual(buffer.frameLength, AVAudioFrameCount(duration * sampleRate))
    }

    func testGenerateBuffer_highNote() throws {
        // Given
        let note = try MIDINote(108) // C8 = 4186 Hz
        let duration: TimeInterval = 0.1
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        XCTAssertEqual(buffer.frameLength, AVAudioFrameCount(duration * sampleRate))
    }

    func testGenerateBuffer_lowNote() throws {
        // Given
        let note = try MIDINote(21) // A0 = 27.5 Hz
        let duration: TimeInterval = 0.2
        let sampleRate: Double = 44100

        // When
        let buffer = sut.generateBuffer(for: note, duration: duration, sampleRate: sampleRate)

        // Then
        XCTAssertEqual(buffer.frameLength, AVAudioFrameCount(duration * sampleRate))
    }

    // MARK: - Helper Methods

    /// Simple zero-crossing frequency detection for testing
    private func detectDominantFrequency(in buffer: AVAudioPCMBuffer, sampleRate: Double) -> Double {
        guard let channelData = buffer.floatChannelData?[0] else { return 0 }

        let frameCount = Int(buffer.frameLength)
        guard frameCount > 100 else { return 0 }

        // Skip attack phase (first 10ms) for stable frequency detection
        let skipSamples = min(Int(0.01 * sampleRate), frameCount / 4)

        // Count zero crossings
        var zeroCrossings = 0
        for i in (skipSamples + 1)..<frameCount {
            if (channelData[i - 1] >= 0 && channelData[i] < 0) ||
               (channelData[i - 1] < 0 && channelData[i] >= 0) {
                zeroCrossings += 1
            }
        }

        // Each full cycle has 2 zero crossings
        let measuredSamples = frameCount - skipSamples
        let measuredDuration = Double(measuredSamples) / sampleRate
        let frequency = Double(zeroCrossings) / (2.0 * measuredDuration)

        return frequency
    }
}
