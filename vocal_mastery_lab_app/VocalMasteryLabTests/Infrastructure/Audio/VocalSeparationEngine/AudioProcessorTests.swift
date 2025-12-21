import XCTest
@testable import VocalMasteryLab

final class AudioProcessorTests: XCTestCase {

    var tempDirectory: URL!

    override func setUp() {
        super.setUp()
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDirectory)
        tempDirectory = nil
        super.tearDown()
    }

    // MARK: - AudioData Tests

    func testAudioData_channelCount() {
        let monoData = AudioProcessor.AudioData(
            samples: [[1.0, 2.0, 3.0]],
            sampleRate: 44100,
            frameCount: 3
        )
        XCTAssertEqual(monoData.channelCount, 1)

        let stereoData = AudioProcessor.AudioData(
            samples: [[1.0, 2.0], [3.0, 4.0]],
            sampleRate: 44100,
            frameCount: 2
        )
        XCTAssertEqual(stereoData.channelCount, 2)
    }

    // MARK: - ConvertToMono Tests

    func testConvertToMono_withStereo_averagesChannels() {
        let left: [Float] = [1.0, 2.0, 3.0]
        let right: [Float] = [3.0, 4.0, 5.0]
        let stereoData = AudioProcessor.AudioData(
            samples: [left, right],
            sampleRate: 44100,
            frameCount: 3
        )

        let monoData = AudioProcessor.convertToMono(stereoData)

        XCTAssertEqual(monoData.channelCount, 1)
        XCTAssertEqual(monoData.samples[0][0], 2.0, accuracy: 0.001) // (1+3)/2
        XCTAssertEqual(monoData.samples[0][1], 3.0, accuracy: 0.001) // (2+4)/2
        XCTAssertEqual(monoData.samples[0][2], 4.0, accuracy: 0.001) // (3+5)/2
    }

    func testConvertToMono_withMono_returnsUnchanged() {
        let monoData = AudioProcessor.AudioData(
            samples: [[1.0, 2.0, 3.0]],
            sampleRate: 44100,
            frameCount: 3
        )

        let result = AudioProcessor.convertToMono(monoData)

        XCTAssertEqual(result.channelCount, 1)
        XCTAssertEqual(result.samples[0], monoData.samples[0])
    }

    // MARK: - ConvertToStereo Tests

    func testConvertToStereo_withMono_duplicatesChannel() {
        let monoData = AudioProcessor.AudioData(
            samples: [[1.0, 2.0, 3.0]],
            sampleRate: 44100,
            frameCount: 3
        )

        let stereoData = AudioProcessor.convertToStereo(monoData)

        XCTAssertEqual(stereoData.channelCount, 2)
        XCTAssertEqual(stereoData.samples[0], stereoData.samples[1])
    }

    func testConvertToStereo_withStereo_returnsUnchanged() {
        let stereoData = AudioProcessor.AudioData(
            samples: [[1.0, 2.0], [3.0, 4.0]],
            sampleRate: 44100,
            frameCount: 2
        )

        let result = AudioProcessor.convertToStereo(stereoData)

        XCTAssertEqual(result.channelCount, 2)
        XCTAssertEqual(result.samples[0], stereoData.samples[0])
        XCTAssertEqual(result.samples[1], stereoData.samples[1])
    }

    // MARK: - Normalize Tests

    func testNormalize_withClippingAudio_normalizesToUnitRange() {
        let samples: [Float] = [0.5, 1.0, -2.0, 0.25]
        let audioData = AudioProcessor.AudioData(
            samples: [samples],
            sampleRate: 44100,
            frameCount: 4
        )

        let normalized = AudioProcessor.normalize(audioData)

        // Max absolute value was 2.0, so all values should be divided by 2.0
        XCTAssertEqual(normalized.samples[0][0], 0.25, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][1], 0.5, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][2], -1.0, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][3], 0.125, accuracy: 0.001)
    }

    func testNormalize_withStereo_normalizesAcrossAllChannels() {
        let left: [Float] = [0.5, -1.0]
        let right: [Float] = [2.0, 0.25]
        let audioData = AudioProcessor.AudioData(
            samples: [left, right],
            sampleRate: 44100,
            frameCount: 2
        )

        let normalized = AudioProcessor.normalize(audioData)

        // Max absolute value across both channels is 2.0
        XCTAssertEqual(normalized.samples[0][0], 0.25, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][1], -0.5, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[1][0], 1.0, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[1][1], 0.125, accuracy: 0.001)
    }

    func testNormalize_withSilence_returnsUnchanged() {
        let samples: [Float] = [0.0, 0.0, 0.0]
        let audioData = AudioProcessor.AudioData(
            samples: [samples],
            sampleRate: 44100,
            frameCount: 3
        )

        let normalized = AudioProcessor.normalize(audioData)

        XCTAssertEqual(normalized.samples[0], samples)
    }

    func testNormalize_withAlreadyNormalized_doesNotAmplify() {
        let samples: [Float] = [0.5, 1.0, -0.75]
        let audioData = AudioProcessor.AudioData(
            samples: [samples],
            sampleRate: 44100,
            frameCount: 3
        )

        let normalized = AudioProcessor.normalize(audioData)

        // Max is already 1.0, so values should be the same
        XCTAssertEqual(normalized.samples[0][0], 0.5, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][1], 1.0, accuracy: 0.001)
        XCTAssertEqual(normalized.samples[0][2], -0.75, accuracy: 0.001)
    }

    // MARK: - Load/Save Integration Tests

    func testLoadAndSave_roundTrip_preservesAudio() throws {
        // Create test audio data
        let sampleRate = 44100.0
        let samples = generateSineWave(frequency: 440.0, sampleRate: sampleRate, duration: 0.1)
        let originalData = AudioProcessor.AudioData(
            samples: [samples, samples], // stereo
            sampleRate: sampleRate,
            frameCount: samples.count
        )

        // Save to temp file
        let fileURL = tempDirectory.appendingPathComponent("test.wav")
        try AudioProcessor.saveAudio(originalData, to: fileURL)

        // Verify file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: fileURL.path))

        // Load back
        let loadedData = try AudioProcessor.loadAudio(from: fileURL)

        // Verify properties
        XCTAssertEqual(loadedData.sampleRate, sampleRate, accuracy: 1.0)
        XCTAssertEqual(loadedData.channelCount, 2)
        XCTAssertEqual(loadedData.frameCount, originalData.frameCount)

        // Verify audio content (correlation should be high)
        let correlation = calculateCorrelation(originalData.samples[0], loadedData.samples[0])
        XCTAssertGreaterThan(correlation, 0.99, "Audio content should be preserved")
    }

    func testLoadAudio_withNonExistentFile_throwsError() {
        let nonExistentURL = tempDirectory.appendingPathComponent("nonexistent.wav")

        XCTAssertThrowsError(try AudioProcessor.loadAudio(from: nonExistentURL)) { error in
            guard case AudioProcessor.ProcessingError.fileNotFound = error else {
                XCTFail("Expected fileNotFound error, got \(error)")
                return
            }
        }
    }

    func testLoadAudio_withResample_changesRate() throws {
        // Create test audio at 44100 Hz
        let originalRate = 44100.0
        let targetRate = 22050.0
        let samples = generateSineWave(frequency: 440.0, sampleRate: originalRate, duration: 0.1)
        let originalData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: originalRate,
            frameCount: samples.count
        )

        // Save
        let fileURL = tempDirectory.appendingPathComponent("test_resample.wav")
        try AudioProcessor.saveAudio(originalData, to: fileURL)

        // Load with different sample rate
        let loadedData = try AudioProcessor.loadAudio(from: fileURL, targetSampleRate: targetRate)

        XCTAssertEqual(loadedData.sampleRate, targetRate, accuracy: 1.0)
        // Frame count should be approximately half
        let expectedFrameCount = Int(Double(samples.count) * targetRate / originalRate)
        XCTAssertEqual(loadedData.frameCount, expectedFrameCount, accuracy: 10)
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
