import XCTest
@testable import VocalMasteryLab
import VocalisDomain

final class CoreMLVocalExtractorTests: XCTestCase {

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

    // MARK: - Protocol Conformance Tests

    func testCoreMLVocalExtractor_conformsToProtocol() {
        // This test verifies the type conforms to VocalExtractorProtocol
        let modelURL = tempDirectory.appendingPathComponent("fake.mlpackage")
        let extractor: VocalExtractorProtocol = CoreMLVocalExtractor(modelURL: modelURL)
        XCTAssertNotNil(extractor)
    }

    // MARK: - Error Handling Tests

    func testExtract_withNonExistentSourceFile_throwsError() async {
        let modelURL = tempDirectory.appendingPathComponent("model.mlpackage")
        let extractor = CoreMLVocalExtractor(modelURL: modelURL)

        let nonExistentURL = tempDirectory.appendingPathComponent("nonexistent.wav")

        do {
            _ = try await extractor.extract(from: nonExistentURL) { _, _ in }
            XCTFail("Expected error to be thrown")
        } catch VocalExtractionError.sourceFileNotFound {
            // Expected
        } catch {
            XCTFail("Expected sourceFileNotFound, got \(error)")
        }
    }

    func testExtract_withInvalidModel_throwsError() async throws {
        // Create a valid audio file
        let audioURL = tempDirectory.appendingPathComponent("test.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 0.5)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        // Use invalid model URL
        let invalidModelURL = tempDirectory.appendingPathComponent("invalid.mlpackage")
        let extractor = CoreMLVocalExtractor(modelURL: invalidModelURL)

        do {
            _ = try await extractor.extract(from: audioURL) { _, _ in }
            XCTFail("Expected error to be thrown")
        } catch VocalExtractionError.extractionFailed {
            // Expected - model load should fail
        } catch {
            // Also acceptable as the underlying error propagates
            XCTAssertTrue(error is VocalExtractionError)
        }
    }

    // MARK: - Progress Handler Tests

    func testExtract_callsProgressHandler() async throws {
        guard let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create test audio
        let audioURL = tempDirectory.appendingPathComponent("test.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 1.0)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        let extractor = CoreMLVocalExtractor(modelURL: modelURL)

        var progressValues: [Double] = []
        var stages: [String] = []

        _ = try await extractor.extract(from: audioURL) { progress, stage in
            progressValues.append(progress)
            stages.append(stage)
        }

        // Verify progress was called
        XCTAssertFalse(progressValues.isEmpty, "Progress handler should be called")
        XCTAssertFalse(stages.isEmpty, "Stage updates should be provided")

        // Progress should increase
        for i in 1..<progressValues.count {
            XCTAssertGreaterThanOrEqual(progressValues[i], progressValues[i-1],
                                         "Progress should not decrease")
        }

        // Should reach completion
        if let lastProgress = progressValues.last {
            XCTAssertEqual(lastProgress, 1.0, accuracy: 0.01,
                           "Final progress should be 1.0")
        } else {
            XCTFail("Progress values should not be empty")
        }
    }

    // MARK: - Full Integration Tests

    func testExtract_withValidInput_producesResult() async throws {
        guard let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create test audio
        let audioURL = tempDirectory.appendingPathComponent("test.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 1.0)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        let extractor = CoreMLVocalExtractor(modelURL: modelURL)

        let result = try await extractor.extract(from: audioURL) { _, _ in }

        // Verify result
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.vocalFileURL.path),
                      "Output file should exist")
        XCTAssertGreaterThan(result.duration.seconds, 0,
                             "Duration should be positive")
    }

    func testExtract_createsOutputInDocumentsDirectory() async throws {
        guard let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create test audio
        let audioURL = tempDirectory.appendingPathComponent("source_audio.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 0.5)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        let extractor = CoreMLVocalExtractor(modelURL: modelURL)

        let result = try await extractor.extract(from: audioURL) { _, _ in }

        // Verify output location
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let extractedDir = documentsURL.appendingPathComponent("ExtractedAudio")

        XCTAssertTrue(result.vocalFileURL.path.hasPrefix(extractedDir.path),
                      "Output should be in ExtractedAudio directory")

        // Cleanup
        try? FileManager.default.removeItem(at: result.vocalFileURL)
    }

    // MARK: - MockVocalExtractor Comparison Tests

    func testMockVocalExtractor_implementsProtocol() async throws {
        let mockExtractor = MockVocalExtractor()

        // Create test audio
        let audioURL = tempDirectory.appendingPathComponent("test.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 0.5)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        var progressCalled = false
        let result = try await mockExtractor.extract(from: audioURL) { _, _ in
            progressCalled = true
        }

        XCTAssertTrue(progressCalled)
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.vocalFileURL.path))
        XCTAssertGreaterThan(result.duration.seconds, 0)

        // Cleanup
        try? FileManager.default.removeItem(at: result.vocalFileURL)
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
}
