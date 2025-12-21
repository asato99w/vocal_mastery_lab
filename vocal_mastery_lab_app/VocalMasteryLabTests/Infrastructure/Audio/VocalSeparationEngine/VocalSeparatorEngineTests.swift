import XCTest
@testable import VocalMasteryLab

final class VocalSeparatorEngineTests: XCTestCase {

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

    // MARK: - ModelConfiguration Tests

    func testModelConfiguration_default_hasExpectedValues() {
        let config = VocalSeparatorEngine.ModelConfiguration.default

        XCTAssertEqual(config.fftSize, 4096)
        XCTAssertEqual(config.hopSize, 1024)
        XCTAssertEqual(config.sampleRate, 44100)
        XCTAssertEqual(config.chunkSize, 256)
    }

    func testModelConfiguration_custom_acceptsValues() {
        let config = VocalSeparatorEngine.ModelConfiguration(
            fftSize: 2048,
            hopSize: 512,
            sampleRate: 22050,
            chunkSize: 128
        )

        XCTAssertEqual(config.fftSize, 2048)
        XCTAssertEqual(config.hopSize, 512)
        XCTAssertEqual(config.sampleRate, 22050)
        XCTAssertEqual(config.chunkSize, 128)
    }

    // MARK: - SeparationResult Tests

    func testSeparationResult_containsVocals() {
        let samples: [[Float]] = [[1.0, 2.0], [3.0, 4.0]]
        let audioData = AudioProcessor.AudioData(
            samples: samples,
            sampleRate: 44100,
            frameCount: 2
        )

        let result = VocalSeparatorEngine.SeparationResult(vocals: audioData)

        XCTAssertEqual(result.vocals.channelCount, 2)
        XCTAssertEqual(result.vocals.frameCount, 2)
    }

    // MARK: - SeparationError Tests

    func testSeparationError_modelLoadFailed_hasDescription() {
        let error = VocalSeparatorEngine.SeparationError.modelLoadFailed("Test error")
        XCTAssertEqual(error.errorDescription, "Model load failed: Test error")
    }

    func testSeparationError_predictionFailed_hasDescription() {
        let error = VocalSeparatorEngine.SeparationError.predictionFailed("Prediction issue")
        XCTAssertEqual(error.errorDescription, "Prediction failed: Prediction issue")
    }

    func testSeparationError_invalidAudioFormat_hasDescription() {
        let error = VocalSeparatorEngine.SeparationError.invalidAudioFormat("Wrong format")
        XCTAssertEqual(error.errorDescription, "Invalid audio format: Wrong format")
    }

    func testSeparationError_processingFailed_hasDescription() {
        let error = VocalSeparatorEngine.SeparationError.processingFailed("Processing issue")
        XCTAssertEqual(error.errorDescription, "Processing failed: Processing issue")
    }

    // MARK: - Initialization Tests

    func testInit_withNonExistentModel_throws() {
        let fakeModelURL = tempDirectory.appendingPathComponent("fake.mlpackage")

        XCTAssertThrowsError(try VocalSeparatorEngine(modelURL: fakeModelURL)) { error in
            // Should throw a separation error
            if case VocalSeparatorEngine.SeparationError.modelLoadFailed = error {
                // Expected
            } else {
                // Also acceptable if it's the underlying error
                XCTAssertTrue(error is VocalSeparatorEngine.SeparationError ||
                              error.localizedDescription.contains("model") ||
                              error.localizedDescription.contains("compile"))
            }
        }
    }

    // MARK: - Integration Tests (Skipped without model)

    func testSeparate_withRealModel_producesOutput() throws {
        // This test requires the actual CoreML model
        // Skip if model is not available in test bundle
        guard let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create test audio file
        let audioURL = tempDirectory.appendingPathComponent("test_input.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 1.0)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        // Initialize engine
        let engine = try VocalSeparatorEngine(modelURL: modelURL)

        // Run separation
        var progressUpdates: [(Double, String)] = []
        let result = try engine.separate(audioURL: audioURL) { progress, stage in
            progressUpdates.append((progress, stage))
        }

        // Verify output
        XCTAssertGreaterThan(result.vocals.frameCount, 0)
        XCTAssertEqual(result.vocals.channelCount, 2)

        // Verify progress was reported
        XCTAssertFalse(progressUpdates.isEmpty)
        if let lastProgress = progressUpdates.last?.0 {
            XCTAssertEqual(lastProgress, 1.0, accuracy: 0.01)
        }
    }

    func testSave_withResult_createsFile() throws {
        guard let modelURL = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create minimal test
        let audioURL = tempDirectory.appendingPathComponent("test_input.wav")
        let samples = generateSineWave(frequency: 440.0, sampleRate: 44100, duration: 0.5)
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: 44100,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        let engine = try VocalSeparatorEngine(modelURL: modelURL)
        let result = try engine.separate(audioURL: audioURL, progressHandler: nil)

        // Save output
        let outputURL = tempDirectory.appendingPathComponent("vocals.wav")
        try engine.save(result: result, to: outputURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
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
