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
        // Try compiled model first, then package
        var modelURL: URL?
        if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlmodelc") {
            modelURL = url
        } else if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") {
            modelURL = url
        }

        guard let modelURL = modelURL else {
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
        // Try compiled model first, then package
        var modelURL: URL?
        if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlmodelc") {
            modelURL = url
        } else if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") {
            modelURL = url
        }

        guard let modelURL = modelURL else {
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

    // MARK: - Diagnostic Tests

    /// Test that verifies vocal separation actually changes the audio
    /// This test checks that the output is different from the input
    func testSeparate_outputDiffersFromInput() throws {
        // Try compiled model first, then package
        var modelURL: URL?
        if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlmodelc") {
            modelURL = url
        } else if let url = Bundle.main.url(forResource: "UVR_MDX_NET", withExtension: "mlpackage") {
            modelURL = url
        }

        guard let modelURL = modelURL else {
            throw XCTSkip("CoreML model not available in test bundle")
        }

        // Create test audio - mix of two frequencies to simulate vocals + accompaniment
        let sampleRate = 44100.0
        let duration = 1.0
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        // Mix 440Hz (A4) + 880Hz (A5) - simulates a more complex signal
        for i in 0..<sampleCount {
            let t = Double(i) / sampleRate
            let vocal = Float(sin(2 * .pi * 440 * t) * 0.7)  // Simulated "vocal"
            let accompaniment = Float(sin(2 * .pi * 880 * t) * 0.3)  // Simulated accompaniment
            samples[i] = vocal + accompaniment
        }

        // Save test audio
        let audioURL = tempDirectory.appendingPathComponent("test_mixed.wav")
        let audioData = AudioProcessor.AudioData(
            samples: [samples, samples],
            sampleRate: sampleRate,
            frameCount: samples.count
        )
        try AudioProcessor.saveAudio(audioData, to: audioURL)

        // Calculate input RMS
        let inputRMS = calculateRMS(samples)
        print("📊 [DIAGNOSTIC] Input RMS: \(inputRMS)")

        // Run separation
        let engine = try VocalSeparatorEngine(modelURL: modelURL)
        let result = try engine.separate(audioURL: audioURL, progressHandler: nil)

        // Calculate output RMS
        let outputSamples = result.vocals.samples[0]
        let outputRMS = calculateRMS(outputSamples)
        print("📊 [DIAGNOSTIC] Output RMS: \(outputRMS)")

        // Calculate correlation between input and output
        let minLength = min(samples.count, outputSamples.count)
        let inputSlice = Array(samples[0..<minLength])
        let outputSlice = Array(outputSamples[0..<minLength])
        let correlation = calculateCorrelation(inputSlice, outputSlice)
        print("📊 [DIAGNOSTIC] Correlation: \(correlation)")

        // Calculate difference
        var sumDiff: Float = 0
        for i in 0..<minLength {
            sumDiff += abs(inputSlice[i] - outputSlice[i])
        }
        let avgDiff = sumDiff / Float(minLength)

        // Write diagnostic output to file for retrieval
        let diagnosticOutput = """
        ===== VOCAL SEPARATION DIAGNOSTIC =====
        Input RMS: \(inputRMS)
        Output RMS: \(outputRMS)
        Correlation: \(correlation)
        Average Difference: \(avgDiff)
        Input samples: \(samples.count)
        Output samples: \(outputSamples.count)
        =======================================
        """
        let diagnosticURL = tempDirectory.appendingPathComponent("diagnostic.txt")
        try? diagnosticOutput.write(to: diagnosticURL, atomically: true, encoding: .utf8)
        print(diagnosticOutput)

        // Also add as XCTest attachment
        let attachment = XCTAttachment(string: diagnosticOutput)
        attachment.name = "Separation Diagnostics"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Assert that output is different from input
        // If correlation is very high (> 0.99), the separation isn't working
        XCTAssertLessThan(
            correlation, 0.99,
            "Output should be different from input (correlation=\(correlation)). " +
            "High correlation indicates separation is not working."
        )

        // Also check that output has some signal (not silence)
        XCTAssertGreaterThan(
            outputRMS, 0.01,
            "Output should have some signal (RMS=\(outputRMS))"
        )
    }

    // MARK: - Helper Methods

    private func calculateRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumSquared = samples.reduce(0) { $0 + $1 * $1 }
        return sqrtf(sumSquared / Float(samples.count))
    }

    private func calculateCorrelation(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        let n = Float(a.count)
        let sumA = a.reduce(0, +)
        let sumB = b.reduce(0, +)
        let sumAB = zip(a, b).reduce(0 as Float) { $0 + $1.0 * $1.1 }
        let sumA2 = a.reduce(0 as Float) { $0 + $1 * $1 }
        let sumB2 = b.reduce(0 as Float) { $0 + $1 * $1 }

        let numerator = n * sumAB - sumA * sumB
        let denominator = sqrtf((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB))

        guard denominator > 0 else { return 0 }
        return numerator / denominator
    }

    private func generateSineWave(frequency: Double, sampleRate: Double, duration: Double) -> [Float] {
        let sampleCount = Int(sampleRate * duration)
        var samples = [Float](repeating: 0, count: sampleCount)

        for i in 0..<sampleCount {
            let phase = 2.0 * Double.pi * frequency * Double(i) / sampleRate
            samples[i] = Float(sin(phase))
        }

        return samples
    }

    // MARK: - POC Comparison Tests

    /// Integration test comparing App engine output with POC output
    /// Uses the same test audio file and compares correlation
    func testSeparate_comparesWithPOC() throws {
        // POC test audio and output paths
        let pocBasePath = "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/poc/uvr_coreml"
        let testAudioURL = URL(fileURLWithPath: "\(pocBasePath)/tests/output/hollow_crown_from_flac.wav")
        let pocOutputURL = URL(fileURLWithPath: "\(pocBasePath)/tests/swift_output/hollow_crown_vocals.wav")

        // Verify test files exist
        guard FileManager.default.fileExists(atPath: testAudioURL.path) else {
            throw XCTSkip("Test audio file not found: \(testAudioURL.path)")
        }
        guard FileManager.default.fileExists(atPath: pocOutputURL.path) else {
            throw XCTSkip("POC output file not found: \(pocOutputURL.path)")
        }

        // Get model URL - use absolute path since Bundle.main doesn't work in unit tests
        let modelPath = "/Users/asatokazu/Documents/dev/mine/music/vocal_mastery_lab/vocal_mastery_lab_app/VocalMasteryLab/Resources/Models/UVR_MDX_NET.mlpackage"
        let modelURL = URL(fileURLWithPath: modelPath)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("CoreML model not available at: \(modelPath)")
        }

        // Initialize engine
        let engine = try VocalSeparatorEngine(modelURL: modelURL)

        // Run separation
        print("🎵 [APP_TEST] Starting vocal extraction...")
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = try engine.separate(audioURL: testAudioURL, progressHandler: nil)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print("✅ [APP_TEST] Extraction completed in \(String(format: "%.2f", elapsed))s")

        // Save App output
        let appOutputURL = tempDirectory.appendingPathComponent("app_vocals.wav")
        try engine.save(result: result, to: appOutputURL)
        print("💾 [APP_TEST] Output saved to: \(appOutputURL.path)")

        // Load POC output for comparison
        let pocAudio = try AudioProcessor.loadAudio(from: pocOutputURL)
        let appAudio = result.vocals

        print("📊 [COMPARISON] POC: \(pocAudio.frameCount) frames, App: \(appAudio.frameCount) frames")

        // Normalize and compare
        let pocLeft = normalize(pocAudio.samples[0])
        let appLeft = normalize(appAudio.samples[0])

        // Align lengths
        let minLen = min(pocLeft.count, appLeft.count)
        let pocSlice = Array(pocLeft[0..<minLen])
        let appSlice = Array(appLeft[0..<minLen])

        // Calculate correlation
        let correlation = calculateCorrelation(pocSlice, appSlice)
        print("📈 [COMPARISON] Correlation: \(correlation)")

        // Calculate RMS
        let pocRMS = calculateRMS(pocSlice)
        let appRMS = calculateRMS(appSlice)
        print("📊 [COMPARISON] POC RMS: \(pocRMS), App RMS: \(appRMS)")

        // Add test attachment
        let report = """
        ===== POC vs App Comparison =====
        Correlation: \(correlation)
        POC RMS: \(pocRMS)
        App RMS: \(appRMS)
        POC frames: \(pocAudio.frameCount)
        App frames: \(appAudio.frameCount)
        App output: \(appOutputURL.path)
        =================================
        """
        let attachment = XCTAttachment(string: report)
        attachment.name = "POC vs App Comparison"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Assert high correlation (target: > 0.95)
        XCTAssertGreaterThan(
            correlation, 0.95,
            "App output should correlate with POC output (correlation=\(correlation), target>0.95)"
        )
    }

    private func normalize(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }
        let maxAbs = samples.map { abs($0) }.max() ?? 1.0
        guard maxAbs > 0 else { return samples }
        return samples.map { $0 / maxAbs }
    }
}
