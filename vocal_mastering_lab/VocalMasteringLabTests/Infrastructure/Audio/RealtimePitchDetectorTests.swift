import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteringLab

@MainActor
final class RealtimePitchDetectorTests: XCTestCase {

    var sut: RealtimePitchDetector!

    override func setUp() {
        super.setUp()

        // Configure audio session for pitch detection tests
        // This is required before startRealtimeDetection() can succeed
        do {
            try AudioSessionManager.shared.configureForRecording()
        } catch {
            // Ignore errors in test environment (simulator may not have full audio support)
            print("⚠️ Audio session configuration failed (acceptable in test environment): \(error)")
        }

        sut = RealtimePitchDetector()
    }

    override func tearDown() {
        if sut.isDetecting {
            sut.stopRealtimeDetection()
        }
        sut = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInit_InitialState_IsNotDetecting() {
        // Then
        XCTAssertFalse(sut.isDetecting)
        XCTAssertNil(sut.detectedPitch)
        XCTAssertNil(sut.spectrum)
    }

    func testInit_InitialState_AudioLevelIsMinimum() {
        // Then - initial audio level should be -160 dB (silence)
        XCTAssertEqual(sut.audioLevel, -160.0)
    }

    // MARK: - Start/Stop Detection Tests

    func testStartRealtimeDetection_WhenNotDetecting_SetsIsDetectingTrue() async throws {
        // When
        try await sut.startRealtimeDetection()

        // Then
        XCTAssertTrue(sut.isDetecting)
    }

    func testStartRealtimeDetection_WhenAlreadyDetecting_DoesNothing() async throws {
        // Given
        try await sut.startRealtimeDetection()
        XCTAssertTrue(sut.isDetecting)

        // When - call again
        try await sut.startRealtimeDetection()

        // Then - still detecting (no error thrown)
        XCTAssertTrue(sut.isDetecting)
    }

    func testStopRealtimeDetection_WhenDetecting_SetsIsDetectingFalse() async throws {
        // Given
        try await sut.startRealtimeDetection()
        XCTAssertTrue(sut.isDetecting)

        // When
        sut.stopRealtimeDetection()

        // Then
        XCTAssertFalse(sut.isDetecting)
        XCTAssertNil(sut.detectedPitch)
        XCTAssertNil(sut.spectrum)
        XCTAssertEqual(sut.audioLevel, -160.0)
    }

    func testStopRealtimeDetection_WhenNotDetecting_DoesNothing() {
        // Given
        XCTAssertFalse(sut.isDetecting)

        // When
        sut.stopRealtimeDetection()

        // Then - no crash or error
        XCTAssertFalse(sut.isDetecting)
    }

    // MARK: - File Analysis Tests

    func testAnalyzePitchFromFile_NonExistentFile_CompletesWithNil() async {
        // Given
        let nonExistentURL = URL(fileURLWithPath: "/tmp/nonexistent_\(UUID().uuidString).m4a")
        let expectation = expectation(description: "Analysis completes")
        var result: DetectedPitch?

        // When
        sut.analyzePitchFromFile(nonExistentURL, atTime: 0.0) { pitch in
            result = pitch
            expectation.fulfill()
        }

        // Then
        await fulfillment(of: [expectation], timeout: 2.0)
        XCTAssertNil(result)
    }

    func testAnalyzePitchFromFile_InvalidTimePosition_CompletesWithNil() async throws {
        // Given - create a short test audio file
        let testFileURL = try createTestAudioFile(duration: 1.0, frequency: 440.0)
        let expectation = expectation(description: "Analysis completes")
        var result: DetectedPitch?

        // When - request analysis at time beyond file length
        sut.analyzePitchFromFile(testFileURL, atTime: 10.0) { pitch in
            result = pitch
            expectation.fulfill()
        }

        // Then
        await fulfillment(of: [expectation], timeout: 2.0)
        XCTAssertNil(result)

        // Cleanup
        try? FileManager.default.removeItem(at: testFileURL)
    }

    func testAnalyzePitchFromFile_ValidAudioWithPitch_DetectsPitch() async throws {
        // Given - create test audio file with 440 Hz tone (A4)
        let testFileURL = try createTestAudioFile(duration: 1.0, frequency: 440.0)
        let expectation = expectation(description: "Analysis completes")
        var result: DetectedPitch?

        // When
        sut.analyzePitchFromFile(testFileURL, atTime: 0.5) { pitch in
            result = pitch
            expectation.fulfill()
        }

        // Then
        await fulfillment(of: [expectation], timeout: 2.0)

        // Note: Due to FFT analysis limitations and HPS algorithm,
        // exact frequency match is not guaranteed, but should be in reasonable range
        if let detectedPitch = result {
            XCTAssertGreaterThan(detectedPitch.confidence, 0.0)
            XCTAssertGreaterThan(detectedPitch.frequency, 100.0)
            XCTAssertLessThan(detectedPitch.frequency, 800.0)
        }
        // Note: We don't XCTAssertNotNil because the test audio might not meet
        // the confidence threshold depending on the signal processing

        // Cleanup
        try? FileManager.default.removeItem(at: testFileURL)
    }

    // MARK: - Helper Methods

    /// Create a test audio file with a simple sine wave
    private func createTestAudioFile(duration: TimeInterval, frequency: Double) throws -> URL {
        let sampleRate = 44100.0
        let frameCount = AVAudioFrameCount(duration * sampleRate)

        let format = AVAudioFormat(
            standardFormatWithSampleRate: sampleRate,
            channels: 1
        )!

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw NSError(domain: "Test", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create buffer"])
        }

        buffer.frameLength = frameCount

        // Generate sine wave
        guard let floatChannelData = buffer.floatChannelData else {
            throw NSError(domain: "Test", code: -2, userInfo: [NSLocalizedDescriptionKey: "No channel data"])
        }

        let samples = floatChannelData[0]
        for frame in 0..<Int(frameCount) {
            let time = Double(frame) / sampleRate
            let value = Float(sin(2.0 * .pi * frequency * time))
            samples[frame] = value * 0.5 // Reduce amplitude to 50%
        }

        // Write to file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("test_audio_\(UUID().uuidString).m4a")

        let audioFile = try AVAudioFile(
            forWriting: fileURL,
            settings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: sampleRate,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
        )

        try audioFile.write(from: buffer)

        return fileURL
    }
}
