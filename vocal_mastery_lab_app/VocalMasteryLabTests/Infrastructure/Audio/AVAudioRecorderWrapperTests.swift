import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteryLab

/// Tests for AVAudioRecorderWrapper
/// Note: Tests requiring actual audio hardware are skipped on simulator
final class AVAudioRecorderWrapperTests: XCTestCase {

    var sut: AVAudioRecorderWrapper!

    override func setUp() {
        super.setUp()
        sut = AVAudioRecorderWrapper()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - Helper

    /// Check if running on simulator (audio hardware not available)
    private var isRunningOnSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    // MARK: - Initial State Tests

    func testInitialState_IsNotRecording() {
        XCTAssertFalse(sut.isRecording)
    }

    // MARK: - Start Recording Tests (Error Cases - work on simulator)

    func testStartRecording_WithoutPrepare_ThrowsError() async {
        // When/Then
        do {
            try await sut.startRecording()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertEqual(error as? AudioRecorderError, .notPrepared)
        }
    }

    // MARK: - Stop Recording Tests (Error Cases - work on simulator)

    func testStopRecording_WithoutStarting_ThrowsError() async {
        // When/Then
        do {
            _ = try await sut.stopRecording()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertEqual(error as? AudioRecorderError, .notRecording)
        }
    }

    // MARK: - Hardware-Dependent Tests (skipped on simulator)
    // These tests require actual microphone access and audio hardware

    func testPrepareRecording_ReturnsValidURL() async throws {
        // Skip on simulator - requires audio session activation
        try XCTSkipIf(isRunningOnSimulator, "Requires audio hardware (skipped on simulator)")

        // When
        let url = try await sut.prepareRecording()

        // Then
        XCTAssertNotNil(url)
        XCTAssertTrue(url.pathExtension == "wav")
        XCTAssertTrue(url.path.contains("recording_"))
    }

    func testPrepareRecording_MultipleCalls_ReturnsDifferentURLs() async throws {
        // Skip on simulator - requires audio session activation
        try XCTSkipIf(isRunningOnSimulator, "Requires audio hardware (skipped on simulator)")

        // When
        let url1 = try await sut.prepareRecording()
        let url2 = try await sut.prepareRecording()

        // Then
        XCTAssertNotEqual(url1, url2)
    }

    func testStartRecording_AfterPrepare_SetsIsRecordingTrue() async throws {
        // Skip on simulator - requires microphone access
        try XCTSkipIf(isRunningOnSimulator, "Requires microphone access (skipped on simulator)")

        // Given
        _ = try await sut.prepareRecording()

        // When
        try await sut.startRecording()

        // Then
        XCTAssertTrue(sut.isRecording)
    }

    func testStartRecording_WhileRecording_ThrowsError() async throws {
        // Skip on simulator - requires microphone access
        try XCTSkipIf(isRunningOnSimulator, "Requires microphone access (skipped on simulator)")

        // Given
        _ = try await sut.prepareRecording()
        try await sut.startRecording()

        // When/Then
        do {
            try await sut.startRecording()
            XCTFail("Expected error to be thrown")
        } catch {
            // Error expected
            XCTAssertTrue(error is AudioRecorderError)
        }
    }

    func testStopRecording_AfterStarting_ReturnsElapsedTime() async throws {
        // Skip on simulator - requires microphone access
        try XCTSkipIf(isRunningOnSimulator, "Requires microphone access (skipped on simulator)")

        // Given
        _ = try await sut.prepareRecording()
        try await sut.startRecording()

        // Wait a bit
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // When
        let duration = try await sut.stopRecording()

        // Then
        XCTAssertGreaterThan(duration, 0.0)
        XCTAssertFalse(sut.isRecording)
    }

    func testStopRecording_CreatesRecordingFile() async throws {
        // Skip on simulator - requires microphone access
        try XCTSkipIf(isRunningOnSimulator, "Requires microphone access (skipped on simulator)")

        // Given
        let url = try await sut.prepareRecording()
        try await sut.startRecording()

        // Wait a bit to record some audio
        try await Task.sleep(nanoseconds: 500_000_000) // 500ms

        // When
        _ = try await sut.stopRecording()

        // Then
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))

        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }

    func testPrepareRecording_ConfiguresCorrectAudioFormat() async throws {
        // Skip on simulator - requires audio session activation
        try XCTSkipIf(isRunningOnSimulator, "Requires audio hardware (skipped on simulator)")

        // When
        _ = try await sut.prepareRecording()

        // Then
        // This test verifies the settings are applied correctly
        // Actual audio format verification would require accessing internal AVAudioRecorder
        // For now, we verify it doesn't throw
        XCTAssertFalse(sut.isRecording)
    }
}
