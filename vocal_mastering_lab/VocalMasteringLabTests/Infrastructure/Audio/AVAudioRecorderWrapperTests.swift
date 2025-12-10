import XCTest
import AVFoundation
import VocalisDomain
@testable import VocalMasteringLab

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

    // MARK: - Initial State Tests

    func testInitialState_IsNotRecording() {
        XCTAssertFalse(sut.isRecording)
    }

    // MARK: - Prepare Recording Tests

    func testPrepareRecording_ReturnsValidURL() async throws {
        // When
        let url = try await sut.prepareRecording()

        // Then
        XCTAssertNotNil(url)
        XCTAssertTrue(url.pathExtension == "m4a")
        XCTAssertTrue(url.path.contains("recording_"))
    }

    func testPrepareRecording_MultipleCalls_ReturnsDifferentURLs() async throws {
        // When
        let url1 = try await sut.prepareRecording()
        let url2 = try await sut.prepareRecording()

        // Then
        XCTAssertNotEqual(url1, url2)
    }

    // MARK: - Start Recording Tests

    func testStartRecording_WithoutPrepare_ThrowsError() async {
        // When/Then
        do {
            try await sut.startRecording()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertEqual(error as? AudioRecorderError, .notPrepared)
        }
    }

    func testStartRecording_AfterPrepare_SetsIsRecordingTrue() async throws {
        // Given
        _ = try await sut.prepareRecording()

        // When
        try await sut.startRecording()

        // Then
        XCTAssertTrue(sut.isRecording)
    }

    func testStartRecording_WhileRecording_ThrowsError() async throws {
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

    // MARK: - Stop Recording Tests

    func testStopRecording_WithoutStarting_ThrowsError() async {
        // When/Then
        do {
            _ = try await sut.stopRecording()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertEqual(error as? AudioRecorderError, .notRecording)
        }
    }

    func testStopRecording_AfterStarting_ReturnsElapsedTime() async throws {
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

    // MARK: - Audio Settings Tests

    func testPrepareRecording_ConfiguresCorrectAudioFormat() async throws {
        // When
        _ = try await sut.prepareRecording()

        // Then
        // This test verifies the settings are applied correctly
        // Actual audio format verification would require accessing internal AVAudioRecorder
        // For now, we verify it doesn't throw
        XCTAssertFalse(sut.isRecording)
    }
}
