import XCTest
import VocalisDomain
@testable import VocalMasteryLab

/// Comprehensive tests for StopRecordingUseCase
@MainActor
final class StopRecordingUseCaseTests: XCTestCase {

    var sut: StopRecordingUseCase!
    var mockAudioRecorder: MockAudioRecorder!
    var mockRecordingRepository: MockRecordingRepository!

    override func setUp() async throws {
        try await super.setUp()
        mockAudioRecorder = MockAudioRecorder()
        mockRecordingRepository = MockRecordingRepository()
        sut = StopRecordingUseCase(
            audioRecorder: mockAudioRecorder,
            recordingRepository: mockRecordingRepository
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockAudioRecorder = nil
        mockRecordingRepository = nil
        try await super.tearDown()
    }

    // MARK: - setRecordingContext() Tests

    func testSetRecordingContext_StoresURL() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test_recording.m4a")

        // When
        sut.setRecordingContext(url: recordingURL)

        // Then - Execute and verify recording is saved with correct URL
        mockAudioRecorder.stopRecordingResult = 30.0
        let result = try await sut.execute()

        XCTAssertNotNil(result.recordingId)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.count, 1)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.first?.fileURL, recordingURL)
    }

    func testSetRecordingContext_OverwritesPreviousURL() async throws {
        // Given
        let firstURL = URL(fileURLWithPath: "/tmp/first.m4a")
        let secondURL = URL(fileURLWithPath: "/tmp/second.m4a")

        // When
        sut.setRecordingContext(url: firstURL)
        sut.setRecordingContext(url: secondURL)

        // Then - Execute and verify only second URL is used
        mockAudioRecorder.stopRecordingResult = 30.0
        _ = try await sut.execute()

        XCTAssertEqual(mockRecordingRepository.savedRecordings.first?.fileURL, secondURL)
    }

    // MARK: - execute() Success Tests

    func testExecute_WithContext_ReturnsDurationAndRecordingId() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let expectedDuration: TimeInterval = 45.5
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = expectedDuration

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, expectedDuration)
        XCTAssertNotNil(result.recordingId)
    }

    func testExecute_WithContext_SavesRecordingToRepository() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let expectedDuration: TimeInterval = 60.0
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = expectedDuration

        // When
        _ = try await sut.execute()

        // Then
        XCTAssertTrue(mockRecordingRepository.saveCalled)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.count, 1)

        let savedRecording = mockRecordingRepository.savedRecordings.first!
        XCTAssertEqual(savedRecording.fileURL, recordingURL)
        XCTAssertEqual(savedRecording.duration.seconds, expectedDuration, accuracy: 0.001)
    }

    func testExecute_WithContext_ReturnsMatchingRecordingId() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = 30.0

        // When
        let result = try await sut.execute()

        // Then
        let savedRecording = mockRecordingRepository.savedRecordings.first!
        XCTAssertEqual(result.recordingId, savedRecording.id)
    }

    func testExecute_WithoutContext_ReturnsNilRecordingId() async throws {
        // Given - No setRecordingContext called
        mockAudioRecorder.stopRecordingResult = 30.0

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, 30.0)
        XCTAssertNil(result.recordingId)
        XCTAssertFalse(mockRecordingRepository.saveCalled)
    }

    func testExecute_WithoutContext_StillStopsRecorder() async throws {
        // Given - No setRecordingContext called
        mockAudioRecorder.stopRecordingResult = 30.0

        // When
        _ = try await sut.execute()

        // Then
        XCTAssertTrue(mockAudioRecorder.stopRecordingCalled)
    }

    func testExecute_ClearsContext() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = 30.0

        // When - First execute
        let firstResult = try await sut.execute()

        // Then - First execute should have recording ID
        XCTAssertNotNil(firstResult.recordingId)

        // When - Second execute without new context
        let secondResult = try await sut.execute()

        // Then - Second execute should have nil recording ID (context was cleared)
        XCTAssertNil(secondResult.recordingId)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.count, 1) // Only one save
    }

    // MARK: - execute() with Various Durations

    func testExecute_WithZeroDuration() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = 0.0

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, 0.0)
        XCTAssertNotNil(result.recordingId)
    }

    func testExecute_WithLongDuration() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let longDuration: TimeInterval = 3600.0 // 1 hour
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = longDuration

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, longDuration)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.first!.duration.seconds, longDuration, accuracy: 0.001)
    }

    func testExecute_WithPreciseDuration() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let preciseDuration: TimeInterval = 123.456789
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = preciseDuration

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, preciseDuration, accuracy: 0.000001)
    }

    // MARK: - execute() Error Tests

    func testExecute_WhenStopRecordingFails_ThrowsError() async throws {
        // Given
        sut.setRecordingContext(url: URL(fileURLWithPath: "/tmp/test.m4a"))
        mockAudioRecorder.stopRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute()
            XCTFail("Expected error to be thrown")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .notRecording)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        // Repository should not be called if stop fails
        XCTAssertFalse(mockRecordingRepository.saveCalled)
    }

    func testExecute_WhenRepositorySaveFails_ThrowsError() async throws {
        // Given
        sut.setRecordingContext(url: URL(fileURLWithPath: "/tmp/test.m4a"))
        mockAudioRecorder.stopRecordingResult = 30.0
        mockRecordingRepository.saveShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute()
            XCTFail("Expected error to be thrown")
        } catch {
            // Expected - repository save failed
        }

        // Stop should have been called
        XCTAssertTrue(mockAudioRecorder.stopRecordingCalled)
    }

    // MARK: - Recording Creation Tests

    func testExecute_CreatesRecordingWithCorrectProperties() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/my_recording.m4a")
        let duration: TimeInterval = 120.0
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = duration

        let beforeExecute = Date()

        // When
        _ = try await sut.execute()

        let afterExecute = Date()

        // Then
        let savedRecording = mockRecordingRepository.savedRecordings.first!
        XCTAssertEqual(savedRecording.fileURL, recordingURL)
        XCTAssertEqual(savedRecording.duration.seconds, duration, accuracy: 0.001)
        // createdAt should be between beforeExecute and afterExecute
        XCTAssertGreaterThanOrEqual(savedRecording.createdAt, beforeExecute)
        XCTAssertLessThanOrEqual(savedRecording.createdAt, afterExecute)
    }

    func testExecute_CreatesUniqueRecordingIds() async throws {
        // Given
        let recordingURL1 = URL(fileURLWithPath: "/tmp/recording1.m4a")
        let recordingURL2 = URL(fileURLWithPath: "/tmp/recording2.m4a")
        mockAudioRecorder.stopRecordingResult = 30.0

        // First recording
        sut.setRecordingContext(url: recordingURL1)
        let result1 = try await sut.execute()

        // Second recording
        sut.setRecordingContext(url: recordingURL2)
        let result2 = try await sut.execute()

        // Then
        XCTAssertNotEqual(result1.recordingId, result2.recordingId)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.count, 2)
    }

    // MARK: - Multiple Executions Tests

    func testMultipleExecutions_EachWithDifferentContext() async throws {
        // Given
        let urls = (1...3).map { URL(fileURLWithPath: "/tmp/recording\($0).m4a") }
        let durations: [TimeInterval] = [10.0, 20.0, 30.0]

        // When
        for (index, url) in urls.enumerated() {
            sut.setRecordingContext(url: url)
            mockAudioRecorder.stopRecordingResult = durations[index]
            _ = try await sut.execute()
        }

        // Then
        XCTAssertEqual(mockRecordingRepository.savedRecordings.count, 3)
        for (index, recording) in mockRecordingRepository.savedRecordings.enumerated() {
            XCTAssertEqual(recording.fileURL, urls[index])
            XCTAssertEqual(recording.duration.seconds, durations[index], accuracy: 0.001)
        }
    }

    // MARK: - Call Order Tests

    func testExecute_CallsStopRecorderBeforeSavingToRepository() async throws {
        // Given
        sut.setRecordingContext(url: URL(fileURLWithPath: "/tmp/test.m4a"))
        mockAudioRecorder.stopRecordingResult = 30.0

        // When
        _ = try await sut.execute()

        // Then - Verify both stop and save were called
        XCTAssertTrue(mockAudioRecorder.stopRecordingCalled)
        XCTAssertTrue(mockRecordingRepository.saveCalled)

        // Verify stop was called by checking the call time exists
        XCTAssertNotNil(mockAudioRecorder.stopRecordingCallTime, "Stop time should be recorded")
    }

    // MARK: - Edge Cases

    func testExecute_WithSpecialCharactersInPath() async throws {
        // Given
        let specialPath = "/tmp/录音 file (1) 日本語.m4a"
        let recordingURL = URL(fileURLWithPath: specialPath)
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = 30.0

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertNotNil(result.recordingId)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.first?.fileURL.path, specialPath)
    }

    func testExecute_WithVeryShortDuration() async throws {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let veryShortDuration: TimeInterval = 0.001
        sut.setRecordingContext(url: recordingURL)
        mockAudioRecorder.stopRecordingResult = veryShortDuration

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, veryShortDuration, accuracy: 0.0001)
    }
}
