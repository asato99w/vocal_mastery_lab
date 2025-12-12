import XCTest
import VocalisDomain
@testable import VocalMasteryLab

final class StopRecordingUseCaseTests: XCTestCase {

    var sut: StopRecordingUseCase!
    var mockAudioRecorder: MockAudioRecorder!
    var mockScalePlayer: MockScalePlayer!
    var mockRecordingRepository: MockRecordingRepository!

    override func setUp() {
        super.setUp()
        mockAudioRecorder = MockAudioRecorder()
        mockScalePlayer = MockScalePlayer()
        mockRecordingRepository = MockRecordingRepository()
        sut = StopRecordingUseCase(
            audioRecorder: mockAudioRecorder,
            scalePlayer: mockScalePlayer,
            recordingRepository: mockRecordingRepository
        )
    }

    override func tearDown() {
        sut = nil
        mockRecordingRepository = nil
        mockScalePlayer = nil
        mockAudioRecorder = nil
        super.tearDown()
    }

    // MARK: - Success Path Tests

    func testExecute_RecordingInProgress_StopsRecordingAndReturnsResult() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 5.5

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertTrue(mockAudioRecorder.stopRecordingCalled)
        XCTAssertEqual(result.duration, 5.5)
    }

    func testExecute_CallsStopRecordingOnAudioRecorder() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 3.0

        // When
        _ = try await sut.execute()

        // Then
        XCTAssertTrue(mockAudioRecorder.stopRecordingCalled)
    }

    func testExecute_ReturnsDurationFromAudioRecorder() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        let expectedDuration = 7.25
        mockAudioRecorder.stopRecordingResult = expectedDuration

        // When
        let result = try await sut.execute()

        // Then
        XCTAssertEqual(result.duration, expectedDuration)
    }

    // MARK: - Error Handling Tests

    func testExecute_NotRecording_ThrowsError() async {
        // Given
        mockAudioRecorder._isRecording = false
        mockAudioRecorder.stopRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is AudioRecorderError)
        }
    }

    func testExecute_StopRecordingFails_ThrowsError() async {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute()
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is AudioRecorderError)
        }
    }

    // MARK: - Playback Timeline Tests

    func testExecute_WithScaleSettings_GetsPlaybackTimeline() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 5.0

        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let settings = ScaleSettings.mvpDefault
        sut.setRecordingContext(url: recordingURL, settings: settings)

        // Set up mock timeline
        let timeline = ScalePlaybackTimeline(events: [], recordingStartTime: Date())
        mockScalePlayer.mockPlaybackTimeline = timeline

        // When
        _ = try await sut.execute()

        // Then: getPlaybackTimeline should be called
        XCTAssertTrue(mockScalePlayer.getPlaybackTimelineCalled)
    }

    func testExecute_WithScaleSettings_StopsTimestampRecording() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 5.0

        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let settings = ScaleSettings.mvpDefault
        sut.setRecordingContext(url: recordingURL, settings: settings)

        // When
        _ = try await sut.execute()

        // Then: stopTimestampRecording should be called
        XCTAssertTrue(mockScalePlayer.stopTimestampRecordingCalled)
    }

    func testExecute_SavesRecordingWithPlaybackTimeline() async throws {
        // Given
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 5.0

        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        let settings = ScaleSettings.mvpDefault
        sut.setRecordingContext(url: recordingURL, settings: settings)

        // Set up mock timeline with events
        let note = try MIDINote(60)
        let events = [
            ScalePlaybackEvent(timestamp: 0.0, note: note, eventType: .noteStart),
            ScalePlaybackEvent(timestamp: 1.0, note: note, eventType: .noteEnd)
        ]
        let timeline = ScalePlaybackTimeline(events: events, recordingStartTime: Date())
        mockScalePlayer.mockPlaybackTimeline = timeline

        // When
        _ = try await sut.execute()

        // Then: Recording should be saved with timeline
        XCTAssertTrue(mockRecordingRepository.saveCalled)
        XCTAssertNotNil(mockRecordingRepository.savedRecordings.last?.playbackTimeline)
        XCTAssertEqual(mockRecordingRepository.savedRecordings.last?.playbackTimeline?.events.count, 2)
    }

    func testExecute_WithoutScaleSettings_DoesNotRecordTimeline() async throws {
        // Given: Recording without scale (settings is nil)
        mockAudioRecorder._isRecording = true
        mockAudioRecorder.stopRecordingResult = 5.0

        let recordingURL = URL(fileURLWithPath: "/tmp/test.m4a")
        sut.setRecordingContext(url: recordingURL, settings: nil) // No scale settings

        // When
        _ = try await sut.execute()

        // Then: getPlaybackTimeline should not be called when settings is nil
        XCTAssertFalse(mockScalePlayer.getPlaybackTimelineCalled)
    }
}
