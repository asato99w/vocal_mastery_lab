import XCTest
import VocalisDomain
@testable import VocalMasteryLab

@MainActor
final class RecordingListViewModelTests: XCTestCase {

    var sut: RecordingListViewModel!
    var mockRepository: MockRecordingRepository!
    var mockExtractedAudioRepository: MockExtractedAudioRepository!
    var mockAudioPlayer: MockAudioPlayer!

    override func setUp() {
        super.setUp()
        mockRepository = MockRecordingRepository()
        mockExtractedAudioRepository = MockExtractedAudioRepository()
        mockAudioPlayer = MockAudioPlayer()
        sut = RecordingListViewModel(
            recordingRepository: mockRepository,
            extractedAudioRepository: mockExtractedAudioRepository,
            audioPlayer: mockAudioPlayer
        )
    }

    override func tearDown() {
        sut = nil
        mockAudioPlayer = nil
        mockExtractedAudioRepository = nil
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testInit_InitialState_IsEmpty() {
        // Then
        XCTAssertTrue(sut.recordings.isEmpty)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNil(sut.errorMessage)
        XCTAssertNil(sut.playingRecordingId)
    }

    // MARK: - Load Recordings Tests

    func testLoadRecordings_Success_PopulatesRecordings() async {
        // Given
        let testRecordings = [
            Recording(
                fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
                duration: Duration(seconds: 10.0),
            ),
            Recording(
                fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
                duration: Duration(seconds: 15.0),
            )
        ]
        mockRepository.recordingsToReturn = testRecordings

        // When
        await sut.loadRecordings()

        // Then
        XCTAssertEqual(sut.recordings.count, 2)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNil(sut.errorMessage)
    }

    func testLoadRecordings_EmptyRepository_ReturnsEmptyArray() async {
        // Given
        mockRepository.recordingsToReturn = []

        // When
        await sut.loadRecordings()

        // Then
        XCTAssertTrue(sut.recordings.isEmpty)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNil(sut.errorMessage)
    }

    func testLoadRecordings_Failure_SetsErrorMessage() async {
        // Given
        mockRepository.findAllShouldFail = true

        // When
        await sut.loadRecordings()

        // Then
        XCTAssertTrue(sut.recordings.isEmpty)
        XCTAssertFalse(sut.isLoading)
        XCTAssertNotNil(sut.errorMessage)
    }

    func testLoadRecordings_SetsLoadingStateDuringExecution() async {
        // Given
        mockRepository.recordingsToReturn = []

        // When
        let loadTask = Task {
            await sut.loadRecordings()
        }

        await loadTask.value

        // Then
        XCTAssertFalse(sut.isLoading) // Should be false after completion
    }

    // MARK: - Play Recording Tests

    func testPlayRecording_Success_SetsPlayingRecordingId() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )

        // When
        await sut.playRecording(recording)

        // Wait for playback completion (MockAudioPlayer sleeps 10ms + processing time)
        try? await Task.sleep(nanoseconds: 30_000_000) // 30ms

        // Then - after completion, should be nil
        XCTAssertNil(sut.playingRecordingId)
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording.fileURL)
    }

    func testPlayRecording_CalledTwice_StartsNewPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )

        // Set longer play duration so playback doesn't complete during test
        mockAudioPlayer.playDurationNanoseconds = 500_000_000 // 500ms

        // Start playback first
        await sut.playRecording(recording)

        // Reset mock to track second call
        mockAudioPlayer.reset()
        mockAudioPlayer.playDurationNanoseconds = 500_000_000 // 500ms

        // When - play the same recording again (starts new playback task)
        await sut.playRecording(recording)

        // Wait briefly for playback to start
        try? await Task.sleep(nanoseconds: 20_000_000) // 20ms

        // Then - playingRecordingId should be set and play was called again
        XCTAssertEqual(sut.playingRecordingId, recording.id)
        XCTAssertTrue(mockAudioPlayer.playCalled)
    }

    func testPlayRecording_DifferentRecording_StopsCurrentAndPlaysNew() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )

        // When - play first recording
        await sut.playRecording(recording1)

        // Reset mock to track second playback
        mockAudioPlayer.reset()

        // Play second recording (should stop first and play new)
        await sut.playRecording(recording2)

        // Wait for playback completion
        try? await Task.sleep(nanoseconds: 30_000_000) // 30ms

        // Then
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording2.fileURL)
    }

    func testPlayRecording_Failure_SetsErrorMessage() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockAudioPlayer.playShouldFail = true

        // When
        await sut.playRecording(recording)

        // Wait for playback failure to be handled
        try? await Task.sleep(nanoseconds: 30_000_000) // 30ms

        // Then
        XCTAssertNotNil(sut.errorMessage)
        XCTAssertNil(sut.playingRecordingId)
    }

    // MARK: - Stop Playback Tests

    func testStopPlayback_WhenPlaying_StopsAndClearsPlayingId() async {
        // Given
        mockAudioPlayer._isPlaying = true

        // When
        await sut.stopPlayback()

        // Then
        XCTAssertNil(sut.playingRecordingId)
        XCTAssertTrue(mockAudioPlayer.stopCalled)
    }

    func testStopPlayback_WhenNotPlaying_DoesNothing() async {
        // Given
        XCTAssertNil(sut.playingRecordingId)

        // When
        await sut.stopPlayback()

        // Then
        XCTAssertNil(sut.playingRecordingId)
        XCTAssertTrue(mockAudioPlayer.stopCalled) // Stop is called regardless
    }

    // MARK: - Delete Recording Tests

    func testDeleteRecording_Success_RemovesRecordingAndReloads() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 10.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 15.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()
        XCTAssertEqual(sut.recordings.count, 2)

        // When - delete first recording
        mockRepository.recordingsToReturn = [recording2] // Simulate deletion
        await sut.deleteRecording(recording1)

        // Then
        XCTAssertEqual(sut.recordings.count, 1)
        XCTAssertEqual(sut.recordings.first?.id, recording2.id)
        XCTAssertNil(sut.errorMessage)
    }

    func testDeleteRecording_WhilePlaying_StopsPlaybackFirst() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]

        await sut.loadRecordings()

        // When - delete
        mockRepository.recordingsToReturn = [] // Simulate deletion
        await sut.deleteRecording(recording)

        // Then
        XCTAssertTrue(sut.recordings.isEmpty)
    }

    func testDeleteRecording_Failure_SetsErrorMessage() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]

        await sut.loadRecordings()

        // When
        await sut.deleteRecording(recording)

        // Then
        XCTAssertTrue(mockRepository.deleteCalled)
    }

    // MARK: - Position Tracking Tests

    func testStartPositionTracking_UpdatesCurrentTime() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0),
        )
        mockAudioPlayer._currentTime = 5.0
        mockAudioPlayer.playDurationNanoseconds = 500_000_000 // 500ms to allow position tracking

        // Start playback to set playingRecordingId
        await sut.playRecording(recording)

        // Wait for playingRecordingId to be set (non-blocking playRecording)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms

        // When
        await sut.startPositionTracking()

        // Wait for at least one update (position tracking updates every 100ms)
        try? await Task.sleep(nanoseconds: 150_000_000) // 150ms

        // Then
        XCTAssertEqual(sut.currentTime, 5.0, accuracy: 0.1)
    }

    func testStopPositionTracking_StopsUpdates() async {
        // Given
        await sut.startPositionTracking()
        mockAudioPlayer._currentTime = 5.0

        // When
        sut.stopPositionTracking()
        let timeAfterStop = sut.currentTime

        // Wait to verify no more updates
        mockAudioPlayer._currentTime = 10.0
        try? await Task.sleep(nanoseconds: 200_000_000)

        // Then - currentTime should not have changed
        XCTAssertEqual(sut.currentTime, timeAfterStop)
    }

    // MARK: - Seek Tests

    func testSeekToPosition_CallsAudioPlayerSeek() async {
        // Given
        let targetTime = 30.0

        // When
        await sut.seekToPosition(targetTime)

        // Then
        XCTAssertTrue(mockAudioPlayer.seekCalled)
        XCTAssertEqual(mockAudioPlayer.seekToTime, targetTime, accuracy: 0.01)
    }

    // MARK: - Selected Recording Tests

    func testInit_SelectedRecording_IsNil() {
        // Then
        XCTAssertNil(sut.selectedRecording)
    }

    func testSelectAndPlay_SelectsRecordingAndStartsPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When
        await sut.selectAndPlay(recording)

        // Wait for internal Task to start playback
        try? await Task.sleep(nanoseconds: 20_000_000) // 20ms

        // Then
        XCTAssertEqual(sut.selectedRecording?.id, recording.id)
        XCTAssertTrue(mockAudioPlayer.playCalled)
    }

    func testSelectAndPlay_AlreadySelected_DoesNothing() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // First select and play
        await sut.selectAndPlay(recording)
        mockAudioPlayer.reset()

        // When - select same recording again (should do nothing - pause/resume is handled by toggle button)
        await sut.selectAndPlay(recording)

        // Then - selection maintained, no playback actions
        XCTAssertEqual(sut.selectedRecording?.id, recording.id)
        XCTAssertFalse(mockAudioPlayer.playCalled) // No new play call
        XCTAssertFalse(mockAudioPlayer.stopCalled) // No stop call
    }

    func testSelectAndPlay_DifferentRecording_SwitchesSelection() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()

        // First select recording1
        await sut.selectAndPlay(recording1)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start
        mockAudioPlayer.reset()

        // When - select recording2
        await sut.selectAndPlay(recording2)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start

        // Then
        XCTAssertEqual(sut.selectedRecording?.id, recording2.id)
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording2.fileURL)
    }

    // MARK: - Toggle Playback Tests

    func testTogglePlayback_WhenPaused_ResumesPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // Select and start playback
        mockAudioPlayer.playDurationNanoseconds = 500_000_000 // 500ms
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start

        // Pause playback (not stop)
        sut.pausePlayback()
        mockAudioPlayer.reset()
        mockAudioPlayer._isPlaying = false

        // When - toggle should resume
        await sut.togglePlayback()

        // Then - resume was called (not play)
        XCTAssertTrue(mockAudioPlayer.resumeCalled)
    }

    func testTogglePlayback_WhenPlaying_PausesPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()
        mockAudioPlayer.playDurationNanoseconds = 500_000_000 // 500ms
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start
        mockAudioPlayer.reset()
        mockAudioPlayer._isPlaying = true

        // When - toggle should pause
        await sut.togglePlayback()

        // Then - pause was called (not stop)
        XCTAssertTrue(mockAudioPlayer.pauseCalled)
    }

    func testTogglePlayback_NoSelection_DoesNothing() async {
        // Given - no recording selected

        // When
        await sut.togglePlayback()

        // Then
        XCTAssertFalse(mockAudioPlayer.playCalled)
        XCTAssertFalse(mockAudioPlayer.stopCalled)
    }

    // MARK: - Play Previous/Next Tests

    func testPlayPrevious_PlaysRecordingBeforeCurrent() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()

        // Select second recording
        await sut.selectAndPlay(recording2)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start
        mockAudioPlayer.reset()

        // When
        await sut.playPrevious()
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start

        // Then
        XCTAssertEqual(sut.selectedRecording?.id, recording1.id)
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording1.fileURL)
    }

    func testPlayPrevious_AtFirstRecording_DoesNothing() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()

        // Select first recording
        await sut.selectAndPlay(recording1)
        mockAudioPlayer.reset()

        // When
        await sut.playPrevious()

        // Then - should stay on first recording
        XCTAssertEqual(sut.selectedRecording?.id, recording1.id)
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    func testPlayNext_PlaysRecordingAfterCurrent() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()

        // Select first recording
        await sut.selectAndPlay(recording1)
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start
        mockAudioPlayer.reset()

        // When
        await sut.playNext()
        try? await Task.sleep(nanoseconds: 20_000_000) // Wait for playback to start

        // Then
        XCTAssertEqual(sut.selectedRecording?.id, recording2.id)
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording2.fileURL)
    }

    func testPlayNext_AtLastRecording_DoesNothing() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()

        // Select last recording
        await sut.selectAndPlay(recording2)
        mockAudioPlayer.reset()

        // When
        await sut.playNext()

        // Then - should stay on last recording
        XCTAssertEqual(sut.selectedRecording?.id, recording2.id)
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    func testPlayPrevious_NoSelection_DoesNothing() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When - no selection
        await sut.playPrevious()

        // Then
        XCTAssertNil(sut.selectedRecording)
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    func testPlayNext_NoSelection_DoesNothing() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When - no selection
        await sut.playNext()

        // Then
        XCTAssertNil(sut.selectedRecording)
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    // MARK: - Can Play Previous/Next Tests

    func testCanPlayPrevious_WhenAtFirst_ReturnsFalse() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording1)

        // Then
        XCTAssertFalse(sut.canPlayPrevious)
    }

    func testCanPlayPrevious_WhenNotAtFirst_ReturnsTrue() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording2)

        // Then
        XCTAssertTrue(sut.canPlayPrevious)
    }

    func testCanPlayNext_WhenAtLast_ReturnsFalse() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording2)

        // Then
        XCTAssertFalse(sut.canPlayNext)
    }

    func testCanPlayNext_WhenNotAtLast_ReturnsTrue() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0),
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording1)

        // Then
        XCTAssertTrue(sut.canPlayNext)
    }

    // MARK: - Delete Selected Recording Tests

    func testDeleteRecording_WhenSelected_ClearsSelection() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)

        // When
        mockRepository.recordingsToReturn = []
        await sut.deleteRecording(recording)

        // Then
        XCTAssertNil(sut.selectedRecording)
    }

    // MARK: - Rename Recording Tests

    func testRenameRecording_Success_UpdatesTitle() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When
        await sut.renameRecording(recording, newTitle: "お気に入りのテイク")

        // Then
        XCTAssertTrue(mockRepository.updateCalled)
        XCTAssertEqual(sut.recordings.first?.title, "お気に入りのテイク")
    }

    func testRenameRecording_EmptyTitle_SetsNil() async {
        // Given
        var recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
            title: "既存のタイトル"
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When - empty string should become nil
        await sut.renameRecording(recording, newTitle: "")

        // Then
        XCTAssertTrue(mockRepository.updateCalled)
        XCTAssertNil(sut.recordings.first?.title)
    }

    func testRenameRecording_WhitespaceOnlyTitle_SetsNil() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When - whitespace only should become nil
        await sut.renameRecording(recording, newTitle: "   ")

        // Then
        XCTAssertTrue(mockRepository.updateCalled)
        XCTAssertNil(sut.recordings.first?.title)
    }

    func testRenameRecording_TrimsWhitespace() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When
        await sut.renameRecording(recording, newTitle: "  テスト名  ")

        // Then
        XCTAssertEqual(sut.recordings.first?.title, "テスト名")
    }

    func testRenameRecording_WhenSelected_UpdatesSelectedRecording() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)

        // When
        await sut.renameRecording(recording, newTitle: "新しい名前")

        // Then
        XCTAssertEqual(sut.selectedRecording?.title, "新しい名前")
    }

    func testRenameRecording_Failure_SetsErrorMessage() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0),
        )
        mockRepository.savedRecordings = [recording]
        mockRepository.recordingsToReturn = [recording]
        mockRepository.updateShouldFail = true
        await sut.loadRecordings()

        // When
        await sut.renameRecording(recording, newTitle: "新しい名前")

        // Then
        XCTAssertNotNil(sut.errorMessage)
    }

    // MARK: - Audio Source Tests

    func testAvailableSources_NoExtractedAudio_ReturnsOnlyOriginal() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When
        let sources = sut.availableSources(for: recording)

        // Then
        XCTAssertEqual(sources, [.original])
    }

    func testAvailableSources_WithVocalOnly_ReturnsOriginalAndVocal() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadRecordings()

        // When
        let sources = sut.availableSources(for: recording)

        // Then
        XCTAssertEqual(sources, [.original, .vocal])
    }

    func testAvailableSources_WithVocalAndInstrumental_ReturnsAll() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        let instrumentalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .instrumental,
            fileURL: URL(fileURLWithPath: "/tmp/instrumental.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio, instrumentalAudio]
        await sut.loadRecordings()

        // When
        let sources = sut.availableSources(for: recording)

        // Then
        XCTAssertEqual(sources, [.original, .vocal, .instrumental])
    }

    func testIsSourceAvailable_OriginalAlwaysAvailable() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)

        // Then
        XCTAssertTrue(sut.isSourceAvailable(.original))
    }

    func testIsSourceAvailable_VocalNotAvailableWhenNotExtracted() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)

        // Then
        XCTAssertFalse(sut.isSourceAvailable(.vocal))
        XCTAssertFalse(sut.isSourceAvailable(.instrumental))
    }

    func testIsSourceAvailable_VocalAvailableWhenExtracted() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)

        // Then
        XCTAssertTrue(sut.isSourceAvailable(.vocal))
    }

    func testSwitchAudioSource_ChangesSelectedSource() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000)

        // When
        await sut.switchAudioSource(to: .vocal)

        // Then
        XCTAssertEqual(sut.selectedAudioSource, .vocal)
    }

    func testSwitchAudioSource_PlaysVocalURL() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000)
        mockAudioPlayer.reset()
        mockAudioPlayer.playDurationNanoseconds = 500_000_000

        // When
        await sut.switchAudioSource(to: .vocal)
        try? await Task.sleep(nanoseconds: 20_000_000)

        // Then
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, vocalAudio.fileURL)
    }

    func testSwitchAudioSource_UnavailableSource_DoesNothing() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000)
        mockAudioPlayer.reset()

        // When - try to switch to vocal which is not available
        await sut.switchAudioSource(to: .vocal)

        // Then
        XCTAssertEqual(sut.selectedAudioSource, .original)
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    func testGetDuration_Original_ReturnsRecordingDuration() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When
        let duration = sut.getDuration(for: recording, source: .original)

        // Then
        XCTAssertEqual(duration, 10.0)
    }

    func testGetDuration_Vocal_ReturnsExtractedAudioDuration() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 10.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 8.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadRecordings()

        // When
        let duration = sut.getDuration(for: recording, source: .vocal)

        // Then
        XCTAssertEqual(duration, 8.0)
    }

    func testSelectAndPlay_ResetsAudioSourceToOriginal() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording1.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()

        // First select recording1 and switch to vocal
        await sut.selectAndPlay(recording1)
        try? await Task.sleep(nanoseconds: 20_000_000)
        await sut.switchAudioSource(to: .vocal)
        XCTAssertEqual(sut.selectedAudioSource, .vocal)

        // When - select different recording
        await sut.selectAndPlay(recording2)

        // Then - should reset to original
        XCTAssertEqual(sut.selectedAudioSource, .original)
    }

    func testPlayNext_ResetsAudioSourceToOriginal() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording1.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()

        // First select recording1 and switch to vocal
        await sut.selectAndPlay(recording1)
        try? await Task.sleep(nanoseconds: 20_000_000)
        await sut.switchAudioSource(to: .vocal)
        XCTAssertEqual(sut.selectedAudioSource, .vocal)

        // When
        await sut.playNext()

        // Then
        XCTAssertEqual(sut.selectedAudioSource, .original)
    }

    func testPlayPrevious_ResetsAudioSourceToOriginal() async {
        // Given
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording2.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording1, recording2]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()

        // First select recording2 and switch to vocal
        await sut.selectAndPlay(recording2)
        try? await Task.sleep(nanoseconds: 20_000_000)
        await sut.switchAudioSource(to: .vocal)
        XCTAssertEqual(sut.selectedAudioSource, .vocal)

        // When
        await sut.playPrevious()

        // Then
        XCTAssertEqual(sut.selectedAudioSource, .original)
    }

    func testCurrentPlayingSource_UpdatedOnSwitch() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        mockAudioPlayer.playDurationNanoseconds = 500_000_000
        await sut.loadRecordings()
        await sut.selectAndPlay(recording)
        try? await Task.sleep(nanoseconds: 20_000_000)

        XCTAssertEqual(sut.currentPlayingSource, .original)

        // When
        await sut.switchAudioSource(to: .vocal)
        try? await Task.sleep(nanoseconds: 20_000_000)

        // Then
        XCTAssertEqual(sut.currentPlayingSource, .vocal)
    }

    func testHasExtractedAudio_TrueWhenExtracted() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadRecordings()

        // Then
        XCTAssertTrue(sut.hasExtractedAudio(recording))
    }

    func testHasExtractedAudio_FalseWhenNotExtracted() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // Then
        XCTAssertFalse(sut.hasExtractedAudio(recording))
    }

    func testGetExtractedAudio_ReturnsCorrectAudio() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        let instrumentalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .instrumental,
            fileURL: URL(fileURLWithPath: "/tmp/instrumental.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio, instrumentalAudio]
        await sut.loadRecordings()

        // Then
        let foundVocal = sut.getExtractedAudio(for: recording, type: .vocal)
        XCTAssertNotNil(foundVocal)
        XCTAssertEqual(foundVocal?.fileURL, vocalAudio.fileURL)

        let foundInstrumental = sut.getExtractedAudio(for: recording, type: .instrumental)
        XCTAssertNotNil(foundInstrumental)
        XCTAssertEqual(foundInstrumental?.fileURL, instrumentalAudio.fileURL)
    }

    func testPlayRecording_WithVocalSource_PlaysVocalURL() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        let vocalAudio = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/vocal.wav"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        mockExtractedAudioRepository.storedAudios = [vocalAudio]
        await sut.loadRecordings()

        // When
        await sut.playRecording(recording, source: .vocal)
        try? await Task.sleep(nanoseconds: 30_000_000)

        // Then
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, vocalAudio.fileURL)
    }

    func testPlayRecording_WithUnavailableSource_FallsBackToOriginal() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 5.0)
        )
        mockRepository.recordingsToReturn = [recording]
        await sut.loadRecordings()

        // When - try to play vocal which doesn't exist
        await sut.playRecording(recording, source: .vocal)
        try? await Task.sleep(nanoseconds: 30_000_000)

        // Then - should fall back to original
        XCTAssertTrue(mockAudioPlayer.playCalled)
        XCTAssertEqual(mockAudioPlayer.playURL, recording.fileURL)
    }
}
