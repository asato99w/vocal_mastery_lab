import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

/// Tests for backing track player functionality in RecordingViewModel
@MainActor
final class BackingTrackPlayerTests: XCTestCase {

    var sut: RecordingViewModel!
    var mockStartRecordingUseCase: MockStartRecordingUseCase!
    var mockStopRecordingUseCase: MockStopRecordingUseCase!
    var mockAudioPlayer: MockAudioPlayer!
    var mockPitchDetector: MockRealtimePitchDetector!
    var mockRecordingRepository: MockRecordingRepository!
    var mockExtractedAudioRepository: MockExtractedAudioRepository!
    var mockBackingTrackPlayer: MockAudioPlayer!

    override func setUp() async throws {
        try await super.setUp()
        mockStartRecordingUseCase = MockStartRecordingUseCase()
        mockStopRecordingUseCase = MockStopRecordingUseCase()
        mockAudioPlayer = MockAudioPlayer()
        mockPitchDetector = MockRealtimePitchDetector()
        mockRecordingRepository = MockRecordingRepository()
        mockExtractedAudioRepository = MockExtractedAudioRepository()
        mockBackingTrackPlayer = MockAudioPlayer()

        let subscriptionVM = SubscriptionViewModel(
            getStatusUseCase: MockGetSubscriptionStatusUseCase(),
            purchaseUseCase: MockPurchaseSubscriptionUseCase(),
            restoreUseCase: MockRestorePurchasesUseCase()
        )

        sut = RecordingViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            pitchDetector: mockPitchDetector,
            subscriptionViewModel: subscriptionVM,
            countdownDuration: 0
        )

        sut.setBackingTrackRepositories(
            recordingRepository: mockRecordingRepository,
            extractedAudioRepository: mockExtractedAudioRepository,
            backingTrackPlayer: mockBackingTrackPlayer
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockBackingTrackPlayer = nil
        mockExtractedAudioRepository = nil
        mockRecordingRepository = nil
        mockPitchDetector = nil
        mockAudioPlayer = nil
        mockStopRecordingUseCase = nil
        mockStartRecordingUseCase = nil
        try await super.tearDown()
    }

    // MARK: - Backing Player State Tests

    func testIsBackingPlaying_InitiallyFalse() {
        XCTAssertFalse(sut.isBackingPlaying)
    }

    func testBackingCurrentTime_InitiallyZero() {
        XCTAssertEqual(sut.backingCurrentTime, 0)
    }

    func testBackingDuration_InitiallyZero() {
        XCTAssertEqual(sut.backingDuration, 0)
    }

    // MARK: - Toggle Playback Tests

    func testToggleBackingPlayback_WithNoTrackSelected_DoesNothing() async {
        // Given
        sut.selectedBackingTrack = nil

        // When
        await sut.toggleBackingPlayback()

        // Then
        XCTAssertFalse(mockBackingTrackPlayer.playCalled)
        XCTAssertFalse(sut.isBackingPlaying)
    }

    func testToggleBackingPlayback_WithTrackSelected_StartsPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._duration = 60.0
        // Make mock play() take long enough so it's still running when we assert
        mockBackingTrackPlayer.playDurationNanoseconds = 1_000_000_000 // 1 second

        // When
        await sut.toggleBackingPlayback()

        // Wait for internal Task to start playing
        try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then
        XCTAssertTrue(mockBackingTrackPlayer.playCalled)
        XCTAssertTrue(sut.isBackingPlaying)
    }

    func testToggleBackingPlayback_WhenPlaying_PausesPlayback() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._duration = 60.0
        mockBackingTrackPlayer._isPlaying = true

        // Simulate playing state
        await sut.toggleBackingPlayback() // Start
        mockBackingTrackPlayer._isPlaying = true

        // When
        await sut.toggleBackingPlayback() // Pause

        // Then
        XCTAssertTrue(mockBackingTrackPlayer.pauseCalled)
        XCTAssertFalse(sut.isBackingPlaying)
    }

    func testToggleBackingPlayback_WhenPaused_ResumesPlayback() async {
        // Given - setup with track selected and paused state
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._duration = 60.0
        // Make mock play() take long enough so it doesn't complete before we pause
        mockBackingTrackPlayer.playDurationNanoseconds = 1_000_000_000 // 1 second

        // Start playback first to set backingHasStarted
        await sut.toggleBackingPlayback()
        try? await Task.sleep(nanoseconds: 50_000_000) // Wait for Task to start

        // Now pause to simulate paused state
        await sut.toggleBackingPlayback()

        // Simulate time has passed (update ViewModel's state via seekBacking)
        sut.seekBacking(to: 30.0)
        mockBackingTrackPlayer.pauseCalled = false // Reset for clean test

        // When - toggle from paused state (currentTime > 0 but not playing)
        await sut.toggleBackingPlayback()

        // Then - should resume, not restart
        XCTAssertTrue(mockBackingTrackPlayer.resumeCalled)
        XCTAssertTrue(sut.isBackingPlaying)
    }

    // MARK: - Seek Tests

    func testSeekBacking_UpdatesCurrentTime() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._duration = 60.0

        // When
        sut.seekBacking(to: 30.0)

        // Then
        XCTAssertTrue(mockBackingTrackPlayer.seekCalled)
        XCTAssertEqual(mockBackingTrackPlayer.seekToTime, 30.0)
    }

    // MARK: - Stop Backing Tests

    func testStopBacking_StopsPlaybackAndResetsTime() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._isPlaying = true

        // When
        await sut.stopBacking()

        // Then
        XCTAssertTrue(mockBackingTrackPlayer.stopCalled)
        XCTAssertFalse(sut.isBackingPlaying)
    }

    // MARK: - Duration Update Tests

    func testBackingDuration_ReturnsPlayerDuration() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 120.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._duration = 120.0

        // When
        sut.updateBackingPlayerState()

        // Then
        XCTAssertEqual(sut.backingDuration, 120.0)
    }

    // MARK: - Clear Track Tests

    func testClearBackingTrack_StopsPlaybackAndClearsSelection() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        sut.selectedBackingTrack = sut.availableBackingTracks.first
        sut.selectedBackingSource = .original
        mockBackingTrackPlayer._isPlaying = true

        // When
        await sut.clearBackingTrackWithStop()

        // Then
        XCTAssertTrue(mockBackingTrackPlayer.stopCalled)
        XCTAssertNil(sut.selectedBackingTrack)
        XCTAssertNil(sut.selectedBackingSource)
        XCTAssertFalse(sut.isBackingPlaying)
    }

    // MARK: - Track Selection Tests

    func testSelectBackingTrack_ResetsPlaybackStateWhenChangingTrack() async {
        // Given - Two recordings available
        let recording1 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test1.m4a"),
            duration: Duration(seconds: 60.0)
        )
        let recording2 = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test2.m4a"),
            duration: Duration(seconds: 90.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording1, recording2]
        await sut.loadBackingTracks()

        // Start playback on first track
        await sut.selectBackingTrack(sut.availableBackingTracks.first)
        mockBackingTrackPlayer.playDurationNanoseconds = 1_000_000_000
        await sut.toggleBackingPlayback()
        try? await Task.sleep(nanoseconds: 50_000_000)

        // Verify playing
        XCTAssertTrue(sut.isBackingPlaying)

        // When - Change to second track
        await sut.selectBackingTrack(sut.availableBackingTracks.last)

        // Then - Playback should stop and state should reset
        XCTAssertTrue(mockBackingTrackPlayer.stopCalled)
        XCTAssertFalse(sut.isBackingPlaying)
        XCTAssertEqual(sut.backingCurrentTime, 0)
        XCTAssertEqual(sut.selectedBackingTrack?.recording.id, recording2.id)
    }

    func testSelectBackingTrack_SetsFirstSourceAutomatically() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        await sut.loadBackingTracks()

        // When
        await sut.selectBackingTrack(sut.availableBackingTracks.first)

        // Then
        XCTAssertNotNil(sut.selectedBackingTrack)
        XCTAssertEqual(sut.selectedBackingSource, .original)
    }

    func testSelectBackingSource_ResetsPlaybackState() async {
        // Given
        let recording = Recording(
            fileURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockRecordingRepository.recordingsToReturn = [recording]
        // Add extracted audio to have multiple sources
        let extractedVocal = ExtractedAudio(
            sourceRecordingId: recording.id,
            type: .vocal,
            fileURL: URL(fileURLWithPath: "/tmp/test_vocal.m4a"),
            duration: Duration(seconds: 60.0)
        )
        mockExtractedAudioRepository.storedAudios = [extractedVocal]
        await sut.loadBackingTracks()

        // Verify tracks were loaded with correct sources
        XCTAssertEqual(sut.availableBackingTracks.count, 1, "Should have 1 backing track")
        let track = sut.availableBackingTracks.first
        XCTAssertNotNil(track, "Track should exist")
        XCTAssertEqual(track?.availableSources.count, 2, "Track should have 2 sources (original + vocal)")
        XCTAssertTrue(track?.availableSources.contains(.original) ?? false, "Should have original source")
        XCTAssertTrue(track?.availableSources.contains(.vocal) ?? false, "Should have vocal source")

        // Set playback duration BEFORE any playback operations
        mockBackingTrackPlayer.playDurationNanoseconds = 5_000_000_000 // 5 seconds
        mockBackingTrackPlayer._duration = 60.0

        await sut.selectBackingTrack(track)

        // Debug: Verify selection state
        XCTAssertNotNil(sut.selectedBackingTrack, "Track should be selected")
        XCTAssertEqual(sut.selectedBackingSource, .original, "Initial source should be original")

        // Start playback
        await sut.toggleBackingPlayback()

        // Wait for Task to start
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // Debug: Check playback state
        let playCalled = mockBackingTrackPlayer.playCalled
        let isPlaying = sut.isBackingPlaying
        XCTAssertTrue(playCalled, "Play should have been called on mock")
        XCTAssertTrue(isPlaying, "isBackingPlaying should be true after toggleBackingPlayback")

        // Only proceed with source change test if playback started
        guard playCalled && isPlaying else {
            XCTFail("Playback did not start properly - playCalled: \(playCalled), isPlaying: \(isPlaying)")
            return
        }

        // Reset stopCalled and playCalled flags
        mockBackingTrackPlayer.stopCalled = false
        mockBackingTrackPlayer.playCalled = false

        // When - Change source to vocal
        await sut.selectBackingSource(.vocal)

        // Wait for auto-restart playback Task to execute
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // Then - Playback should stop, state should reset, then auto-restart on new source
        XCTAssertTrue(mockBackingTrackPlayer.stopCalled, "Stop should be called when changing source while playing")
        // Note: selectBackingSource auto-restarts playback if wasPlaying was true (seamless source switch)
        XCTAssertTrue(mockBackingTrackPlayer.playCalled, "Play should be called again for new source")
        XCTAssertTrue(sut.isBackingPlaying, "Should continue playing after seamless source change")
        XCTAssertEqual(sut.selectedBackingSource, .vocal, "Source should be updated to vocal")
    }
}
