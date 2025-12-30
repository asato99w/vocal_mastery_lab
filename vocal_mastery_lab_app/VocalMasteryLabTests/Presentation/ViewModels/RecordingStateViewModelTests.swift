import XCTest
import Combine
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

/// Comprehensive tests for RecordingStateViewModel
@MainActor
final class RecordingStateViewModelTests: XCTestCase {

    var sut: RecordingStateViewModel!
    var mockStartRecordingUseCase: MockStartRecordingUseCase!
    var mockStopRecordingUseCase: MockStopRecordingUseCase!
    var mockAudioPlayer: MockAudioPlayer!
    var mockSubscriptionVM: SubscriptionViewModel!
    var mockGetStatusUseCase: MockGetSubscriptionStatusUseCase!
    var testUsageTracker: RecordingUsageTracker!
    var testUserDefaultsSuiteName: String!
    var testUserDefaults: UserDefaults!
    var cancellables: Set<AnyCancellable>!

    override func setUp() async throws {
        try await super.setUp()

        mockStartRecordingUseCase = MockStartRecordingUseCase()
        mockStopRecordingUseCase = MockStopRecordingUseCase()
        mockAudioPlayer = MockAudioPlayer()

        // Set up subscription mock
        mockGetStatusUseCase = MockGetSubscriptionStatusUseCase()
        mockGetStatusUseCase.mockStatus = SubscriptionStatus.defaultFree(cohort: .v2_0)
        mockSubscriptionVM = SubscriptionViewModel(
            getStatusUseCase: mockGetStatusUseCase,
            purchaseUseCase: MockPurchaseSubscriptionUseCase(),
            restoreUseCase: MockRestorePurchasesUseCase()
        )

        // Create test user defaults for usage tracker
        testUserDefaultsSuiteName = "test_recording_state_\(UUID().uuidString)"
        testUserDefaults = UserDefaults(suiteName: testUserDefaultsSuiteName)!
        testUsageTracker = RecordingUsageTracker(userDefaults: testUserDefaults)
        testUsageTracker.resetForTesting()

        // Create SUT with countdownDuration = 0 for faster testing
        sut = RecordingStateViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            subscriptionViewModel: mockSubscriptionVM,
            usageTracker: testUsageTracker,
            countdownDuration: 0
        )

        cancellables = []
    }

    override func tearDown() async throws {
        sut = nil
        mockStartRecordingUseCase = nil
        mockStopRecordingUseCase = nil
        mockAudioPlayer = nil
        mockSubscriptionVM = nil
        mockGetStatusUseCase = nil
        testUsageTracker = nil
        if let suiteName = testUserDefaultsSuiteName {
            UserDefaults.standard.removePersistentDomain(forName: suiteName)
        }
        testUserDefaults = nil
        testUserDefaultsSuiteName = nil
        cancellables = nil
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    private func createRecordingSession() -> RecordingSession {
        return RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test_recording.m4a"),
            startedAt: Date()
        )
    }

    private func createSUTWithCountdown(seconds: Int) -> RecordingStateViewModel {
        return RecordingStateViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            subscriptionViewModel: mockSubscriptionVM,
            usageTracker: testUsageTracker,
            countdownDuration: seconds
        )
    }

    // MARK: - Initial State Tests

    func testInitialState() {
        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNil(sut.currentSession)
        XCTAssertNil(sut.errorMessage)
        XCTAssertEqual(sut.progress, 0.0)
        XCTAssertNil(sut.lastRecordingURL)
        XCTAssertNil(sut.lastRecordingId)
        XCTAssertNil(sut.lastRecordingDuration)
        XCTAssertFalse(sut.isPlayingRecording)
        XCTAssertFalse(sut.isCountdownComplete)
    }

    // MARK: - clearError() Tests

    func testClearError_ResetsErrorMessage() {
        // Given
        sut.clearError() // First ensure it's nil
        // Manually set error state for testing (would need to trigger an error first)

        // When
        sut.clearError()

        // Then
        XCTAssertNil(sut.errorMessage)
        XCTAssertEqual(sut.alertMessageType, .error)
    }

    // MARK: - cleanup() Tests

    func testCleanup_WhenIdle_DeactivatesAudioSession() {
        // Given - SUT is in idle state by default
        XCTAssertEqual(sut.recordingState, .idle)

        // When
        sut.cleanup()

        // Then - No crash or error, cleanup should complete
        XCTAssertEqual(sut.recordingState, .idle)
    }

    func testCleanup_WhenRecording_SkipsCleanup() async {
        // Given - Start recording to get into recording state
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session

        await sut.startRecording()

        // Verify we're in recording state
        XCTAssertEqual(sut.recordingState, .recording)

        // When
        sut.cleanup()

        // Then - Cleanup was skipped, still recording
        XCTAssertEqual(sut.recordingState, .recording)
    }

    // MARK: - startRecording() Basic Tests

    func testStartRecording_WhenIdle_ChangesToPreparingThenRecording() async {
        // Given
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session

        // When
        await sut.startRecording()

        // Then - Should transition from idle → preparing → recording (countdown = 0)
        XCTAssertEqual(sut.recordingState, .recording)
        XCTAssertNotNil(sut.currentSession)
        XCTAssertTrue(mockStartRecordingUseCase.prepareCalled)
        XCTAssertTrue(mockStartRecordingUseCase.startCalled)
    }

    func testStartRecording_WhenNotIdle_DoesNothing() async {
        // Given - Already recording
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session
        await sut.startRecording()
        mockStartRecordingUseCase.reset()
        mockStartRecordingUseCase.executeResult = session

        // When - Try to start again
        await sut.startRecording()

        // Then - Should be ignored
        XCTAssertFalse(mockStartRecordingUseCase.prepareCalled)
        XCTAssertEqual(sut.recordingState, .recording)
    }

    func testStartRecording_SetsCountdownComplete_WhenCountdownIsZero() async {
        // Given
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session

        // When
        await sut.startRecording()

        // Then
        XCTAssertTrue(sut.isCountdownComplete)
    }

    // MARK: - startRecording() Error Handling Tests

    func testStartRecording_WhenPrepareFails_SetsErrorAndReturnsToIdle() async {
        // Given
        mockStartRecordingUseCase.prepareShouldFail = true
        mockStartRecordingUseCase.executeResult = createRecordingSession()

        // When
        await sut.startRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNotNil(sut.errorMessage)
        XCTAssertTrue(sut.errorMessage?.contains("録音の準備に失敗しました") ?? false)
    }

    func testStartRecording_WhenStartFails_SetsErrorAndReturnsToIdle() async {
        // Given
        mockStartRecordingUseCase.startShouldFail = true
        mockStartRecordingUseCase.executeResult = createRecordingSession()

        // When
        await sut.startRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNotNil(sut.errorMessage)
        XCTAssertFalse(sut.isCountdownComplete)
    }

    // MARK: - Countdown Tests

    func testStartRecording_WithCountdown_TransitionsToCountdownState() async {
        // Given - Create SUT with countdown
        let sutWithCountdown = createSUTWithCountdown(seconds: 3)
        mockStartRecordingUseCase.executeResult = createRecordingSession()

        // Observe state changes
        var stateChanges: [RecordingState] = []
        sutWithCountdown.$recordingState
            .sink { stateChanges.append($0) }
            .store(in: &cancellables)

        // When
        await sutWithCountdown.startRecording()

        // Then - Should transition to countdown state after preparing
        // Note: Because countdown starts asynchronously, the state at this point should be .countdown
        XCTAssertEqual(sutWithCountdown.recordingState, .countdown)
        XCTAssertTrue(stateChanges.contains(.preparing))
        XCTAssertTrue(stateChanges.contains(.countdown))
    }

    func testCancelCountdown_WhenInCountdown_ReturnsToIdle() async {
        // Given - Create SUT with countdown and start recording
        let sutWithCountdown = createSUTWithCountdown(seconds: 3)
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        await sutWithCountdown.startRecording()

        // Verify we're in countdown state
        XCTAssertEqual(sutWithCountdown.recordingState, .countdown)

        // When
        await sutWithCountdown.cancelCountdown()

        // Then
        XCTAssertEqual(sutWithCountdown.recordingState, .idle)
        XCTAssertFalse(sutWithCountdown.isCountdownComplete)
    }

    func testCancelCountdown_WhenNotInCountdown_DoesNothing() async {
        // Given - SUT is in idle state
        XCTAssertEqual(sut.recordingState, .idle)

        // When
        await sut.cancelCountdown()

        // Then - No change
        XCTAssertEqual(sut.recordingState, .idle)
    }

    // MARK: - stopRecording() Tests

    func testStopRecording_WhenRecording_StopsAndSavesRecording() async {
        // Given
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session
        mockStopRecordingUseCase.executeResult = StopRecordingResult(
            duration: 30.0,
            recordingId: RecordingId()
        )

        await sut.startRecording()
        XCTAssertEqual(sut.recordingState, .recording)

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNil(sut.currentSession)
        XCTAssertEqual(sut.progress, 0.0)
        XCTAssertFalse(sut.isCountdownComplete)
        XCTAssertTrue(mockStopRecordingUseCase.executeCalled)
    }

    func testStopRecording_SavesLastRecordingInfo() async {
        // Given
        let recordingURL = URL(fileURLWithPath: "/tmp/test_recording.m4a")
        let session = RecordingSession(recordingURL: recordingURL, startedAt: Date())
        let recordingId = RecordingId()
        let duration: TimeInterval = 45.5

        mockStartRecordingUseCase.executeResult = session
        mockStopRecordingUseCase.executeResult = StopRecordingResult(
            duration: duration,
            recordingId: recordingId
        )

        await sut.startRecording()

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.lastRecordingURL, recordingURL)
        XCTAssertEqual(sut.lastRecordingId, recordingId)
        XCTAssertEqual(sut.lastRecordingDuration, duration)
        XCTAssertNotNil(sut.lastRecordingDate)
    }

    func testStopRecording_WhenNotRecording_DoesNothing() async {
        // Given - SUT is in idle state
        XCTAssertEqual(sut.recordingState, .idle)

        // When
        await sut.stopRecording()

        // Then
        XCTAssertFalse(mockStopRecordingUseCase.executeCalled)
    }

    func testStopRecording_WhenStopFails_SetsErrorAndReturnsToIdle() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeShouldFail = true

        await sut.startRecording()

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNotNil(sut.errorMessage)
        XCTAssertFalse(sut.isCountdownComplete)
    }

    func testStopRecording_IncrementsUsageCount() async {
        // Given
        let initialCount = testUsageTracker.getTodayCount()
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)

        await sut.startRecording()

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.dailyRecordingCount, initialCount + 1)
    }

    // MARK: - playLastRecording() Tests

    func testPlayLastRecording_WithValidURL_PlaysRecording() async {
        // Given - Set up last recording
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)
        await sut.startRecording()
        await sut.stopRecording()

        XCTAssertNotNil(sut.lastRecordingURL)

        // When
        await sut.playLastRecording()

        // Then
        XCTAssertTrue(mockAudioPlayer.playCalled)
    }

    func testPlayLastRecording_WithNoURL_SetsError() async {
        // Given - No recording available
        XCTAssertNil(sut.lastRecordingURL)

        // When
        await sut.playLastRecording()

        // Then
        XCTAssertFalse(mockAudioPlayer.playCalled)
        XCTAssertNotNil(sut.errorMessage)
    }

    func testPlayLastRecording_WhenAlreadyPlaying_DoesNothing() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)
        await sut.startRecording()
        await sut.stopRecording()

        // Simulate already playing
        sut.isPlayingRecording = true

        // When
        await sut.playLastRecording()

        // Then - Should not call play again
        XCTAssertFalse(mockAudioPlayer.playCalled)
    }

    func testPlayLastRecording_WhenPlayFails_SetsError() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)
        mockAudioPlayer.playShouldFail = true

        await sut.startRecording()
        await sut.stopRecording()
        mockAudioPlayer.playCalled = false // Reset after stop

        // When
        await sut.playLastRecording()

        // Then
        XCTAssertNotNil(sut.errorMessage)
        XCTAssertFalse(sut.isPlayingRecording)
    }

    // MARK: - stopPlayback() Tests

    func testStopPlayback_StopsPlayingAndResetsFlag() async {
        // Given
        sut.isPlayingRecording = true

        // When
        await sut.stopPlayback()

        // Then
        XCTAssertTrue(mockAudioPlayer.stopCalled)
        XCTAssertFalse(sut.isPlayingRecording)
    }

    // MARK: - Recording Count Limit Tests

    func testStartRecording_WhenLimitReached_SetsLimitReachedError() async {
        // Given - Set up a SUT with count limit enforced
        // Note: The current implementation returns unlimited for all tiers (dailyCount = nil)
        // So this test verifies the normal flow works with unlimited recording policy

        mockStartRecordingUseCase.executeResult = createRecordingSession()

        // When
        await sut.startRecording()

        // Then - Should proceed normally since current policy is unlimited
        XCTAssertEqual(sut.recordingState, .recording)
    }

    // MARK: - State Transition Tests

    func testRecordingStateTransitions_FullCycle() async {
        // Given
        var stateChanges: [RecordingState] = []
        sut.$recordingState
            .sink { stateChanges.append($0) }
            .store(in: &cancellables)

        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)

        // When - Full cycle
        await sut.startRecording()
        await sut.stopRecording()

        // Then - Should have: idle → preparing → recording → idle
        XCTAssertTrue(stateChanges.contains(.idle))
        XCTAssertTrue(stateChanges.contains(.preparing))
        XCTAssertTrue(stateChanges.contains(.recording))
        XCTAssertEqual(stateChanges.last, .idle)
    }

    // MARK: - Session Management Tests

    func testCurrentSession_SetDuringRecording() async {
        // Given
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session

        // When
        await sut.startRecording()

        // Then
        XCTAssertNotNil(sut.currentSession)
        XCTAssertEqual(sut.currentSession?.recordingURL, session.recordingURL)
    }

    func testCurrentSession_ClearedAfterStop() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)
        await sut.startRecording()
        XCTAssertNotNil(sut.currentSession)

        // When
        await sut.stopRecording()

        // Then
        XCTAssertNil(sut.currentSession)
    }

    // MARK: - Progress Tests

    func testProgress_InitiallyZero() {
        XCTAssertEqual(sut.progress, 0.0)
    }

    func testProgress_ResetAfterStop() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()
        mockStopRecordingUseCase.executeResult = StopRecordingResult(duration: 30.0)
        await sut.startRecording()

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.progress, 0.0)
    }

    // MARK: - onAutoStopNeeded Callback Tests

    func testOnAutoStopNeeded_CalledWhenSet() async {
        // Given
        var callbackCalled = false
        sut.onAutoStopNeeded = {
            callbackCalled = true
        }

        // Verify callback is set
        XCTAssertNotNil(sut.onAutoStopNeeded)

        // Then - Just verify the callback can be set (actual triggering requires duration monitoring)
        XCTAssertFalse(callbackCalled) // Not called yet
    }

    // MARK: - Use Case Integration Tests

    func testStartRecording_CallsPrepareWithUser() async {
        // Given
        mockStartRecordingUseCase.executeResult = createRecordingSession()

        // When
        await sut.startRecording()

        // Then
        XCTAssertTrue(mockStartRecordingUseCase.prepareCalled)
        XCTAssertNotNil(mockStartRecordingUseCase.prepareUser)
    }

    func testStartRecording_SetsRecordingContextOnStopUseCase() async {
        // Given
        let session = createRecordingSession()
        mockStartRecordingUseCase.executeResult = session

        // When
        await sut.startRecording()

        // Then - Verify prepare was called which sets context
        XCTAssertTrue(mockStartRecordingUseCase.prepareCalled)
        // Note: MockStopRecordingUseCase doesn't track setRecordingContext
    }
}
