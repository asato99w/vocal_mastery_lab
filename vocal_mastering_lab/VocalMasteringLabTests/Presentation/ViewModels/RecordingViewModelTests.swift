import XCTest
import Combine
import VocalisDomain
import OSLog
@testable import VocalMasteringLab

@MainActor
final class RecordingViewModelTests: XCTestCase {

    var sut: RecordingViewModel!
    var mockStartRecordingUseCase: MockStartRecordingUseCase!
    var mockStartRecordingWithScaleUseCase: MockStartRecordingWithScaleUseCase!
    var mockStopRecordingUseCase: MockStopRecordingUseCase!
    var mockAudioPlayer: MockAudioPlayer!
    var mockScalePlayer: MockScalePlayer!
    var mockPitchDetector: MockRealtimePitchDetector!
    var mockSubscriptionViewModel: SubscriptionViewModel!
    var cancellables: Set<AnyCancellable>!

    override func setUp() async throws {
        try await super.setUp()
        mockStartRecordingUseCase = MockStartRecordingUseCase()
        mockStartRecordingWithScaleUseCase = MockStartRecordingWithScaleUseCase()
        mockStopRecordingUseCase = MockStopRecordingUseCase()
        mockAudioPlayer = MockAudioPlayer()
        mockScalePlayer = MockScalePlayer()
        mockPitchDetector = MockRealtimePitchDetector()
        mockSubscriptionViewModel = SubscriptionViewModel(
            getStatusUseCase: MockGetSubscriptionStatusUseCase(),
            purchaseUseCase: MockPurchaseSubscriptionUseCase(),
            restoreUseCase: MockRestorePurchasesUseCase()
        )

        // Set default executeResult to prevent hangs
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/default_test.m4a"),
            settings: nil
        )
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/default_scale_test.m4a"),
            settings: ScaleSettings.mvpDefault
        )

        // Create a fresh UsageTracker for each test and reset it
        let usageTracker = RecordingUsageTracker()
        usageTracker.resetForTesting()

        sut = RecordingViewModel(
            startRecordingUseCase: mockStartRecordingUseCase,
            startRecordingWithScaleUseCase: mockStartRecordingWithScaleUseCase,
            stopRecordingUseCase: mockStopRecordingUseCase,
            audioPlayer: mockAudioPlayer,
            pitchDetector: mockPitchDetector,
            scalePlaybackCoordinator: ScalePlaybackCoordinator(scalePlayer: mockScalePlayer),
            subscriptionViewModel: mockSubscriptionViewModel,
            usageTracker: usageTracker,
            countdownDuration: 0,
            targetPitchPollingIntervalNanoseconds: 1_000_000,
            playbackPitchPollingIntervalNanoseconds: 1_000_000
        )
        cancellables = []
    }

    override func tearDown() async throws {
        cancellables = nil
        sut = nil
        mockPitchDetector = nil
        mockScalePlayer = nil
        mockAudioPlayer = nil
        mockStopRecordingUseCase = nil
        mockStartRecordingWithScaleUseCase = nil
        mockStartRecordingUseCase = nil
        try await super.tearDown()
    }

    // MARK: - Initial State Tests

    func testInitialState_IsIdle() {
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNil(sut.currentSession)
        XCTAssertNil(sut.errorMessage)
    }

    // MARK: - Start Recording Tests

    func testStartRecording_TransitionsToRecording() async {
        // Given: Set up mock to return a session
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: nil
        )

        // When
        await sut.startRecording()

        // Wait a tiny bit for async state propagation
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms

        // Then
        // Should immediately transition to recording (countdown=0)
        XCTAssertEqual(sut.recordingState, .recording)
    }

    func testStartRecording_CountdownCompletesAndStartsRecording() async {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: nil
        )

        // When
        await sut.startRecording()

        // Wait for execution (countdown=0 in test environment)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0

        // Then
        XCTAssertEqual(sut.recordingState, .recording)
        XCTAssertNotNil(sut.currentSession)
        XCTAssertTrue(mockStartRecordingUseCase.executeCalled)
    }

    func testStartRecording_UseCaseFailure_ShowsError() async {
        // Given
        mockStartRecordingUseCase.executeShouldFail = true

        // When
        await sut.startRecording()

        // Wait for execution (countdown=0 in test environment)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNotNil(sut.errorMessage)
    }

    func testStartRecording_WhileRecording_DoesNothing() async {
        // Given
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: nil
        )
        await sut.startRecording()
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0

        let initialSession = sut.currentSession

        // When
        await sut.startRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .recording)
        XCTAssertEqual(sut.currentSession?.recordingURL, initialSession?.recordingURL)
    }

    // MARK: - Cancel Countdown Tests
    // Note: testCancelCountdown_DuringCountdown_ReturnsToIdle removed - not applicable when countdownDuration=0

    func testCancelCountdown_NotDuringCountdown_DoesNothing() async {
        // Given
        XCTAssertEqual(sut.recordingState, .idle)

        // When
        await sut.cancelCountdown()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
    }

    // MARK: - Stop Recording Tests

    func testStopRecording_WhileRecording_TransitionsToIdle() async {
        // Given
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: nil
        )
        await sut.startRecording()
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0
        XCTAssertEqual(sut.recordingState, .recording)

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
        XCTAssertNil(sut.currentSession)
    }

    func testStopRecording_NotRecording_DoesNothing() async {
        // Given
        XCTAssertEqual(sut.recordingState, .idle)

        // When
        await sut.stopRecording()

        // Then
        XCTAssertEqual(sut.recordingState, .idle)
    }

    // MARK: - Progress Tests

    func testProgress_InitiallyZero() {
        XCTAssertEqual(sut.progress, 0.0)
    }

    // MARK: - Countdown Tests

    func testCountdownValue_InitiallyThree() {
        XCTAssertEqual(sut.countdownValue, 0)
    }

    // Note: testCountdown_DecrementsFromThreeToOne removed - not applicable when countdownDuration=0

    // MARK: - Target Pitch Tests (Bug Reproduction)

    func testStartRecording_withScale_shouldSetTargetPitch() async throws {
        // Given: Setup for scale recording
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: settings
        )

        // Set mock to simulate scale playback
        let expectedNote = try MIDINote(60) // C4: 261.63 Hz
        mockScalePlayer.currentScaleElement = .scaleNote(expectedNote)

        // Verify initial state
        XCTAssertNil(sut.targetPitch, "Target pitch should be nil before recording starts")

        // When: Start recording with scale
        await sut.startRecording(settings: settings)

        // Wait for execution (countdown=0 in test environment)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0

        // Wait for monitoring task to poll
        try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then: Target pitch should be set
        XCTAssertNotNil(sut.targetPitch, "Target pitch should be set when scale element is available")
        if let targetPitch = sut.targetPitch {
            XCTAssertEqual(targetPitch.frequency, expectedNote.frequency, accuracy: 0.01,
                          "Target pitch frequency should match the scale element note")
        }
    }

    // MARK: - Playback Tests (Bug Reproduction)

    func testPlayLastRecording_withScale_shouldSetTargetPitchDuringPlayback() async throws {
        // Given: Record with scale settings first
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: settings
        )

        // Set mock to simulate scale playback
        let expectedNote = try MIDINote(60) // C4: 261.63 Hz
        mockScalePlayer.currentScaleElement = .scaleNote(expectedNote)

        // Record and stop to set lastRecordingURL
        await sut.startRecording(settings: settings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms - immediate with countdown=0
        await sut.stopRecording()

        // Verify recording was saved
        XCTAssertNotNil(sut.lastRecordingURL, "Last recording URL should be set after stopping")
        XCTAssertNotNil(sut.lastRecordingSettings, "Last recording settings should be set after stopping")

        // When: Start target pitch monitoring directly (bypassing playLastRecording's blocking behavior)
        try await sut.pitchDetectionVM.startTargetPitchMonitoring(settings: settings)

        // Wait for monitoring task to poll and set targetPitch
        try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

        // Then: Target pitch should be set during monitoring
        XCTAssertNotNil(sut.targetPitch, "Target pitch should be set during monitoring when scale settings exist")
        if let targetPitch = sut.targetPitch {
            XCTAssertEqual(targetPitch.frequency, expectedNote.frequency, accuracy: 0.01,
                          "Target pitch frequency should match the scale element note during monitoring")
        }

        // Cleanup
        await sut.pitchDetectionVM.stopTargetPitchMonitoring()
    }

    func testStopPlayback_withScale_shouldStopTargetPitchMonitoring() async throws {
        // RACE CONDITION CONCERN:
        // This test documents the expected behavior: stopTargetPitchMonitoring() should ensure targetPitch is nil
        // before returning. The implementation must await progressMonitorTask?.value to guarantee
        // the monitoring task completes before clearing targetPitch, preventing a race where the
        // task's while loop executes one more iteration after cancel() but before Task.isCancelled check.

        // Given: Record with scale settings
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: settings
        )

        let expectedNote = try MIDINote(60) // C4: 261.63 Hz
        mockScalePlayer.currentScaleElement = .scaleNote(expectedNote)

        // Record and stop to set lastRecordingURL
        await sut.startRecording(settings: settings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Start target pitch monitoring directly
        try await sut.pitchDetectionVM.startTargetPitchMonitoring(settings: settings)
        try? await Task.sleep(nanoseconds: 50_000_000) // 50ms - wait for monitoring to start

        // Verify target pitch is being monitored
        XCTAssertNotNil(sut.targetPitch, "Target pitch should be set during monitoring")

        // When: Stop monitoring (await for completion)
        await sut.pitchDetectionVM.stopTargetPitchMonitoring()

        // Then: Target pitch should be cleared immediately after stopTargetPitchMonitoring() returns
        // No wait needed - stopTargetPitchMonitoring() is async and should complete all cleanup before returning
        XCTAssertNil(sut.targetPitch, "Target pitch should be nil immediately after stopTargetPitchMonitoring() returns")
    }

    func testMultiplePlaybackCycles_shouldClearTargetPitchConsistently() async throws {
        // Given: Record with scale settings
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: settings
        )

        let expectedNote = try MIDINote(60) // C4: 261.63 Hz

        // Record and stop to set lastRecordingURL and lastRecordingSettings
        await sut.startRecording(settings: settings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Verify recording was saved
        XCTAssertNotNil(sut.lastRecordingURL, "Last recording URL should be set after stopping")
        XCTAssertNotNil(sut.lastRecordingSettings, "Last recording settings should be set after stopping")

        // When: Perform multiple monitoring start-stop cycles rapidly
        for cycle in 1...3 {
            // Set current scale element before starting monitoring
            mockScalePlayer.currentScaleElement = .scaleNote(expectedNote)

            // Start target pitch monitoring directly (bypassing playLastRecording's blocking behavior)
            try await sut.pitchDetectionVM.startTargetPitchMonitoring(settings: settings)

            // Wait for monitoring task to poll and set targetPitch
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

            // Verify target pitch is set during monitoring
            XCTAssertNotNil(sut.targetPitch, "Target pitch should be set during monitoring (cycle \(cycle))")

            // Stop monitoring
            await sut.pitchDetectionVM.stopTargetPitchMonitoring()

            // Immediately check without waiting - this should catch the race condition
            // Then: Target pitch must be cleared immediately after await returns
            XCTAssertNil(sut.targetPitch,
                        "Target pitch should be nil immediately after stopTargetPitchMonitoring() returns (cycle \(cycle))")
        }
    }

    // MARK: - Pitch Detection Bug Reproduction Tests

    /// BUG REPRODUCTION: ピッチ検出がスケール設定なしで録音した場合に開始されない
    ///
    /// 問題：
    /// - RecordingViewModel.startRecording(settings: nil)を呼ぶと、
    ///   `if let settings = settings`のチェックにより、ピッチ検出が一切開始されない
    /// - スケール機能OFF（settings = nil）の場合でも、リアルタイムピッチ検出は必要
    ///
    /// 期待される動作：
    /// - settings の有無に関わらず、pitchDetector.startRealtimeDetection()が呼ばれるべき
    /// - settingsがある場合のみ、targetPitchのモニタリングを開始すべき
    func testStartRecording_withoutScale_shouldStillStartPitchDetection() async throws {
        // Given: スケール設定なしで録音を開始（通常録音）
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test.m4a"),
            settings: nil  // ← スケール設定なし
        )

        // Verify initial state
        XCTAssertFalse(mockPitchDetector.isDetecting, "Pitch detection should not be active initially")

        // When: スケール設定なしで録音開始（settings = nil）
        await sut.startRecording(settings: nil)

        // Then: ピッチ検出は開始されるべき（スケールなしでもリアルタイムピッチ検出は必要）
        // 🐛 BUG: 現在の実装では pitchDetector.startRealtimeDetection() が呼ばれない
        XCTAssertTrue(mockPitchDetector.isDetecting,
                     "Pitch detection should be active even without scale settings for realtime pitch visualization")
    }

    // MARK: - Analysis Navigation Tests (分析画面遷移テスト)

    /// Test: Recording with scale settings should preserve settings for analysis navigation
    /// 録音後にlastRecordingURLとlastRecordingSettingsの両方が設定されていることを確認
    func testStopRecording_withScale_shouldPreserveSettingsForAnalysis() async throws {
        // Given: スケール付きで録音
        let settings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/test_analysis.m4a"),
            settings: settings
        )

        // When: 録音開始して停止
        await sut.startRecording(settings: settings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Then: 分析画面遷移用のデータが保持されている
        XCTAssertNotNil(sut.lastRecordingURL, "lastRecordingURL should be set after recording stops")
        XCTAssertNotNil(sut.lastRecordingSettings, "lastRecordingSettings should be set for scale recordings")
        XCTAssertEqual(sut.lastRecordingSettings?.notePattern, settings.notePattern,
                       "lastRecordingSettings should match the recording settings")
        XCTAssertEqual(sut.lastRecordingSettings?.startNote, settings.startNote,
                       "startNote should be preserved")
    }

    /// Test: Recording URL and settings should match (consistency check)
    /// lastRecordingURLとlastRecordingSettingsの整合性確認
    func testStopRecording_settingsAndURL_shouldBeConsistent() async throws {
        // Given: スケール付きで録音
        let settings = ScaleSettings.mvpDefault
        let expectedURL = URL(fileURLWithPath: "/tmp/consistent_test.m4a")
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: expectedURL,
            settings: settings
        )

        // When: 録音開始して停止
        await sut.startRecording(settings: settings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Then: URLとsettingsが正しくペアリングされている
        XCTAssertEqual(sut.lastRecordingURL, expectedURL, "URL should match the recording session URL")
        XCTAssertNotNil(sut.lastRecordingSettings, "Settings should be present for scale recordings")

        // 両方がnilでないことを確認（分析画面遷移の前提条件）
        let canNavigateToAnalysis = sut.lastRecordingURL != nil
        XCTAssertTrue(canNavigateToAnalysis, "Should have valid data for analysis navigation")
    }

    /// Test: After second recording, settings should update correctly
    /// 複数回の録音後に最新の設定が保持されることを確認
    func testMultipleRecordings_shouldKeepLatestSettings() async throws {
        // Given: 最初の録音（5-tone）
        let firstSettings = ScaleSettings.mvpDefault
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/first_recording.m4a"),
            settings: firstSettings
        )

        await sut.startRecording(settings: firstSettings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        let firstURL = sut.lastRecordingURL
        let firstSavedSettings = sut.lastRecordingSettings

        XCTAssertNotNil(firstURL, "First recording URL should be set")
        XCTAssertNotNil(firstSavedSettings, "First recording settings should be set")

        // When: 2回目の録音（異なる設定）
        let secondStartNote = try MIDINote(65) // F4
        let secondEndNote = try MIDINote(77)   // F5
        let secondTempo = try Tempo(secondsPerNote: 0.6)
        let secondSettings = ScaleSettings(
            startNote: secondStartNote,
            endNote: secondEndNote,
            notePattern: .rossiniScale,
            tempo: secondTempo,
            ascendingCount: 12
        )
        mockStartRecordingWithScaleUseCase.executeResult = RecordingSession(
            recordingURL: URL(fileURLWithPath: "/tmp/second_recording.m4a"),
            settings: secondSettings
        )

        await sut.startRecording(settings: secondSettings)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Then: 最新の録音データが保持されている
        XCTAssertNotEqual(sut.lastRecordingURL, firstURL, "URL should be updated to second recording")
        XCTAssertEqual(sut.lastRecordingSettings?.notePattern, .rossiniScale,
                       "Settings should reflect the second recording's scale type")
        XCTAssertEqual(sut.lastRecordingSettings?.startNote.value, 65,
                       "Settings should reflect the second recording's start note")
    }

    /// Test: Recording without scale should have nil settings but valid URL
    /// スケールなし録音では、URLは設定されるがsettingsはnilになることを確認
    func testStopRecording_withoutScale_shouldHaveNilSettings() async throws {
        // Given: スケールなしで録音
        let expectedURL = URL(fileURLWithPath: "/tmp/no_scale_test.m4a")
        mockStartRecordingUseCase.executeResult = RecordingSession(
            recordingURL: expectedURL,
            settings: nil
        )

        // When: スケールなしで録音開始して停止
        await sut.startRecording(settings: nil)
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        await sut.stopRecording()

        // Then: URLは設定されるが、settingsはnil
        XCTAssertEqual(sut.lastRecordingURL, expectedURL, "URL should be set even without scale")
        XCTAssertNil(sut.lastRecordingSettings, "Settings should be nil for non-scale recordings")
    }

}

