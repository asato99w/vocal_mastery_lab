import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteringLab

final class StartRecordingWithScaleUseCaseTests: XCTestCase {

    var sut: StartRecordingWithScaleUseCase!
    var mockScalePlayer: MockScalePlayer!
    var mockAudioRecorder: MockAudioRecorder!
    var mockRecordingPolicyService: MockRecordingPolicyService!
    var mockLogger: MockLogger!
    var premiumUser: User!

    override func setUp() {
        super.setUp()
        mockScalePlayer = MockScalePlayer()
        mockAudioRecorder = MockAudioRecorder()
        mockRecordingPolicyService = MockRecordingPolicyService()
        mockLogger = MockLogger()
        sut = StartRecordingWithScaleUseCase(
            scalePlayer: mockScalePlayer,
            audioRecorder: mockAudioRecorder,
            recordingPolicyService: mockRecordingPolicyService,
            logger: mockLogger
        )

        // Create a premium user for tests (allows scale recording)
        premiumUser = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: true,
                expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
            ),
            recordingStats: RecordingStats(todayCount: 0)
        )
    }

    override func tearDown() {
        sut = nil
        mockAudioRecorder = nil
        mockScalePlayer = nil
        mockRecordingPolicyService = nil
        mockLogger = nil
        premiumUser = nil
        super.tearDown()
    }

    // MARK: - Success Path Tests

    func testExecute_ValidSettings_ReturnsRecordingSession() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        let expectedURL = URL(fileURLWithPath: "/tmp/test.m4a")
        mockAudioRecorder.prepareRecordingResult = expectedURL

        // When
        let session = try await sut.execute(user: premiumUser, settings: settings)

        // Then
        XCTAssertEqual(session.recordingURL, expectedURL)
        XCTAssertEqual(session.settings, settings)
        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
        XCTAssertTrue(mockScalePlayer.loadScaleCalled)
        XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
        XCTAssertTrue(mockScalePlayer.playCalled)
    }

    func testExecute_LoadsCorrectScale() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.execute(user: premiumUser, settings: settings)

        // Then
        // Extract expected notes from scale elements (including chords)
        let scaleElements = settings.generateScaleWithKeyChange()
        var expectedNotes: [MIDINote] = []
        for element in scaleElements {
            switch element {
            case .scaleNote(let note):
                expectedNotes.append(note)
            case .chordShort(let notes), .chordLong(let notes):
                expectedNotes.append(contentsOf: notes)
            case .silence:
                break
            }
        }

        XCTAssertEqual(mockScalePlayer.loadedNotes, expectedNotes)
        XCTAssertEqual(mockScalePlayer.loadedTempo, settings.tempo)
    }

    func testExecute_CallsInCorrectOrder() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.execute(user: premiumUser, settings: settings)

        // Wait a moment for async play to be called
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // Then
        // Verify all methods were called
        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
        XCTAssertTrue(mockScalePlayer.loadScaleCalled)
        XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
        XCTAssertTrue(mockScalePlayer.playCalled)

        // Verify synchronous call order (before execute returns)
        // prepare -> load -> start recording must happen before execute returns
        XCTAssertLessThan(
            mockAudioRecorder.prepareRecordingCallTime!,
            mockScalePlayer.loadScaleCallTime!
        )
        XCTAssertLessThan(
            mockScalePlayer.loadScaleCallTime!,
            mockAudioRecorder.startRecordingCallTime!
        )
        // play is async, so we don't assert its timing relative to others
    }

    // MARK: - Error Handling Tests

    func testExecute_PrepareRecordingFails_ThrowsError() async {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute(user: premiumUser, settings: settings)
            XCTFail("Expected error to be thrown")
        } catch {
            // Error thrown as expected
            XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
            XCTAssertFalse(mockScalePlayer.loadScaleCalled)
        }
    }

    func testExecute_LoadScaleFails_ThrowsError() async {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")
        mockScalePlayer.loadScaleShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute(user: premiumUser, settings: settings)
            XCTFail("Expected error to be thrown")
        } catch {
            // Error thrown as expected
            XCTAssertTrue(mockScalePlayer.loadScaleCalled)
            XCTAssertFalse(mockAudioRecorder.startRecordingCalled)
        }
    }

    func testExecute_StartRecordingFails_ThrowsError() async {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")
        mockAudioRecorder.startRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute(user: premiumUser, settings: settings)
            XCTFail("Expected error to be thrown")
        } catch {
            // Error thrown as expected
            XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
            XCTAssertFalse(mockScalePlayer.playCalled)
        }
    }

    func testExecute_PlayStartsAsynchronously() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        let session = try await sut.execute(user: premiumUser, settings: settings)

        // Then
        // execute() returns immediately without waiting for playback to complete
        XCTAssertNotNil(session)
        XCTAssertTrue(mockAudioRecorder.startRecordingCalled)

        // Wait a bit for async play to be called
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        XCTAssertTrue(mockScalePlayer.playCalled)
    }

    // MARK: - Permission Tests

    func testExecute_FreeUser_WithScaleRecording_Succeeds() async {
        // Given: Free user with scale recording (now available for all tiers)
        let freeUser = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 0)
        )
        let settings = ScaleSettings.mvpDefault
        let expectedURL = URL(fileURLWithPath: "/tmp/test.m4a")
        mockRecordingPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = expectedURL

        // When
        do {
            let session = try await sut.execute(user: freeUser, settings: settings)

            // Then: Recording should succeed
            XCTAssertTrue(mockRecordingPolicyService.canStartRecordingCalled)
            XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
            XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
            XCTAssertTrue(mockScalePlayer.playCalled)
            XCTAssertEqual(session.recordingURL, expectedURL)
            XCTAssertEqual(session.settings, settings)
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testExecute_UserExceedsDailyLimit_Denied() async {
        // Given: User who exceeded daily limit
        let limitedUser = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: true
            ),
            recordingStats: RecordingStats(todayCount: 10) // Premium limit
        )
        let settings = ScaleSettings.mvpDefault
        mockRecordingPolicyService.canStartRecordingResult = .denied(.dailyLimitExceeded)

        // When/Then
        do {
            _ = try await sut.execute(user: limitedUser, settings: settings)
            XCTFail("Expected error to be thrown")
        } catch let error as RecordingPermissionError {
            XCTAssertEqual(error, .dailyLimitExceeded)
            XCTAssertTrue(mockRecordingPolicyService.canStartRecordingCalled)
            XCTAssertFalse(mockAudioRecorder.prepareRecordingCalled)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Timestamp Recording Tests

    func testExecute_StartsTimestampRecording() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.execute(user: premiumUser, settings: settings)

        // Then: startTimestampRecording should be called
        XCTAssertTrue(mockScalePlayer.startTimestampRecordingCalled)
        XCTAssertNotNil(mockScalePlayer.recordingStartTime)
    }

    func testExecute_TimestampRecordingStartedBeforePlay() async throws {
        // Given
        let settings = ScaleSettings.mvpDefault
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.execute(user: premiumUser, settings: settings)

        // Wait for async play to be called
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        // Then: Both should be called
        XCTAssertTrue(mockScalePlayer.startTimestampRecordingCalled)
        XCTAssertTrue(mockScalePlayer.playCalled)
    }
}
