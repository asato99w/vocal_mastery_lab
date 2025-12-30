import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

/// Comprehensive tests for StartRecordingUseCase
@MainActor
final class StartRecordingUseCaseTests: XCTestCase {

    var sut: StartRecordingUseCase!
    var mockAudioRecorder: MockAudioRecorder!
    var mockPolicyService: MockRecordingPolicyService!

    override func setUp() async throws {
        try await super.setUp()
        mockAudioRecorder = MockAudioRecorder()
        mockPolicyService = MockRecordingPolicyService()
        sut = StartRecordingUseCase(
            audioRecorder: mockAudioRecorder,
            recordingPolicyService: mockPolicyService
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockAudioRecorder = nil
        mockPolicyService = nil
        try await super.tearDown()
    }

    // MARK: - Helper Methods

    private func createTestUser() -> User {
        return User.new(cohort: .v2_0)
    }

    // MARK: - prepare() Success Tests

    func testPrepare_WhenPermissionAllowed_ReturnsRecordingURL() async throws {
        // Given
        let testUser = createTestUser()
        let expectedURL = URL(fileURLWithPath: "/tmp/test_recording.m4a")
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = expectedURL

        // When
        let resultURL = try await sut.prepare(user: testUser)

        // Then
        XCTAssertEqual(resultURL, expectedURL)
        XCTAssertTrue(mockPolicyService.canStartRecordingCalled)
        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
    }

    func testPrepare_ChecksPermissionBeforePreparing() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.prepare(user: testUser)

        // Then - Permission check should happen first
        XCTAssertTrue(mockPolicyService.canStartRecordingCalled)
        XCTAssertEqual(mockPolicyService.lastUser?.id, testUser.id)
    }

    func testPrepare_CallsPrepareRecordingOnAudioRecorder() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.prepare(user: testUser)

        // Then
        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
    }

    // MARK: - prepare() Permission Denied Tests

    func testPrepare_WhenDailyLimitExceeded_ThrowsDailyLimitExceededError() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .denied(.dailyLimitExceeded)

        // When/Then
        do {
            _ = try await sut.prepare(user: testUser)
            XCTFail("Expected dailyLimitExceeded error to be thrown")
        } catch let error as RecordingPermissionError {
            XCTAssertEqual(error, .dailyLimitExceeded)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        // Verify permission was checked but audio recorder was not called
        XCTAssertTrue(mockPolicyService.canStartRecordingCalled)
        XCTAssertFalse(mockAudioRecorder.prepareRecordingCalled)
    }

    func testPrepare_WhenPremiumRequired_ThrowsPremiumRequiredError() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .denied(.premiumRequired)

        // When/Then
        do {
            _ = try await sut.prepare(user: testUser)
            XCTFail("Expected premiumRequired error to be thrown")
        } catch let error as RecordingPermissionError {
            XCTAssertEqual(error, .premiumRequired)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        XCTAssertFalse(mockAudioRecorder.prepareRecordingCalled)
    }

    func testPrepare_WhenInvalidSettings_ThrowsInvalidSettingsError() async throws {
        // Given
        let testUser = createTestUser()
        let errorMessage = "Invalid sample rate"
        mockPolicyService.canStartRecordingResult = .denied(.invalidSettings(errorMessage))

        // When/Then
        do {
            _ = try await sut.prepare(user: testUser)
            XCTFail("Expected invalidSettings error to be thrown")
        } catch let error as RecordingPermissionError {
            XCTAssertEqual(error, .invalidSettings(errorMessage))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - prepare() Audio Recorder Error Tests

    func testPrepare_WhenAudioRecorderPrepareFails_ThrowsError() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.prepare(user: testUser)
            XCTFail("Expected audio recorder error to be thrown")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .notPrepared)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testPrepare_WhenAudioRecorderReturnsNil_ThrowsNotPreparedError() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = nil // No URL set

        // When/Then
        do {
            _ = try await sut.prepare(user: testUser)
            XCTFail("Expected notPrepared error to be thrown")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .notPrepared)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - start() Success Tests

    func testStart_AfterPrepare_ReturnsRecordingSession() async throws {
        // Given
        let testUser = createTestUser()
        let expectedURL = URL(fileURLWithPath: "/tmp/test_recording.m4a")
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = expectedURL

        // Prepare first
        _ = try await sut.prepare(user: testUser)

        // When
        let session = try await sut.start()

        // Then
        XCTAssertEqual(session.recordingURL, expectedURL)
        XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
    }

    func testStart_RecordingSessionHasCurrentTimestamp() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")
        _ = try await sut.prepare(user: testUser)

        // When
        let beforeStart = Date()
        let session = try await sut.start()
        let afterStart = Date()

        // Then - startedAt should be between beforeStart and afterStart
        XCTAssertGreaterThanOrEqual(session.startedAt, beforeStart)
        XCTAssertLessThanOrEqual(session.startedAt, afterStart)
    }

    func testStart_ClearsPreparedURL() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")
        _ = try await sut.prepare(user: testUser)

        // When - First start should succeed
        _ = try await sut.start()

        // Then - Second start without prepare should fail
        do {
            _ = try await sut.start()
            XCTFail("Expected notPrepared error on second start")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .notPrepared)
        }
    }

    // MARK: - start() Error Tests

    func testStart_WithoutPrepare_ThrowsNotPreparedError() async throws {
        // Given - No prepare() called

        // When/Then
        do {
            _ = try await sut.start()
            XCTFail("Expected notPrepared error to be thrown")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .notPrepared)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        XCTAssertFalse(mockAudioRecorder.startRecordingCalled)
    }

    func testStart_WhenAudioRecorderStartFails_ThrowsError() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")
        _ = try await sut.prepare(user: testUser)
        mockAudioRecorder.startRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.start()
            XCTFail("Expected recording error to be thrown")
        } catch let error as AudioRecorderError {
            XCTAssertEqual(error, .recordingFailed("Mock recording error"))
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - execute() Legacy Method Tests

    func testExecute_CombinesPrepareAndStart() async throws {
        // Given
        let testUser = createTestUser()
        let expectedURL = URL(fileURLWithPath: "/tmp/test_recording.m4a")
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = expectedURL

        // When
        let session = try await sut.execute(user: testUser)

        // Then
        XCTAssertEqual(session.recordingURL, expectedURL)
        XCTAssertTrue(mockPolicyService.canStartRecordingCalled)
        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
        XCTAssertTrue(mockAudioRecorder.startRecordingCalled)
    }

    func testExecute_WhenPermissionDenied_ThrowsWithoutStarting() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .denied(.dailyLimitExceeded)

        // When/Then
        do {
            _ = try await sut.execute(user: testUser)
            XCTFail("Expected error to be thrown")
        } catch is RecordingPermissionError {
            // Expected
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        XCTAssertFalse(mockAudioRecorder.prepareRecordingCalled)
        XCTAssertFalse(mockAudioRecorder.startRecordingCalled)
    }

    func testExecute_WhenPrepareFails_DoesNotCallStart() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingShouldFail = true

        // When/Then
        do {
            _ = try await sut.execute(user: testUser)
            XCTFail("Expected error to be thrown")
        } catch is AudioRecorderError {
            // Expected
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }

        XCTAssertTrue(mockAudioRecorder.prepareRecordingCalled)
        XCTAssertFalse(mockAudioRecorder.startRecordingCalled)
    }

    // MARK: - Multiple prepare() Calls Tests

    func testPrepare_MultipleCalls_OverwritesPreviousURL() async throws {
        // Given
        let testUser = createTestUser()
        let firstURL = URL(fileURLWithPath: "/tmp/first.m4a")
        let secondURL = URL(fileURLWithPath: "/tmp/second.m4a")
        mockPolicyService.canStartRecordingResult = .allowed

        // First prepare
        mockAudioRecorder.prepareRecordingResult = firstURL
        _ = try await sut.prepare(user: testUser)

        // Second prepare (overwrites)
        mockAudioRecorder.prepareRecordingResult = secondURL
        _ = try await sut.prepare(user: testUser)

        // When
        let session = try await sut.start()

        // Then - Should use the second URL
        XCTAssertEqual(session.recordingURL, secondURL)
    }

    // MARK: - Call Order Tests

    func testPrepareAndStart_CallOrder() async throws {
        // Given
        let testUser = createTestUser()
        mockPolicyService.canStartRecordingResult = .allowed
        mockAudioRecorder.prepareRecordingResult = URL(fileURLWithPath: "/tmp/test.m4a")

        // When
        _ = try await sut.prepare(user: testUser)
        _ = try await sut.start()

        // Then - Verify call order through timestamps
        guard let prepareTime = mockAudioRecorder.prepareRecordingCallTime,
              let startTime = mockAudioRecorder.startRecordingCallTime else {
            XCTFail("Call times not recorded")
            return
        }
        XCTAssertLessThan(prepareTime, startTime)
    }
}
