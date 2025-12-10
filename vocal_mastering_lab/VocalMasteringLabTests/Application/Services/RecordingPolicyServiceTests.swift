import XCTest
import SubscriptionDomain
import VocalisDomain
@testable import VocalMasteringLab

final class RecordingPolicyServiceTests: XCTestCase {

    var sut: RecordingPolicyService!

    override func setUp() {
        super.setUp()
        sut = RecordingPolicyServiceImpl()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    // MARK: - canStartRecording Tests

    func testCanStartRecording_FreeUser_NoScale_WithinDailyLimit_ReturnsAllowed() async throws {
        // Given: Free user, no scale, within daily limit
        let user = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 2) // Under limit (3)
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user, settings: nil)

        // Then: Should be allowed
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_FreeUser_NoScale_ExceedsDailyLimit_ReturnsDenied() async throws {
        // Given: Free user, no scale, exceeds daily limit
        let user = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 5) // At limit (free = 5/day)
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user, settings: nil)

        // Then: Should be denied with daily limit reason
        if case .denied(let reason) = permission {
            XCTAssertEqual(reason, .dailyLimitExceeded)
        } else {
            XCTFail("Expected denied permission, got \(permission)")
        }
    }

    func testCanStartRecording_FreeUser_WithScale_ReturnsAllowed() async throws {
        // Given: Free user with scale recording (now available for all tiers)
        let user = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 0)
        )
        let settings = ScaleSettings.mvpDefault

        // When: Check if can start recording with scale
        let permission = try await sut.canStartRecording(user: user, settings: settings)

        // Then: Should be allowed (scale recording available for all tiers)
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_PremiumUser_WithScale_WithinDailyLimit_ReturnsAllowed() async throws {
        // Given: Premium user, with scale, within daily limit
        let user = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: true,
                expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
            ),
            recordingStats: RecordingStats(todayCount: 5) // Under premium limit (10)
        )
        let settings = ScaleSettings.mvpDefault

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user, settings: settings)

        // Then: Should be allowed
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_PremiumUser_UnlimitedRecordings_ReturnsAllowed() async throws {
        // Given: Premium user with many recordings (premium has unlimited daily count)
        let user = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: true,
                expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
            ),
            recordingStats: RecordingStats(todayCount: 100) // Many recordings
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user, settings: nil)

        // Then: Should be allowed (premium has unlimited daily count)
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_GrandfatherUser_WithScale_ReturnsAllowed() async throws {
        // Given: v1.0 Grandfather user
        let user = User(
            id: UserId(),
            subscriptionStatus: .grandfatherFree,
            recordingStats: RecordingStats(todayCount: 5)
        )
        let settings = ScaleSettings.mvpDefault

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user, settings: settings)

        // Then: Should be allowed (grandfather privileges)
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_ExpiredPremium_WithScale_ReturnsAllowed() async throws {
        // Given: Expired Premium subscription (scale recording now available for all tiers)
        let user = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: false,
                expirationDate: Date().addingTimeInterval(-24 * 3600) // Yesterday
            ),
            recordingStats: RecordingStats(todayCount: 0)
        )
        let settings = ScaleSettings.mvpDefault

        // When: Check if can start recording with scale
        let permission = try await sut.canStartRecording(user: user, settings: settings)

        // Then: Should be allowed (scale recording available for all tiers)
        XCTAssertEqual(permission, .allowed)
    }

    // MARK: - validateDuration Tests

    func testValidateDuration_FreeTier_Under30Seconds_Succeeds() throws {
        // Given: Free tier, 25 seconds duration
        let duration = Duration(seconds: 25.0)
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When/Then: Should not throw
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
    }

    func testValidateDuration_FreeTier_Exactly30Seconds_Succeeds() throws {
        // Given: Free tier, exactly 30 seconds
        let duration = Duration(seconds: 30.0)
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When/Then: Should not throw
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
    }

    func testValidateDuration_FreeTier_Over30Seconds_ThrowsError() throws {
        // Given: Free tier, 31 seconds duration
        let duration = Duration(seconds: 31.0)
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When/Then: Should throw duration limit exceeded error
        XCTAssertThrowsError(try sut.validateDuration(duration, for: status)) { error in
            guard let policyError = error as? RecordingPolicyError else {
                XCTFail("Expected RecordingPolicyError, got \(error)")
                return
            }
            if case .durationLimitExceeded(let limit) = policyError {
                XCTAssertEqual(limit.seconds, 30.0)
            } else {
                XCTFail("Expected durationLimitExceeded error, got \(policyError)")
            }
        }
    }

    func testValidateDuration_PremiumTier_Under5Minutes_Succeeds() throws {
        // Given: Premium tier, 4 minutes duration
        let duration = Duration(seconds: 240.0)
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then: Should not throw
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
    }

    func testValidateDuration_PremiumTier_Over5Minutes_ThrowsError() throws {
        // Given: Premium tier, 6 minutes duration
        let duration = Duration(seconds: 360.0)
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then: Should throw duration limit exceeded error
        XCTAssertThrowsError(try sut.validateDuration(duration, for: status)) { error in
            guard let policyError = error as? RecordingPolicyError else {
                XCTFail("Expected RecordingPolicyError, got \(error)")
                return
            }
            if case .durationLimitExceeded(let limit) = policyError {
                XCTAssertEqual(limit.seconds, 300.0)
            } else {
                XCTFail("Expected durationLimitExceeded error, got \(policyError)")
            }
        }
    }

    func testValidateDuration_PremiumPlusTier_AnyDuration_Succeeds() throws {
        // Given: Premium Plus tier, very long duration
        let duration = Duration(seconds: 10000.0)
        let status = SubscriptionStatus(
            tier: .premiumPlus,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then: Should not throw
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
    }

    func testValidateDuration_GrandfatherUser_AnyDuration_Succeeds() throws {
        // Given: Grandfather user, long duration
        let duration = Duration(seconds: 600.0)
        let status = SubscriptionStatus.grandfatherFree

        // When/Then: Should not throw (grandfather privileges)
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
    }
}
