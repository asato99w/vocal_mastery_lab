import XCTest
import SubscriptionDomain
import VocalisDomain
@testable import VocalMasteryLab

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

    func testCanStartRecording_FreeUser_WithinDailyLimit_ReturnsAllowed() async throws {
        // Given: Free user within daily limit
        let user = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 2) // Under limit
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user)

        // Then: Should be allowed
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_FreeUser_ExceedsDailyLimit_ReturnsAllowed() async throws {
        // Given: Free user exceeds daily limit
        // Note: Test updated to reflect "all features free" policy - no limits
        let user = User(
            id: UserId(),
            subscriptionStatus: .defaultFree(cohort: .v2_0),
            recordingStats: RecordingStats(todayCount: 5) // Was at limit, now unlimited
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user)

        // Then: Should be allowed (current policy: all free, unlimited recording)
        XCTAssertEqual(permission, .allowed, "Should be allowed (current policy: unlimited recording)")
    }

    func testCanStartRecording_PremiumUser_WithinDailyLimit_ReturnsAllowed() async throws {
        // Given: Premium user within daily limit
        let user = User(
            id: UserId(),
            subscriptionStatus: SubscriptionStatus(
                tier: .premium,
                cohort: .v2_0,
                isActive: true,
                expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
            ),
            recordingStats: RecordingStats(todayCount: 5) // Under premium limit
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user)

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
        let permission = try await sut.canStartRecording(user: user)

        // Then: Should be allowed (premium has unlimited daily count)
        XCTAssertEqual(permission, .allowed)
    }

    func testCanStartRecording_GrandfatherUser_ReturnsAllowed() async throws {
        // Given: v1.0 Grandfather user
        let user = User(
            id: UserId(),
            subscriptionStatus: .grandfatherFree,
            recordingStats: RecordingStats(todayCount: 5)
        )

        // When: Check if can start recording
        let permission = try await sut.canStartRecording(user: user)

        // Then: Should be allowed (grandfather privileges)
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

    func testValidateDuration_FreeTier_AnyDuration_Succeeds() throws {
        // Given: Free tier, any duration (current policy: unlimited)
        // Note: Test updated to reflect "all features free" policy - no limits
        let duration = Duration(seconds: 10000.0)
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When/Then: Should not throw (current policy: unlimited)
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
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

    func testValidateDuration_PremiumTier_AnyDuration_Succeeds() throws {
        // Given: Premium tier, any duration (current policy: unlimited)
        // Note: Test updated to reflect "all features free" policy - no limits
        let duration = Duration(seconds: 10000.0)
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then: Should not throw (current policy: unlimited)
        XCTAssertNoThrow(try sut.validateDuration(duration, for: status))
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
