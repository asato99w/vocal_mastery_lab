import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

/// Recording Limit Tests
/// Note: Currently all tiers have unlimited recording. Tests updated to reflect "all free" policy.
/// Structure preserved for future paid plan restoration.
final class RecordingLimitTests: XCTestCase {

    // MARK: - Duration Limit Tests (All Unlimited)

    func testIsWithinDurationLimit_FreeTier_AnyDuration_ReturnsTrue() {
        // Given: All tiers now have unlimited duration (current policy: all free)
        let limit = RecordingLimit.forTier(.free)
        let duration: TimeInterval = 10000.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then: All durations are within limit (unlimited)
        XCTAssertTrue(result, "All durations should be within limit (current policy: all free)")
    }

    func testIsWithinDurationLimit_PremiumTier_AnyDuration_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premium)
        let duration: TimeInterval = 10000.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result, "All durations should be within limit (current policy: all free)")
    }

    func testIsWithinDurationLimit_PremiumPlusTier_AnyDuration_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premiumPlus)
        let duration: TimeInterval = 10000.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result, "All durations should be within limit (unlimited)")
    }

    // MARK: - Count Limit Tests (All Unlimited)

    func testIsCountWithinLimit_FreeTier_AnyCount_ReturnsTrue() {
        // Given: All tiers now have unlimited daily count (current policy: all free)
        let limit = RecordingLimit.forTier(.free)

        // When: Very high count
        let result = limit.isCountWithinLimit(Int.max - 1)

        // Then: Should return true (unlimited)
        XCTAssertTrue(result, "All counts should be within limit (current policy: all free)")
    }

    func testIsCountWithinLimit_PremiumTier_AnyCount_ReturnsTrue() {
        // Given: Premium has unlimited daily count
        let limit = RecordingLimit.forTier(.premium)

        // When: Very high count
        let result = limit.isCountWithinLimit(Int.max - 1)

        // Then: Should return true (unlimited)
        XCTAssertTrue(result, "Premium unlimited should always return true")
    }

    func testIsCountWithinLimit_PremiumPlusTier_AnyCount_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premiumPlus)

        // When
        let result = limit.isCountWithinLimit(1000000)

        // Then
        XCTAssertTrue(result, "PremiumPlus unlimited should always return true")
    }

    func testIsCountWithinLimit_ZeroCount_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: No recordings yet
        let result = limit.isCountWithinLimit(0)

        // Then: Should return true
        XCTAssertTrue(result, "Zero count should return true")
    }

    func testIsCountWithinLimit_NegativeCount_ShouldReturnTrue() {
        // Given: This is an edge case that shouldn't happen in practice
        let limit = RecordingLimit.forTier(.free)

        // When: Negative count (invalid but test behavior)
        let result = limit.isCountWithinLimit(-1)

        // Then: Should return true (unlimited)
        XCTAssertTrue(result, "Negative count should return true (unlimited)")
    }

    // MARK: - remainingCount Display Tests (All Unlimited)

    func testRemainingCount_AllTiers_ShowsUnlimited() {
        // Given: All tiers are now unlimited (current policy: all free)
        let freeTier = RecordingLimit.forTier(.free)
        let premiumTier = RecordingLimit.forTier(.premium)
        let premiumPlusTier = RecordingLimit.forTier(.premiumPlus)

        // When/Then: All should show "無制限"
        XCTAssertEqual(freeTier.remainingCount(100), "無制限")
        XCTAssertEqual(premiumTier.remainingCount(100), "無制限")
        XCTAssertEqual(premiumPlusTier.remainingCount(100), "無制限")
    }

    // MARK: - Duration Description Tests (All Unlimited)

    func testDurationDescription_AllTiers_ShowsUnlimited() {
        // Given: All tiers are now unlimited (current policy: all free)
        let freeTier = RecordingLimit.forTier(.free)
        let premiumTier = RecordingLimit.forTier(.premium)
        let premiumPlusTier = RecordingLimit.forTier(.premiumPlus)

        // When/Then: All should show "無制限"
        XCTAssertEqual(freeTier.durationDescription, "無制限")
        XCTAssertEqual(premiumTier.durationDescription, "無制限")
        XCTAssertEqual(premiumPlusTier.durationDescription, "無制限")
    }

    // MARK: - Limit Properties Tests

    func testAllTiers_HaveNoLimits() {
        // Given: All tiers now have unlimited recording (current policy: all free)
        let freeTier = RecordingLimit.forTier(.free)
        let premiumTier = RecordingLimit.forTier(.premium)
        let premiumPlusTier = RecordingLimit.forTier(.premiumPlus)

        // When/Then: All limits should be nil (unlimited)
        XCTAssertNil(freeTier.dailyCount, "Free tier should have unlimited daily count")
        XCTAssertNil(freeTier.maxDuration, "Free tier should have unlimited duration")

        XCTAssertNil(premiumTier.dailyCount, "Premium tier should have unlimited daily count")
        XCTAssertNil(premiumTier.maxDuration, "Premium tier should have unlimited duration")

        XCTAssertNil(premiumPlusTier.dailyCount, "PremiumPlus tier should have unlimited daily count")
        XCTAssertNil(premiumPlusTier.maxDuration, "PremiumPlus tier should have unlimited duration")
    }
}
