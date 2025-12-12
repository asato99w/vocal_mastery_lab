import XCTest
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

final class RecordingLimitTests: XCTestCase {

    // MARK: - Duration Limit Tests

    func testIsWithinDurationLimit_FreeTier_Under30Seconds_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.free)
        let duration: TimeInterval = 25.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result)
    }

    func testIsWithinDurationLimit_FreeTier_Exactly30Seconds_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.free)
        let duration: TimeInterval = 30.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result)
    }

    func testIsWithinDurationLimit_FreeTier_Over30Seconds_ReturnsFalse() {
        // Given
        let limit = RecordingLimit.forTier(.free)
        let duration: TimeInterval = 31.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertFalse(result)
    }

    func testIsWithinDurationLimit_PremiumTier_Under5Minutes_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premium)
        let duration: TimeInterval = 250.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result)
    }

    func testIsWithinDurationLimit_PremiumTier_Exactly5Minutes_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premium)
        let duration: TimeInterval = 300.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result)
    }

    func testIsWithinDurationLimit_PremiumTier_Over5Minutes_ReturnsFalse() {
        // Given
        let limit = RecordingLimit.forTier(.premium)
        let duration: TimeInterval = 301.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertFalse(result)
    }

    func testIsWithinDurationLimit_PremiumPlusTier_AnyDuration_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premiumPlus)
        let duration: TimeInterval = 10000.0

        // When
        let result = limit.isWithinDurationLimit(duration)

        // Then
        XCTAssertTrue(result)
    }

    // MARK: - Bug Detection Tests: Count Boundary Values

    /// BUG DETECTION TEST: Count at exactly the limit
    /// isCountWithinLimit uses `count < limit`, so count == limit should return false
    func testIsCountWithinLimit_FreeTier_AtExactLimit_ReturnsFalse() {
        // Given: Free tier has 5 daily recordings
        let limit = RecordingLimit.forTier(.free)

        // When: Count is exactly at the limit (5)
        let result = limit.isCountWithinLimit(5)

        // Then: Should return false (can't record more)
        XCTAssertFalse(result, "At exact limit (5), should return false")
    }

    /// BUG DETECTION TEST: Count one below limit
    func testIsCountWithinLimit_FreeTier_OneBelowLimit_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: Count is one below limit (4)
        let result = limit.isCountWithinLimit(4)

        // Then: Should return true (can still record)
        XCTAssertTrue(result, "One below limit (4), should return true")
    }

    /// BUG DETECTION TEST: Count one above limit
    func testIsCountWithinLimit_FreeTier_OneAboveLimit_ReturnsFalse() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: Count is above limit (6)
        let result = limit.isCountWithinLimit(6)

        // Then: Should return false
        XCTAssertFalse(result, "Above limit (6), should return false")
    }

    /// BUG DETECTION TEST: Count at zero
    func testIsCountWithinLimit_FreeTier_ZeroCount_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: No recordings yet
        let result = limit.isCountWithinLimit(0)

        // Then: Should return true
        XCTAssertTrue(result, "Zero count should return true")
    }

    /// BUG DETECTION TEST: Premium tier unlimited count
    func testIsCountWithinLimit_PremiumTier_AnyCount_ReturnsTrue() {
        // Given: Premium has unlimited daily count (nil)
        let limit = RecordingLimit.forTier(.premium)

        // When: Very high count
        let result = limit.isCountWithinLimit(Int.max - 1)

        // Then: Should return true (unlimited)
        XCTAssertTrue(result, "Premium unlimited should always return true")
    }

    /// BUG DETECTION TEST: PremiumPlus tier unlimited count
    func testIsCountWithinLimit_PremiumPlusTier_AnyCount_ReturnsTrue() {
        // Given
        let limit = RecordingLimit.forTier(.premiumPlus)

        // When
        let result = limit.isCountWithinLimit(1000000)

        // Then
        XCTAssertTrue(result, "PremiumPlus unlimited should always return true")
    }

    /// BUG DETECTION TEST: Negative count (edge case)
    func testIsCountWithinLimit_NegativeCount_ShouldReturnTrue() {
        // Given: This is an edge case that shouldn't happen in practice
        let limit = RecordingLimit.forTier(.free)

        // When: Negative count (invalid but test behavior)
        let result = limit.isCountWithinLimit(-1)

        // Then: Should return true (negative < 5)
        XCTAssertTrue(result, "Negative count should return true (less than limit)")
    }

    // MARK: - Bug Detection Tests: remainingCount Display

    /// BUG DETECTION TEST: remainingCount at boundary
    func testRemainingCount_AtExactLimit_ShowsZero() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: At exact limit
        let result = limit.remainingCount(5)

        // Then: Should show 0/5
        XCTAssertEqual(result, "0/5")
    }

    /// BUG DETECTION TEST: remainingCount over limit (should not go negative)
    func testRemainingCount_OverLimit_ShowsZeroNotNegative() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When: Over the limit
        let result = limit.remainingCount(10)

        // Then: Should show 0/5, not -5/5
        XCTAssertEqual(result, "0/5", "Should clamp to 0, not show negative")
    }

    /// BUG DETECTION TEST: remainingCount for unlimited tier
    func testRemainingCount_UnlimitedTier_ShowsUnlimited() {
        // Given
        let limit = RecordingLimit.forTier(.premium)

        // When
        let result = limit.remainingCount(100)

        // Then: Should show "無制限"
        XCTAssertEqual(result, "無制限")
    }

    // MARK: - Bug Detection Tests: Duration Description

    /// BUG DETECTION TEST: durationDescription for minutes
    func testDurationDescription_FreeTier_Shows30Seconds() {
        // Given
        let limit = RecordingLimit.forTier(.free)

        // When
        let result = limit.durationDescription

        // Then: 30 seconds should show as "30秒"
        XCTAssertEqual(result, "30秒")
    }

    /// BUG DETECTION TEST: durationDescription for 5 minutes
    func testDurationDescription_PremiumTier_Shows5Minutes() {
        // Given
        let limit = RecordingLimit.forTier(.premium)

        // When
        let result = limit.durationDescription

        // Then: 300 seconds should show as "5分"
        XCTAssertEqual(result, "5分")
    }

    /// BUG DETECTION TEST: durationDescription edge case at exactly 60 seconds
    func testDurationDescription_Exactly60Seconds_Shows1Minute() {
        // Given: Custom configuration with 60 second limit
        let config = RecordingLimit.Configuration(
            freeDailyCount: 5,
            freeMaxDuration: 60,
            premiumDailyCount: .max,
            premiumMaxDuration: 300
        )
        let limit = RecordingLimit.forTier(.free, configuration: config)

        // When
        let result = limit.durationDescription

        // Then: Should show "1分", not "60秒"
        XCTAssertEqual(result, "1分")
    }

    /// BUG DETECTION TEST: durationDescription for unlimited
    func testDurationDescription_PremiumPlusTier_ShowsUnlimited() {
        // Given
        let limit = RecordingLimit.forTier(.premiumPlus)

        // When
        let result = limit.durationDescription

        // Then
        XCTAssertEqual(result, "無制限")
    }
}
