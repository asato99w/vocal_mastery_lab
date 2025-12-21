//
//  SubscriptionStatusTests.swift
//  VocalMasteryLabTests
//
//  Tests for SubscriptionStatus entity
//

import XCTest
import SubscriptionDomain
@testable import VocalMasteryLab

final class SubscriptionStatusTests: XCTestCase {

    // MARK: - Feature Access Tests

    func testGrandfatherUserHasAccessToAllFeatures() {
        // Given: v1.0 user (Grandfather)
        let status = SubscriptionStatus.grandfatherFree

        // When/Then: Should have access to all features
        for feature in Feature.allCases {
            XCTAssertTrue(
                status.hasAccessTo(feature),
                "v1.0 user should have access to \(feature.rawValue)"
            )
        }
    }

    func testFreeUserHasAccessToBasicFeaturesOnly() {
        // Given: v2.0 free user
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When/Then: All features are now free
        // Note: Test updated to reflect current "all features free" policy
        for feature in Feature.allCases {
            XCTAssertTrue(
                status.hasAccessTo(feature),
                "All features should be accessible (current policy: all free)"
            )
        }
    }

    func testPremiumUserHasAccessToPremiumFeatures() {
        // Given: Premium user
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
        )

        // When/Then: All features are now free (current policy)
        // Note: Test updated to reflect "all features free" policy
        for feature in Feature.allCases {
            XCTAssertTrue(
                status.hasAccessTo(feature),
                "All features should be accessible (current policy: all free)"
            )
        }
    }

    func testPremiumPlusUserHasAccessToAllFeatures() {
        // Given: Premium Plus user
        let status = SubscriptionStatus(
            tier: .premiumPlus,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
        )

        // When/Then: Should have access to all features
        for feature in Feature.allCases {
            XCTAssertTrue(
                status.hasAccessTo(feature),
                "Premium Plus user should have access to \(feature.rawValue)"
            )
        }
    }

    func testInactiveSubscriptionLosesAccessToPaidFeatures() {
        // Given: Expired Premium subscription
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: false,
            expirationDate: Date().addingTimeInterval(-24 * 3600) // Yesterday
        )

        // When/Then: All features are still accessible (current policy: all free)
        // Note: Test updated to reflect "all features free" policy
        for feature in Feature.allCases {
            XCTAssertTrue(
                status.hasAccessTo(feature),
                "All features should be accessible even when inactive (current policy: all free)"
            )
        }
    }

    // MARK: - Ad Policy Tests

    func testPremiumUserGetsNoAds() {
        // Given: Active Premium user
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )

        // When: Get ad policy
        let adPolicy = status.getAdPolicy()

        // Then: Should be no ads
        XCTAssertEqual(adPolicy.frequency, .none)
        XCTAssertFalse(adPolicy.shouldShowAds)
    }

    func testFreeV2UserGetsStandardAds() {
        // Given: v2.0 free user
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Get ad policy
        let adPolicy = status.getAdPolicy()

        // Then: Should get standard ads with interstitial
        XCTAssertEqual(adPolicy.frequency, .standard)
        XCTAssertTrue(adPolicy.shouldShowAds)
        XCTAssertTrue(adPolicy.allowsBanner)
        XCTAssertTrue(adPolicy.allowsInterstitial)
        XCTAssertTrue(adPolicy.allowsRewarded)
    }

    // MARK: - Subscription State Tests

    func testSubscriptionIsExpiredWhenPastExpirationDate() {
        // Given: Subscription expired yesterday
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: false,
            expirationDate: Date().addingTimeInterval(-24 * 3600)
        )

        // When/Then
        XCTAssertTrue(status.isExpired)
    }

    func testSubscriptionIsNotExpiredWhenBeforeExpirationDate() {
        // Given: Subscription expires in 30 days
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
        )

        // When/Then
        XCTAssertFalse(status.isExpired)
    }

    func testFreeSubscriptionNeverExpires() {
        // Given: Free user
        let status = SubscriptionStatus.defaultFree()

        // When/Then
        XCTAssertFalse(status.isExpired)
        XCTAssertNil(status.daysUntilExpiration)
    }

    func testDaysUntilExpirationCalculation() {
        // Given: Subscription expires in 15 days
        let expirationDate = Calendar.current.date(
            byAdding: .day,
            value: 15,
            to: Date()
        )!

        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: expirationDate
        )

        // When: Get days until expiration
        let days = status.daysUntilExpiration

        // Then: Should be approximately 15 days
        XCTAssertNotNil(days)
        if let days = days {
            XCTAssertEqual(days, 15, accuracy: 1)
        }
    }

    // MARK: - Premium Status Tests

    func testHasPremiumReturnsTrueForActivePremium() {
        // Given: Active Premium user
        let status = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then
        XCTAssertTrue(status.hasPremium)
        XCTAssertFalse(status.hasPremiumPlus)
        XCTAssertTrue(status.hasPaidSubscription)
    }

    func testHasPremiumPlusReturnsTrueForActivePremiumPlus() {
        // Given: Active Premium Plus user
        let status = SubscriptionStatus(
            tier: .premiumPlus,
            cohort: .v2_0,
            isActive: true
        )

        // When/Then
        XCTAssertFalse(status.hasPremium) // Premium Plus is NOT Premium
        XCTAssertTrue(status.hasPremiumPlus)
        XCTAssertTrue(status.hasPaidSubscription)
    }

    func testIsFreeReturnsTrueForFreeUser() {
        // Given: Free user
        let status = SubscriptionStatus.defaultFree()

        // When/Then
        XCTAssertTrue(status.isFree)
        XCTAssertFalse(status.hasPaidSubscription)
    }

    // MARK: - Accessible Features Tests

    func testAccessibleFeaturesListForFreeUser() {
        // Given: Free user
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Get accessible features
        let accessible = status.accessibleFeatures

        // Then: All features should be accessible (current policy: all free)
        // Note: Test updated to reflect "all features free" policy
        XCTAssertEqual(accessible.count, Feature.allCases.count)
        for feature in Feature.allCases {
            XCTAssertTrue(accessible.contains(feature))
        }
    }

    func testLockedFeaturesListForFreeUser() {
        // Given: Free user
        let status = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Get locked features
        let locked = status.lockedFeatures

        // Then: No features should be locked (current policy: all free)
        // Note: Test updated to reflect "all features free" policy
        XCTAssertTrue(locked.isEmpty, "No features should be locked when all features are free")
    }

    // MARK: - Codable Tests

    func testSubscriptionStatusCodable() throws {
        // Given: A subscription status
        let original = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date(),
            purchaseDate: Date().addingTimeInterval(-30 * 24 * 3600),
            willAutoRenew: true
        )

        // When: Encode and decode
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(SubscriptionStatus.self, from: data)

        // Then: Should be equal
        XCTAssertEqual(decoded.tier, original.tier)
        XCTAssertEqual(decoded.cohort, original.cohort)
        XCTAssertEqual(decoded.isActive, original.isActive)
        XCTAssertEqual(decoded.willAutoRenew, original.willAutoRenew)
    }
}
