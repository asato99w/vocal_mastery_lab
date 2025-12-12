//
//  StoreKitPurchaseTests.swift
//  VocalisStudioTests
//
//  StoreKit Testing for subscription purchase flow
//

import XCTest
import StoreKitTest
import SubscriptionDomain
@testable import VocalisStudio

@available(iOS 15.0, *)
final class StoreKitPurchaseTests: XCTestCase {

    var session: SKTestSession!
    var sut: SubscriptionViewModel!

    override func setUp() async throws {
        try await super.setUp()

        // Initialize StoreKit test session
        session = try SKTestSession(configurationFileNamed: "Configuration")
        session.clearTransactions()
        session.disableDialogs = true
        session.resetToDefaultState()

        // Set fast renewal rate for testing (1 hour real time = 1 month subscription)
        // Options: .oneSecond, .thirtySeconds, .oneMinute, .fiveMinutes, .oneHour
        session.timeRate = .oneSecond

        // Initialize ViewModel with real use cases (they will use StoreKit test environment)
        sut = SubscriptionViewModel(
            getStatusUseCase: GetSubscriptionStatusUseCase(
                subscriptionRepository: AppStoreSubscriptionRepository()
            ),
            purchaseUseCase: PurchaseSubscriptionUseCase(
                subscriptionRepository: AppStoreSubscriptionRepository()
            ),
            restoreUseCase: RestorePurchasesUseCase(
                subscriptionRepository: AppStoreSubscriptionRepository()
            )
        )
    }

    override func tearDown() async throws {
        session.clearTransactions()
        session = nil
        sut = nil
        try await super.tearDown()
    }

    // MARK: - Purchase Flow Tests

    func testPurchaseMonthlySubscription_shouldUnlockPremium() async throws {
        // Given: User is on free tier
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .free, "Should start as free tier")

        // When: User purchases monthly subscription
        await sut.purchaseSelectedTier()

        // Then: User should be upgraded to premium
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium, "Should be upgraded to premium")
        XCTAssertTrue(sut.currentStatus?.isActive ?? false, "Subscription should be active")
        XCTAssertNotNil(sut.currentStatus?.expirationDate, "Should have expiration date")
    }

    func testPurchase_shouldUpdateUIImmediately() async throws {
        // Given: User is on free tier
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .free)

        // When: User purchases subscription
        let expectation = XCTestExpectation(description: "Status updated")

        Task {
            await sut.purchaseSelectedTier()
            await sut.loadStatus()

            if sut.currentStatus?.tier == .premium {
                expectation.fulfill()
            }
        }

        // Then: UI should update within reasonable time
        await fulfillment(of: [expectation], timeout: 5.0)
    }

    // MARK: - Restore Tests

    func testRestorePurchases_withExistingPurchase_shouldRestoreAccess() async throws {
        // Given: User has purchased subscription (simulated by StoreKit test)
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")

        // Clear local state to simulate app reinstall
        session.clearTransactions()

        // When: User taps restore button
        await sut.restorePurchases()

        // Then: Premium access should be restored
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium, "Should restore premium tier")
        XCTAssertTrue(sut.currentStatus?.isActive ?? false, "Restored subscription should be active")
    }

    func testRestorePurchases_withoutPurchase_shouldRemainFree() async throws {
        // Given: User has never purchased (fresh install)
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .free)

        // When: User taps restore button
        await sut.restorePurchases()

        // Then: Should remain on free tier
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .free, "Should remain free tier")
    }

    // MARK: - Cancellation & Expiration Tests

    func testSubscriptionExpiration_shouldRevertToFree() async throws {
        // Given: User has active subscription
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium)

        // When: Subscription expires (fast-forward time)
        try await session.expireSubscription(productIdentifier: "com.kazuasato.VocalisStudio.premium.monthly")

        // Then: Should revert to free tier
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .free, "Should revert to free after expiration")
        XCTAssertFalse(sut.currentStatus?.isActive ?? true, "Should not be active")
    }

    func testCancelSubscription_beforeRenewal_shouldContinueUntilExpiration() async throws {
        // Given: User has active subscription
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")
        await sut.loadStatus()
        let initialExpirationDate = sut.currentStatus?.expirationDate

        // When: User cancels auto-renewal (but subscription hasn't expired yet)
        // Note: In StoreKit test, we simulate by disabling renewal
        session.disableAutoRenewForProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")

        // Then: Should still have access until expiration
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium, "Should keep premium until expiration")
        XCTAssertFalse(sut.currentStatus?.willAutoRenew ?? true, "Auto-renewal should be disabled")
        XCTAssertEqual(sut.currentStatus?.expirationDate, initialExpirationDate, "Expiration date unchanged")
    }

    // MARK: - Error Handling Tests

    func testPurchase_whenNetworkError_shouldShowError() async throws {
        // Given: StoreKit configured to fail purchases
        session.failTransactionsEnabled = true
        session.failureError = .networkError(.networkConnectionLost)

        // When: User attempts purchase
        await sut.purchaseSelectedTier()

        // Then: Should show error message
        XCTAssertNotNil(sut.errorMessage, "Should have error message")
        XCTAssertTrue(sut.errorMessage?.contains("ネットワーク") ?? false, "Error should mention network")

        // Cleanup
        session.failTransactionsEnabled = false
    }

    func testPurchase_whenUserCancels_shouldNotShowError() async throws {
        // Given: User will cancel purchase dialog
        session.failTransactionsEnabled = true
        session.failureError = .userCancelled

        // When: User cancels purchase
        await sut.purchaseSelectedTier()

        // Then: Should not show error (cancellation is normal)
        XCTAssertNil(sut.errorMessage, "Should not show error for user cancellation")
        XCTAssertEqual(sut.currentStatus?.tier, .free, "Should remain free")

        // Cleanup
        session.failTransactionsEnabled = false
    }

    // MARK: - Recording Limit Integration Tests

    func testRecordingLimit_afterPurchase_shouldBeUnlimited() async throws {
        // Given: User is on free tier with 5 recordings/day limit
        await sut.loadStatus()
        let freeLimit = RecordingLimit.forTier(.free)
        XCTAssertEqual(freeLimit.dailyCount, 100, "Free tier should have count limit (100 for testing)")
        XCTAssertEqual(freeLimit.maxDuration, 30, "Free tier should have 30s duration limit")

        // When: User purchases premium
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")
        await sut.loadStatus()

        // Then: Recording limits should be lifted
        let premiumLimit = RecordingLimit.forTier(.premium)
        XCTAssertNil(premiumLimit.dailyCount, "Premium should have unlimited recordings")
        XCTAssertEqual(premiumLimit.maxDuration, 300, "Premium should have 5min duration limit")
    }

    // MARK: - Subscription Group Tests

    func testUpgrade_fromMonthlyToYearly_shouldReplaceSubscription() async throws {
        // Given: User has monthly subscription
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium)

        // When: User upgrades to yearly
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.yearly")

        // Then: Should have yearly subscription (monthly should be replaced)
        let transactions = try await session.allTransactions()
        let activeSubscriptions = transactions.filter { !$0.revocationDate.map { _ in true } ?? false }
        XCTAssertEqual(activeSubscriptions.count, 1, "Should have only one active subscription")

        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium, "Should maintain premium tier")
    }

    // MARK: - Auto-Renewal Tests

    func testAutoRenewal_whenEnabled_shouldRenewAutomatically() async throws {
        // Given: User has subscription with auto-renewal enabled
        try await session.buyProduct(identifier: "com.kazuasato.VocalisStudio.premium.monthly")
        await sut.loadStatus()
        XCTAssertTrue(sut.currentStatus?.willAutoRenew ?? false)

        // When: Subscription period ends (simulate fast-forward)
        // Note: With timeRate = .oneSecond, 1 second = 1 month
        try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds = 2 months

        // Then: Should automatically renew
        await sut.loadStatus()
        XCTAssertEqual(sut.currentStatus?.tier, .premium, "Should auto-renew")
        XCTAssertTrue(sut.currentStatus?.isActive ?? false, "Should remain active")
    }
}
