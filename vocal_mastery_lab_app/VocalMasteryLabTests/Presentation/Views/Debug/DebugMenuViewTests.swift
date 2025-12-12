//
//  DebugMenuViewTests.swift
//  VocalMasteryLabTests
//
//  ViewInspector-based tests for DebugMenuView
//

import XCTest
import SwiftUI
import ViewInspector
import VocalisDomain
import SubscriptionDomain
@testable import VocalMasteryLab

@MainActor
final class DebugMenuViewTests: XCTestCase {

    var subscriptionViewModel: SubscriptionViewModel!

    override func setUp() async throws {
        // Create subscription view model with mock dependencies
        // Note: Using mock classes defined in SubscriptionViewModelTests.swift
        let mockGetStatusUseCase = DebugMenuMockGetSubscriptionStatusUseCase()
        let mockPurchaseUseCase = DebugMenuMockPurchaseSubscriptionUseCase()
        let mockRestoreUseCase = DebugMenuMockRestorePurchasesUseCase()

        subscriptionViewModel = SubscriptionViewModel(
            getStatusUseCase: mockGetStatusUseCase,
            purchaseUseCase: mockPurchaseUseCase,
            restoreUseCase: mockRestoreUseCase
        )

        // Set initial tier to Free
        subscriptionViewModel.setDebugTier(.free)
    }

    // MARK: - Debug Tier Persistence Tests

    func testDebugTierPersistsAcrossViewUpdates() async throws {
        // Given: Initial Free tier
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier, .free)

        // When: Switch to Premium tier
        subscriptionViewModel.setDebugTier(.premium)

        // Wait for state update
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Tier should persist to Premium
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier, .premium)

        // When: Switch to Premium Plus tier
        subscriptionViewModel.setDebugTier(.premiumPlus)

        // Wait for state update
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Tier should persist to Premium Plus
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier, .premiumPlus)
    }

    func testDebugTierDisplaysCorrectName() async throws {
        // Given: Free tier
        subscriptionViewModel.setDebugTier(.free)
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Display name key should be correct (use key instead of localized string for test stability)
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier.displayNameKey, "paywall.price_free")

        // When: Switch to Premium
        subscriptionViewModel.setDebugTier(.premium)
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Display name key should be correct
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier.displayNameKey, "paywall.premium")

        // When: Switch to Premium Plus
        subscriptionViewModel.setDebugTier(.premiumPlus)
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Display name key should be correct
        XCTAssertEqual(subscriptionViewModel.currentStatus?.tier.displayNameKey, "paywall.premium_plus")
    }

    func testDebugTierStateIsAccessibleDirectly() async throws {
        // This is the key advantage of ViewInspector:
        // We can directly access ViewModel state without relying on UI elements

        // Given: Free tier
        subscriptionViewModel.setDebugTier(.free)
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: Can directly access tier without UI interaction
        let currentTier = subscriptionViewModel.currentStatus?.tier
        XCTAssertNotNil(currentTier)
        XCTAssertEqual(currentTier, .free)

        // When: Change tier
        subscriptionViewModel.setDebugTier(.premium)
        try await Task.sleep(nanoseconds: 100_000_000)

        // Then: State change is immediately observable
        let newTier = subscriptionViewModel.currentStatus?.tier
        XCTAssertNotNil(newTier)
        XCTAssertEqual(newTier, .premium)
    }
}

// MARK: - DebugMenuView-specific Mock Use Cases
// Note: Using unique names to avoid conflicts with mocks in SubscriptionViewModelTests.swift

private class DebugMenuMockGetSubscriptionStatusUseCase: GetSubscriptionStatusUseCaseProtocol {
    func execute() async throws -> SubscriptionStatus {
        return SubscriptionStatus(
            tier: .free,
            cohort: .v2_0,
            isActive: false,
            expirationDate: nil,
            purchaseDate: nil,
            willAutoRenew: false
        )
    }
}

private class DebugMenuMockPurchaseSubscriptionUseCase: PurchaseSubscriptionUseCaseProtocol {
    func execute(tier: SubscriptionTier) async throws {
        // Mock implementation
    }
}

private class DebugMenuMockRestorePurchasesUseCase: RestorePurchasesUseCaseProtocol {
    func execute() async throws {
        // Mock implementation
    }
}
