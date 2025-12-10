//
//  PurchaseSubscriptionUseCaseTests.swift
//  VocalMasteringLabTests
//
//  Tests for PurchaseSubscriptionUseCase
//

import XCTest
import SubscriptionDomain
@testable import VocalMasteringLab

final class PurchaseSubscriptionUseCaseTests: XCTestCase {

    var useCase: PurchaseSubscriptionUseCase!
    var mockRepository: MockSubscriptionRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockSubscriptionRepository()
        useCase = PurchaseSubscriptionUseCase(repository: mockRepository)
    }

    override func tearDown() {
        useCase = nil
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Basic Purchase Tests

    func testExecutePurchasesPremiumSubscription() async throws {
        // Given: User wants to purchase Premium
        let tier = SubscriptionTier.premium
        mockRepository.mockPurchaseResult = .success(())

        // When: Execute purchase
        try await useCase.execute(tier: tier)

        // Then: Should purchase correct product
        XCTAssertEqual(mockRepository.lastPurchasedProductId, tier.productId)
    }

    func testExecutePurchasesPremiumPlusSubscription() async throws {
        // Given: User wants to purchase Premium Plus
        let tier = SubscriptionTier.premiumPlus
        mockRepository.mockPurchaseResult = .success(())

        // When: Execute purchase
        try await useCase.execute(tier: tier)

        // Then: Should purchase correct product
        XCTAssertEqual(mockRepository.lastPurchasedProductId, tier.productId)
    }

    func testExecuteCannotPurchaseFreeTier() async {
        // Given: User tries to purchase Free tier
        let tier = SubscriptionTier.free

        // When/Then: Should throw error
        do {
            try await useCase.execute(tier: tier)
            XCTFail("Should have thrown error for free tier")
        } catch let error as PurchaseError {
            XCTAssertEqual(error, PurchaseError.cannotPurchaseFreeTier)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Error Handling Tests

    func testExecuteThrowsErrorWhenProductNotFound() async {
        // Given: Repository cannot find product
        mockRepository.mockPurchaseResult = .failure(SubscriptionError.productNotFound)

        // When/Then: Should propagate error
        do {
            try await useCase.execute(tier: .premium)
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.productNotFound)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testExecuteThrowsErrorWhenPurchaseFails() async {
        // Given: Purchase fails
        mockRepository.mockPurchaseResult = .failure(
            SubscriptionError.purchaseFailed("User cancelled")
        )

        // When/Then: Should propagate error
        do {
            try await useCase.execute(tier: .premium)
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            if case .purchaseFailed(let reason) = error {
                XCTAssertEqual(reason, "User cancelled")
            } else {
                XCTFail("Wrong error case")
            }
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    func testExecuteThrowsErrorWhenNetworkFails() async {
        // Given: Network error occurs
        mockRepository.mockPurchaseResult = .failure(SubscriptionError.networkError)

        // When/Then: Should propagate error
        do {
            try await useCase.execute(tier: .premium)
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.networkError)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Product ID Validation Tests

    func testExecutePassesCorrectProductIdForPremium() async throws {
        // Given: Premium tier purchase
        mockRepository.mockPurchaseResult = .success(())

        // When: Execute purchase
        try await useCase.execute(tier: .premium)

        // Then: Should use correct product ID
        XCTAssertEqual(
            mockRepository.lastPurchasedProductId,
            "com.vocalisstudio.premium.monthly"
        )
    }

    func testExecutePassesCorrectProductIdForPremiumPlus() async throws {
        // Given: Premium Plus tier purchase
        mockRepository.mockPurchaseResult = .success(())

        // When: Execute purchase
        try await useCase.execute(tier: .premiumPlus)

        // Then: Should use correct product ID
        XCTAssertEqual(
            mockRepository.lastPurchasedProductId,
            "com.vocalisstudio.premiumplus.monthly"
        )
    }
}
