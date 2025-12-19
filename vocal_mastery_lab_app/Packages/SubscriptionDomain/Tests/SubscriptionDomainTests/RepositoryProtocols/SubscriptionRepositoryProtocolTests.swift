//
//  SubscriptionRepositoryProtocolTests.swift
//  VocalisStudioTests
//
//  Tests for SubscriptionRepositoryProtocol through mock implementation
//

import XCTest
@testable import SubscriptionDomain

final class SubscriptionRepositoryProtocolTests: XCTestCase {

    var mockRepository: MockSubscriptionRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockSubscriptionRepository()
    }

    override func tearDown() {
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Get Current Status Tests

    func testGetCurrentStatusReturnsDefaultFreeForNewUser() async throws {
        // Given: New user with no subscription
        mockRepository.mockStatus = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Get current status
        let status = try await mockRepository.getCurrentStatus()

        // Then: Should return free status
        XCTAssertEqual(status.tier, .free)
        XCTAssertEqual(status.cohort, .v2_0)
        XCTAssertTrue(status.isActive)
    }

    func testGetCurrentStatusReturnsPremiumForSubscribedUser() async throws {
        // Given: User with active Premium subscription
        let premiumStatus = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
        )
        mockRepository.mockStatus = premiumStatus

        // When: Get current status
        let status = try await mockRepository.getCurrentStatus()

        // Then: Should return premium status
        XCTAssertEqual(status.tier, .premium)
        XCTAssertTrue(status.isActive)
        XCTAssertNotNil(status.expirationDate)
    }

    // MARK: - Purchase Tests

    func testPurchaseProductSucceedsForValidProduct() async throws {
        // Given: Valid product ID
        let productId = SubscriptionTier.premium.productId
        mockRepository.mockPurchaseResult = .success(())

        // When: Purchase product
        try await mockRepository.purchaseProduct(productId: productId)

        // Then: Should succeed (no error thrown)
        XCTAssertEqual(mockRepository.lastPurchasedProductId, productId)
    }

    func testPurchaseProductThrowsErrorForInvalidProduct() async {
        // Given: Invalid product ID
        let productId = "invalid_product_id"
        mockRepository.mockPurchaseResult = .failure(SubscriptionError.productNotFound)

        // When/Then: Purchase should throw error
        do {
            try await mockRepository.purchaseProduct(productId: productId)
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.productNotFound)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - Restore Purchases Tests

    func testRestorePurchasesSucceedsWhenPurchasesExist() async throws {
        // Given: User has previous purchases
        mockRepository.mockRestoreResult = .success(())

        // When: Restore purchases
        try await mockRepository.restorePurchases()

        // Then: Should succeed
        XCTAssertTrue(mockRepository.didCallRestorePurchases)
    }

    func testRestorePurchasesThrowsErrorWhenNoPurchases() async {
        // Given: No previous purchases
        mockRepository.mockRestoreResult = .failure(SubscriptionError.noPurchasesToRestore)

        // When/Then: Should throw error
        do {
            try await mockRepository.restorePurchases()
            XCTFail("Should have thrown error")
        } catch let error as SubscriptionError {
            XCTAssertEqual(error, SubscriptionError.noPurchasesToRestore)
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
    }

    // MARK: - User Cohort Tests

    func testGetUserCohortReturnsStoredCohort() async throws {
        // Given: User cohort is v1.0
        mockRepository.mockCohort = .v1_0

        // When: Get user cohort
        let cohort = try await mockRepository.getUserCohort()

        // Then: Should return v1.0
        XCTAssertEqual(cohort, .v1_0)
    }

    func testGetUserCohortDeterminesNewUserCohort() async throws {
        // Given: New user (cohort not set)
        mockRepository.mockCohort = .v2_0 // Determined based on current date

        // When: Get user cohort
        let cohort = try await mockRepository.getUserCohort()

        // Then: Should determine and return cohort
        XCTAssertNotNil(cohort)
    }
}

// MARK: - Mock Repository

final class MockSubscriptionRepository: SubscriptionRepositoryProtocol {
    var mockStatus: SubscriptionStatus!
    var mockPurchaseResult: Result<Void, Error>!
    var mockRestoreResult: Result<Void, Error>!
    var mockCohort: UserCohort!

    var lastPurchasedProductId: String?
    var didCallRestorePurchases = false

    func getCurrentStatus() async throws -> SubscriptionStatus {
        return mockStatus
    }

    func purchaseProduct(productId: String) async throws {
        lastPurchasedProductId = productId
        try mockPurchaseResult.get()
    }

    func restorePurchases() async throws {
        didCallRestorePurchases = true
        try mockRestoreResult.get()
    }

    func getUserCohort() async throws -> UserCohort {
        return mockCohort
    }

    func getAvailableProducts() async throws -> [SubscriptionProduct] {
        return []
    }
}
