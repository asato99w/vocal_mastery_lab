//
//  GetSubscriptionStatusUseCaseTests.swift
//  VocalMasteryLabTests
//
//  Tests for GetSubscriptionStatusUseCase
//

import XCTest
import SubscriptionDomain
@testable import VocalMasteryLab

final class GetSubscriptionStatusUseCaseTests: XCTestCase {

    var useCase: GetSubscriptionStatusUseCase!
    var mockRepository: MockSubscriptionRepository!

    override func setUp() {
        super.setUp()
        mockRepository = MockSubscriptionRepository()
        useCase = GetSubscriptionStatusUseCase(repository: mockRepository)
    }

    override func tearDown() {
        useCase = nil
        mockRepository = nil
        super.tearDown()
    }

    // MARK: - Basic Execution Tests

    func testExecuteReturnsCurrentStatusFromRepository() async throws {
        // Given: Repository has a Premium subscription
        let expectedStatus = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: Date().addingTimeInterval(30 * 24 * 3600)
        )
        mockRepository.mockStatus = expectedStatus

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should return the status from repository
        XCTAssertEqual(result.tier, expectedStatus.tier)
        XCTAssertEqual(result.cohort, expectedStatus.cohort)
        XCTAssertEqual(result.isActive, expectedStatus.isActive)
    }

    func testExecuteReturnsDefaultFreeForNewUser() async throws {
        // Given: New user with no subscription
        mockRepository.mockStatus = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should return free tier status
        XCTAssertEqual(result.tier, .free)
        XCTAssertEqual(result.cohort, .v2_0)
        XCTAssertTrue(result.isActive)
    }

    func testExecuteReturnsGrandfatherFreeForV1User() async throws {
        // Given: v1.0 user (Grandfather clause)
        mockRepository.mockStatus = SubscriptionStatus.grandfatherFree

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should return v1.0 cohort with all feature access
        XCTAssertEqual(result.tier, .free)
        XCTAssertEqual(result.cohort, .v1_0)
        XCTAssertTrue(result.isActive)
    }

    // MARK: - Error Handling Tests

    func testExecuteThrowsErrorWhenRepositoryFails() async {
        // Given: Repository throws error
        mockRepository.mockStatus = SubscriptionStatus.defaultFree()
        // Simulate error by making repository throw (will implement in mock)

        // When/Then: Should propagate error
        // Note: This test will be expanded when we add error handling to mock
    }

    // MARK: - Feature Access Tests

    func testExecuteReturnsStatusWithCorrectFeatureAccess() async throws {
        // Given: Premium user
        let premiumStatus = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true
        )
        mockRepository.mockStatus = premiumStatus

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: All features should be accessible (current policy: all free)
        // Note: Test updated to reflect "all features free" policy
        XCTAssertTrue(result.hasAccessTo(.basicRecording))
        XCTAssertTrue(result.hasAccessTo(.spectrumVisualization))
        XCTAssertTrue(result.hasAccessTo(.aiPitchSuggestions)) // Now accessible to all
    }

    func testExecuteReturnsStatusWithCorrectAdPolicy() async throws {
        // Given: Free user
        mockRepository.mockStatus = SubscriptionStatus.defaultFree(cohort: .v2_0)

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should have standard ads
        let adPolicy = result.getAdPolicy()
        XCTAssertTrue(adPolicy.shouldShowAds)
        XCTAssertEqual(adPolicy.frequency, .standard)
    }

    // MARK: - Expiration Tests

    func testExecuteReturnsExpiredStatusCorrectly() async throws {
        // Given: Expired Premium subscription
        let expiredStatus = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: false,
            expirationDate: Date().addingTimeInterval(-24 * 3600) // Yesterday
        )
        mockRepository.mockStatus = expiredStatus

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should be expired and inactive
        XCTAssertTrue(result.isExpired)
        XCTAssertFalse(result.isActive)
        XCTAssertFalse(result.hasPremium)
    }

    func testExecuteReturnsActiveSubscriptionWithDaysUntilExpiration() async throws {
        // Given: Active subscription expiring in 15 days
        let expirationDate = Calendar.current.date(
            byAdding: .day,
            value: 15,
            to: Date()
        )!
        let activeStatus = SubscriptionStatus(
            tier: .premium,
            cohort: .v2_0,
            isActive: true,
            expirationDate: expirationDate
        )
        mockRepository.mockStatus = activeStatus

        // When: Execute use case
        let result = try await useCase.execute()

        // Then: Should have correct days until expiration
        XCTAssertFalse(result.isExpired)
        XCTAssertNotNil(result.daysUntilExpiration)
        if let days = result.daysUntilExpiration {
            XCTAssertEqual(days, 15, accuracy: 1)
        }
    }
}
